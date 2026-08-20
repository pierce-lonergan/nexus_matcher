"""
tests.unit.presentation.api.test_score_contract | Layer: TEST
The three things a response has to say about itself before its numbers mean anything.

## Relationships
# TESTS → presentation/api/matching :: absoluteScore, fieldDecisions, scoring
# TESTS → presentation/api/schemas  :: the published models for all three

## What is under test, and why each piece is not optional

**An absolute score on every candidate.** `confidence` is min-max normalised inside one
field's shortlist, so its rank-1 value has a structural floor -- 0.63 for the shipped
configuration -- and a field whose best candidate is unrelated still scores above it.
There was exactly one number in this response comparable across fields, and it was behind
an opt-in `explain` block, so the common case was a client thresholding on the one number
that cannot carry a threshold.

**A field-level verdict.** The consumer writes one decision per column. `decision` is per
CANDIDATE, so rolling it up was a rule each client invented for itself -- and the rule
could not express "nothing matched" at all, because rank 1 cannot be REJECT while the
confidence floor (0.63) sits above the review threshold (0.50).

**A scale contract.** The library documents `confidence` as rank-relative and says do not
threshold on it, then ships `auto_approve_threshold = 0.87`, a threshold on it. A consumer
cannot act on both statements. The `scoring` block is where that is settled, in the
response rather than in a docstring nobody downstream reads.

The numbers below come from the fictional Lumenport glossary in `_support.py`. Its field
`misc.zzz_unmatchable` is named for the property this file needs: nothing in the glossary
describes it, and it still comes back REVIEW at confidence 0.71 with an absolute
similarity of 0.12.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.application.use_cases.match_schema import MatchingConfig
from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.schemas import MatchResponseView
from tests.unit.presentation.api._support import (
    GLOSSARY,
    STAND_IN_ABSOLUTE_COSINE,
    FakeMatcher,
    build_api_matcher,
    request_fields,
)

# The field nothing in the fictional glossary describes, and the one this file leans on.
UNMATCHABLE = "misc.zzz_unmatchable"

# A floor above what `misc.zzz_unmatchable` can reach (~0.12) and below what every other
# field in the fixture reaches (0.64 and up). Chosen from the measured spread rather than
# picked round, because a floor that separates nothing proves nothing.
FLOOR = 0.30

SCORING_KEYS = (
    "confidenceFloor",
    "absoluteScoreFloor",
    "absoluteScoreMetric",
    "absoluteScorePooledOverAliases",
    "thresholdableAcrossFields",
    "comparabilityScopesNarrowestFirst",
    "comparability",
)


def client_for(matcher: object, **kwargs: object) -> TestClient:
    app = create_app(configure_logs=False, matcher=matcher, environ={}, **kwargs)
    return TestClient(app)


def post_match(client: TestClient, **body: object):
    payload = {"fields": request_fields()}
    payload.update(body)
    return client.post("/api/v1/match", json=payload)


@pytest.fixture
def real_client():
    with client_for(build_api_matcher()) as client:
        yield client


@pytest.fixture
def floored_client():
    """The same real matcher, with an absolute floor the caller chose."""
    with client_for(build_api_matcher(config=MatchingConfig(absolute_score_floor=FLOOR))) as client:
        yield client


# =============================================================================
# SC-1 -- AN ABSOLUTE SCORE ON EVERY CANDIDATE
# =============================================================================


class TestAbsoluteScore:
    """Present without `explain`, and the same number `explain` has always reported."""

    def test_every_candidate_carries_it_without_asking_for_explain(self, real_client):
        """
        WITHOUT `explain`. That is the whole change: the number existed, and the request
        that gets it was the one a batch client does not send, because `explain` also
        carries five components and a weight map per candidate.
        """
        body = post_match(real_client, top_k=5).json()
        candidates = [c for row in body["results"].values() for c in row]

        assert candidates, "no candidates at all, so nothing was proven"
        for candidate in candidates:
            assert "explain" not in candidate
            assert "absoluteScore" in candidate, (
                "the key was omitted rather than nulled, so a client cannot tell 'the "
                "dense arm did not return this' from 'this response did not say'"
            )

    def test_it_is_the_same_number_explain_reports(self, real_client):
        """
        Two paths to one quantity. A response that reported the same similarity twice and
        disagreed with itself in the sixth decimal would be a governance artifact arguing
        with itself -- and `explain.absoluteCosine` is kept precisely because clients read
        it, so the duplication has to be exact rather than approximately exact.
        """
        body = post_match(real_client, top_k=5, explain=True).json()
        pairs = [
            (c["absoluteScore"], c["explain"]["absoluteCosine"])
            for row in body["results"].values()
            for c in row
        ]

        assert pairs
        for promoted, explained in pairs:
            assert promoted == explained

    def test_it_is_not_the_confidence(self, real_client):
        """
        The distinction the feature exists for, on the one field that shows it.

        `misc.zzz_unmatchable` matches nothing, and its rank-1 confidence is above the
        0.63 structural floor -- it looks like a real, if unexciting, match. The absolute
        score is an order of magnitude lower and is what says the shortlist was bad.
        A response carrying only the first number cannot express that.
        """
        top = post_match(real_client).json()["results"][UNMATCHABLE][0]

        assert top["confidence"] > 0.63
        assert top["decision"] != "REJECT"
        assert top["absoluteScore"] < 0.2
        assert top["absoluteScore"] < top["confidence"] / 3

    def test_null_when_the_dense_arm_did_not_return_the_candidate(self):
        """
        Null, not zero. Zero is a similarity the retriever measured; null is one it never
        took, because the candidate reached the shortlist through the lexical arm alone.
        Collapsing them would let a client read "we looked and found nothing in common"
        where the truth is "we did not look".
        """
        with client_for(FakeMatcher(absolute_cosine=None)) as client:
            candidates = [c for row in post_match(client).json()["results"].values() for c in row]

        assert candidates
        assert all(c["absoluteScore"] is None for c in candidates)

    def test_the_emitted_value_is_the_breakdowns_own(self):
        """Pinned against the fixture's constant, not against whatever came back."""
        with client_for(FakeMatcher()) as client:
            top = post_match(client).json()["results"]["account.resident_nm"][0]
        assert top["absoluteScore"] == pytest.approx(STAND_IN_ABSOLUTE_COSINE, abs=1e-6)


# =============================================================================
# SC-4 / SC-5 -- ONE VERDICT PER FIELD, INCLUDING "NOTHING MATCHED"
# =============================================================================


class TestFieldDecisions:
    """The value a consumer writes into a per-column decision."""

    def test_one_verdict_per_field_keyed_and_ordered_like_the_results(self, real_client):
        """
        The conservation law applied to the verdict. A column present in `results` and
        absent from `fieldDecisions` has candidates and no decision, and a client that
        defaulted an absent verdict to "nothing matched" would silently unclassify it.

        Compared against the paths this test SENT, not against the response's own
        `results` keys -- an expectation read back off the code under test holds just as
        well when both halves are wrong.
        """
        sent = [f["path"] for f in request_fields()]
        body = post_match(real_client).json()

        assert list(body["fieldDecisions"]) == sent
        assert list(body["fieldDecisions"]) == list(body["results"])

    def test_the_verdict_is_rank_ones_own_decision(self, real_client):
        body = post_match(real_client, top_k=5).json()
        for path, candidates in body["results"].items():
            if candidates:
                assert body["fieldDecisions"][path] == candidates[0]["decision"]

    def test_a_field_with_no_candidates_reads_no_match(self):
        """
        The empty list already said this; the verdict names it. And it is the piece that
        makes the empty list unambiguous: an empty `results` row with a `NO_MATCH` verdict
        is a field that was processed and matched nothing, while a field missing from both
        maps was not processed at all. Returning an empty candidate list ALONE cannot tell
        those apart, which is why the recommended shape is candidates plus a verdict.
        """
        with client_for(FakeMatcher(empty_paths=(UNMATCHABLE,))) as client:
            body = post_match(client).json()

        assert body["results"][UNMATCHABLE] == []
        assert body["fieldDecisions"][UNMATCHABLE] == "NO_MATCH"

    def test_no_configured_floor_means_no_score_driven_no_match(self, real_client):
        """
        THE DEFAULT MUST NOT INVENT A FLOOR. Every field on this fixture has candidates,
        including the one nothing describes, so with no floor configured NO_MATCH must not
        appear anywhere -- a library that started unclassifying fields on a threshold
        nobody chose would be shipping a calibration for a corpus it has never seen.
        """
        body = post_match(real_client, top_k=5).json()
        assert body["scoring"]["absoluteScoreFloor"] is None
        assert "NO_MATCH" not in set(body["fieldDecisions"].values())

    def test_a_configured_floor_turns_the_hopeless_field_into_no_match(self, floored_client):
        """
        The same matcher, the same glossary, the same request -- one number added to the
        server's configuration. Only the field whose absolute score is below it changes,
        which is what separates "the floor works" from "the floor rejects everything".
        """
        body = post_match(floored_client, top_k=5).json()

        assert body["scoring"]["absoluteScoreFloor"] == FLOOR
        assert body["fieldDecisions"][UNMATCHABLE] == "NO_MATCH"

        others = {p: v for p, v in body["fieldDecisions"].items() if p != UNMATCHABLE}
        assert others, "fixture no longer has fields other than the unmatchable one"
        assert "NO_MATCH" not in set(others.values())

    def test_the_candidates_survive_a_no_match(self, floored_client):
        """
        NO_MATCH must NOT be spelled as an empty candidate list. Dropping the candidates
        would collide with the conservation law and make "nothing matched" look identical
        to "this field was never processed"; the candidates stay as evidence for whoever
        decides whether a new glossary term is needed.
        """
        body = post_match(floored_client, top_k=5).json()
        assert body["fieldDecisions"][UNMATCHABLE] == "NO_MATCH"
        assert len(body["results"][UNMATCHABLE]) > 0

    def test_a_no_match_field_still_reports_what_its_candidates_carry(self, floored_client):
        """
        The half a reader has to be warned about, pinned so the warning stays true.

        `governance` on a candidate answers "what class does this entry carry", which is
        still a fact under a NO_MATCH verdict. The FIELD inherits nothing -- that is what
        `fieldDecisions` says -- and `GovernanceView`'s description is where a generated
        client learns not to read rank 1's class as an instruction here.
        """
        body = post_match(floored_client, top_k=5).json()
        top = body["results"][UNMATCHABLE][0]

        assert body["fieldDecisions"][UNMATCHABLE] == "NO_MATCH"
        assert "governance" in top

        schemas = None
        with client_for(build_api_matcher()) as client:
            schemas = client.get("/openapi.json").json()["components"]["schemas"]
        assert "NO_MATCH" in schemas["GovernanceView"]["description"]

    def test_the_published_vocabulary_names_no_match(self, real_client):
        """
        A Java client generates its enum from `/openapi.json`. A verdict the spec does not
        list is a deserialisation failure in the field, and this endpoint's own history
        has an example: "REJECT" appeared nowhere in the published schema while the service
        shipped it.
        """
        schemas = real_client.get("/openapi.json").json()["components"]["schemas"]
        assert set(schemas["FieldDecision"]["enum"]) == {
            "AUTO_APPROVE",
            "REVIEW",
            "REJECT",
            "NO_MATCH",
        }

    def test_the_per_candidate_enum_is_not_widened(self, real_client):
        """
        The other half of that decision. `decision` keeps its three values: adding one to
        an enum a client already generated a closed Java `enum` from turns an ordinary 200
        into a deserialisation failure on a build nobody redeployed. The wider vocabulary
        lives on the NEW field, where it is additive.
        """
        schemas = real_client.get("/openapi.json").json()["components"]["schemas"]
        assert set(schemas["MatchDecision"]["enum"]) == {"AUTO_APPROVE", "REVIEW", "REJECT"}


# =============================================================================
# SC-2 / SC-3 -- THE RESPONSE DESCRIBES ITS OWN NUMBERS
# =============================================================================


class TestScoringContract:
    """`scoring` -- the block that makes the rest of the body readable on its own."""

    def test_the_keys_are_the_contract_in_order(self, real_client):
        """A literal, not something read back off the response."""
        assert tuple(post_match(real_client).json()["scoring"]) == SCORING_KEYS

    def test_the_structural_floor_is_published(self, real_client):
        """
        0.63, stated so a consumer never has to read this library's source to learn it --
        and never sets a confidence threshold beneath it. A threshold at or below the floor
        selects nothing however bad the matches are, which is a review queue that reports
        "nothing to see" on a schema where nothing is trustworthy. This library shipped
        that defect once (NM-0027) with a default of 0.6 against a floor of 0.63.
        """
        assert post_match(real_client).json()["scoring"]["confidenceFloor"] == pytest.approx(0.63)

    def test_the_floor_is_null_when_this_response_disproves_it(self):
        """
        A BOUND VERIFIED AGAINST ITS OWN RESPONSE. The 0.63 derivation has preconditions --
        no reranker, and at least two distinct dense scores so min-max maps the winner to
        1.0 rather than to 0.0. A one-entry dictionary violates the second, and the real
        confidences then sit far below while the configuration still computes 0.63.

        Publishing it there would tell a client to set every threshold above a floor its
        own fields are underneath, which is NM-0027's failure re-shipped on the wire. So it
        is checked against the rank-1 confidences actually emitted and nulled if any is
        below.
        """
        with client_for(build_api_matcher(entries=GLOSSARY[:1])) as client:
            body = post_match(client).json()

        tops = [row[0]["confidence"] for row in body["results"].values() if row]
        assert tops and min(tops) < 0.63, "fixture no longer violates the bound"
        assert body["scoring"]["confidenceFloor"] is None

    def test_the_metric_behind_the_absolute_score_is_named(self, real_client):
        """
        `absoluteScore` is only a cosine while the store says it is. A deployment wiring a
        dot-product store gets a number that is monotone in similarity and neither bounded
        nor a cosine, and an absolute floor measured under one metric means nothing under
        the other. Naming it is what stops a client assuming.
        """
        assert post_match(real_client).json()["scoring"]["absoluteScoreMetric"] == "cosine"

    def test_alias_pooling_is_declared(self, real_client):
        """
        With index-time aliasing on, `absoluteScore` is the best score over an entry's
        FABRICATED spellings rather than its own text, so an entry can look confident on a
        spelling invented for it. Off in the shipped configuration, and a client comparing
        a floor across two deployments needs to know which one it is looking at.
        """
        assert post_match(real_client).json()["scoring"]["absoluteScorePooledOverAliases"] is False

    def test_it_declares_a_scope_for_every_number_the_body_can_carry(self, real_client):
        """
        Every numeric leaf a client can reach, keyed by its path in the body. A number with
        no declared scope is one a consumer will compare anyway.
        """
        scoring = post_match(real_client, explain=True).json()["scoring"]
        assert set(scoring["comparability"]) == {
            "confidence",
            "absoluteScore",
            "explain.absoluteCosine",
            "explain.scores.fusedRetrieval",
            "explain.scores.lexical",
            "explain.scores.editDistance",
            "explain.scores.type",
            "explain.scores.domain",
        }

    def test_it_settles_the_contradiction_between_the_two_scores(self, real_client):
        """
        THE POINT OF THE BLOCK. `confidence` is WITHIN_FIELD: it is min-max normalised
        inside one field's shortlist, so 0.72 on one field and 0.72 on another are not the
        same claim -- which is why the library's own docs say do not threshold on it, while
        the server nonetheless applies a fixed per-field cut point to it.

        `absoluteScore` is ACROSS_FIELDS, and it is the number a client may compare against
        a constant between columns. `explain.scores.fusedRetrieval` shares the confidence's
        scope for the same reason: it is the normalised component that creates the floor.
        """
        comparability = post_match(real_client, explain=True).json()["scoring"]["comparability"]

        assert comparability["confidence"] == "WITHIN_FIELD"
        assert comparability["explain.scores.fusedRetrieval"] == "WITHIN_FIELD"
        assert comparability["absoluteScore"] == "ACROSS_FIELDS"
        assert comparability["explain.absoluteCosine"] == "ACROSS_FIELDS"

    def test_the_scope_ordering_ships_so_a_client_need_not_hard_code_it(self, real_client):
        """
        The same pattern as `vocabulary.tiersMostOpenFirst`, extended rather than
        duplicated: the library ships the ordering so a client does not sort strings and
        conclude that ACROSS_FIELDS outranks WITHIN_FIELD by alphabet.
        """
        scoring = post_match(real_client).json()["scoring"]
        assert scoring["comparabilityScopesNarrowestFirst"] == [
            "WITHIN_FIELD",
            "ACROSS_FIELDS",
            "ACROSS_RUNS",
        ]

    def test_nothing_is_claimed_comparable_across_runs(self, real_client):
        """
        The honest half. `ACROSS_RUNS` is in the vocabulary and nothing reaches it: none of
        these numbers is calibrated, so none behaves like a probability that a match is
        correct, and every one moves with the configuration, the model or the dictionary.
        A scope declared and never used is a gap this library is stating rather than one it
        is hiding.
        """
        scoring = post_match(real_client, explain=True).json()["scoring"]
        assert "ACROSS_RUNS" not in set(scoring["comparability"].values())

    def test_the_thresholdable_list_agrees_with_the_scope_map(self, real_client):
        """
        Derived from the map rather than typed twice, so a number whose declared scope
        narrows cannot keep a stale entry telling a client it is still safe to threshold.
        """
        scoring = post_match(real_client, explain=True).json()["scoring"]
        expected = [
            key
            for key, scope in scoring["comparability"].items()
            if scope in ("ACROSS_FIELDS", "ACROSS_RUNS")
        ]
        assert scoring["thresholdableAcrossFields"] == expected
        assert "confidence" not in scoring["thresholdableAcrossFields"]
        assert "absoluteScore" in scoring["thresholdableAcrossFields"]


# =============================================================================
# THE PROPERTIES THE NEW KEYS MUST NOT COST
# =============================================================================


class TestNothingElseMoved:
    """Determinism, the published schema, and the shape of a degraded matcher."""

    def test_two_identical_requests_are_still_byte_identical(self, floored_client):
        """
        Three new maps went into the body, and a map built by iterating a set or a dict of
        unstable order makes every diff of this artifact unreadable.
        """
        first = post_match(floored_client, top_k=5, explain=True)
        second = post_match(floored_client, top_k=5, explain=True)
        assert first.content == second.content
        first.content.decode("ascii")  # raises if any byte is not ASCII

    def test_the_published_model_still_describes_what_is_sent(self, floored_client):
        MatchResponseView.model_validate(post_match(floored_client, explain=True).json())

    def test_a_matcher_that_declares_nothing_degrades_instead_of_failing(self):
        """
        `FakeMatcher` exposes no `absolute_score_floor` and no `absolute_score_metric` --
        it is the shape of a caller's own object, or of a lane that has not landed yet. The
        endpoint reports "no floor" and "unknown", which are the documented defaults, and
        serves the request. A 500 here would take matching down for a diagnostic string.
        """
        with client_for(FakeMatcher()) as client:
            body = post_match(client).json()

        assert body["scoring"]["absoluteScoreFloor"] is None
        assert body["scoring"]["absoluteScoreMetric"] == "unknown"
        assert "NO_MATCH" not in set(body["fieldDecisions"].values())
