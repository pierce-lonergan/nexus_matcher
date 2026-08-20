"""
tests.unit.presentation.api.test_bypassed_candidate_on_the_wire | Layer: TEST
Tests: what `/api/v1/match` says about a candidate that did not come from retrieval
Target: presentation/api/matching.py, application/use_cases/match_schema.py

AR-7's third hazard -- "how does it show on the wire?" -- and the two defects an
adjudication found in the first answer to it.

WHY THIS FILE EXISTS AT ALL

The shipped default attaches no feedback consumer, so nothing below is reachable on a
server this package starts. But `create_app(matcher=...)` takes a caller-built matcher, so
a deployment that opts into a bypass reaches it immediately.

  WHAT THE FIRST ANSWER GOT WRONG, TWICE.

  1. THE IDENTIFICATION WAS AN INFERENCE FROM TWO NUMBERS, AND THE INFERENCE WAS INVALID.
     A bypassed candidate carried `(confidence 1.0, decision AUTO_APPROVE)` and the source
     asserted that pair was unreachable by the scorer. It is not: the five default weights
     sum to exactly 1.0, so ordinary retrieval reaches 1.0 whenever all five signals are
     maximal. `_MAXIMAL_*` below constructs exactly that, over HTTP, and the pair then
     identifies nothing. `provenance` is the fix -- a VALUE that says where the answer came
     from, which cannot collide with a score.

  2. `explain=true` TOOK THE WHOLE REQUEST DOWN. The reproduction guard refuses a candidate
     whose emitted components and weights do not give its emitted confidence, and a
     bypassed candidate cannot satisfy that by construction. The refusal was right and its
     blast radius was not: an opted-in deployment lost MATCHING ENTIRELY for any batch
     containing one decided field. It is now narrowed by PROVENANCE rather than loosened --
     a candidate that never went through scoring is not evidence of scoring drift, and
     `TestTheGuardKeepsItsTeeth` is the half that proves the narrowing did not disarm it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, field_result_key
from nexus_matcher.domain.models.entities import DictionaryEntry
from nexus_matcher.domain.ports import (
    ApprovedPair,
    BaseFeedbackConsumer,
    ReviewVerdict,
    approval_binding,
)
from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.shared.types.base import DataType
from tests.unit.presentation.api._support import FakeMatcher, build_api_matcher, request_fields

SPECS = request_fields()
BYPASSED_PATH = SPECS[0]["path"]


# =============================================================================
# THE MAXIMAL FIXTURE -- A RETRIEVED CANDIDATE THAT REACHES 1.0
# =============================================================================

# Invented for this file, like everything else in this directory: a fictional utility's
# billing vocabulary, shaped for one purpose. It drives all five scoring signals to their
# maximum for one column at once, which is what puts a RETRIEVED candidate at confidence
# 1.0 and destroys the old identification.
#
#   fusedRetrieval  1.0  the same entry is rank 1 in BOTH arms, so min-max maps it to 1.0
#                        in each and the fused score is alpha + (1 - alpha)
#   lexical         1.0  the column's tokens are a subset of the entry's
#   editDistance    1.0  the column's normalised tokens ARE the entry's -- which needs a
#                        single-token column name
#   type            1.0  string against string
#   domain          1.0  the request's `domain` signal contains the entry's own domain
#
# `tariff` appears in exactly TWO definitions on purpose: in one, BM25's IDF leaves the
# lexical arm with a single result, min-max over which is 0.0, and the fused score stops at
# alpha; in three of six the term goes below the IDF floor and the arm returns nothing.
_MAXIMAL_ROWS = (
    ("MX-1", "Tariff", "the tariff the premises is billed on"),
    ("MX-2", "Meter", "the meter installed at the premises under a tariff"),
    ("MX-3", "Premises", "the premises the supply serves"),
    ("MX-4", "Invoice", "the invoice raised for one billing period"),
    ("MX-5", "Arrears", "the arrears carried on the account"),
    ("MX-6", "Settlement", "the settlement run a charge belongs to"),
)

_MAXIMAL_GLOSSARY = tuple(
    DictionaryEntry(
        id=entry_id,
        business_name=name,
        logical_name=name.lower(),
        definition=f"The governed element recording {definition}.",
        data_type=DataType.STRING,
        domain="billing",
    )
    for entry_id, name, definition in _MAXIMAL_ROWS
)

_REACHING_PATH = "tariff"
_MAXIMAL_BYPASSED_PATH = "arrears"
_MAXIMAL_BODY = {
    "fields": [
        {"name": _REACHING_PATH, "path": _REACHING_PATH, "type": "string"},
        {"name": _MAXIMAL_BYPASSED_PATH, "path": _MAXIMAL_BYPASSED_PATH, "type": "string"},
    ],
    "signals": {"domain": "billing"},
}


class _OneApprovedPair(BaseFeedbackConsumer):
    """Answers for exactly one field, with a term retrieval does not rank first for it."""

    def __init__(self, entry, path: str = BYPASSED_PATH) -> None:
        self._entry = entry
        self._path = path

    def approved_pair(self, field):
        key = field_result_key(field)
        if key != self._path:
            return None
        return ApprovedPair(
            field_key=key,
            entry=self._entry,
            verdict=ReviewVerdict.MANUAL_OVERRIDE,
            binding=approval_binding(self._entry),
            reviewer="steward-a",
            decided_at="2026-08-10T09:15:02+00:00",
        )


@pytest.fixture
def bypassing_client():
    matcher = build_api_matcher()
    entry = next(iter(matcher._dictionary_entries.values()))
    matcher._feedback_consumer = _OneApprovedPair(entry)
    app = create_app(configure_logs=False, matcher=matcher, environ={})
    with TestClient(app) as client:
        yield client, entry


@pytest.fixture
def plain_client():
    app = create_app(configure_logs=False, matcher=build_api_matcher(), environ={})
    with TestClient(app) as client:
        yield client


@pytest.fixture
def maximal_client():
    """
    One request carrying BOTH a retrieved candidate at confidence 1.0 and a bypassed one.

    That is the only arrangement in which the collision is visible rather than argued: the
    two candidates come back from a single call, and every member a client might read to
    tell them apart is right there to compare.
    """
    matcher = build_api_matcher(_MAXIMAL_GLOSSARY)
    matcher._feedback_consumer = _OneApprovedPair(
        matcher._dictionary_entries["MX-4"], _MAXIMAL_BYPASSED_PATH
    )
    app = create_app(configure_logs=False, matcher=matcher, environ={})
    with TestClient(app) as client:
        yield client


def _results(client: TestClient, body: dict | None = None, **extra) -> dict:
    payload = dict(body or {"fields": SPECS})
    payload.update(extra)
    response = client.post("/api/v1/match", json=payload)
    assert response.status_code == 200, response.text
    return response.json()["results"]


# =============================================================================
# WHAT THE RESPONSE SAYS
# =============================================================================


class TestABypassedCandidateIsIdentifiable:
    def test_the_humans_answer_is_the_only_candidate_and_it_is_theirs(self, bypassing_client):
        client, entry = bypassing_client
        candidates = _results(client)[BYPASSED_PATH]
        assert len(candidates) == 1
        assert candidates[0]["governanceId"] == entry.id

    def test_the_absolute_score_is_null_because_nothing_retrieved_it(self, bypassing_client):
        client, _entry = bypassing_client
        assert _results(client)[BYPASSED_PATH][0]["absoluteScore"] is None

    def test_the_candidate_says_where_its_answer_came_from(self, bypassing_client):
        """
        THE MEMBER, and the whole point of it: a client reads a value, not a conjunction.
        `MatchResult.performance.retrieval_stage` used to hold this fact and stop at the
        application boundary.
        """
        client, _entry = bypassing_client
        assert _results(client)[BYPASSED_PATH][0]["provenance"] == "APPROVED_PAIR"

    def test_every_retrieved_candidate_says_so_too_and_none_of_them_is_absent(
        self, bypassing_client
    ):
        """
        NON-VACUITY, in two directions. The value has to distinguish something, so the
        other fields in the same response must say `RETRIEVAL`; and it is published on
        EVERY candidate rather than only on the interesting one, because a member that
        appears only when something unusual happened is a member a client learns about in
        production.
        """
        client, _entry = bypassing_client
        results = _results(client)

        others = [c for path, cs in results.items() if path != BYPASSED_PATH for c in cs]
        assert others
        assert {c["provenance"] for c in others} == {"RETRIEVAL"}
        assert all("provenance" in c for cs in results.values() for c in cs)

    def test_a_response_with_no_consumer_attached_still_carries_the_member(self, plain_client):
        """The shipped default publishes it too, with the value that describes it."""
        results = _results(plain_client)
        assert {c["provenance"] for cs in results.values() for c in cs} == {"RETRIEVAL"}

    def test_the_other_fields_still_come_back_retrieved(self, bypassing_client):
        """One bypassed field must not turn the response into a bypassed response."""
        client, _entry = bypassing_client
        results = _results(client)

        assert len(results) == len(SPECS)
        others = [c for path, cs in results.items() if path != BYPASSED_PATH for c in cs]
        assert others and all(c["confidence"] < 1.0 for c in others)


# =============================================================================
# WHY THE TWO NUMBERS COULD NOT DO THE JOB
# =============================================================================


class TestTheConfidenceAndDecisionPairIdentifiesNothing:
    """
    The claim the member replaces, refuted on the wire it was made about.

    The old test asked one fixture whether anything in it happened to reach 1.0 and
    recorded that nothing did. That cannot distinguish "unreachable" from "not reached
    here", and the difference is the whole claim. `maximal_client` constructs the reaching
    case deliberately instead.
    """

    def test_ordinary_retrieval_reaches_the_confidence_that_was_called_unreachable(
        self, maximal_client
    ):
        candidate = _results(maximal_client, _MAXIMAL_BODY)[_REACHING_PATH][0]

        assert candidate["confidence"] == 1.0
        assert candidate["decision"] == "AUTO_APPROVE"
        assert candidate["provenance"] == "RETRIEVAL"

    def test_the_bypassed_candidate_in_the_same_response_is_equal_on_both_members(
        self, maximal_client
    ):
        """
        The collision itself. Two candidates, one request, identical on the exact pair a
        client was told to read -- and a client reading only those two members has no way
        to tell a human's answer from a very good match.
        """
        results = _results(maximal_client, _MAXIMAL_BODY)
        retrieved = results[_REACHING_PATH][0]
        bypassed = results[_MAXIMAL_BYPASSED_PATH][0]

        assert (
            (retrieved["confidence"], retrieved["decision"])
            == (
                bypassed["confidence"],
                bypassed["decision"],
            )
            == (1.0, "AUTO_APPROVE")
        )

    def test_provenance_separates_the_two_where_the_numbers_cannot(self, maximal_client):
        results = _results(maximal_client, _MAXIMAL_BODY)
        assert results[_REACHING_PATH][0]["provenance"] == "RETRIEVAL"
        assert results[_MAXIMAL_BYPASSED_PATH][0]["provenance"] == "APPROVED_PAIR"

    def test_the_absolute_score_conjunction_is_not_a_substitute_for_the_member(
        self, maximal_client
    ):
        """
        The undocumented fallback a client could have reached for -- `confidence == 1.0`
        AND `absoluteScore is null` -- pinned as the wrong question rather than left to be
        discovered. The reaching candidate HAS an absolute score, so the conjunction
        happens to separate this pair; but `absoluteScore: null` independently means "the
        dense arm never returned this candidate", which a lexical-only rank 1 satisfies
        while having been fully scored. It agrees by accident, and nothing should be built
        on it.
        """
        candidate = _results(maximal_client, _MAXIMAL_BODY)[_REACHING_PATH][0]
        assert candidate["confidence"] == 1.0
        assert candidate["absoluteScore"] is not None


# =============================================================================
# EXPLAIN, NARROWED RATHER THAN LOOSENED
# =============================================================================


class TestExplainSurvivesABypassedField:
    def test_the_whole_request_still_answers(self, bypassing_client):
        """
        THE DEFECT, driven directly. This was a 500 for every field in the batch, with an
        error telling the operator the library had drifted -- so an opted-in deployment
        whose client sets `explain=true` lost matching entirely for any batch containing
        one decided field.
        """
        client, _entry = bypassing_client
        response = client.post("/api/v1/match", json={"fields": SPECS, "explain": True})

        assert response.status_code == 200, response.text
        assert len(response.json()["results"]) == len(SPECS)

    def test_the_bypassed_candidate_carries_no_explain_block(self, bypassing_client):
        """
        ABSENT, not zeroed and not faked. `explain` promises the caller can recompute the
        confidence from the components beside it; this candidate's confidence did not come
        from a computation, so there is nothing to publish. Emitting components of 1.0 so
        the arithmetic closed would publish five measurements nobody took, into fields the
        scoring contract declares comparable ACROSS fields.
        """
        client, _entry = bypassing_client
        candidate = _results(client, explain=True)[BYPASSED_PATH][0]

        assert "explain" not in candidate
        assert candidate["provenance"] == "APPROVED_PAIR"

    def test_every_scored_candidate_in_the_same_response_still_has_one(self, bypassing_client):
        """
        NON-VACUITY: the omission is about the bypass, not about `explain`. Every other
        candidate in the same response carries its block and its own arithmetic still
        closes -- checked here on the emitted numbers rather than trusting that the server
        checked, because "the server verified it" is what the server would also say if the
        verification had been narrowed into nothing.
        """
        client, _entry = bypassing_client
        results = _results(client, explain=True)

        scored = [c for path, cs in results.items() if path != BYPASSED_PATH for c in cs]
        assert scored
        for candidate in scored:
            explain = candidate["explain"]
            recomputed = sum(
                explain["scores"][key] * weight for key, weight in explain["weights"].items()
            )
            assert min(max(recomputed, 0.0), 1.0) == pytest.approx(
                candidate["confidence"], abs=1e-5
            )

    def test_explain_is_fine_when_nothing_was_bypassed(self, plain_client):
        response = plain_client.post("/api/v1/match", json={"fields": SPECS, "explain": True})
        assert response.status_code == 200, response.text
        assert all("explain" in c for cs in response.json()["results"].values() for c in cs)


class TestTheGuardKeepsItsTeeth:
    """
    The narrowing must not be a loosening, so the check is shown still firing.

    It caught a real defect once -- a matcher whose weights the response could not read,
    explained with the shipped defaults that do not describe its confidences -- and that
    is a class of drift no name matching can see.
    """

    def test_a_scored_candidate_whose_numbers_do_not_close_is_still_refused(self):
        """
        `FakeMatcher` emits a confidence computed with the SHIPPED weights; a matcher
        configured with different ones cannot explain it. Its candidates come from
        RETRIEVAL, so the provenance exclusion does not reach them and the response is
        still refused rather than sent self-consistently wrong.
        """
        tuned = MatchingConfig(
            semantic_weight=0.60,
            lexical_weight=0.10,
            edit_distance_weight=0.10,
            type_weight=0.10,
            domain_weight=0.10,
        )
        app = create_app(configure_logs=False, matcher=FakeMatcher(config=tuned), environ={})
        with TestClient(app) as client:
            response = client.post("/api/v1/match", json={"fields": SPECS, "explain": True})

        assert response.status_code == 500, response.text
        assert response.json()["error"]["code"] == "NEXUS-1003"
        assert "reproducible confidence" in response.json()["error"]["message"]
