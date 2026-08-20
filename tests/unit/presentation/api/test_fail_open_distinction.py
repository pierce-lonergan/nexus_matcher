"""
tests.unit.presentation.api.test_fail_open_distinction | Layer: TEST
The four ways a response can decline to confer a protection class, and the members that
tell them apart WITHOUT the vocabulary file.

## Relationships
# TESTS → presentation/api/matching :: governance, fieldDecisions, vocabulary, scoring
# TESTS → presentation/api/schemas  :: GovernanceView, VocabularyView, FieldDecision

## The definition this file pins

The second of the two enterprise-adoption BLOCKERS was "what IS `governance.code`?", and
the maintainer's answer describes a CLASS of thing: *a description of the level of
protection on data that an organisation defines in order to attach read permissions to
columns.* That is the answer, and it settles what the member is FOR: a consumer maps
`governance.code` onto who may read the column.

Which makes the interesting question not "what is a code" but "what does the ABSENCE of
one mean", because a consumer mapping codes onto read permissions has to do something when
no code arrives -- and the cheap something is to grant the most permissive reading. That is
failing open, and it is silent: nobody files a ticket saying they were allowed to see data.

There are FOUR distinct reasons this service returns no class, and they demand three
different actions:

  OPEN        the matched entry carries no protection code. The field IS governed -- at the
              caller's own open tier -- and sending it to a human would be a reviewer's
              hour spent on a published price band. ACTION: apply the open tier.
  REJECTED    the rank-1 candidate was REJECTed, so no entry in the glossary describes this
              field and nothing is conferred. ACTION: a human.
  NO_MATCH    no candidate cleared the deployment's absolute floor. The candidates are
              still attached as evidence and rank 1 may carry a perfectly real class.
              ACTION: a human, and do NOT read rank 1's class.
  UNCONFIGURED the deployment has no vocabulary at all, so nothing could be classified,
              and "no class" is a statement about the SERVER rather than about the field.
              ACTION: fix the deployment. Applying an open tier here grants read access on
              the strength of a configuration nobody completed.

This file proves each is distinguishable from the response alone, names in every assertion
message WHICH members do the distinguishing, and -- because the answer turned out to be
uneven -- says out loud where the whole weight rests.

## What actually distinguishes them, measured rather than assumed

The four readings, taken from the four servers built below:

    case          governance   decision  fieldDecision  openClassification  tiers
    OPEN          null         AUTO_APP  AUTO_APPROVE   LUMENPORT_OPEN      3
    REJECTED      null         REJECT    REJECT         LUMENPORT_OPEN      3
    NO_MATCH      POPULATED    REVIEW    NO_MATCH       LUMENPORT_OPEN      3
    UNCONFIGURED  null         AUTO_APP  AUTO_APPROVE   UNCLASSIFIED        0

Three of the six pairs are separated by `fieldDecisions[path]` alone. The fourth --
NO_MATCH against everything -- is separated twice over, since its rank-1 candidate carries
a REAL class while the field inherits nothing, which is exactly the trap
`FieldDecision.NO_MATCH` exists to warn about.

**OPEN and UNCONFIGURED are separated by NOTHING at the field level.** Same rank-1 entry,
same confidence, same per-candidate verdict, same null class, same `fieldDecisions` map.
The ONLY discriminator is the response-level `vocabulary` block: `openClassification` is
the caller's own tier in one and the library's `UNCLASSIFIED` sentinel in the other, and
`tiersMostOpenFirst` is their declared ordering in one and empty in the other. So the
fail-open case that matters most is guarded by a single block, and
`test_open_and_unconfigured_are_separated_by_the_vocabulary_block_alone` says so by
asserting the collapse rather than by describing it.

There is one inference that appears to substitute for it and does not. On a BATCH request
over a glossary that has some coded rows, the configured server confers a class on some
other field's candidate while the unconfigured one confers none anywhere, so "did any class
arrive in this body?" answers the question. That is a coincidence of the request, and it
evaporates on exactly the request a classification pipeline sends when it is deciding one
column: a single field at `top_k` 1 matching an uncoded entry produces two bodies that are
equal in every member except `vocabulary`.
`test_a_class_seen_elsewhere_in_the_body_is_not_a_substitute_for_the_block` pins that.

None of this is a defect -- the discriminator exists, it rides on every response, and the
sentinel is deliberately a word no real taxonomy uses -- but it is a thin margin, and a
reader who does not know it is thin will not check it.

## The four servers

Each is the SAME real matcher over the SAME fictional Lumenport glossary. They differ by
one configuration decision apiece, which is what makes the comparison worth making: the
readings above differ because of the server's configuration and not because four different
fixtures were built to produce four different answers.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.governance import OPEN_CLASSIFICATION, GovernanceVocabulary
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.presentation.api.app import create_app
from tests.properties._support import DIMENSION, BagOfTokensProvider
from tests.unit.presentation.api._support import (
    GLOSSARY,
    TIER_GUARDED,
    TIER_OPEN,
    TIER_SEALED,
    build_api_matcher,
    request_fields,
)

# =============================================================================
# THE FOUR CASES, AND THE FIELD EACH IS READ ON
# =============================================================================

OPEN = "open"
REJECTED = "rejected"
NO_MATCH = "no_match"
UNCONFIGURED = "unconfigured"

CASES = (OPEN, REJECTED, NO_MATCH, UNCONFIGURED)

# The field whose reading demonstrates each case, chosen from the fixture's own behaviour:
#
#   billing.tariff_band   matches LWP-0005, the one glossary row with NO governance code.
#   account.resident_nm   matches LWP-0001, which carries RESIDENT -- so a null here is
#                         withholding a class that exists, not reporting one that does not.
#   misc.zzz_unmatchable  is the field nothing in the glossary describes; its rank-1
#                         absolute score (~0.12) is the only one under the floor below.
PATH_FOR: dict[str, str] = {
    OPEN: "billing.tariff_band",
    REJECTED: "account.resident_nm",
    NO_MATCH: "misc.zzz_unmatchable",
    UNCONFIGURED: "billing.tariff_band",
}

# A floor above what `misc.zzz_unmatchable` reaches (~0.12) and below what every other
# field reaches (0.64 and up). The same number and the same reasoning as
# `test_score_contract.FLOOR`: a floor that separates nothing proves nothing.
FLOOR = 0.30

# A review threshold above the structural rank-1 confidence floor of
# `semantic_weight * fusion_alpha` = 0.63, which is the ONLY way a rank-1 REJECT is
# reachable at all -- no setting of a threshold below that floor can produce one. 0.90 is
# chosen rather than something that rejects everything, so the REJECTED server still
# answers REVIEW on four of its six fields and the case is visibly per-FIELD.
REJECTING_REVIEW_THRESHOLD = 0.90
REJECTING_AUTO_APPROVE_THRESHOLD = 0.99


# =============================================================================
# THE FOUR SERVERS
# =============================================================================


def client_for(matcher: object) -> TestClient:
    return TestClient(create_app(configure_logs=False, matcher=matcher, environ={}))


def unconfigured_matcher() -> NexusMatcher:
    """
    A deployment that wired NO vocabulary, built the only way the library allows one.

    The glossary is the Lumenport one with every `governance_code` stripped. That is not a
    convenience: `NexusMatcher._index_dictionary` REFUSES to index an entry whose code the
    vocabulary cannot resolve, and its refusal names this exact confusion -- "indexing
    anyway would return every match with governance=None, which reads as 'this entry has no
    class' rather than 'nobody told me what its class means'". So the only unconfigured
    deployment that can exist is one whose glossary carries no codes either, and this
    fixture is that deployment rather than a mock of it.
    """
    matcher = NexusMatcher(
        embedding_provider=BagOfTokensProvider(),
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=DIMENSION)
        ),
        sparse_retriever=BM25Retriever(),
        config=MatchingConfig(results_per_field=5),
        governance=GovernanceVocabulary.empty(),
    )
    matcher._index_dictionary(
        [dataclasses.replace(entry, governance_code=None) for entry in GLOSSARY]
    )
    return matcher


def matcher_for(case: str) -> object:
    if case == OPEN:
        return build_api_matcher()
    if case == REJECTED:
        return build_api_matcher(
            config=MatchingConfig(
                results_per_field=5,
                review_threshold=REJECTING_REVIEW_THRESHOLD,
                auto_approve_threshold=REJECTING_AUTO_APPROVE_THRESHOLD,
            )
        )
    if case == NO_MATCH:
        return build_api_matcher(
            config=MatchingConfig(results_per_field=5, absolute_score_floor=FLOOR)
        )
    if case == UNCONFIGURED:
        return unconfigured_matcher()
    raise AssertionError(f"unknown case {case!r}")


def body_for(case: str) -> dict[str, Any]:
    with client_for(matcher_for(case)) as client:
        response = client.post("/api/v1/match", json={"fields": request_fields(), "top_k": 5})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def bodies() -> dict[str, dict[str, Any]]:
    """One match response per case, built once."""
    return {case: body_for(case) for case in CASES}


# =============================================================================
# THE READING -- EVERY MEMBER A CONSUMER MAY USE, AND NOTHING ELSE
# =============================================================================

# The members of a match response that carry any part of "may this column inherit a class,
# and if not, why not". Named as a closed list so that "distinguishable from the response
# alone" is checked against a FIXED set of members rather than against whatever happens to
# differ: a difference in `confidence` is not a discrimination a consumer may act on, and a
# test that accepted one would report the four cases as distinguishable on a number whose
# own contract says it is not comparable across fields.
#
# `scoring.absoluteScoreFloor` is deliberately NOT here, although a consumer may read it.
# It says whether the SERVER can emit a score-driven NO_MATCH at all; it says nothing about
# THIS field, and including it would let the pairwise check below pass on a deployment-wide
# setting after the per-field discrimination had been lost. It is asserted separately, as
# the explanation for the NO_MATCH case rather than as a discriminator for it.
READING_MEMBERS = (
    "governance",
    "decision",
    "fieldDecision",
    "openClassification",
    "tiersMostOpenFirst",
)


def reading(body: dict[str, Any], path: str) -> dict[str, Any]:
    """What a consumer mapping `governance.code` onto read permissions gets to look at."""
    top = body["results"][path][0]
    return {
        # The class itself. Null is the whole subject of this file.
        "governance": top["governance"],
        # The rank-1 candidate's own verdict.
        "decision": top["decision"],
        # THE FIELD-LEVEL AUTHORITY. The one a consumer writes into a per-column decision.
        "fieldDecision": body["fieldDecisions"][path],
        # What a null class MEANS on this deployment -- a real tier, or the sentinel.
        "openClassification": body["vocabulary"]["openClassification"],
        # The caller's own tier ordering. Empty is itself a signal.
        "tiersMostOpenFirst": body["vocabulary"]["tiersMostOpenFirst"],
    }


def differing_members(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """The members of the reading on which two cases disagree, in declaration order."""
    return [member for member in READING_MEMBERS if left[member] != right[member]]


# Which members separate each pair, measured from the four servers above and written down
# so that LOSING one is a failure rather than a silently narrower margin. A pair that still
# differs on something is still distinguishable -- but a pair that used to differ on three
# members and now differs on one has moved a lot closer to the collapse this file exists to
# prevent, and nothing else in the suite would report it.
SEPARATED_BY: dict[tuple[str, str], list[str]] = {
    (OPEN, REJECTED): ["decision", "fieldDecision"],
    (OPEN, NO_MATCH): ["governance", "decision", "fieldDecision"],
    (OPEN, UNCONFIGURED): ["openClassification", "tiersMostOpenFirst"],
    (REJECTED, NO_MATCH): ["governance", "decision", "fieldDecision"],
    (REJECTED, UNCONFIGURED): [
        "decision",
        "fieldDecision",
        "openClassification",
        "tiersMostOpenFirst",
    ],
    (NO_MATCH, UNCONFIGURED): [
        "governance",
        "decision",
        "fieldDecision",
        "openClassification",
        "tiersMostOpenFirst",
    ],
}


# =============================================================================
# EACH CASE, PINNED ON ITS OWN
# =============================================================================


class TestEachCaseReadsTheWayItsDefinitionSays:
    """One test per case. The definition, in assertions, on a real response."""

    def test_open_is_a_class_the_deployment_declared_and_not_a_gap(self, bodies):
        """
        The matched entry carries no protection code, and this field IS governed.

        The consumer's correct action is to apply the open tier, and the response is what
        tells them which tier that is. Without `vocabulary.openClassification` the null is
        a value they cannot resolve without a file on the server.
        """
        body = bodies[OPEN]
        path = PATH_FOR[OPEN]
        top = body["results"][path][0]

        assert top["governance"] is None, (
            "`results[path][0].governance` is the member that says no class was conferred; "
            "the fixture entry for this field carries no protection code"
        )
        assert top["decision"] != "REJECT", (
            "`results[path][0].decision` is what separates OPEN from REJECTED, and it must "
            "not be REJECT here or the two cases have collapsed"
        )
        assert body["fieldDecisions"][path] in {"AUTO_APPROVE", "REVIEW"}, (
            "`fieldDecisions[path]` is the field-level authority and it must be an ordinary "
            "verdict here -- NO_MATCH or REJECT would mean this field inherits nothing, "
            "which is a different case entirely"
        )
        assert body["vocabulary"]["openClassification"] == TIER_OPEN, (
            "`vocabulary.openClassification` is the member that RESOLVES this null into a "
            "tier. It must name the caller's own open tier; the UNCLASSIFIED sentinel here "
            "would mean nothing was ever classified, which is the UNCONFIGURED case."
        )
        assert body["vocabulary"]["tiersMostOpenFirst"] == [TIER_OPEN, TIER_GUARDED, TIER_SEALED]

    def test_rejected_confers_nothing_and_says_so_on_two_members(self, bodies):
        """
        The rank-1 candidate was rejected: no entry describes this field.

        The entry behind this candidate DOES carry a protection code -- it is the resident
        name row -- so the null here is a class being WITHHELD, not a class that does not
        exist. Reading it as the open tier would file a column the server has just said it
        cannot identify as "governed, openly".
        """
        body = bodies[REJECTED]
        path = PATH_FOR[REJECTED]
        top = body["results"][path][0]

        assert top["rank"] == 1
        assert top["decision"] == "REJECT", (
            "`results[path][0].decision` is the member that makes this null a WITHHOLDING; "
            "it is also the only member that separates this case from OPEN at the "
            "candidate level"
        )
        assert top["governance"] is None, (
            "a REJECTED rank 1 confers nothing, so `results[path][0].governance` is null "
            "even though the matched entry carries a real code"
        )
        assert body["fieldDecisions"][path] == "REJECT", (
            "`fieldDecisions[path]` carries rank 1's verdict up to the field level, and it "
            "is the member a consumer should branch on rather than re-deriving the roll-up"
        )
        assert body["vocabulary"]["openClassification"] == TIER_OPEN, (
            "the deployment IS configured here; the vocabulary block must not be what "
            "changed, or this case would be indistinguishable from UNCONFIGURED"
        )

    def test_rejected_is_per_field_and_not_a_property_of_the_server(self, bodies):
        """
        Non-vacuity for the case above. If this server rejected every field, "rank 1 was
        rejected" would be a fact about its configuration rather than about this column,
        and the test above would be reading a constant.
        """
        verdicts = bodies[REJECTED]["fieldDecisions"]

        assert verdicts[PATH_FOR[REJECTED]] == "REJECT"
        assert set(verdicts.values()) - {"REJECT"}, (
            f"every field on the REJECTED server came back REJECT ({verdicts}), so the "
            f"verdict is now a property of the threshold rather than of the field and the "
            f"case above proves less than it claims"
        )

    def test_no_match_withholds_a_class_its_own_candidates_still_carry(self, bodies):
        """
        THE TRAP, pinned. No candidate cleared the floor, so the field inherits nothing --
        and its rank-1 candidate carries a POPULATED protection class anyway.

        `governance` on a candidate answers "what class does this entry carry", which stays
        true under any verdict. The field-level answer is `fieldDecisions[path]`, and the
        two disagree here on purpose. A consumer reading rank 1's class would classify a
        column from an entry the server has just said describes nothing.
        """
        body = bodies[NO_MATCH]
        path = PATH_FOR[NO_MATCH]
        top = body["results"][path][0]

        assert body["fieldDecisions"][path] == "NO_MATCH", (
            "`fieldDecisions[path]` is the ONLY member that carries this case; the "
            "per-candidate `decision` vocabulary cannot express 'nothing matched' at all"
        )
        assert top["governance"] is not None, (
            "the rank-1 candidate on a NO_MATCH field must still report the class its "
            "entry carries -- that is what makes `fieldDecisions` load-bearing rather than "
            "derivable from `governance`, and it is why this case cannot be read off the "
            "candidate"
        )
        assert body["results"][path], (
            "NO_MATCH must not be spelled as an empty candidate list; the candidates are "
            "the evidence for the human who now has to decide"
        )
        assert body["scoring"]["absoluteScoreFloor"] == FLOOR, (
            "`scoring.absoluteScoreFloor` is the member that explains WHY this verdict was "
            "reachable at all -- the library ships no floor, so a null here means a "
            "score-driven NO_MATCH cannot occur on this server"
        )

    def test_unconfigured_says_nothing_was_classifiable_rather_than_nothing_applies(self, bodies):
        """
        No vocabulary at all. Every candidate's class is null and every one of those nulls
        is a statement about the SERVER.

        The distinguishing members are both on the response-level `vocabulary` block, and
        neither of them is inside `results`. That is the whole margin: see
        `test_open_and_unconfigured_are_separated_by_the_vocabulary_block_alone`.
        """
        body = bodies[UNCONFIGURED]
        path = PATH_FOR[UNCONFIGURED]

        assert body["vocabulary"]["openClassification"] == OPEN_CLASSIFICATION, (
            "`vocabulary.openClassification` is the FIRST of the two members that carry "
            "this case, and it must be the library's sentinel rather than a plausible "
            "tier name -- a real-sounding default would read in an audit as a decision "
            "somebody made instead of as a gap in configuration"
        )
        assert body["vocabulary"]["tiersMostOpenFirst"] == [], (
            "`vocabulary.tiersMostOpenFirst` is the SECOND, and an empty ordering is what "
            "tells a consumer that no two tiers on this deployment are comparable"
        )
        assert body["results"][path][0]["governance"] is None
        assert all(
            candidate["governance"] is None
            for candidates in body["results"].values()
            for candidate in candidates
        ), (
            "a class arrived from a deployment with no vocabulary, so something is "
            "resolving codes against a taxonomy this library supplied"
        )

    def test_the_sentinel_is_learnable_from_the_published_schema(self, bodies):
        """
        The claim this file rests on is that the four cases are distinguishable WITHOUT
        the vocabulary file. That is only true if the sentinel is discoverable from
        something the consumer already has -- and the published schema is what a generated
        client is built from.
        """
        with client_for(matcher_for(OPEN)) as client:
            schemas = client.get("/openapi.json").json()["components"]["schemas"]

        described = schemas["VocabularyView"]["properties"]["openClassification"]["description"]
        assert OPEN_CLASSIFICATION in described, (
            f"`VocabularyView.openClassification` no longer names {OPEN_CLASSIFICATION!r} "
            f"in its published description, so a consumer cannot learn what the value "
            f"means without a file on the server -- which is the exact dependency the "
            f"vocabulary block was added to remove"
        )


# =============================================================================
# PAIRWISE -- NO TWO CASES SHARE A READING
# =============================================================================


class TestNoTwoCasesAreIndistinguishable:
    """All six pairs, and the members that separate each."""

    @pytest.mark.parametrize("left,right", list(itertools.combinations(CASES, 2)))
    def test_the_pair_differs_on_at_least_one_member_a_consumer_may_act_on(
        self, bodies, left, right
    ):
        """
        THE REQUIREMENT, literally: no two of the four look alike from the response alone.
        """
        left_reading = reading(bodies[left], PATH_FOR[left])
        right_reading = reading(bodies[right], PATH_FOR[right])

        differences = differing_members(left_reading, right_reading)

        assert differences, (
            f"{left} and {right} produce the IDENTICAL reading on every member a consumer "
            f"may act on ({list(READING_MEMBERS)}): {left_reading}. A consumer mapping "
            f"`governance.code` onto read permissions cannot tell these two apart from the "
            f"response alone, and the safe action for one of them is not the safe action "
            f"for the other. This is a finding about the CONTRACT, not about the test: "
            f"report it rather than relaxing READING_MEMBERS."
        )

    @pytest.mark.parametrize("left,right", list(itertools.combinations(CASES, 2)))
    def test_the_pair_is_separated_by_the_members_this_file_says_it_is(self, bodies, left, right):
        """
        The same six pairs, but pinning WHICH members do the separating rather than only
        that something does.

        A pair that used to differ on three members and now differs on one is still
        "distinguishable", so the test above stays green through most of the way to a
        collapse. This is the one that reports the margin narrowing, and its message names
        the members that were lost.
        """
        expected = SEPARATED_BY[(left, right)]
        differences = differing_members(
            reading(bodies[left], PATH_FOR[left]),
            reading(bodies[right], PATH_FOR[right]),
        )

        assert differences == expected, (
            f"{left} and {right} are now separated by {differences}, not by {expected}. "
            f"Lost: {sorted(set(expected) - set(differences))}. Gained: "
            f"{sorted(set(differences) - set(expected))}. A member that stopped separating "
            f"these two is a member a consumer was entitled to branch on, so update "
            f"SEPARATED_BY only after deciding the remaining ones are enough."
        )

    def test_the_field_level_verdict_separates_three_of_the_four(self, bodies):
        """
        `fieldDecisions[path]` on its own, named because it is the member a consumer should
        reach for first and it does most of the work.
        """
        verdicts = {case: bodies[case]["fieldDecisions"][PATH_FOR[case]] for case in CASES}

        assert verdicts[REJECTED] == "REJECT"
        assert verdicts[NO_MATCH] == "NO_MATCH"
        assert verdicts[OPEN] not in {"REJECT", "NO_MATCH"}
        assert verdicts[OPEN] == verdicts[UNCONFIGURED], (
            f"`fieldDecisions` now separates OPEN from UNCONFIGURED ({verdicts[OPEN]} vs "
            f"{verdicts[UNCONFIGURED]}). That is an IMPROVEMENT, not a failure -- but this "
            f"file and the Java client's GovernanceOutcome both document the vocabulary "
            f"block as the sole discriminator, and that documentation is now wrong."
        )

    def test_open_and_unconfigured_are_separated_by_the_vocabulary_block_alone(self, bodies):
        """
        THE THIN MARGIN, asserted rather than described.

        For the field this case is read on, the two servers return the IDENTICAL rank-1
        candidate -- same entry, same confidence, same verdict, same null class -- and the
        identical `fieldDecisions` map. Rank 1 is what a classification pipeline reads, so
        a consumer that reads it and never reads `vocabulary` sees no difference at all
        between "this column is governed, openly" and "this deployment classified
        nothing". Granting read access on the second is the fail-open this whole file is
        about, and this test exists so that the single member standing between them cannot
        be removed, renamed or emptied without a failure that says why.

        The runner-ups on this batch DO differ, because they are other fields' coded
        entries surfacing as low-ranked candidates. That is the coincidence the next test
        takes away.
        """
        open_body, unconfigured_body = bodies[OPEN], bodies[UNCONFIGURED]
        path = PATH_FOR[OPEN]
        assert path == PATH_FOR[UNCONFIGURED], "the two cases are no longer read on one field"

        assert open_body["fieldDecisions"] == unconfigured_body["fieldDecisions"], (
            "the two servers no longer agree on every verdict; re-derive which members "
            "separate them before trusting the claim below"
        )
        assert open_body["results"][path][0] == unconfigured_body["results"][path][0], (
            "the rank-1 candidate for the field whose entry carries no code no longer "
            "matches, so something other than `vocabulary` separates these two cases and "
            "the claim below needs re-deriving rather than relaxing"
        )

        differences = differing_members(
            reading(open_body, path),
            reading(unconfigured_body, path),
        )
        assert differences == ["openClassification", "tiersMostOpenFirst"], (
            f"OPEN and UNCONFIGURED are now separated by {differences} rather than by the "
            f"vocabulary block alone. If that list GREW, this file's warning is stale and "
            f"should be relaxed; if it SHRANK, the two cases have collapsed and a consumer "
            f"mapping `governance.code` onto read permissions can no longer tell an open "
            f"tier from an unconfigured server."
        )

    def test_a_class_seen_elsewhere_in_the_body_is_not_a_substitute_for_the_block(self):
        """
        The inference a clever consumer would reach for instead, and why it does not hold.

        On a BATCH request over this fixture the two servers are separable without reading
        `vocabulary` at all: the configured one confers a class on some other field's
        candidate and the unconfigured one confers none anywhere, so "did any class arrive
        in this body?" answers the question. That inference is a coincidence of the
        request, not a property of the contract -- and it evaporates on the request shape a
        governance pipeline actually sends when it is classifying one column.

        A single field at `top_k` 1, matching the one entry that carries no code: the two
        bodies are byte-identical except for the vocabulary block. There is no class
        anywhere to infer from, and nothing else to read.
        """
        payload = {
            "fields": [f for f in request_fields() if f["path"] == PATH_FOR[OPEN]],
            "top_k": 1,
        }
        assert len(payload["fields"]) == 1

        with client_for(matcher_for(OPEN)) as client:
            configured = client.post("/api/v1/match", json=payload).json()
        with client_for(matcher_for(UNCONFIGURED)) as client:
            unconfigured = client.post("/api/v1/match", json=payload).json()

        assert not any(
            candidate["governance"] is not None
            for body in (configured, unconfigured)
            for candidates in body["results"].values()
            for candidate in candidates
        ), (
            "a class arrived in one of these two bodies, so the 'is there a class "
            "anywhere?' inference is available here and this test is no longer describing "
            "the case it names -- pick a field whose rank-1 entry carries no code"
        )

        assert configured["results"] == unconfigured["results"]
        assert configured["fieldDecisions"] == unconfigured["fieldDecisions"]
        assert configured["vocabulary"] != unconfigured["vocabulary"], (
            "the ONLY difference between a configured deployment reporting an open tier "
            "and a deployment that classified nothing has disappeared. A consumer mapping "
            "`governance.code` onto read permissions now grants the open tier's access on "
            "the strength of a configuration nobody completed."
        )
        assert {key: value for key, value in configured.items() if key != "vocabulary"} == {
            key: value for key, value in unconfigured.items() if key != "vocabulary"
        }


# =============================================================================
# THE LOOKUP PLANE
# =============================================================================


class TestTheLookupPlaneCarriesTheSameDistinction:
    """
    An entry has no verdict, so only two of the four cases can reach this plane -- and the
    two that can are the two a candidate cannot separate on its own.
    """

    def test_an_uncoded_entry_and_an_unconfigured_server_differ_only_in_the_vocabulary(self):
        """
        The same collapse as on the match plane, and here it is even starker: a lookup
        carries no `decision` and no `fieldDecisions` at all, so `vocabulary` is not merely
        the strongest discriminator, it is the only one there is.
        """
        with client_for(matcher_for(OPEN)) as client:
            configured = client.post("/api/v1/lookup", json={"ids": ["LWP-0005"]}).json()
        with client_for(matcher_for(UNCONFIGURED)) as client:
            unconfigured = client.post("/api/v1/lookup", json={"ids": ["LWP-0005"]}).json()

        assert configured["results"]["LWP-0005"] == unconfigured["results"]["LWP-0005"], (
            "the entry payloads no longer match, so `vocabulary` is not the only "
            "discriminator on this plane and this test's premise needs re-deriving"
        )
        assert configured["vocabulary"]["openClassification"] == TIER_OPEN
        assert unconfigured["vocabulary"]["openClassification"] == OPEN_CLASSIFICATION
        assert configured["vocabulary"] != unconfigured["vocabulary"], (
            "`vocabulary` is the ONLY member of a lookup response that says whether a null "
            "class means the open tier or an unconfigured server; if the two are equal, a "
            "caller resolving ids has no way to tell them apart at all"
        )

    def test_a_coded_entry_still_confers_its_class_on_the_configured_server(self):
        """
        Non-vacuity: the comparison above is only interesting while this plane CAN carry a
        class. A lookup that returned null for everything would satisfy it trivially.
        """
        with client_for(matcher_for(OPEN)) as client:
            body = client.post("/api/v1/lookup", json={"ids": ["LWP-0001"]}).json()

        governance = body["results"]["LWP-0001"]["governance"]
        assert governance is not None
        assert governance["classification"] == TIER_GUARDED
