"""
tests.unit.presentation.api.test_review_evidence_wire | Layer: TEST
The two optional evidence blocks on POST /api/v1/match: the pairwise contrast between
rank 1 and rank 2, and the cross-field consistency report.

## Relationships
# TESTS → presentation/api/matching        :: projection of both blocks
# TESTS → presentation/api/schemas         :: the request flags and the published shape
# TESTS → domain/services/review_evidence  :: through the endpoint, as a caller reaches it

## The property that outranks every other assertion here

A REQUEST THAT DOES NOT ASK FOR THESE MUST GET THE BYTES IT GOT BEFORE THEY EXISTED.
`TestAdditive` is that assertion, and it is written byte-for-byte rather than key-by-key
because this response is a governance artifact: it gets pasted into a ticket and diffed,
and a key that appears, an order that shifts or a float that gains a decimal makes every
one of those diffs unreadable. Asserting on parsed JSON would let a re-ordering through.

## Why a scripted matcher and not only the real one

Both blocks are arithmetic over score breakdowns, and the interesting cases are the ones
a correct matcher will not produce on demand: two candidates level to the last decimal, a
signal that differs below the resolution of the published numbers, a group of columns
that got three different answers. `ScriptedMatcher` builds those states exactly.
`build_api_matcher` is still used wherever the question is "does this hold on the thing
that ships" -- the shape of the block, the conservation of its keys, determinism.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.domain.models.entities import DictionaryEntry, MatchResult, SchemaField
from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.schemas import MatchResponseView
from nexus_matcher.shared.types.base import (
    DataType,
    MatchDecision,
    PerformanceMetrics,
    ProtectionLevel,
    ScoreBreakdown,
)
from tests.unit.presentation.api._support import (
    GLOSSARY,
    build_api_matcher,
    governance_vocabulary,
    request_fields,
)

# The five weighted signals, in the order the response declares them. A literal, not read
# back off the code under test: an expectation derived from the thing it checks is an
# identity and holds just as well when both sides are wrong.
SIGNAL_NAMES = ("fusedRetrieval", "lexical", "editDistance", "type", "domain")

# The shipped weights, written out for the same reason. Any change to them is a red test
# here, which is the point: the contrast's arithmetic is stated against these numbers.
WEIGHTS = {
    "fusedRetrieval": 0.70,
    "lexical": 0.05,
    "editDistance": 0.05,
    "type": 0.05,
    "domain": 0.15,
}

# Wire name to the config attribute that holds its weight. Written out rather than
# imported from `matching._SCORE_COMPONENTS`, which is the pairing this file exists to
# check: an expectation read off the code under test is an identity, and an identity holds
# just as well when both sides are wrong.
WEIGHT_ATTRS = {
    "fusedRetrieval": "semantic_weight",
    "lexical": "lexical_weight",
    "editDistance": "edit_distance_weight",
    "type": "type_weight",
    "domain": "domain_weight",
}

CONTRAST_KEYS = (
    "topGovernanceId",
    "runnerUpGovernanceId",
    "topConfidence",
    "runnerUpConfidence",
    "confidenceGap",
    "signalGap",
    "separation",
    "largestDifference",
    "decidingSignals",
    "governanceDiffers",
    "domainDiffers",
    "signals",
)

SIGNAL_KEYS = (
    "signal",
    "topScore",
    "runnerUpScore",
    "delta",
    "weight",
    "weightedDelta",
    "separating",
    "deciding",
)

GROUP_KEYS = (
    "concept",
    "fields",
    "answers",
    "distinctAnswers",
    "agreement",
    "majorityGovernanceId",
    "majorityCount",
)


# =============================================================================
# A MATCHER WHOSE SCORES THIS FILE CHOOSES
# =============================================================================


def breakdown(scores: dict[str, float]) -> ScoreBreakdown:
    """A breakdown from the five wire names, so a test states what a reviewer would read."""
    return ScoreBreakdown(
        fused_retrieval_score=scores["fusedRetrieval"],
        lexical_score=scores["lexical"],
        edit_distance_score=scores["editDistance"],
        type_compatibility_score=scores["type"],
        domain_score=scores["domain"],
        absolute_cosine=0.5,
    )


def confidence_of(scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """The confidence those scores imply under these weights, clamped as the matcher
    clamps. Computed here rather than asserted from the response, so the fixture states an
    absolute expectation."""
    weights = weights or WEIGHTS
    return min(max(sum(scores[k] * weights[k] for k in SIGNAL_NAMES), 0.0), 1.0)


class ScriptedMatcher:
    """
    Serves, for every field, the candidate list this test wrote.

    `_config` is present because the endpoint reads `results_per_field` and the weights
    off it; without it every test would go down the weights-unavailable fallback instead
    of the path that ships.

    `weights` produce the confidences; `config_weights` are what the endpoint will READ.
    They are separate knobs so a matcher whose confidences the published weights do not
    explain can be constructed, which is the one state the contrast's arithmetic check
    exists to refuse and which no correct matcher will produce on demand.
    """

    def __init__(
        self,
        script: list[tuple[str, dict[str, float]]],
        *,
        per_path: dict[str, list[tuple[str, dict[str, float]]]] | None = None,
        weights: dict[str, float] | None = None,
        config_weights: dict[str, float] | None = None,
    ) -> None:
        from dataclasses import replace

        from nexus_matcher.application.use_cases.match_schema import MatchingConfig

        self._weights = weights or WEIGHTS
        published = config_weights or self._weights
        self._config = replace(
            MatchingConfig(), **{WEIGHT_ATTRS[k]: published[k] for k in SIGNAL_NAMES}
        )
        self._script = script
        self._per_path = per_path or {}
        self._entries = {entry.id: entry for entry in GLOSSARY}
        self._vocabulary = governance_vocabulary()

    @property
    def _governance(self) -> Any:
        return self._vocabulary

    def _match_fields(
        self, fields: list[SchemaField], signals: dict[str, Any] | None = None
    ) -> dict[str, tuple[MatchResult, ...]]:
        out: dict[str, tuple[MatchResult, ...]] = {}
        for field in fields:
            key = field.source_metadata.get("flattened_name", field.full_path)
            script = self._per_path.get(key, self._script)
            matches = []
            for rank, (entry_id, scores) in enumerate(script, 1):
                entry = self._entries[entry_id]
                matches.append(
                    MatchResult(
                        schema_field=field,
                        dictionary_entry=entry,
                        rank=rank,
                        final_confidence=confidence_of(scores, self._weights),
                        score_breakdown=breakdown(scores),
                        decision=MatchDecision.REVIEW,
                        performance=PerformanceMetrics(latency_ms=1.0),
                        governance=self._vocabulary.get(entry.governance_code),
                    )
                )
            out[key] = tuple(matches)
        return out


def client_for(matcher: object) -> TestClient:
    return TestClient(create_app(configure_logs=False, matcher=matcher, environ={}))


@pytest.fixture
def real_client():
    with client_for(build_api_matcher()) as client:
        yield client


def post(client: TestClient, fields: list[dict[str, Any]] | None = None, **body: Any):
    payload: dict[str, Any] = {"fields": fields if fields is not None else request_fields()}
    payload.update(body)
    return client.post("/api/v1/match", json=payload)


# =============================================================================
# THE BLOCKS ARE ADDITIVE
# =============================================================================


class TestAdditive:
    """
    Nothing in the shipped response moves unless the caller asked for more.

    Byte-for-byte, because this body is diffed by hand.
    """

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"contrast": False},
            {"consistency": False},
            {"contrast": False, "consistency": False},
        ],
    )
    def test_the_shipped_body_is_byte_identical_without_the_flags(self, real_client, body):
        baseline = post(real_client).content
        assert post(real_client, **body).content == baseline

    @pytest.mark.parametrize("flag", ["contrast", "consistency"])
    def test_asking_for_one_block_changes_nothing_else(self, real_client, flag):
        baseline = post(real_client).json()
        asked = post(real_client, **{flag: True}).json()
        for key in ("results", "vocabulary", "fieldDecisions", "scoring"):
            assert json.dumps(asked[key]) == json.dumps(baseline[key]), key

    def test_the_new_keys_are_appended_last_and_only_when_asked(self, real_client):
        assert tuple(post(real_client).json()) == (
            "results",
            "vocabulary",
            "fieldDecisions",
            "scoring",
        )
        assert tuple(post(real_client, contrast=True, consistency=True).json()) == (
            "results",
            "vocabulary",
            "fieldDecisions",
            "scoring",
            "contrast",
            "consistency",
        )

    def test_both_blocks_are_deterministic(self, real_client):
        first = post(real_client, contrast=True, consistency=True).content
        second = post(real_client, contrast=True, consistency=True).content
        assert first == second

    def test_the_published_schema_accepts_both_blocks(self, real_client):
        MatchResponseView.model_validate(post(real_client, contrast=True).json())
        MatchResponseView.model_validate(post(real_client, consistency=True).json())
        MatchResponseView.model_validate(post(real_client, contrast=True, consistency=True).json())


# =============================================================================
# CONTRAST -- WHY THE RUNNER-UP LOST
# =============================================================================

# Two candidates that differ on `domain` alone. Everything else is identical, so the
# whole margin is one signal's and the deciding signal is not a matter of opinion.
DOMAIN_ONLY = [
    (
        "LWP-0001",
        {"fusedRetrieval": 0.9, "lexical": 0.4, "editDistance": 0.4, "type": 1.0, "domain": 1.0},
    ),
    (
        "LWP-0004",
        {"fusedRetrieval": 0.9, "lexical": 0.4, "editDistance": 0.4, "type": 1.0, "domain": 0.2},
    ),
]

# A pair whose confidences are identical to the last decimal while two signals differ and
# cancel: 0.05 * (+0.4) against 0.15 * (-0.133333...). The order came from the sort.
CANCELLING = [
    (
        "LWP-0001",
        {"fusedRetrieval": 0.9, "lexical": 0.8, "editDistance": 0.4, "type": 1.0, "domain": 0.2},
    ),
    (
        "LWP-0004",
        {
            "fusedRetrieval": 0.9,
            "lexical": 0.4,
            "editDistance": 0.4,
            "type": 1.0,
            "domain": 0.333333,
        },
    ),
]

# Two candidates separated by less than the published resolution on every signal.
BELOW_RESOLUTION = [
    (
        "LWP-0001",
        {"fusedRetrieval": 0.9, "lexical": 0.4, "editDistance": 0.4, "type": 1.0, "domain": 0.2},
    ),
    (
        "LWP-0004",
        {
            "fusedRetrieval": 0.9000001,
            "lexical": 0.4,
            "editDistance": 0.4,
            "type": 1.0,
            "domain": 0.2,
        },
    ),
]


class TestContrastShape:
    def test_every_field_carries_a_contrast_key(self, real_client):
        body = post(real_client, contrast=True).json()
        assert list(body["contrast"]["fields"]) == list(body["results"])

    def test_a_field_with_one_candidate_gets_an_explicit_null(self):
        single = [DOMAIN_ONLY[0]]
        with client_for(ScriptedMatcher(single)) as client:
            body = post(client, contrast=True).json()
        assert set(body["contrast"]["fields"].values()) == {None}

    def test_the_contrast_object_has_the_contract_key_order(self):
        with client_for(ScriptedMatcher(DOMAIN_ONLY)) as client:
            body = post(client, contrast=True).json()
        contrast = body["contrast"]["fields"]["account.resident_nm"]
        assert tuple(contrast) == CONTRAST_KEYS
        assert tuple(contrast["signals"][0]) == SIGNAL_KEYS

    def test_the_block_declares_its_resolution_and_scopes(self):
        with client_for(ScriptedMatcher(DOMAIN_ONLY)) as client:
            block = post(client, contrast=True).json()["contrast"]
        assert block["resolution"] == 1e-06
        # A gap between two confidences is only as comparable as the confidences are, and
        # `confidence` is declared WITHIN_FIELD. Publishing the gap without saying so
        # would invite a client to threshold it across fields.
        assert block["comparability"]["confidenceGap"] == "WITHIN_FIELD"
        assert block["comparability"]["signals"]["fusedRetrieval"] == "WITHIN_FIELD"
        assert block["comparability"]["signals"]["domain"] == "ACROSS_FIELDS"


class TestContrastArithmetic:
    def test_the_signal_gap_reproduces_the_confidence_gap(self):
        with client_for(ScriptedMatcher(DOMAIN_ONLY)) as client:
            contrast = post(client, contrast=True).json()["contrast"]["fields"][
                "account.resident_nm"
            ]
        assert contrast["signalGap"] == pytest.approx(contrast["confidenceGap"], abs=1e-6)
        assert sum(s["weightedDelta"] for s in contrast["signals"]) == pytest.approx(
            contrast["confidenceGap"], abs=1e-6
        )

    def test_the_deltas_are_the_subtraction_a_reviewer_would_do(self):
        """
        A reviewer holding `explain` subtracts two published component scores. The delta
        printed beside them has to be that subtraction, not a full-precision number
        rounded afterwards -- which would disagree in the last place and read as the tool
        getting its own arithmetic wrong.
        """
        with client_for(ScriptedMatcher(DOMAIN_ONLY)) as client:
            body = post(client, contrast=True, explain=True).json()
        candidates = body["results"]["account.resident_nm"]
        contrast = body["contrast"]["fields"]["account.resident_nm"]
        by_name = {s["signal"]: s for s in contrast["signals"]}
        for name in SIGNAL_NAMES:
            top = candidates[0]["explain"]["scores"][name]
            runner_up = candidates[1]["explain"]["scores"][name]
            assert by_name[name]["delta"] == pytest.approx(round(top - runner_up, 6), abs=0)
            assert by_name[name]["weight"] == candidates[0]["explain"]["weights"][name]

    def test_the_only_differing_signal_is_named_as_the_cause(self):
        with client_for(ScriptedMatcher(DOMAIN_ONLY)) as client:
            contrast = post(client, contrast=True).json()["contrast"]["fields"][
                "account.resident_nm"
            ]
        assert contrast["separation"] == "SEPARATED"
        assert contrast["largestDifference"] == "domain"
        assert contrast["decidingSignals"] == ["domain"]
        separating = [s["signal"] for s in contrast["signals"] if s["separating"]]
        assert separating == ["domain"]

    def test_the_largest_difference_is_first_in_the_list(self):
        with client_for(ScriptedMatcher(DOMAIN_ONLY)) as client:
            contrast = post(client, contrast=True).json()["contrast"]["fields"][
                "account.resident_nm"
            ]
        assert contrast["signals"][0]["signal"] == "domain"

    def test_a_difference_below_the_resolution_is_not_a_cause(self):
        """
        The response rounds every number it publishes. A 'cause' below that rounding is a
        difference the reviewer cannot see in the artifact they are holding, and naming
        one would be inventing a reason.
        """
        with client_for(ScriptedMatcher(BELOW_RESOLUTION)) as client:
            contrast = post(client, contrast=True).json()["contrast"]["fields"][
                "account.resident_nm"
            ]
        assert not any(s["separating"] for s in contrast["signals"])
        assert contrast["largestDifference"] is None
        assert contrast["decidingSignals"] == []

    def test_two_level_candidates_name_nothing_and_say_so(self):
        """
        Signals that disagree and cancel are exactly what a reviewer wants to see -- and
        exactly where a naive contrast would invent a winner. The differences are still
        reported; the CAUSE is not, because the sort order decided this one.
        """
        with client_for(ScriptedMatcher(CANCELLING)) as client:
            contrast = post(client, contrast=True).json()["contrast"]["fields"][
                "account.resident_nm"
            ]
        assert contrast["confidenceGap"] == pytest.approx(0.0, abs=1e-6)
        assert contrast["separation"] == "TIED"
        assert contrast["largestDifference"] is None
        assert contrast["decidingSignals"] == []
        # The evidence survives the refusal to name a cause.
        assert {s["signal"] for s in contrast["signals"] if s["separating"]} == {
            "lexical",
            "domain",
        }

    def test_a_margin_no_single_signal_carries_names_no_deciding_signal(self):
        """
        Two signals each worth less than the margin. Removing either leaves the winner
        ahead, so neither decided it, and `decidingSignals` is empty -- which is a real
        answer rather than a gap.
        """
        script = [
            (
                "LWP-0001",
                {
                    "fusedRetrieval": 0.9,
                    "lexical": 1.0,
                    "editDistance": 1.0,
                    "type": 1.0,
                    "domain": 1.0,
                },
            ),
            (
                "LWP-0004",
                {
                    "fusedRetrieval": 0.9,
                    "lexical": 0.0,
                    "editDistance": 0.0,
                    "type": 1.0,
                    "domain": 1.0,
                },
            ),
        ]
        with client_for(ScriptedMatcher(script)) as client:
            contrast = post(client, contrast=True).json()["contrast"]["fields"][
                "account.resident_nm"
            ]
        assert contrast["separation"] == "SEPARATED"
        assert contrast["decidingSignals"] == []
        assert {s["signal"] for s in contrast["signals"] if s["separating"]} == {
            "lexical",
            "editDistance",
        }

    def test_the_two_entries_own_facts_are_reported(self):
        with client_for(ScriptedMatcher(DOMAIN_ONLY)) as client:
            contrast = post(client, contrast=True).json()["contrast"]["fields"][
                "account.resident_nm"
            ]
        assert contrast["topGovernanceId"] == "LWP-0001"
        assert contrast["runnerUpGovernanceId"] == "LWP-0004"
        # RESIDENT against OUTAGENOTE, CUSTOMER against OPERATIONS: usually the fact that
        # settles a review, and not derivable from any weight.
        assert contrast["governanceDiffers"] is True
        assert contrast["domainDiffers"] is True

    def test_contrast_does_not_require_explain(self, real_client):
        body = post(real_client, contrast=True).json()
        assert "explain" not in body["results"]["account.resident_nm"][0]
        assert body["contrast"]["fields"]["account.resident_nm"] is not None

    def test_a_contrast_that_cannot_close_its_arithmetic_is_refused(self):
        """
        The same posture as `explain`: a governance document whose arithmetic does not
        close is worse than none, because it is the one used as evidence. A matcher whose
        confidences the published weights do not explain must not get a contrast that
        quietly disagrees with them.
        """
        published = {**WEIGHTS, "domain": 0.99}
        with client_for(ScriptedMatcher(DOMAIN_ONLY, config_weights=published)) as client:
            assert post(client, contrast=True).status_code == 500

    def test_a_clamped_pair_is_not_reported_as_an_arithmetic_failure(self):
        """
        `_weighted_confidence` clamps to [0, 1], so a deployment whose weights sum above
        1.0 can produce two candidates that both land on 1.0: the gap is legitimately 0
        while the weighted differences are not. That is the one case where the two routes
        to the margin are allowed to disagree, and refusing it would turn a tuned-but-
        working configuration into a 500.
        """
        heavy = {**WEIGHTS, "domain": 0.5}  # sums to 1.35
        top = dict.fromkeys(SIGNAL_NAMES, 1.0)
        runner_up = {**top, "domain": 0.9}
        script = [("LWP-0001", top), ("LWP-0004", runner_up)]
        with client_for(ScriptedMatcher(script, weights=heavy)) as client:
            response = post(client, contrast=True)
        assert response.status_code == 200, response.text[:400]
        contrast = response.json()["contrast"]["fields"]["account.resident_nm"]
        assert (contrast["topConfidence"], contrast["runnerUpConfidence"]) == (1.0, 1.0)
        assert contrast["confidenceGap"] == 0.0
        assert contrast["signalGap"] == pytest.approx(0.05, abs=1e-6)
        assert contrast["separation"] == "TIED"

    def test_the_contrast_reads_past_top_k(self):
        """
        The runner-up is a property of what the matcher FOUND, not of how many candidates
        the caller asked to see -- the same reading `fieldDecisions` takes of rank 1.
        """
        with client_for(ScriptedMatcher(DOMAIN_ONLY)) as client:
            body = post(client, contrast=True, top_k=1).json()
        assert len(body["results"]["account.resident_nm"]) == 1
        assert body["contrast"]["fields"]["account.resident_nm"]["runnerUpGovernanceId"] == (
            "LWP-0004"
        )


# =============================================================================
# CONSISTENCY -- THE SAME CONCEPT, ANSWERED TWICE
# =============================================================================

# The LOOSE leaf-only key, which every test in `TestConsistency` asks for explicitly.
#
# It is no longer the default -- it was measured against schemas whose answers are known
# by construction and, on a repeated-leaf schema, produced four findings of which four
# were collisions (`tests/unit/domain/test_review_evidence_grouping.py`). These tests are
# about the ASSESSMENT the block performs once a group exists: agreement, majority,
# silence, key order. They need a group, so they name the key that makes one, rather than
# reaching for a default whose value is a policy decision they are not testing.
LOOSE_KEY = 0

# Three columns that are the same concept under three parents, plus one that is not.
REPEATED_FIELDS = [
    {"name": "resident_nm", "path": "account.resident_nm", "type": "string"},
    {"name": "resident_nm", "path": "billing.resident_nm", "type": "string"},
    {"name": "resident_nm", "path": "ops.resident_nm", "type": "string"},
    {"name": "usage_litres", "path": "meter.usage_litres", "type": "bigint"},
]

RESIDENT = {
    "fusedRetrieval": 0.9,
    "lexical": 0.5,
    "editDistance": 0.5,
    "type": 1.0,
    "domain": 0.5,
}
OUTAGE = {
    "fusedRetrieval": 0.9,
    "lexical": 0.4,
    "editDistance": 0.5,
    "type": 1.0,
    "domain": 0.5,
}


class TestConsistency:
    def test_a_group_that_agrees_is_reported_as_agreeing(self):
        script = [("LWP-0001", RESIDENT), ("LWP-0004", OUTAGE)]
        with client_for(ScriptedMatcher(script)) as client:
            block = post(
                client, REPEATED_FIELDS, consistency=True, consistency_qualifier_segments=LOOSE_KEY
            ).json()["consistency"]
        groups = block["groups"]
        assert len(groups) == 1, groups
        assert groups[0]["fields"] == [
            "account.resident_nm",
            "billing.resident_nm",
            "ops.resident_nm",
        ]
        assert groups[0]["agreement"] == "AGREE"
        assert groups[0]["distinctAnswers"] == 1
        assert block["groupsDisagreeing"] == 0

    def test_a_group_that_disagrees_is_reported_without_ground_truth(self):
        per_path = {
            "ops.resident_nm": [("LWP-0004", OUTAGE), ("LWP-0001", RESIDENT)],
        }
        script = [("LWP-0001", RESIDENT), ("LWP-0004", OUTAGE)]
        with client_for(ScriptedMatcher(script, per_path=per_path)) as client:
            block = post(
                client, REPEATED_FIELDS, consistency=True, consistency_qualifier_segments=LOOSE_KEY
            ).json()["consistency"]
        group = block["groups"][0]
        assert group["agreement"] == "DISAGREE"
        assert group["distinctAnswers"] == 2
        assert group["answers"] == {
            "account.resident_nm": "LWP-0001",
            "billing.resident_nm": "LWP-0001",
            "ops.resident_nm": "LWP-0004",
        }
        assert group["majorityGovernanceId"] == "LWP-0001"
        assert group["majorityCount"] == 2
        assert block["groupsDisagreeing"] == 1

    def test_a_tied_group_has_no_majority(self):
        per_path = {"ops.resident_nm": [("LWP-0002", OUTAGE), ("LWP-0001", RESIDENT)]}
        script = [("LWP-0001", RESIDENT), ("LWP-0004", OUTAGE)]
        fields = REPEATED_FIELDS[:1] + REPEATED_FIELDS[2:3]
        with client_for(ScriptedMatcher(script, per_path=per_path)) as client:
            block = post(
                client, fields, consistency=True, consistency_qualifier_segments=LOOSE_KEY
            ).json()["consistency"]
        group = block["groups"][0]
        assert group["distinctAnswers"] == 2
        assert group["majorityGovernanceId"] is None
        assert group["majorityCount"] == 0

    def test_the_block_reports_and_never_overrides(self):
        """
        Reporting cannot be wrong in a way that changes a classification; promotion can.
        The published `results` and `fieldDecisions` must be identical whether or not the
        consistency pass ran and whatever it found.
        """
        per_path = {"ops.resident_nm": [("LWP-0004", OUTAGE), ("LWP-0001", RESIDENT)]}
        script = [("LWP-0001", RESIDENT), ("LWP-0004", OUTAGE)]
        with client_for(ScriptedMatcher(script, per_path=per_path)) as client:
            plain = post(client, REPEATED_FIELDS).json()
            asked = post(
                client, REPEATED_FIELDS, consistency=True, consistency_qualifier_segments=LOOSE_KEY
            ).json()
        assert json.dumps(asked["results"]) == json.dumps(plain["results"])
        assert json.dumps(asked["fieldDecisions"]) == json.dumps(plain["fieldDecisions"])
        assert asked["consistency"]["promotionApplied"] is False
        assert asked["consistency"]["groups"][0]["agreement"] == "DISAGREE"

    def test_the_group_object_has_the_contract_key_order(self):
        script = [("LWP-0001", RESIDENT), ("LWP-0004", OUTAGE)]
        with client_for(ScriptedMatcher(script)) as client:
            block = post(
                client, REPEATED_FIELDS, consistency=True, consistency_qualifier_segments=LOOSE_KEY
            ).json()["consistency"]
        assert tuple(block["groups"][0]) == GROUP_KEYS

    def test_a_field_with_no_candidates_is_silent_not_contradictory(self):
        """
        A column that matched nothing has no answer. Counting its blank as a second
        answer would report a disagreement in a group where only one column was answered.
        """
        empty: list[tuple[str, dict[str, float]]] = []
        per_path = {"ops.resident_nm": empty, "billing.resident_nm": empty}
        script = [("LWP-0001", RESIDENT), ("LWP-0004", OUTAGE)]
        with client_for(ScriptedMatcher(script, per_path=per_path)) as client:
            block = post(
                client, REPEATED_FIELDS, consistency=True, consistency_qualifier_segments=LOOSE_KEY
            ).json()["consistency"]
        group = block["groups"][0]
        assert group["answers"]["ops.resident_nm"] is None
        assert group["agreement"] == "UNDECIDED"
        assert block["groupsDisagreeing"] == 0

    def test_the_grouping_policy_is_published_and_tunable(self):
        script = [("LWP-0001", RESIDENT), ("LWP-0004", OUTAGE)]
        with client_for(ScriptedMatcher(script)) as client:
            loose = post(
                client, REPEATED_FIELDS, consistency=True, consistency_qualifier_segments=LOOSE_KEY
            ).json()["consistency"]
            tight = post(
                client,
                REPEATED_FIELDS,
                consistency=True,
                consistency_qualifier_segments=1,
            ).json()["consistency"]
        assert loose["grouping"]["qualifierSegments"] == 0
        assert tight["grouping"]["qualifierSegments"] == 1
        # The parent segment differs on all three, so a tighter key finds no group at all.
        assert loose["groups"] and not tight["groups"]

    def test_the_default_grouping_reports_nothing_on_a_repeated_leaf(self):
        """
        THE DEFAULT, ON THE WIRE, ON THE SHAPE THE FEATURE EXISTS FOR.

        These three columns share a leaf under three different records -- the miniature of
        the generated repeated-leaf schema where the loose key merges 87 columns spanning
        29 answers and reports the merge as a contradiction. A caller who switches
        `consistency` on and sends this gets an EMPTY report rather than a confident wrong
        one, and the policy that produced the emptiness is published beside it so the
        emptiness is legible rather than mysterious.
        """
        script = [("LWP-0001", RESIDENT), ("LWP-0004", OUTAGE)]
        with client_for(ScriptedMatcher(script)) as client:
            block = post(client, REPEATED_FIELDS, consistency=True).json()["consistency"]
        assert block["grouping"]["qualifierSegments"] == 1
        assert block["groups"] == []
        assert (block["groupsFound"], block["fieldsGrouped"], block["groupsDisagreeing"]) == (
            0,
            0,
            0,
        )

    def test_columns_that_share_nothing_produce_no_groups(self, real_client):
        block = post(
            real_client, consistency=True, consistency_qualifier_segments=LOOSE_KEY
        ).json()["consistency"]
        assert block["groups"] == []
        assert block["fieldsGrouped"] == 0

    def test_the_counts_agree_with_the_groups_they_summarise(self):
        per_path = {"ops.resident_nm": [("LWP-0004", OUTAGE), ("LWP-0001", RESIDENT)]}
        script = [("LWP-0001", RESIDENT), ("LWP-0004", OUTAGE)]
        with client_for(ScriptedMatcher(script, per_path=per_path)) as client:
            block = post(
                client, REPEATED_FIELDS, consistency=True, consistency_qualifier_segments=LOOSE_KEY
            ).json()["consistency"]
        assert block["groupsFound"] == len(block["groups"])
        assert block["fieldsGrouped"] == sum(len(g["fields"]) for g in block["groups"])
        assert block["groupsDisagreeing"] == sum(
            1 for g in block["groups"] if g["agreement"] == "DISAGREE"
        )


# =============================================================================
# THE REQUEST CONTRACT
# =============================================================================


class TestRequestContract:
    def test_the_qualifier_bound_is_refused_rather_than_clamped(self, real_client):
        """
        Below the floor is a mistake and is refused. Above the ceiling is refused too --
        and the ceiling is DERIVED from the `path` length bound rather than picked, so it
        is the largest value that can still change a concept key rather than a literal
        that happened to look big enough.
        """
        from nexus_matcher.presentation.api.schemas import MAX_QUALIFIER_SEGMENTS

        assert post(
            real_client, consistency=True, consistency_qualifier_segments=-1
        ).status_code == (422)
        assert (
            post(
                real_client,
                consistency=True,
                consistency_qualifier_segments=MAX_QUALIFIER_SEGMENTS + 1,
            ).status_code
            == 422
        )
        # ... and everything inside it is accepted, including the values the previous
        # literal ceiling of 8 refused for no reason it could state.
        for segments in (0, 1, 9, MAX_QUALIFIER_SEGMENTS):
            assert (
                post(
                    real_client, consistency=True, consistency_qualifier_segments=segments
                ).status_code
                == 200
            ), segments

    def test_the_published_bound_follows_from_the_path_bound(self, real_client):
        """
        The bound is derived, and the derivation is checked against the live document
        rather than against the constant that produced it. A ceiling that stopped tracking
        `_MAX_PATH` would publish a range that refuses values this endpoint can
        distinguish, which is the defect the literal 8 was.
        """
        from nexus_matcher.domain.services.review_evidence import max_qualifier_segments
        from nexus_matcher.presentation.api.schemas import _MAX_PATH, MAX_QUALIFIER_SEGMENTS

        assert max_qualifier_segments(_MAX_PATH) == MAX_QUALIFIER_SEGMENTS
        published = real_client.get("/openapi.json").json()["components"]["schemas"]
        parameter = published["MatchRequest"]["properties"]["consistency_qualifier_segments"]
        assert parameter["maximum"] == MAX_QUALIFIER_SEGMENTS
        assert parameter["minimum"] == 0
        assert parameter["default"] == 1

    def test_a_no_match_field_contributes_no_answer(self):
        """
        `fieldDecisions` is the field-level authority. A NO_MATCH column inherits nothing,
        so its rank-1 candidate is evidence for a reviewer and not an answer -- feeding it
        to the consistency pass would manufacture a disagreement with a column that has one.
        """
        entry = DictionaryEntry(
            id="LWP-0009",
            business_name="Unrelated",
            logical_name="unrelated",
            definition="Nothing.",
            data_type=DataType.STRING,
            protection_level=ProtectionLevel.PUBLIC,
            governance_code=None,
            domain="NONE",
        )
        assert entry.id not in {e.id for e in GLOSSARY}
        empty: list[tuple[str, dict[str, float]]] = []
        script = [("LWP-0001", RESIDENT), ("LWP-0004", OUTAGE)]
        with client_for(ScriptedMatcher(script, per_path={"ops.resident_nm": empty})) as client:
            body = post(
                client, REPEATED_FIELDS, consistency=True, consistency_qualifier_segments=LOOSE_KEY
            ).json()
        assert body["fieldDecisions"]["ops.resident_nm"] == "NO_MATCH"
        assert body["consistency"]["groups"][0]["answers"]["ops.resident_nm"] is None


# =============================================================================
# THE TWO VOCABULARIES THESE BLOCKS PUBLISH
# =============================================================================


class TestTheClosedVocabularies:
    """
    `agreement` and `separation` are the LIBRARY'S OWN words, so they close.

    The rule this repository holds to is not "enums are good" or "enums are bad", it is
    WHOSE VOCABULARY IT IS. `MatchDecision` and `FieldDecision` close because the library
    computes them and has committed to the value set; a governance code, a protection
    class, an encoder tier and a domain stay open strings on both sides because they are
    the caller's taxonomy and closing one would hard-code a customer into the schema.

    `AGREE/DISAGREE/UNDECIDED` and `SEPARATED/TIED` are in the first category and were
    published as bare `str` with nothing gating their values -- the safe direction, but
    the same category of thing published two different ways, which is how one of them
    quietly acquires a sixth value nobody publishes. They are named components now.
    """

    @pytest.fixture
    def schemas(self, real_client):
        return real_client.get("/openapi.json").json()["components"]["schemas"]

    def test_agreement_and_separation_are_named_components_like_match_decision(self, schemas):
        from nexus_matcher.domain.services.review_evidence import Agreement, Separation

        for view, member, source in (
            ("ConceptGroupView", "agreement", Agreement),
            ("ContrastView", "separation", Separation),
        ):
            reference = schemas[view]["properties"][member]
            assert reference["$ref"] == f"#/components/schemas/{source.__name__}", reference
            # Typed as the DOMAIN enum, not as a hand-written Literal beside it: a second
            # copy of the value list is the drift that makes an audit surface
            # self-consistently wrong.
            assert set(schemas[source.__name__]["enum"]) == {m.value for m in source}

    def test_the_values_published_are_the_values_the_service_emits(self, schemas):
        """
        A closed enum a client generates from is a promise that no other value arrives.
        Asserted against real bodies rather than against the enum class, because the class
        is what produced the schema and comparing it to itself proves nothing.
        """
        script = [("LWP-0001", RESIDENT), ("LWP-0004", OUTAGE)]
        per_path = {"ops.resident_nm": [("LWP-0004", OUTAGE), ("LWP-0001", RESIDENT)]}
        with client_for(ScriptedMatcher(script, per_path=per_path)) as client:
            body = post(
                client,
                REPEATED_FIELDS,
                contrast=True,
                consistency=True,
                consistency_qualifier_segments=LOOSE_KEY,
            ).json()

        seen_agreement = {group["agreement"] for group in body["consistency"]["groups"]}
        seen_separation = {
            contrast["separation"]
            for contrast in body["contrast"]["fields"].values()
            if contrast is not None
        }
        assert seen_agreement and seen_separation
        assert seen_agreement <= set(schemas["Agreement"]["enum"])
        assert seen_separation <= set(schemas["Separation"]["enum"])

    def test_no_caller_supplied_vocabulary_closed_alongside_them(self, schemas):
        """
        The other half of the rule, checked so that "make the library's words consistent"
        cannot drift into "close everything". Every enum component the document publishes
        must be one the library owns; a governance code, a protection class or a domain
        appearing here would be a customer's taxonomy frozen into the contract.

        The set is pinned EXACTLY, not as a subset, so a new closed vocabulary has to be
        argued here before it can ship. `MatchProvenanceView` was added that way: it is
        RETRIEVAL | APPROVED_PAIR, decided by this library and by nothing else, and it
        exists because `confidence` turned out NOT to discriminate a scored candidate from
        a reviewer's answer -- the weights sum to exactly 1.0, so retrieval can reach the
        same 1.0 the approved-pair path writes.
        """
        published = {name for name, schema in schemas.items() if "enum" in schema}
        assert published == {
            "MatchDecision",
            "FieldDecision",
            "Agreement",
            "Separation",
            "MatchProvenanceView",
        }, published
        for view, member in (
            ("GovernanceView", "code"),
            ("GovernanceView", "classification"),
            ("VocabularyView", "openClassification"),
        ):
            assert "enum" not in json.dumps(schemas[view]["properties"][member]), (view, member)
