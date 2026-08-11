"""
tests.unit.application.test_governance_on_match_results | Layer: TEST
The two fields a caller actually came for: which entry, and what class it confers.

A caller matches a field IN ORDER TO inherit the entry's governance. If that answer is not
on the MatchResult, every consumer has to re-derive it -- and the ones that forget quietly
publish a schema with no classification on it, which is NM-0005's failure with a different
first line.

Four properties, each with a silent failure mode:

  1. `governance_id` is always populated, and always names the matched entry
  2. `governance` is resolved on EVERY returned candidate, not only rank 1
  3. a REJECT confers NOTHING -- a novel field must never inherit the class of the
     least-bad candidate the matcher itself rejected
  4. indexing entries whose codes the wired vocabulary cannot resolve is refused, rather
     than degrading to `governance=None` on every match

The vocabulary is fictional (see tests/unit/domain/test_governance.py). The encoder is a
stub with hand-chosen orthogonal vectors, so no model is downloaded and no ranking depends
on a learned embedding.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.governance import GovernanceVocabulary
from nexus_matcher.domain.models.entities import (
    DictionaryEntry,
    MatchResult,
    SchemaField,
)
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import (
    DataType,
    MatchDecision,
    PerformanceMetrics,
    Result,
    ScoreBreakdown,
)

THORNBURY = {
    "open_classification": "Open",
    "classes": [
        {
            "code": "METERID",
            "name": "Meter Serial Identifier",
            "classification": "Sealed",
            "personal_information": True,
            "direct_identifier": True,
        },
        {
            "code": "USAGE",
            "name": "Metered Consumption Reading",
            "classification": "Guarded",
            "personal_information": True,
            "direct_identifier": False,
        },
        {
            "code": "PUBMAP",
            "name": "Published Network Map Reference",
            "classification": "Open",
            "personal_information": False,
            "direct_identifier": False,
        },
    ],
}

# Hand-chosen orthogonal vectors in 4 dimensions, one reserved dimension per entry, so the
# ranking is fixed by construction and checkable by hand rather than learned.
#
# The query's components ARE the four cosines. They are spaced so the fixture exercises
# every case this file is about at once: the top three clear `review_threshold` (0.50) and
# the fourth does not, and the UNCODED entry lands at rank 2 -- inside the reviewed set --
# so "an entry with no code confers no class" is proved on a match that was not rejected.
# Were the uncoded entry last, the REJECT rule alone would satisfy that test and it would
# prove nothing about the open tier.
_VECTORS = {
    "Meter Serial": (1.0, 0.0, 0.0, 0.0),
    "Depot Rota Note": (0.0, 1.0, 0.0, 0.0),
    "Quarterly Reading": (0.0, 0.0, 1.0, 0.0),
    "Trunk Main Reference": (0.0, 0.0, 0.0, 1.0),
}
_QUERY = (0.80, 0.70, 0.55, 0.10)


class _StubProvider:
    """Model-free encoder with DISTINCT vectors: no download, no network, no ties."""

    dimension = 4
    model_name = "stub"

    def _vector(self, text: str) -> tuple[float, ...]:
        for name, vector in _VECTORS.items():
            if name in text:
                return vector
        return _QUERY

    def embed(self, texts):
        rows = np.array([self._vector(t) for t in texts], dtype=np.float32)

        class _Batch:
            embeddings = rows

        return Result.success(_Batch())

    def embed_single(self, text):
        return Result.success(np.array(self._vector(text), dtype=np.float32))


def _entry(eid: str, business: str, code: str | None) -> DictionaryEntry:
    return DictionaryEntry(
        id=eid,
        business_name=business,
        logical_name="",
        definition=f"Definition of {business.lower()} in the supply network.",
        data_type=DataType.STRING,
        governance_code=code,
        domain="NETWORK",
    )


ENTRIES = [
    _entry("TWA-1001", "Meter Serial", "METERID"),
    # No code at all: the "unclassified" case, which sits at the open tier and confers
    # no class. It must be distinguishable from "we could not resolve the code".
    _entry("TWA-1002", "Depot Rota Note", None),
    _entry("TWA-1003", "Quarterly Reading", "USAGE"),
    _entry("TWA-1004", "Trunk Main Reference", "PUBMAP"),
]

FIELD = SchemaField(
    name="meter_serial",
    data_type=DataType.STRING,
    full_path="supply.meter_serial",
    parent_path="supply",
    description="Serial of the meter at the supply point",
)


def _matcher(governance=THORNBURY, entries=ENTRIES) -> NexusMatcher:
    matcher = NexusMatcher(
        embedding_provider=_StubProvider(),
        # No sparse retriever and no reranker: the fused score is then exactly the min-max
        # normalised dense score, so the ranking is the cosine ranking and nothing here
        # depends on BM25 or a learned reranker.
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=4)
        ),
        config=MatchingConfig(results_per_field=4),
        governance=governance,
    )
    matcher._index_dictionary(entries)
    return matcher


@pytest.fixture(scope="module")
def matches() -> tuple[MatchResult, ...]:
    return tuple(_matcher()._match_field(FIELD))


# =============================================================================
# THE PREMISE
# =============================================================================


def test_the_premise_all_four_candidates_come_back_ranked(matches):
    """
    Guards every test below against passing vacuously. If the fixture returned one
    candidate, "every candidate carries a class" would be a statement about one object,
    and the runner-up assertions would prove nothing.
    """
    assert [m.dictionary_entry.id for m in matches] == [
        "TWA-1001",
        "TWA-1002",
        "TWA-1003",
        "TWA-1004",
    ]
    assert [m.rank for m in matches] == [1, 2, 3, 4]


def test_the_premise_three_are_reviewed_and_the_last_is_rejected(matches):
    """
    The other half of the premise, pinned so the fixture cannot drift into proving less.

    Both the runner-up tests and the REJECT tests need a mixed decision set; a fixture
    where everything scored the same would satisfy each of them for the wrong reason.
    """
    assert [m.decision for m in matches] == [
        MatchDecision.REVIEW,
        MatchDecision.REVIEW,
        MatchDecision.REVIEW,
        MatchDecision.REJECT,
    ]


# =============================================================================
# GOVERNANCE ID
# =============================================================================


class TestGovernanceId:
    def test_it_is_populated_on_every_result(self, matches):
        assert all(m.governance_id for m in matches)

    def test_it_names_the_matched_entry(self, matches):
        assert [m.governance_id for m in matches] == [m.dictionary_entry.id for m in matches]

    def test_it_is_derived_rather_than_left_to_a_caller_to_remember(self):
        """A MatchResult built by hand, with no governance_id given, still has one."""
        result = _hand_built(decision=MatchDecision.REVIEW)
        assert result.governance_id == "TWA-1001"

    def test_a_governance_id_that_disagrees_with_the_entry_is_refused(self):
        """
        Two answers to "whose class is this?" is worse than none: the consumer that reads
        `governance_id` and the consumer that reads `dictionary_entry.id` would classify
        the same field differently and neither would be visibly wrong.
        """
        with pytest.raises(ValueError, match="does not name the matched entry"):
            _hand_built(decision=MatchDecision.REVIEW, governance_id="TWA-9999")


# =============================================================================
# GOVERNANCE CLASS
# =============================================================================


class TestGovernanceClass:
    def test_the_top_match_confers_the_class_its_code_derives(self, matches):
        """
        Pinned absolutely against the fictional catalog, not derived from the vocabulary
        under test (H-004): computing the expectation with `classification_for` would
        agree with an implementation that read the tier off the glossary row.
        """
        top = matches[0].governance
        assert top is not None
        assert (top.code, top.classification) == ("METERID", "Sealed")
        assert (top.personal_information, top.direct_identifier) == (True, True)

    def test_every_runner_up_carries_its_class_too(self, matches):
        """
        A consumer deciding between rank 1 and rank 2 needs to see that one is a direct
        identifier and the other is not. That is usually the deciding fact, and having it
        on rank 1 alone makes the comparison impossible without a second lookup.
        """
        reviewable = [m for m in matches if not m.is_rejected]
        assert len(reviewable) >= 2, "fixture must return more than one non-rejected match"
        assert all(
            m.governance is not None for m in reviewable if m.dictionary_entry.governance_code
        )

    def test_the_ranks_carry_DIFFERENT_classes(self, matches):
        """
        Guards the shape a constant would satisfy: if `governance` were wired to a single
        value, "every candidate has one" would still pass.
        """
        classes = {m.governance.code for m in matches if m.governance is not None}
        assert len(classes) >= 2, f"every candidate resolved to the same class: {classes}"

    def test_an_entry_with_no_code_confers_no_class(self, matches):
        """
        The open-tier case, and it must be distinguishable from a resolution failure: the
        entry is present, matched, ranked and NOT rejected, and simply carries nothing.
        """
        uncoded = next(m for m in matches if m.dictionary_entry.id == "TWA-1002")
        assert uncoded.decision is not MatchDecision.REJECT, (
            "a rejected match confers nothing anyway, so this would prove nothing"
        )
        assert uncoded.dictionary_entry.governance_code is None
        assert uncoded.governance is None
        assert uncoded.governance_id == "TWA-1002"


# =============================================================================
# A REJECT INHERITS NOTHING
# =============================================================================


class TestRejectConfersNothing:
    def test_a_rejected_match_carries_no_class(self, matches):
        rejected = [m for m in matches if m.is_rejected]
        assert rejected, (
            "the fixture produced no REJECT, so this file proves nothing about them -- "
            f"decisions were {[m.decision.value for m in matches]}"
        )
        for match in rejected:
            assert match.governance is None, (
                f"{match.dictionary_entry.id} was REJECTED and still confers "
                f"{match.governance}. A novel field would inherit the class of the "
                f"least-bad candidate the matcher itself rejected."
            )

    def test_a_rejected_match_still_says_which_entry_it_was(self, matches):
        """
        Dropping the id as well would leave a reviewer unable to see what was rejected.
        The CLASS is what must not be inherited; the identity is evidence.
        """
        for match in (m for m in matches if m.is_rejected):
            assert match.governance_id == match.dictionary_entry.id

    def test_the_domain_model_enforces_it_rather_than_the_matcher_remembering_to(self):
        """
        Constructed directly, with a class explicitly handed in. It is dropped anyway --
        so a future call site that forgets cannot reintroduce the defect.
        """
        assert (
            _hand_built(decision=MatchDecision.REJECT, governance=_class("METERID")).governance
            is None
        )

    def test_a_non_rejected_match_keeps_the_class_it_was_given(self):
        """The control. A rule that dropped governance unconditionally would satisfy the above."""
        for decision in (MatchDecision.REVIEW, MatchDecision.AUTO_APPROVE):
            kept = _hand_built(decision=decision, governance=_class("METERID"))
            assert kept.governance is not None and kept.governance.code == "METERID"

    def test_a_field_nothing_matches_produces_no_result_to_inherit_from(self):
        """
        The other half of "a novel field inherits nothing": with nothing in the
        dictionary there is no candidate at all, so there is no MatchResult and nothing
        to read a class off. `MatchingSession.get_low_confidence_fields()` already flags
        such a field, so it reaches a human rather than being silently unclassified.
        """
        assert _matcher(governance=None, entries=[])._match_field(FIELD) == []


# =============================================================================
# WIRING
# =============================================================================


class TestVocabularyWiring:
    def test_indexing_a_code_the_matcher_cannot_resolve_is_refused(self):
        """
        The half-wired case: a glossary validated against one vocabulary handed to a
        matcher holding another. Left unchecked it degrades silently -- every MatchResult
        comes back with `governance=None`, which is indistinguishable from "this entry has
        no class", and a field inherits nothing where it should have inherited something.
        """
        with pytest.raises(ValueError, match="not in this matcher's governance vocabulary"):
            _matcher(governance=None)

    def test_the_refusal_names_the_offending_codes_and_the_fix(self):
        with pytest.raises(ValueError) as excinfo:
            _matcher(governance=GovernanceVocabulary.from_json(THORNBURY["classes"][:1]))

        message = str(excinfo.value)
        assert "USAGE" in message and "PUBMAP" in message
        assert "METERID" not in message.split("Vocabulary declares:", maxsplit=1)[0]
        assert "from_config(governance=" in message

    def test_entries_with_no_codes_need_no_vocabulary(self):
        """
        The feature is opt-in. A caller who has never heard of it must not be made to
        configure one, so a dictionary carrying no codes indexes against `empty()`.
        """
        matcher = _matcher(governance=None, entries=[_entry("TWA-1002", "Depot Rota Note", None)])
        assert matcher._match_field(FIELD)[0].governance is None


# =============================================================================
# HELPERS
# =============================================================================


def _class(code: str):
    return GovernanceVocabulary.from_json(THORNBURY).get(code)


def _hand_built(*, decision: MatchDecision, **overrides) -> MatchResult:
    """A MatchResult built without the matcher, to pin the DOMAIN model's own rules."""
    return MatchResult(
        schema_field=FIELD,
        dictionary_entry=ENTRIES[0],
        rank=1,
        final_confidence=0.9,
        score_breakdown=ScoreBreakdown(),
        decision=decision,
        performance=PerformanceMetrics(latency_ms=0.0),
        **overrides,
    )
