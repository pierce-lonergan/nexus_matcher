"""
tests.unit.application.test_scoring_hot_path | Layer: TEST
The per-candidate loop carries its five signals as a plain tuple and only widens the
ones it actually returns into a ScoreBreakdown. These pin what makes that safe.

Both failure modes below are silent -- they produce a plausible number rather than an
exception, which is why they get tests instead of trust:

  * widening the five signals into ScoreBreakdown positionally lands `domain_score` in
    `colbert_score`, because the dataclass declares two reranker fields in between; the
    result reports a reranker score that never ran, and loses the domain signal
  * the confidence a result is RANKED by is computed from the tuple, while the breakdown
    a caller INSPECTS is built later; if those two ever disagree, every reported score
    becomes a plausible-looking lie about why the match won

An earlier revision of this file also pinned an index-time entry-token cache and a
per-field hoist of the field tokens and inferred domain. Both were reverted: they moved
rankings on the FHIR corpus, and a scoring change does not get to arrive inside an
object-allocation refactor. Do not reintroduce them without a per-query identity check.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application.use_cases.match_schema import (
    MatchingConfig,
    NexusMatcher,
    _breakdown,
    _signal_weights,
    _weighted_confidence,
)
from nexus_matcher.core.fusion import FusionStats
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import DataType, ProtectionLevel, Result, ScoreBreakdown


class _StubProvider:
    """Model-free encoder: identical unit vectors, so no download and no network."""

    dimension = 8
    model_name = "stub"

    def embed(self, texts):
        rows = np.tile(np.eye(1, 8, 0, dtype=np.float32), (len(list(texts)), 1))

        class _Batch:
            embeddings = rows

        return Result.success(_Batch())

    def embed_single(self, text):
        return Result.success(np.eye(1, 8, 0, dtype=np.float32)[0])


def _entry(eid: str, business: str, logical: str, domain: str) -> DictionaryEntry:
    return DictionaryEntry(
        id=eid,
        business_name=business,
        logical_name=logical,
        definition=f"Definition of {business}",
        data_type=DataType.STRING,
        protection_level=ProtectionLevel.INTERNAL,
        domain=domain,
    )


FIRST = [
    _entry("a1", "Customer Email Address", "cust_email", "CONTACT"),
    _entry("a2", "Customer Identifier", "cust_id", "CUSTOMER"),
]
SECOND = [
    _entry("b1", "Merchant Postal Code", "merch_zip", "ADDRESS"),
]


@pytest.fixture
def matcher() -> NexusMatcher:
    return NexusMatcher(
        embedding_provider=_StubProvider(),
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=8)
        ),
        config=MatchingConfig(results_per_field=3),
    )


FIELD = SchemaField(
    name="CustEmailAddr",
    data_type=DataType.STRING,
    full_path="customer.CustEmailAddr",
    parent_path="customer",
)


class TestSignalsAreScoredFromTheEntryInFront:
    """Scoring reads the entry it was handed, with nothing carried over from before."""

    def test_reindexing_rescopes_the_dictionary(self, matcher):
        """A second load_dictionary() leaves nothing of the first one behind."""
        matcher._index_dictionary(FIRST)
        assert set(matcher._dictionary_entries) == {"a1", "a2"}

        matcher._index_dictionary(SECOND)
        assert set(matcher._dictionary_entries) == {"b1"}

    def test_unindexed_entry_scores_like_its_indexed_twin(self, matcher):
        """An entry never seen by the index scores identically to an identical indexed one."""
        matcher._index_dictionary(FIRST)
        emb = np.eye(1, 8, 0, dtype=np.float32)[0]
        stranger = _entry("zz", "Customer Email Address", "cust_email", "CONTACT")

        assert matcher._calculate_scores(FIELD, stranger, emb, 0.77) == matcher._calculate_scores(
            FIELD, FIRST[0], emb, 0.77
        )

    def test_lexical_signal_uses_both_names_and_splits_case(self, matcher):
        """Token overlap unions logical and business names, and CustEmailAddr is three tokens."""
        matcher._index_dictionary(FIRST)
        emb = np.eye(1, 8, 0, dtype=np.float32)[0]

        # FIELD tokens {cust, email, addr} vs "Customer Email Address" + "cust_email"
        # -> {customer, email, address, cust}. Intersection {cust, email} over 3 tokens.
        scores = matcher._calculate_scores(FIELD, FIRST[0], emb, 0.77)
        assert scores.lexical_score == pytest.approx(2 / 3)


class TestBreakdownWiring:
    """ScoreBreakdown has two reranker fields between type and domain."""

    def test_domain_signal_lands_in_domain_score(self):
        """Widening the signal tuple must not shift domain into colbert_score."""
        bd = _breakdown((0.11, 0.22, 0.33, 0.44, 0.55))

        assert bd.semantic_score == 0.11
        assert bd.lexical_score == 0.22
        assert bd.edit_distance_score == 0.33
        assert bd.type_compatibility_score == 0.44
        assert bd.domain_score == 0.55
        assert bd.colbert_score is None, "domain score leaked into the ColBERT slot"
        assert bd.cross_encoder_score is None

    def test_confidence_agrees_between_tuple_and_breakdown(self, matcher):
        """The ranking loop and _calculate_final_confidence share one weighted sum."""
        signals = (0.9, 0.5, 0.25, 1.0, 0.5)
        from_tuple = _weighted_confidence(signals, _signal_weights(matcher._config))
        from_breakdown = matcher._calculate_final_confidence(_breakdown(signals))

        assert from_tuple == from_breakdown

    def test_confidence_is_clamped(self, matcher):
        """A weight set that oversums must still yield a probability-shaped number."""
        weights = (10.0, 10.0, 10.0, 10.0, 10.0)
        assert _weighted_confidence((1.0, 1.0, 1.0, 1.0, 1.0), weights) == 1.0
        assert _weighted_confidence((-5.0, 0.0, 0.0, 0.0, 0.0), weights) == 0.0


class TestEmittedResultsCarryTheirBreakdown:
    """Only returned results get a ScoreBreakdown built; it must still be the right one."""

    def test_result_breakdown_reproduces_its_confidence(self, matcher):
        """Every returned match's breakdown re-derives the confidence it was ranked by."""
        matcher._index_dictionary(FIRST)
        results = matcher._match_field(FIELD)

        assert results
        for r in results:
            assert isinstance(r.score_breakdown, ScoreBreakdown)
            assert matcher._calculate_final_confidence(r.score_breakdown) == pytest.approx(
                r.final_confidence
            )
            assert r.score_breakdown.colbert_score is None


class TestFusionStatsMean:
    """avg_overlap_ratio is a diagnostic, so it has to actually be the mean it claims."""

    def test_recorded_mean_matches_the_naive_mean(self):
        """avg_overlap_ratio equals the plain average of every recorded overlap."""
        stats = FusionStats()
        ratios = []
        for both, sem_only, lex_only in ((3, 1, 1), (0, 5, 5), (7, 2, 1), (1, 0, 9)):
            stats.record(
                num_items=both + sem_only + lex_only,
                num_in_both=both,
                num_in_semantic_only=sem_only,
                num_in_lexical_only=lex_only,
            )
            ratios.append(both / (both + sem_only + lex_only))

        assert stats.total_fusions == 4
        assert stats.avg_overlap_ratio == pytest.approx(sum(ratios) / len(ratios))

    def test_empty_fusion_does_not_enter_the_mean(self):
        """A fusion with no candidates has no overlap ratio to average."""
        stats = FusionStats()
        stats.record(num_items=0, num_in_both=0, num_in_semantic_only=0, num_in_lexical_only=0)

        assert stats.total_fusions == 1
        assert stats.avg_overlap_ratio == 0.0
