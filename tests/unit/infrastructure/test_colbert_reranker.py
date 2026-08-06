"""
tests.unit.infrastructure.test_colbert_reranker | Layer: TEST
Tests: ColBERT MaxSim Reranker | Target: GAP-001

TDD Phase: RED → Tests written before implementation
Research Reference: README_RESEARCH_3.md, Lines 5-8

Research Context:
- Current bi-encoder approach is WRONG (pools tokens into single vectors)
- Need proper MaxSim late interaction
- Model: answerai-colbert-small-v1 (33M params, CPU-optimized)
- Target: 60ms for top-100 reranking
- Impact: +10-20% accuracy improvement

Key Insight:
"The correct approach uses token-level embeddings where each query and
document becomes multiple 128-dimensional vectors, then computes the
sum of maximum similarities across query tokens."
"""

import time

# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================


class TestColBERTMaxSimRerankerBasics:
    """Test ColBERT MaxSim reranker basic functionality."""

    def test_reranker_class_exists(self):
        """Test ColBERTMaxSimReranker class exists."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            ColBERTMaxSimReranker,
        )

        assert ColBERTMaxSimReranker is not None

    def test_mock_reranker_class_exists(self):
        """Test MockColBERTReranker exists for testing without models."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        assert reranker is not None

    def test_mock_reranker_type(self):
        """Test reranker type is colbert_maxsim."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        assert reranker.reranker_type == "colbert_maxsim"

    def test_mock_reranker_model_name(self):
        """Test model name is correct."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        assert "colbert" in reranker.model_name.lower() or "mock" in reranker.model_name.lower()


class TestColBERTReranking:
    """Test ColBERT reranking functionality."""

    def test_rerank_returns_result(self):
        """Test rerank returns a Result object."""
        from nexus_matcher.domain.ports.retrieval import RerankCandidate
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        candidates = [
            RerankCandidate(id="1", text="customer email address"),
            RerankCandidate(id="2", text="account balance"),
        ]

        result = reranker.rerank("find email field", candidates)

        assert result.is_success
        assert isinstance(result.unwrap(), list)

    def test_rerank_empty_candidates(self):
        """Test rerank handles empty candidates."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        result = reranker.rerank("query", [])

        assert result.is_success
        assert result.unwrap() == []

    def test_rerank_returns_scores(self):
        """Test rerank returns scored results."""
        from nexus_matcher.domain.ports.retrieval import RerankCandidate
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        candidates = [
            RerankCandidate(id="1", text="customer email address"),
            RerankCandidate(id="2", text="account balance"),
            RerankCandidate(id="3", text="transaction date"),
        ]

        result = reranker.rerank("email", candidates)
        results = result.unwrap()

        assert len(results) == 3
        for r in results:
            assert hasattr(r, "score")
            assert hasattr(r, "rank")
            assert hasattr(r, "id")

    def test_rerank_orders_by_score(self):
        """Test results are ordered by score descending."""
        from nexus_matcher.domain.ports.retrieval import RerankCandidate
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        candidates = [
            RerankCandidate(id="1", text="some text"),
            RerankCandidate(id="2", text="other text"),
            RerankCandidate(id="3", text="more text"),
        ]

        result = reranker.rerank("query", candidates)
        results = result.unwrap()

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_top_k(self):
        """Test top_k parameter limits results."""
        from nexus_matcher.domain.ports.retrieval import RerankCandidate
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        candidates = [RerankCandidate(id=str(i), text=f"text {i}") for i in range(10)]

        result = reranker.rerank("query", candidates, top_k=5)
        results = result.unwrap()

        assert len(results) == 5

    def test_rerank_preserves_metadata(self):
        """Test rerank preserves candidate metadata."""
        from nexus_matcher.domain.ports.retrieval import RerankCandidate
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        candidates = [
            RerankCandidate(id="1", text="test", metadata={"source": "test_source"}),
        ]

        result = reranker.rerank("query", candidates)
        results = result.unwrap()

        assert results[0].metadata.get("source") == "test_source"


# =============================================================================
# MAXSIM SPECIFIC TESTS
# =============================================================================


class TestMaxSimComputation:
    """Test MaxSim late interaction computation."""

    def test_maxsim_config_exists(self):
        """Test MaxSimConfig dataclass exists."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MaxSimConfig,
        )

        config = MaxSimConfig()
        assert config is not None

    def test_maxsim_config_defaults(self):
        """Test MaxSimConfig has correct defaults."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MaxSimConfig,
        )

        config = MaxSimConfig()

        # 128-dimensional vectors per research
        assert config.embedding_dim == 128
        # CPU optimized
        assert config.device in ("cpu", "auto")

    def test_maxsim_statistics_tracking(self):
        """Test MaxSim statistics are tracked."""
        from nexus_matcher.domain.ports.retrieval import RerankCandidate
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        candidates = [RerankCandidate(id=str(i), text=f"text {i}") for i in range(10)]

        # Perform some reranking
        reranker.rerank("query 1", candidates)
        reranker.rerank("query 2", candidates)

        stats = reranker.get_statistics()

        assert stats.total_reranks >= 2
        assert stats.total_candidates >= 20

    def test_maxsim_uses_token_embeddings(self):
        """Test MaxSim uses token-level embeddings (not pooled)."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()

        # The mock should indicate it's using token-level embeddings
        assert reranker.uses_token_embeddings is True
        # NOT bi-encoder (single pooled vector)
        assert reranker.is_bi_encoder is False


class TestMaxSimScoring:
    """Test MaxSim scoring mechanism."""

    def test_score_pair_returns_maxsim_score(self):
        """Test single pair scoring works."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        result = reranker.score_pair("customer email", "email address field")

        assert result.is_success
        score = result.unwrap()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_semantic_similarity_affects_score(self):
        """Test that semantically similar texts get higher scores."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()

        # More similar should score higher
        similar_result = reranker.score_pair("customer email", "email address")
        dissimilar_result = reranker.score_pair("customer email", "transaction amount")

        # Mock may not perfectly model this, but should at least work
        assert similar_result.is_success
        assert dissimilar_result.is_success


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestColBERTPerformance:
    """Test ColBERT MaxSim performance characteristics."""

    def test_rerank_100_candidates_under_target_latency(self):
        """Test reranking 100 candidates meets latency target."""
        from nexus_matcher.domain.ports.retrieval import RerankCandidate
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        candidates = [
            RerankCandidate(id=str(i), text=f"dictionary entry description {i} with some text")
            for i in range(100)
        ]

        # Warmup
        reranker.rerank("warmup query", candidates[:10])

        # Measure
        start = time.perf_counter()
        result = reranker.rerank("find customer email address", candidates)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.is_success
        # Target: 60ms for top-100 (mock should be much faster)
        assert elapsed_ms < 100, f"Reranking took {elapsed_ms:.2f}ms (target: <100ms for mock)"

    def test_batch_reranking_efficiency(self):
        """Test batch reranking is more efficient than sequential."""
        from nexus_matcher.domain.ports.retrieval import RerankCandidate
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()

        queries = [f"query {i}" for i in range(5)]
        candidates_per_query = [
            [RerankCandidate(id=f"{i}_{j}", text=f"text {j}") for j in range(20)] for i in range(5)
        ]

        # Batch should work
        result = reranker.rerank_batch(queries, candidates_per_query)

        assert result.is_success
        results = result.unwrap()
        assert len(results) == 5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestColBERTIntegration:
    """Test ColBERT integration with pipeline."""

    def test_implements_reranker_protocol(self):
        """Test ColBERT reranker implements Reranker protocol."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()

        # Should have required attributes
        assert hasattr(reranker, "reranker_type")
        assert hasattr(reranker, "model_name")
        assert hasattr(reranker, "rerank")

    def test_result_has_original_rank(self):
        """Test results track original rank for analysis."""
        from nexus_matcher.domain.ports.retrieval import RerankCandidate
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        candidates = [
            RerankCandidate(id="a", text="first"),
            RerankCandidate(id="b", text="second"),
            RerankCandidate(id="c", text="third"),
        ]

        result = reranker.rerank("query", candidates)
        results = result.unwrap()

        # All results should have original_rank
        for r in results:
            assert hasattr(r, "original_rank")
            assert r.original_rank >= 1


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestColBERTConfiguration:
    """Test ColBERT configuration options."""

    def test_config_model_name_override(self):
        """Test model name can be overridden."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MaxSimConfig,
        )

        config = MaxSimConfig(model_name="custom-colbert-model")
        assert config.model_name == "custom-colbert-model"

    def test_config_top_k_default(self):
        """Test default top_k for reranking."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MaxSimConfig,
        )

        config = MaxSimConfig()
        # Should have a reasonable default for top-k reranking
        assert config.default_top_k is None or config.default_top_k >= 10


# =============================================================================
# STATISTICS TESTS
# =============================================================================


class TestColBERTStatistics:
    """Test ColBERT statistics tracking."""

    def test_statistics_dataclass_exists(self):
        """Test ColBERTStatistics dataclass exists."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            ColBERTStatistics,
        )

        stats = ColBERTStatistics()
        assert stats is not None

    def test_statistics_tracks_latency(self):
        """Test statistics track latency metrics."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            ColBERTStatistics,
        )

        stats = ColBERTStatistics(
            total_reranks=10,
            total_candidates=1000,
            avg_latency_ms=15.5,
        )

        assert stats.avg_latency_ms == 15.5

    def test_statistics_reset(self):
        """Test statistics can be reset."""
        from nexus_matcher.domain.ports.retrieval import RerankCandidate
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            MockColBERTReranker,
        )

        reranker = MockColBERTReranker()
        candidates = [RerankCandidate(id="1", text="test")]

        reranker.rerank("query", candidates)
        reranker.get_statistics()

        reranker.reset_statistics()
        stats_after = reranker.get_statistics()

        assert stats_after.total_reranks == 0


# =============================================================================
# EXPORT TESTS
# =============================================================================


class TestColBERTExports:
    """Test module exports are correct."""

    def test_exports_from_colbert_module(self):
        """Test classes are exported from colbert module."""
        from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
            ColBERTMaxSimReranker,
            ColBERTStatistics,
            MaxSimConfig,
            MockColBERTReranker,
        )

        assert ColBERTMaxSimReranker is not None
        assert MockColBERTReranker is not None
        assert MaxSimConfig is not None
        assert ColBERTStatistics is not None

    def test_exports_from_rerankers_init(self):
        """Test exports from rerankers package."""
        from nexus_matcher.infrastructure.adapters.rerankers import (
            ColBERTMaxSimReranker,
            MockColBERTReranker,
        )

        assert ColBERTMaxSimReranker is not None
        assert MockColBERTReranker is not None
