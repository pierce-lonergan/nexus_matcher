"""
tests.unit.infrastructure.test_cross_encoder_reranker | Layer: TEST
Tests: CrossEncoder reranker | Target: src/infrastructure/adapters/rerankers/cross_encoder.py

TDD Phase: RED → Tests written before implementation
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from nexus_matcher.domain.ports.retrieval import (
    RerankCandidate,
    RerankResult,
)

# Check if sentence-transformers is available
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TestCrossEncoderRerankerProperties:
    """Test basic reranker properties."""

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_reranker_type(self):
        """Test reranker type is 'cross_encoder'."""
        from nexus_matcher.infrastructure.adapters.rerankers.cross_encoder import (
            CrossEncoderReranker,
        )

        with patch("nexus_matcher.infrastructure.adapters.rerankers.cross_encoder.CrossEncoder"):
            reranker = CrossEncoderReranker(model_name="test-model")
            assert reranker.reranker_type == "cross_encoder"

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_model_name(self):
        """Test model name is stored."""
        from nexus_matcher.infrastructure.adapters.rerankers.cross_encoder import (
            CrossEncoderReranker,
        )

        with patch("nexus_matcher.infrastructure.adapters.rerankers.cross_encoder.CrossEncoder"):
            reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


class TestCrossEncoderRerankerRerank:
    """Test reranking operations."""

    @pytest.fixture
    def mock_model(self):
        """Create mock CrossEncoder model."""
        with patch("nexus_matcher.infrastructure.adapters.rerankers.cross_encoder.CrossEncoder") as mock:
            model = MagicMock()
            mock.return_value = model
            yield model

    @pytest.fixture
    def reranker(self, mock_model):
        """Create reranker with mock model."""
        from nexus_matcher.infrastructure.adapters.rerankers.cross_encoder import (
            CrossEncoderReranker,
        )

        return CrossEncoderReranker(model_name="test-model")

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_returns_results(self, reranker, mock_model):
        """Test reranking returns results."""
        # Mock model to return scores
        mock_model.predict.return_value = [0.9, 0.5, 0.7]

        candidates = [
            RerankCandidate(id="1", text="document one"),
            RerankCandidate(id="2", text="document two"),
            RerankCandidate(id="3", text="document three"),
        ]

        result = reranker.rerank("test query", candidates)

        assert result.is_success
        results = result.unwrap()
        assert len(results) == 3

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_orders_by_score(self, reranker, mock_model):
        """Test reranking orders by score descending."""
        mock_model.predict.return_value = [0.5, 0.9, 0.3]

        candidates = [
            RerankCandidate(id="1", text="low score"),
            RerankCandidate(id="2", text="high score"),
            RerankCandidate(id="3", text="lowest score"),
        ]

        result = reranker.rerank("test query", candidates)

        assert result.is_success
        results = result.unwrap()
        
        # Should be ordered: id=2 (0.9), id=1 (0.5), id=3 (0.3)
        assert results[0].id == "2"
        assert results[1].id == "1"
        assert results[2].id == "3"

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_assigns_ranks(self, reranker, mock_model):
        """Test reranking assigns correct ranks."""
        mock_model.predict.return_value = [0.5, 0.9, 0.3]

        candidates = [
            RerankCandidate(id="1", text="doc one"),
            RerankCandidate(id="2", text="doc two"),
            RerankCandidate(id="3", text="doc three"),
        ]

        result = reranker.rerank("query", candidates)
        results = result.unwrap()

        # Check ranks are 1, 2, 3
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_preserves_original_ranks(self, reranker, mock_model):
        """Test reranking preserves original ranks."""
        mock_model.predict.return_value = [0.5, 0.9, 0.3]

        candidates = [
            RerankCandidate(id="1", text="doc one"),
            RerankCandidate(id="2", text="doc two"),
            RerankCandidate(id="3", text="doc three"),
        ]

        result = reranker.rerank("query", candidates)
        results = result.unwrap()

        # id=2 was originally rank 2, now rank 1
        assert results[0].original_rank == 2
        # id=1 was originally rank 1, now rank 2
        assert results[1].original_rank == 1

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_with_top_k(self, reranker, mock_model):
        """Test reranking with top_k limit."""
        mock_model.predict.return_value = [0.5, 0.9, 0.3, 0.7]

        candidates = [
            RerankCandidate(id="1", text="doc one"),
            RerankCandidate(id="2", text="doc two"),
            RerankCandidate(id="3", text="doc three"),
            RerankCandidate(id="4", text="doc four"),
        ]

        result = reranker.rerank("query", candidates, top_k=2)

        assert result.is_success
        results = result.unwrap()
        assert len(results) == 2
        # Should be top 2: id=2 (0.9), id=4 (0.7)
        assert results[0].id == "2"
        assert results[1].id == "4"

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_empty_candidates(self, reranker, mock_model):
        """Test reranking with empty candidates."""
        result = reranker.rerank("query", [])

        assert result.is_success
        assert result.unwrap() == []

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_preserves_metadata(self, reranker, mock_model):
        """Test reranking preserves candidate metadata."""
        mock_model.predict.return_value = [0.9]

        candidates = [
            RerankCandidate(
                id="1", 
                text="doc one",
                metadata={"domain": "customer", "source": "database"}
            ),
        ]

        result = reranker.rerank("query", candidates)
        results = result.unwrap()

        assert results[0].metadata == {"domain": "customer", "source": "database"}


class TestCrossEncoderRerankerScorePair:
    """Test single pair scoring."""

    @pytest.fixture
    def mock_model(self):
        """Create mock CrossEncoder model."""
        with patch("nexus_matcher.infrastructure.adapters.rerankers.cross_encoder.CrossEncoder") as mock:
            model = MagicMock()
            mock.return_value = model
            yield model

    @pytest.fixture
    def reranker(self, mock_model):
        """Create reranker with mock model."""
        from nexus_matcher.infrastructure.adapters.rerankers.cross_encoder import (
            CrossEncoderReranker,
        )

        return CrossEncoderReranker(model_name="test-model")

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_score_pair(self, reranker, mock_model):
        """Test scoring a single pair."""
        mock_model.predict.return_value = [0.85]

        result = reranker.score_pair("query text", "document text")

        assert result.is_success
        assert result.unwrap() == 0.85


class TestCrossEncoderRerankerBatching:
    """Test batch processing."""

    @pytest.fixture
    def mock_model(self):
        """Create mock CrossEncoder model."""
        with patch("nexus_matcher.infrastructure.adapters.rerankers.cross_encoder.CrossEncoder") as mock:
            model = MagicMock()
            mock.return_value = model
            yield model

    @pytest.fixture
    def reranker(self, mock_model):
        """Create reranker with mock model."""
        from nexus_matcher.infrastructure.adapters.rerankers.cross_encoder import (
            CrossEncoderReranker,
        )

        return CrossEncoderReranker(model_name="test-model", batch_size=2)

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_batch_size_parameter(self, reranker):
        """Test batch size is configurable."""
        assert reranker._batch_size == 2


class TestCrossEncoderRerankerErrorHandling:
    """Test error handling."""

    @pytest.fixture
    def mock_model(self):
        """Create mock CrossEncoder model."""
        with patch("nexus_matcher.infrastructure.adapters.rerankers.cross_encoder.CrossEncoder") as mock:
            model = MagicMock()
            mock.return_value = model
            yield model

    @pytest.fixture
    def reranker(self, mock_model):
        """Create reranker with mock model."""
        from nexus_matcher.infrastructure.adapters.rerankers.cross_encoder import (
            CrossEncoderReranker,
        )

        return CrossEncoderReranker(model_name="test-model")

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_handles_model_error(self, reranker, mock_model):
        """Test reranking handles model errors gracefully."""
        mock_model.predict.side_effect = Exception("Model error")

        candidates = [RerankCandidate(id="1", text="test")]
        result = reranker.rerank("query", candidates)

        assert result.is_failure
        assert "failed" in result.error.lower()

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_score_pair_handles_error(self, reranker, mock_model):
        """Test score_pair handles errors gracefully."""
        mock_model.predict.side_effect = Exception("Scoring failed")

        result = reranker.score_pair("query", "document")

        assert result.is_failure
