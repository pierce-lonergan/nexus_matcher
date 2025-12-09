"""
tests.unit.infrastructure.test_quantized_embedding_provider | Layer: TEST
Tests: INT8 Quantized Embedding Provider | Target: GAP-002

TDD Phase: RED → Tests written before implementation
Research Reference: README_RESEARCH_2.md, Lines 9-18; README_RESEARCH_3.md, Lines 9-11

Research Targets:
- 3-10x speedup over FP32
- <2% accuracy loss
- ≤15ms batch-32 latency on 8-core CPU
"""

import time
import pytest
import numpy as np

from nexus_matcher.domain.ports.embedding_provider import EmbeddingConfig


# =============================================================================
# HARDWARE DETECTION TESTS
# =============================================================================


class TestVNNIDetection:
    """Test VNNI/CPU feature detection."""

    def test_detect_cpu_features_returns_dict(self):
        """Test CPU feature detection returns feature dict."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            detect_cpu_features,
        )
        
        features = detect_cpu_features()
        assert isinstance(features, dict)
        assert "vnni" in features
        assert "avx2" in features
        assert "avx512" in features

    def test_detect_cpu_features_boolean_values(self):
        """Test CPU features are boolean."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            detect_cpu_features,
        )
        
        features = detect_cpu_features()
        for key, value in features.items():
            assert isinstance(value, bool), f"{key} should be boolean"

    def test_quantization_recommended_checks_vnni(self):
        """Test quantization recommendation considers VNNI."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            is_quantization_recommended,
        )
        
        # Should return boolean
        recommended = is_quantization_recommended()
        assert isinstance(recommended, bool)


# =============================================================================
# QUANTIZATION CONFIG TESTS
# =============================================================================


class TestQuantizationConfig:
    """Test quantization configuration."""

    def test_config_exists(self):
        """Test QuantizationConfig class exists."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizationConfig,
        )
        
        config = QuantizationConfig()
        assert config is not None

    def test_config_default_values(self):
        """Test default configuration values."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizationConfig,
        )
        
        config = QuantizationConfig()
        assert config.precision == "int8"
        assert config.backend in ("onnx", "openvino", "auto")

    def test_config_supports_int8(self):
        """Test INT8 precision is supported."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizationConfig,
        )
        
        config = QuantizationConfig(precision="int8")
        assert config.precision == "int8"

    def test_config_supports_multiple_backends(self):
        """Test multiple backends are supported."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizationConfig,
        )
        
        for backend in ("onnx", "openvino", "auto"):
            config = QuantizationConfig(backend=backend)
            assert config.backend == backend


# =============================================================================
# QUANTIZED PROVIDER BASIC TESTS
# =============================================================================


class TestQuantizedEmbeddingProviderBasic:
    """Test basic QuantizedEmbeddingProvider functionality."""

    def test_provider_class_exists(self):
        """Test QuantizedEmbeddingProvider class exists."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizedEmbeddingProvider,
        )
        
        assert QuantizedEmbeddingProvider is not None

    def test_provider_implements_protocol(self):
        """Test provider implements EmbeddingProvider protocol."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizedEmbeddingProvider,
            QuantizationConfig,
        )
        from nexus_matcher.domain.ports.embedding_provider import EmbeddingProvider
        
        # Should be able to instantiate
        provider = QuantizedEmbeddingProvider(
            model_name="test-model",
            quantization_config=QuantizationConfig(),
        )
        
        # Should implement protocol (duck typing check)
        assert hasattr(provider, "embed")
        assert hasattr(provider, "embed_single")
        assert hasattr(provider, "model_name")
        assert hasattr(provider, "dimension")

    def test_provider_has_quantization_info(self):
        """Test provider exposes quantization info."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizedEmbeddingProvider,
            QuantizationConfig,
        )
        
        provider = QuantizedEmbeddingProvider(
            model_name="test-model",
            quantization_config=QuantizationConfig(precision="int8"),
        )
        
        assert hasattr(provider, "is_quantized")
        assert hasattr(provider, "quantization_precision")


# =============================================================================
# MOCK QUANTIZED PROVIDER TESTS
# =============================================================================


class TestMockQuantizedProvider:
    """Test mock quantized provider for unit testing."""

    def test_mock_provider_exists(self):
        """Test MockQuantizedProvider class exists."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            MockQuantizedProvider,
        )
        
        provider = MockQuantizedProvider()
        assert provider is not None

    def test_mock_provider_generates_embeddings(self):
        """Test mock provider generates embeddings."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            MockQuantizedProvider,
        )
        
        provider = MockQuantizedProvider(dimension=768)
        result = provider.embed(["test text", "another text"])
        
        assert result.is_success
        embeddings = result.unwrap()
        assert embeddings.count == 2
        assert embeddings.dimension == 768

    def test_mock_provider_simulates_speedup(self):
        """Test mock provider can simulate quantization speedup."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            MockQuantizedProvider,
        )
        
        provider = MockQuantizedProvider(
            dimension=768,
            simulated_speedup=3.0,
        )
        
        assert provider.simulated_speedup == 3.0
        assert provider.is_quantized is True

    def test_mock_provider_tracks_latency(self):
        """Test mock provider tracks inference latency."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            MockQuantizedProvider,
        )
        
        provider = MockQuantizedProvider(dimension=768)
        _ = provider.embed(["test"] * 32)
        
        assert hasattr(provider, "last_inference_time_ms")
        assert provider.last_inference_time_ms is not None


# =============================================================================
# QUANTIZATION STATISTICS TESTS
# =============================================================================


class TestQuantizationStatistics:
    """Test quantization statistics tracking."""

    def test_statistics_class_exists(self):
        """Test QuantizationStats class exists."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizationStats,
        )
        
        stats = QuantizationStats()
        assert stats is not None

    def test_statistics_tracks_inference_count(self):
        """Test statistics tracks total inferences."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizationStats,
        )
        
        stats = QuantizationStats()
        stats.record_inference(batch_size=32, latency_ms=10.0)
        stats.record_inference(batch_size=16, latency_ms=5.0)
        
        assert stats.total_inferences == 2
        assert stats.total_texts_processed == 48

    def test_statistics_computes_average_latency(self):
        """Test statistics computes average latency."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizationStats,
        )
        
        stats = QuantizationStats()
        stats.record_inference(batch_size=32, latency_ms=10.0)
        stats.record_inference(batch_size=32, latency_ms=20.0)
        
        assert stats.avg_latency_ms == 15.0

    def test_statistics_tracks_throughput(self):
        """Test statistics tracks throughput."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizationStats,
        )
        
        stats = QuantizationStats()
        stats.record_inference(batch_size=100, latency_ms=100.0)
        
        # 100 texts in 100ms = 1000 texts/s
        assert stats.throughput_texts_per_second == 1000.0

    def test_statistics_resets(self):
        """Test statistics can be reset."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizationStats,
        )
        
        stats = QuantizationStats()
        stats.record_inference(batch_size=32, latency_ms=10.0)
        stats.reset()
        
        assert stats.total_inferences == 0


# =============================================================================
# ONNX BACKEND TESTS (INTEGRATION)
# =============================================================================


@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="ONNX Runtime required",
)
class TestONNXBackend:
    """Test ONNX Runtime quantization backend."""

    def test_onnx_backend_available(self):
        """Test ONNX backend availability detection."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            is_backend_available,
        )
        
        available = is_backend_available("onnx")
        assert isinstance(available, bool)

    def test_onnx_backend_config(self):
        """Test ONNX backend configuration."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizationConfig,
        )
        
        config = QuantizationConfig(
            backend="onnx",
            precision="int8",
            optimize_for_inference=True,
        )
        assert config.backend == "onnx"


# =============================================================================
# PERFORMANCE BENCHMARK TESTS
# =============================================================================


class TestQuantizationPerformance:
    """Test quantization performance characteristics."""

    def test_batch_32_latency_target(self):
        """Test batch-32 latency target is achievable (mock).
        
        Research Target: ≤15ms for batch-32 on 8-core CPU.
        """
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            MockQuantizedProvider,
        )
        
        provider = MockQuantizedProvider(
            dimension=768,
            simulated_latency_per_text_us=100,  # 100µs per text
        )
        
        texts = ["test text"] * 32
        start = time.perf_counter()
        result = provider.embed(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert result.is_success
        # Mock should be fast; real impl needs to meet 15ms target
        assert elapsed_ms < 100  # Give generous margin for mock

    def test_speedup_ratio_tracking(self):
        """Test speedup ratio tracking vs baseline."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            QuantizationStats,
        )
        
        stats = QuantizationStats()
        stats.set_baseline_latency_ms(45.0)  # FP32 baseline
        stats.record_inference(batch_size=32, latency_ms=15.0)  # INT8
        
        # 45ms / 15ms = 3x speedup
        assert stats.speedup_ratio == 3.0


# =============================================================================
# ACCURACY PRESERVATION TESTS
# =============================================================================


class TestAccuracyPreservation:
    """Test accuracy preservation after quantization."""

    def test_embedding_similarity_preserved(self):
        """Test embedding similarity is preserved after quantization (mock)."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            MockQuantizedProvider,
        )
        
        provider = MockQuantizedProvider(dimension=768)
        
        # Same text should produce consistent embeddings
        result1 = provider.embed_single("customer email address")
        result2 = provider.embed_single("customer email address")
        
        assert result1.is_success
        assert result2.is_success
        
        embedding1 = result1.unwrap()
        embedding2 = result2.unwrap()
        
        # Dot product should be 1.0 for identical normalized vectors
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        assert similarity > 0.99  # Allow small float precision

    def test_quantization_accuracy_loss_within_bounds(self):
        """Test accuracy loss is within acceptable bounds (mock).
        
        Research Target: <2% accuracy degradation.
        """
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            MockQuantizedProvider,
        )
        
        provider = MockQuantizedProvider(
            dimension=768,
            simulated_accuracy_loss=0.015,  # 1.5% simulated loss
        )
        
        # Mock provider tracks simulated accuracy loss
        assert provider.simulated_accuracy_loss < 0.02  # <2%


# =============================================================================
# INTEGRATION WITH SEMANTIC CACHE TESTS
# =============================================================================


class TestCacheIntegration:
    """Test integration with SemanticContentCache."""

    def test_provider_compatible_with_cache(self):
        """Test quantized provider works with semantic cache."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
            MockQuantizedProvider,
        )
        from nexus_matcher.infrastructure.adapters.caches.content import (
            SemanticContentCache,
        )
        
        provider = MockQuantizedProvider(dimension=768)
        cache = SemanticContentCache(max_size=100)
        
        def compute_embedding(text: str) -> np.ndarray:
            result = provider.embed_single(text)
            return result.unwrap() if result.is_success else np.zeros(768)
        
        # Use cache with provider
        embedding1 = cache.get_or_compute("test text", compute_embedding)
        embedding2 = cache.get_or_compute("test text", compute_embedding)  # Should hit cache
        
        assert np.allclose(embedding1, embedding2)
        # SemanticContentCache uses get_stats() method
        stats = cache.get_stats()
        assert stats.hits == 1


# =============================================================================
# EXPORT TESTS
# =============================================================================


class TestQuantizedProviderExports:
    """Test module exports are correct."""

    def test_exports_from_init(self):
        """Test classes are exported from package __init__."""
        from nexus_matcher.infrastructure.adapters.embedding_providers import (
            QuantizedEmbeddingProvider,
            QuantizationConfig,
            MockQuantizedProvider,
        )
        
        assert QuantizedEmbeddingProvider is not None
        assert QuantizationConfig is not None
        assert MockQuantizedProvider is not None
