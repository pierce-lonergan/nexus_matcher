"""
nexus_matcher.infrastructure.adapters.embedding_providers | Layer: INFRASTRUCTURE
Embedding provider implementations.
"""

from nexus_matcher.infrastructure.adapters.embedding_providers.sentence_transformers import (
    MockEmbeddingProvider,
    SentenceTransformersProvider,
)
from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
    MockQuantizedProvider,
    QuantizedEmbeddingProvider,
    QuantizationConfig,
    QuantizationStats,
    create_quantized_provider,
    detect_cpu_features,
    get_quantization_info,
    is_backend_available,
    is_quantization_recommended,
)

__all__ = [
    # Standard providers
    "SentenceTransformersProvider",
    "MockEmbeddingProvider",
    # Quantized providers (GAP-002)
    "QuantizedEmbeddingProvider",
    "QuantizationConfig",
    "QuantizationStats",
    "MockQuantizedProvider",
    # Utility functions
    "create_quantized_provider",
    "detect_cpu_features",
    "is_quantization_recommended",
    "is_backend_available",
    "get_quantization_info",
]
