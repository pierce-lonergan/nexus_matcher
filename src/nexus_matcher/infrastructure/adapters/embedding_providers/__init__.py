"""
nexus_matcher.infrastructure.adapters.embedding_providers | Layer: INFRASTRUCTURE
Embedding provider implementations.
"""

from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
    MockQuantizedProvider,
    QuantizationConfig,
    QuantizationStats,
    QuantizedEmbeddingProvider,
    create_quantized_provider,
    detect_cpu_features,
    get_quantization_info,
    is_backend_available,
    is_quantization_recommended,
)
from nexus_matcher.infrastructure.adapters.embedding_providers.sentence_transformers import (
    MockEmbeddingProvider,
    SentenceTransformersProvider,
)

__all__ = [
    "MockEmbeddingProvider",
    "MockQuantizedProvider",
    "QuantizationConfig",
    "QuantizationStats",
    # Quantized providers (GAP-002)
    "QuantizedEmbeddingProvider",
    # Standard providers
    "SentenceTransformersProvider",
    # Utility functions
    "create_quantized_provider",
    "detect_cpu_features",
    "get_quantization_info",
    "is_backend_available",
    "is_quantization_recommended",
]
