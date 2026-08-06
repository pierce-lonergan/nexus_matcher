"""
nexus_matcher.infrastructure.adapters.rerankers | Layer: INFRASTRUCTURE
Reranker implementations.

## Relationships
# EXPORTS → CrossEncoderReranker :: sentence-transformers CrossEncoder (optional)
# EXPORTS → ColBERTMaxSimReranker :: ColBERT MaxSim late interaction (GAP-001)
# EXPORTS → MockColBERTReranker :: Mock for testing without models
"""

__all__ = []

# Optional CrossEncoder support
try:
    from nexus_matcher.infrastructure.adapters.rerankers.cross_encoder import (
        CrossEncoderReranker,
    )

    __all__.append("CrossEncoderReranker")
except ImportError:
    pass  # sentence-transformers not installed

# ColBERT MaxSim support (GAP-001)
from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
    ColBERTMaxSimReranker,
    ColBERTStatistics,
    MaxSimConfig,
    MockColBERTReranker,
)

__all__.extend(
    [
        "ColBERTMaxSimReranker",
        "ColBERTStatistics",
        "MaxSimConfig",
        "MockColBERTReranker",
    ]
)
