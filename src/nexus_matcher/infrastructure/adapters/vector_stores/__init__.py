"""
nexus_matcher.infrastructure.adapters.vector_stores | Layer: INFRASTRUCTURE
Vector store implementations.

## Relationships
# EXPORTS → InMemoryVectorStore :: In-memory store for testing
# EXPORTS → QdrantVectorStore :: Qdrant production store (optional)
"""

from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore"]

# Optional Qdrant support
try:
    from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import QdrantVectorStore

    __all__.append("QdrantVectorStore")
except ImportError:
    pass  # qdrant-client not installed
