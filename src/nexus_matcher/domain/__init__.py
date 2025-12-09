"""
nexus_matcher.domain | Layer: DOMAIN
Pure business logic layer with no external dependencies.

The domain layer contains:
- models/: Core domain entities (SchemaField, DictionaryEntry, MatchResult)
- ports/: Abstract interfaces for external dependencies
- services/: Domain services for business logic

This layer should be completely independent of infrastructure concerns.
All external dependencies are abstracted behind ports.
"""

from nexus_matcher.domain.models import (
    DictionaryEntry,
    MatchingSession,
    MatchResult,
    Schema,
    SchemaField,
)
from nexus_matcher.domain.ports import (
    Cache,
    DictionaryLoader,
    EmbeddingProvider,
    Reranker,
    SchemaParser,
    SparseRetriever,
    VectorStore,
)

__all__ = [
    # Models
    "SchemaField",
    "DictionaryEntry",
    "MatchResult",
    "Schema",
    "MatchingSession",
    # Ports
    "SchemaParser",
    "DictionaryLoader",
    "EmbeddingProvider",
    "VectorStore",
    "SparseRetriever",
    "Reranker",
    "Cache",
]
