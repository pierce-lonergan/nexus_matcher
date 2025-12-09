"""
nexus_matcher.shared.types | Layer: SHARED
Common type definitions used across all layers.
"""

from nexus_matcher.shared.types.base import (
    ChangeType,
    DataType,
    DocumentId,
    EmbeddingVector,
    EntityId,
    MatchDecision,
    Metadata,
    PagedResult,
    PerformanceMetrics,
    ProtectionLevel,
    Result,
    Score,
    ScoreBreakdown,
)

__all__ = [
    "DataType",
    "MatchDecision",
    "ProtectionLevel",
    "ChangeType",
    "EmbeddingVector",
    "DocumentId",
    "Score",
    "EntityId",
    "Metadata",
    "ScoreBreakdown",
    "PerformanceMetrics",
    "Result",
    "PagedResult",
]
