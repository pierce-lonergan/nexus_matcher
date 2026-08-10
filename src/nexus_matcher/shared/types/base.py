"""
nexus_matcher.shared.types.base | Layer: SHARED
Core type definitions used across all layers.

## Relationships
# USED_BY    → domain/* :: base types for domain models
# USED_BY    → application/* :: DTO type hints
# USED_BY    → infrastructure/* :: adapter implementations
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeAlias, TypeVar
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray


# Helper for timezone-aware UTC datetime
def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Embedding vector type (768-dim typical for BERT-based models)
EmbeddingVector: TypeAlias = NDArray[np.float32]

# Document/Entry ID type
DocumentId: TypeAlias = str

# Score type (0.0 to 1.0 typically)
Score: TypeAlias = float

# Generic type variable for containers
T = TypeVar("T")


# =============================================================================
# ENUMS
# =============================================================================


class DataType(str, Enum):
    """Normalized data types for schema fields."""

    STRING = "string"
    INTEGER = "integer"
    LONG = "long"
    FLOAT = "float"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    BYTES = "bytes"
    ARRAY = "array"
    RECORD = "record"
    ENUM = "enum"
    UUID = "uuid"
    JSON = "json"
    DECIMAL = "decimal"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, type_str: str) -> DataType:
        """Normalize a type string to DataType enum."""
        if not type_str:
            return cls.UNKNOWN

        type_lower = type_str.lower().strip()

        # String types
        if any(s in type_lower for s in ("string", "char", "text", "varchar")):
            return cls.STRING

        # Integer types
        if "int" in type_lower and "long" not in type_lower and "big" not in type_lower:
            return cls.INTEGER

        # Long types
        if any(s in type_lower for s in ("long", "bigint", "int64")):
            return cls.LONG

        # Float types
        if "float" in type_lower:
            return cls.FLOAT

        # Double/Decimal types
        if any(s in type_lower for s in ("double", "decimal", "numeric", "number")):
            return cls.DOUBLE

        # Boolean types
        if any(s in type_lower for s in ("bool", "boolean", "bit")):
            return cls.BOOLEAN

        # Date types
        if "date" in type_lower and "time" not in type_lower:
            return cls.DATE

        # Timestamp types
        if any(s in type_lower for s in ("timestamp", "datetime", "time")):
            return cls.TIMESTAMP

        # Binary types
        if any(s in type_lower for s in ("bytes", "binary", "blob")):
            return cls.BYTES

        # Array types
        if any(s in type_lower for s in ("array", "list", "[]")):
            return cls.ARRAY

        # Record/Struct types
        if any(s in type_lower for s in ("record", "struct", "object")):
            return cls.RECORD

        # Enum types
        if "enum" in type_lower:
            return cls.ENUM

        # UUID types
        if "uuid" in type_lower:
            return cls.UUID

        # JSON types
        if "json" in type_lower:
            return cls.JSON

        return cls.UNKNOWN


class MatchDecision(str, Enum):
    """Decision classification for match results."""

    AUTO_APPROVE = "AUTO_APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class ProtectionLevel(str, Enum):
    """Data protection classification levels."""

    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    PII = "PII"
    RESTRICTED = "RESTRICTED"

    @classmethod
    def from_string(cls, level_str: str) -> ProtectionLevel:
        """Parse protection level string."""
        if not level_str:
            return cls.INTERNAL

        level_upper = level_str.upper().strip()

        for member in cls:
            if member.value == level_upper:
                return member

        return cls.INTERNAL


class ChangeType(str, Enum):
    """Type of change detected in incremental updates."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


# =============================================================================
# BASE DATACLASSES
# =============================================================================


@dataclass(frozen=True, slots=True)
class EntityId:
    """Value object for entity identification."""

    value: str = field(default_factory=lambda: str(uuid4()))

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(frozen=True, slots=True)
class Metadata:
    """Common metadata for entities."""

    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    version: int = 1
    tags: frozenset[str] = field(default_factory=frozenset)

    def with_update(self) -> Metadata:
        """Create new metadata with updated timestamp."""
        return Metadata(
            created_at=self.created_at,
            updated_at=_utc_now(),
            version=self.version + 1,
            tags=self.tags,
        )


# Distinguishes "caller passed 0.0" from "caller passed nothing", so ScoreBreakdown can
# tell whether BOTH the new field name and the deprecated alias were supplied.
_UNSET: Any = object()

_SEMANTIC_SCORE_DEPRECATION = (
    "ScoreBreakdown.semantic_score is deprecated and will be removed in 3.0; it is "
    "renamed to fused_retrieval_score. The number was never a semantic similarity: it "
    "is the min-max-normalised FUSED RETRIEVAL score, which is rank-relative, so 0.9 "
    "means 'ranked first among the candidates for this field', not '90% similar'. For "
    "absolute similarity read absolute_cosine."
)


@dataclass(frozen=True, slots=True, init=False)
class ScoreBreakdown:
    """
    Detailed breakdown of scoring components.

    `fused_retrieval_score` used to be called `semantic_score`, and that name misled
    every reader of it -- including this project's own documentation. It is NOT a cosine
    similarity between the field and the entry. It is the min-max-normalised fused
    retrieval score: the dense and lexical arms are each rescaled by their own min and
    max over the candidates retrieved FOR THIS FIELD, then combined with
    `MatchingConfig.fusion_alpha`. It is therefore RANK-RELATIVE. The rank-1 candidate
    lands at or just above `fusion_alpha` for essentially every field, whether the match
    is excellent or barely plausible, which is why the recurring value is 0.9 -- 0.9 is
    `fusion_alpha`. Move fusion_alpha to 0.5 and the rank-1 floor moves with it; a cosine
    would not move at all.

    Telling an auditor "semantic score 0.9" therefore claims 90% semantic similarity and
    delivers "ranked first among N candidates". `absolute_cosine` is the number that
    answers "how similar were they really?".

    `semantic_score` still works, as a deprecated alias for reading and for construction,
    and emits a DeprecationWarning. It is scheduled for removal in 3.0.
    """

    fused_retrieval_score: Score = 0.0
    lexical_score: Score = 0.0
    edit_distance_score: Score = 0.0
    type_compatibility_score: Score = 0.0
    colbert_score: Score | None = None
    cross_encoder_score: Score | None = None
    domain_score: Score = 0.0
    graph_boost: Score = 0.0
    # The RAW dense-retrieval score for this candidate, before any normalisation: under
    # the shipped wiring the vector store's distance metric is "cosine", so this is the
    # actual cosine similarity between the query vector and the entry vector. It is the
    # only number here that is comparable ACROSS fields, and it is the one an auditor
    # asking "how similar were they really?" needs. None when the dense arm did not
    # return this candidate at all (it reached the result list through the lexical arm),
    # and not a cosine if a caller supplied a vector store configured for a different
    # distance metric.
    absolute_cosine: Score | None = None

    def __init__(
        self,
        fused_retrieval_score: Score = _UNSET,
        lexical_score: Score = 0.0,
        edit_distance_score: Score = 0.0,
        type_compatibility_score: Score = 0.0,
        colbert_score: Score | None = None,
        cross_encoder_score: Score | None = None,
        domain_score: Score = 0.0,
        graph_boost: Score = 0.0,
        absolute_cosine: Score | None = None,
        *,
        semantic_score: Score = _UNSET,
    ) -> None:
        """
        Hand-written so `semantic_score=` keeps working for one more major version.

        A rename that breaks construction breaks every caller at once, with a TypeError
        that names a field they never heard of. The old keyword is accepted, warned
        about, and routed to the new field; supplying both is an error rather than a
        coin toss, because the two would silently disagree about which number the
        breakdown reports.
        """
        if semantic_score is not _UNSET:
            if fused_retrieval_score is not _UNSET:
                raise TypeError(
                    "ScoreBreakdown got both fused_retrieval_score and its deprecated "
                    "alias semantic_score. Pass one -- they are the same field."
                )
            warnings.warn(_SEMANTIC_SCORE_DEPRECATION, DeprecationWarning, stacklevel=2)
            fused_retrieval_score = semantic_score
        if fused_retrieval_score is _UNSET:
            fused_retrieval_score = 0.0

        setattr_ = object.__setattr__  # frozen: assign through the base implementation
        setattr_(self, "fused_retrieval_score", fused_retrieval_score)
        setattr_(self, "lexical_score", lexical_score)
        setattr_(self, "edit_distance_score", edit_distance_score)
        setattr_(self, "type_compatibility_score", type_compatibility_score)
        setattr_(self, "colbert_score", colbert_score)
        setattr_(self, "cross_encoder_score", cross_encoder_score)
        setattr_(self, "domain_score", domain_score)
        setattr_(self, "graph_boost", graph_boost)
        setattr_(self, "absolute_cosine", absolute_cosine)

    @property
    def semantic_score(self) -> Score:
        """Deprecated alias for `fused_retrieval_score`. Removed in 3.0."""
        warnings.warn(_SEMANTIC_SCORE_DEPRECATION, DeprecationWarning, stacklevel=2)
        return self.fused_retrieval_score

    @property
    def has_reranking(self) -> bool:
        """Check if reranking scores are present."""
        return self.colbert_score is not None or self.cross_encoder_score is not None


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Performance metrics for operations."""

    latency_ms: float
    cache_hit: bool = False
    retrieval_stage: str = "unknown"
    candidates_evaluated: int = 0
    reranking_applied: bool = False


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass(frozen=True)
class Result(Generic[T]):
    """Generic result wrapper for operations that may fail."""

    value: T | None = None
    error: str | None = None
    error_code: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    @property
    def is_failure(self) -> bool:
        return self.error is not None

    @classmethod
    def success(cls, value: T) -> Result[T]:
        return cls(value=value)

    @classmethod
    def failure(cls, error: str, error_code: str | None = None) -> Result[T]:
        return cls(error=error, error_code=error_code)

    def unwrap(self) -> T:
        """Get value or raise if failure."""
        if self.is_failure:
            raise ValueError(f"Result is failure: {self.error}")
        return self.value  # type: ignore


@dataclass(frozen=True)
class PagedResult(Generic[T]):
    """Paginated result wrapper."""

    items: list[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool

    @property
    def total_pages(self) -> int:
        return (self.total + self.page_size - 1) // self.page_size
