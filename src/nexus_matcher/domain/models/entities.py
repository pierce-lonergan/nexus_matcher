"""
nexus_matcher.domain.models.entities | Layer: DOMAIN
Core domain entities representing schemas and dictionary entries.

## Relationships
# DEPENDS_ON → shared/types/base :: base types and enums
# USED_BY    → domain/ports/* :: port interfaces use these models
# USED_BY    → application/use_cases/* :: use cases manipulate these
# USED_BY    → infrastructure/adapters/* :: adapters convert to/from these

## Attributes
# Security: Models contain no secrets, but DictionaryEntry may have PII classification
# Performance: Frozen dataclasses for immutability and hashability
# Reliability: Full validation in __post_init__
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from nexus_matcher.shared.types.base import (
    DataType,
    DocumentId,
    EntityId,
    MatchDecision,
    Metadata,
    PerformanceMetrics,
    ProtectionLevel,
    Score,
    ScoreBreakdown,
)

# =============================================================================
# SCHEMA FIELD - Represents a field from any schema format
# =============================================================================


@dataclass(frozen=True, slots=True)
class SchemaField:
    """
    Domain model representing a field from a schema.

    This is the normalized representation of fields from any schema format
    (Avro, JSON Schema, SQL DDL, CSV headers, etc.).

    Attributes:
        name: Field name (e.g., "customer_email")
        data_type: Normalized data type
        full_path: Dot-separated path (e.g., "customer.contact.email")
        parent_path: Path to parent record (e.g., "customer.contact")
        description: Optional documentation/description
        is_nullable: Whether the field can be null
        is_array: Whether the field is an array type
        array_item_type: If array, the type of items
        default_value: Default value if specified
        constraints: Additional constraints (e.g., min/max, pattern)
        metadata: Additional metadata from source schema
    """

    name: str
    data_type: DataType
    full_path: str = ""
    parent_path: str = ""
    description: str = ""
    is_nullable: bool = True
    is_array: bool = False
    array_item_type: DataType | None = None
    default_value: Any = None
    constraints: frozenset[str] = field(default_factory=frozenset)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the schema field."""
        # Ensure full_path is set
        if not self.full_path:
            object.__setattr__(self, "full_path", self.name)

    @property
    def is_nested(self) -> bool:
        """Check if field is part of a nested structure."""
        return "." in self.full_path

    @property
    def depth(self) -> int:
        """Get nesting depth (0 for root fields)."""
        return self.full_path.count(".")

    @property
    def root_name(self) -> str:
        """Get the root-level name."""
        return self.full_path.split(".")[0]

    def with_path(self, parent: str) -> SchemaField:
        """Create new field with updated path."""
        new_path = f"{parent}.{self.name}" if parent else self.name
        new_parent = parent if parent else ""
        return SchemaField(
            name=self.name,
            data_type=self.data_type,
            full_path=new_path,
            parent_path=new_parent,
            description=self.description,
            is_nullable=self.is_nullable,
            is_array=self.is_array,
            array_item_type=self.array_item_type,
            default_value=self.default_value,
            constraints=self.constraints,
            source_metadata=self.source_metadata,
        )

    def to_searchable_text(self) -> str:
        """Generate text representation for embedding/search."""
        parts = [
            self.name.replace("_", " "),
            self.description,
        ]
        return " ".join(filter(None, parts))


# =============================================================================
# DICTIONARY ENTRY - Represents an entry from a data dictionary
# =============================================================================


@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    """
    Domain model representing an entry in a data dictionary.

    Attributes:
        id: Unique identifier for this entry
        business_name: Human-readable business name
        logical_name: Technical/logical name (often abbreviated)
        definition: Full definition/description
        data_type: Expected data type
        protection_level: Data classification level
        domain: Business domain (e.g., "Sales", "Finance")
        parent_table: Parent table/entity name
        sample_values: Example values
        synonyms: Alternative names/terms
        metadata: Additional metadata
    """

    id: DocumentId
    business_name: str
    logical_name: str
    definition: str
    data_type: DataType
    protection_level: ProtectionLevel = ProtectionLevel.INTERNAL
    domain: str = ""
    parent_table: str = ""
    sample_values: tuple[str, ...] = field(default_factory=tuple)
    synonyms: frozenset[str] = field(default_factory=frozenset)
    is_enum: bool = False
    enum_values: tuple[str, ...] = field(default_factory=tuple)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the dictionary entry."""
        if not self.id:
            raise ValueError("DictionaryEntry.id cannot be empty")
        if not self.business_name:
            raise ValueError("DictionaryEntry.business_name cannot be empty")

    @property
    def content_hash(self) -> str:
        """Generate content hash for change detection."""
        content = (
            f"{self.business_name}|{self.logical_name}|{self.definition}|{self.data_type.value}"
        )
        return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()

    def to_searchable_text(self) -> str:
        """Generate text representation for embedding/search."""
        parts = [
            self.business_name,
            self.logical_name.replace("_", " "),
            self.definition,
        ]
        if self.synonyms:
            parts.extend(self.synonyms)
        return " ".join(filter(None, parts))

    def matches_type(self, other_type: DataType, strict: bool = False) -> float:
        """
        Calculate type compatibility score.

        Args:
            other_type: Type to compare against
            strict: If True, require exact match

        Returns:
            Compatibility score (0.0 to 1.0)
        """
        if self.data_type == other_type:
            return 1.0

        if strict:
            return 0.0

        # Define compatible type groups
        numeric_types = {
            DataType.INTEGER,
            DataType.LONG,
            DataType.FLOAT,
            DataType.DOUBLE,
            DataType.DECIMAL,
        }
        string_types = {DataType.STRING, DataType.UUID, DataType.JSON}
        temporal_types = {DataType.DATE, DataType.TIMESTAMP}

        # Check group compatibility
        if self.data_type in numeric_types and other_type in numeric_types:
            return 0.8

        if self.data_type in string_types and other_type in string_types:
            return 0.9

        if self.data_type in temporal_types and other_type in temporal_types:
            return 0.9

        # String can represent most types
        if DataType.STRING in (self.data_type, other_type):
            return 0.5

        return 0.0


# =============================================================================
# MATCH RESULT - Represents a matching result
# =============================================================================


@dataclass(frozen=True, slots=True)
class MatchResult:
    """
    Domain model representing a match between a schema field and dictionary entry.

    Attributes:
        schema_field: The source schema field
        dictionary_entry: The matched dictionary entry
        rank: Rank among all matches for this field (1-based)
        final_confidence: Overall confidence score (0.0 to 1.0)
        score_breakdown: Detailed component scores
        decision: Classification decision
        performance: Performance metrics for this match
    """

    schema_field: SchemaField
    dictionary_entry: DictionaryEntry
    rank: int
    final_confidence: Score
    score_breakdown: ScoreBreakdown
    decision: MatchDecision
    performance: PerformanceMetrics

    def __post_init__(self) -> None:
        """Validate match result."""
        if not 0.0 <= self.final_confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.final_confidence}")
        if self.rank < 1:
            raise ValueError(f"Rank must be >= 1, got {self.rank}")

    @property
    def is_auto_approved(self) -> bool:
        """Check if this match is auto-approved."""
        return self.decision == MatchDecision.AUTO_APPROVE

    @property
    def needs_review(self) -> bool:
        """Check if this match needs human review."""
        return self.decision == MatchDecision.REVIEW

    @property
    def is_rejected(self) -> bool:
        """Check if this match is rejected."""
        return self.decision == MatchDecision.REJECT


# =============================================================================
# SCHEMA - Container for a complete schema
# =============================================================================


@dataclass(frozen=True, slots=True)
class Schema:
    """
    Domain model representing a complete schema.

    Attributes:
        name: Schema name
        namespace: Schema namespace (e.g., "com.company.domain")
        fields: All fields in the schema (flattened)
        source_format: Original format (avro, json_schema, sql_ddl, etc.)
        source_metadata: Original schema metadata
    """

    name: str
    fields: tuple[SchemaField, ...]
    namespace: str = ""
    source_format: str = "unknown"
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def field_count(self) -> int:
        """Get total number of fields."""
        return len(self.fields)

    @property
    def field_paths(self) -> frozenset[str]:
        """Get all field paths."""
        return frozenset(f.full_path for f in self.fields)

    def get_field(self, path: str) -> SchemaField | None:
        """Get field by path."""
        for f in self.fields:
            if f.full_path == path:
                return f
        return None

    def get_fields_by_parent(self, parent_path: str) -> tuple[SchemaField, ...]:
        """Get all fields with given parent path."""
        return tuple(f for f in self.fields if f.parent_path == parent_path)


# =============================================================================
# MATCHING SESSION - Container for a complete matching operation
# =============================================================================


@dataclass(frozen=True, slots=True)
class MatchingSession:
    """
    Domain model representing a complete schema matching session.

    Attributes:
        session_id: Unique session identifier
        schema: The source schema being matched
        results: All match results grouped by field path
        started_at: Session start timestamp
        completed_at: Session completion timestamp
        metadata: Session metadata
    """

    session_id: EntityId
    schema: Schema
    results: dict[str, tuple[MatchResult, ...]]
    total_duration_ms: float
    metadata: Metadata = field(default_factory=Metadata)

    @property
    def field_count(self) -> int:
        """Get number of fields matched."""
        return len(self.results)

    @property
    def total_matches(self) -> int:
        """Get total number of match results."""
        return sum(len(matches) for matches in self.results.values())

    @property
    def auto_approval_rate(self) -> float:
        """Calculate auto-approval rate."""
        if not self.results:
            return 0.0

        auto_approved = sum(
            1 for matches in self.results.values() if matches and matches[0].is_auto_approved
        )
        return auto_approved / len(self.results)

    def get_top_matches(self) -> dict[str, MatchResult]:
        """Get top match for each field."""
        return {path: matches[0] for path, matches in self.results.items() if matches}

    def get_low_confidence_fields(self, threshold: float = 0.6) -> list[str]:
        """Get fields with low confidence top matches."""
        return [
            path
            for path, matches in self.results.items()
            if matches and matches[0].final_confidence < threshold
        ]
