"""
nexus_matcher.domain.services.type_compatibility | Layer: DOMAIN
Type compatibility scoring for data type matching.

## Relationships
# DEPENDS_ON → shared/types/base :: DataType enum
# USED_BY    → domain/models/entities :: DictionaryEntry.matches_type()
# USED_BY    → application/use_cases/match_schema :: type scoring

## Invariants
# 1. Scores are always between 0.0 and 1.0
# 2. Exact type matches always score 1.0
# 3. Widening conversions score higher than narrowing

## Attributes
# Security: No external I/O, pure computation
# Performance: O(1) lookups using precomputed matrices
# Reliability: Unknown types handled gracefully with neutral scores
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from nexus_matcher.shared.types.base import DataType


# =============================================================================
# ENUMS
# =============================================================================


class TypeCategory(Enum):
    """Categories of data types for compatibility grouping."""

    NUMERIC = "numeric"
    STRING = "string"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    BINARY = "binary"
    COMPLEX = "complex"
    NULL = "null"

    @classmethod
    def from_data_type(cls, data_type: DataType) -> TypeCategory:
        """Get category for a data type."""
        return TYPE_TO_CATEGORY.get(data_type, cls.NULL)


class CompatibilityLevel(Enum):
    """Levels of type compatibility."""

    EXACT = "exact"           # 1.00 - Same type
    EQUIVALENT = "equivalent"  # 0.95-0.99 - Semantically identical
    COMPATIBLE = "compatible"  # 0.80-0.94 - Safe conversion
    CONVERTIBLE = "convertible"  # 0.50-0.79 - Possible conversion
    INCOMPATIBLE = "incompatible"  # 0.00-0.49 - Likely incompatible

    @classmethod
    def from_score(cls, score: float) -> CompatibilityLevel:
        """Get level from score."""
        if score >= 1.0:
            return cls.EXACT
        if score >= 0.95:
            return cls.EQUIVALENT
        if score >= 0.80:
            return cls.COMPATIBLE
        if score >= 0.50:
            return cls.CONVERTIBLE
        return cls.INCOMPATIBLE


# =============================================================================
# TYPE TO CATEGORY MAPPING
# =============================================================================

TYPE_TO_CATEGORY: dict[DataType, TypeCategory] = {
    # Numeric types
    DataType.INTEGER: TypeCategory.NUMERIC,
    DataType.LONG: TypeCategory.NUMERIC,
    DataType.FLOAT: TypeCategory.NUMERIC,
    DataType.DOUBLE: TypeCategory.NUMERIC,
    DataType.DECIMAL: TypeCategory.NUMERIC,

    # String types
    DataType.STRING: TypeCategory.STRING,

    # Temporal types
    DataType.DATE: TypeCategory.TEMPORAL,
    DataType.TIMESTAMP: TypeCategory.TEMPORAL,

    # Boolean
    DataType.BOOLEAN: TypeCategory.BOOLEAN,

    # Binary
    DataType.BYTES: TypeCategory.BINARY,

    # Complex types
    DataType.ARRAY: TypeCategory.COMPLEX,
    DataType.RECORD: TypeCategory.COMPLEX,
    DataType.JSON: TypeCategory.COMPLEX,

    # Enum type (treat as string-like)
    DataType.ENUM: TypeCategory.STRING,

    # UUID type (treat as string-like)
    DataType.UUID: TypeCategory.STRING,

    # Unknown
    DataType.UNKNOWN: TypeCategory.NULL,
}


# =============================================================================
# VALUE OBJECTS
# =============================================================================


@dataclass(frozen=True)
class TypeCompatibilityResult:
    """
    Result of type compatibility check.

    Contains the score, level, and reason for the compatibility assessment.
    """

    source_type: DataType
    target_type: DataType
    score: float
    level: CompatibilityLevel
    reason: str

    @property
    def is_compatible(self) -> bool:
        """Check if types are compatible (score >= 0.5)."""
        return self.score >= 0.5

    @property
    def is_exact(self) -> bool:
        """Check if this is an exact match."""
        return self.level == CompatibilityLevel.EXACT


# =============================================================================
# COMPATIBILITY MATRIX
# =============================================================================

# Pre-computed compatibility scores for type pairs
# Key: (source_type, target_type), Value: (score, reason)
COMPATIBILITY_MATRIX: dict[tuple[DataType, DataType], tuple[float, str]] = {
    # Numeric conversions
    (DataType.INTEGER, DataType.LONG): (0.95, "Widening numeric conversion"),
    (DataType.INTEGER, DataType.FLOAT): (0.90, "Integer to float conversion"),
    (DataType.INTEGER, DataType.DOUBLE): (0.90, "Integer to double conversion"),
    (DataType.INTEGER, DataType.DECIMAL): (0.85, "Integer to decimal conversion"),

    (DataType.LONG, DataType.INTEGER): (0.75, "Narrowing conversion (possible overflow)"),
    (DataType.LONG, DataType.FLOAT): (0.85, "Long to float (precision loss)"),
    (DataType.LONG, DataType.DOUBLE): (0.90, "Long to double conversion"),
    (DataType.LONG, DataType.DECIMAL): (0.85, "Long to decimal conversion"),

    (DataType.FLOAT, DataType.INTEGER): (0.70, "Float to integer truncation"),
    (DataType.FLOAT, DataType.LONG): (0.70, "Float to long truncation"),
    (DataType.FLOAT, DataType.DOUBLE): (0.95, "Widening float to double"),
    (DataType.FLOAT, DataType.DECIMAL): (0.80, "Float to decimal"),

    (DataType.DOUBLE, DataType.INTEGER): (0.65, "Double to integer truncation"),
    (DataType.DOUBLE, DataType.LONG): (0.70, "Double to long truncation"),
    (DataType.DOUBLE, DataType.FLOAT): (0.75, "Narrowing double to float"),
    (DataType.DOUBLE, DataType.DECIMAL): (0.80, "Double to decimal"),

    (DataType.DECIMAL, DataType.INTEGER): (0.65, "Decimal to integer truncation"),
    (DataType.DECIMAL, DataType.LONG): (0.70, "Decimal to long truncation"),
    (DataType.DECIMAL, DataType.FLOAT): (0.70, "Decimal to float (precision loss)"),
    (DataType.DECIMAL, DataType.DOUBLE): (0.75, "Decimal to double (precision loss)"),

    # Temporal conversions
    (DataType.DATE, DataType.TIMESTAMP): (0.85, "Date to timestamp (adds time component)"),
    (DataType.TIMESTAMP, DataType.DATE): (0.70, "Timestamp to date (loses time)"),

    # Cross-category conversions (generally lower scores)
    # Numeric to String
    (DataType.INTEGER, DataType.STRING): (0.35, "Integer to string conversion"),
    (DataType.LONG, DataType.STRING): (0.35, "Long to string conversion"),
    (DataType.FLOAT, DataType.STRING): (0.35, "Float to string conversion"),
    (DataType.DOUBLE, DataType.STRING): (0.35, "Double to string conversion"),
    (DataType.DECIMAL, DataType.STRING): (0.40, "Decimal to string conversion"),

    # String to Numeric (parsing required)
    (DataType.STRING, DataType.INTEGER): (0.25, "String to integer (parsing)"),
    (DataType.STRING, DataType.LONG): (0.25, "String to long (parsing)"),
    (DataType.STRING, DataType.FLOAT): (0.25, "String to float (parsing)"),
    (DataType.STRING, DataType.DOUBLE): (0.25, "String to double (parsing)"),
    (DataType.STRING, DataType.DECIMAL): (0.30, "String to decimal (parsing)"),

    # Boolean conversions
    (DataType.BOOLEAN, DataType.INTEGER): (0.40, "Boolean to integer (0/1)"),
    (DataType.BOOLEAN, DataType.LONG): (0.40, "Boolean to long (0/1)"),
    (DataType.BOOLEAN, DataType.STRING): (0.45, "Boolean to string (true/false)"),
    (DataType.INTEGER, DataType.BOOLEAN): (0.35, "Integer to boolean"),
    (DataType.STRING, DataType.BOOLEAN): (0.35, "String to boolean (parsing)"),

    # Temporal to String
    (DataType.DATE, DataType.STRING): (0.50, "Date to string (format dependent)"),
    (DataType.TIMESTAMP, DataType.STRING): (0.50, "Timestamp to string (format dependent)"),
    (DataType.STRING, DataType.DATE): (0.40, "String to date (parsing)"),
    (DataType.STRING, DataType.TIMESTAMP): (0.40, "String to timestamp (parsing)"),

    # UNKNOWN handling (like NULL)
    (DataType.UNKNOWN, DataType.STRING): (0.50, "Unknown can represent string"),
    (DataType.UNKNOWN, DataType.INTEGER): (0.45, "Unknown to numeric (needs default)"),
    (DataType.STRING, DataType.UNKNOWN): (0.45, "String can be unknown"),
    (DataType.INTEGER, DataType.UNKNOWN): (0.40, "Integer can be unknown"),

    # Complex types
    (DataType.ARRAY, DataType.JSON): (0.60, "Array to JSON representation"),
    (DataType.RECORD, DataType.JSON): (0.70, "Record to JSON representation"),
    (DataType.JSON, DataType.ARRAY): (0.55, "JSON to array (if array)"),
    (DataType.JSON, DataType.RECORD): (0.65, "JSON to record (if object)"),
    (DataType.ARRAY, DataType.RECORD): (0.20, "Array vs structured record"),
    (DataType.RECORD, DataType.ARRAY): (0.20, "Record vs array"),

    # Binary type
    (DataType.BYTES, DataType.STRING): (0.30, "Bytes to string (encoding)"),
    (DataType.STRING, DataType.BYTES): (0.30, "String to bytes (encoding)"),

    # UUID special handling
    (DataType.UUID, DataType.STRING): (0.90, "UUID to string representation"),
    (DataType.STRING, DataType.UUID): (0.70, "String to UUID (parsing)"),

    # ENUM handling
    (DataType.ENUM, DataType.STRING): (0.85, "Enum to string value"),
    (DataType.STRING, DataType.ENUM): (0.60, "String to enum (if valid)"),
}

# Default scores for categories when specific mapping not found
CATEGORY_CROSS_SCORES: dict[tuple[TypeCategory, TypeCategory], float] = {
    # Same category defaults (if not exact match)
    (TypeCategory.NUMERIC, TypeCategory.NUMERIC): 0.70,
    (TypeCategory.STRING, TypeCategory.STRING): 0.90,
    (TypeCategory.TEMPORAL, TypeCategory.TEMPORAL): 0.75,
    (TypeCategory.COMPLEX, TypeCategory.COMPLEX): 0.30,

    # Cross-category defaults
    (TypeCategory.NUMERIC, TypeCategory.STRING): 0.30,
    (TypeCategory.STRING, TypeCategory.NUMERIC): 0.25,
    (TypeCategory.NUMERIC, TypeCategory.BOOLEAN): 0.35,
    (TypeCategory.BOOLEAN, TypeCategory.NUMERIC): 0.35,
    (TypeCategory.TEMPORAL, TypeCategory.STRING): 0.45,
    (TypeCategory.STRING, TypeCategory.TEMPORAL): 0.35,
    (TypeCategory.BOOLEAN, TypeCategory.STRING): 0.40,
    (TypeCategory.STRING, TypeCategory.BOOLEAN): 0.30,

    # Binary vs others (generally low compatibility)
    (TypeCategory.BINARY, TypeCategory.STRING): 0.25,
    (TypeCategory.STRING, TypeCategory.BINARY): 0.25,
    (TypeCategory.BINARY, TypeCategory.NUMERIC): 0.15,
    (TypeCategory.NUMERIC, TypeCategory.BINARY): 0.15,
    (TypeCategory.BINARY, TypeCategory.BOOLEAN): 0.10,
    (TypeCategory.BOOLEAN, TypeCategory.BINARY): 0.10,
    (TypeCategory.BINARY, TypeCategory.TEMPORAL): 0.10,
    (TypeCategory.TEMPORAL, TypeCategory.BINARY): 0.10,
    (TypeCategory.BINARY, TypeCategory.COMPLEX): 0.15,
    (TypeCategory.COMPLEX, TypeCategory.BINARY): 0.15,

    # Complex vs others
    (TypeCategory.COMPLEX, TypeCategory.STRING): 0.20,
    (TypeCategory.STRING, TypeCategory.COMPLEX): 0.15,
    (TypeCategory.COMPLEX, TypeCategory.NUMERIC): 0.10,
    (TypeCategory.NUMERIC, TypeCategory.COMPLEX): 0.10,
    (TypeCategory.COMPLEX, TypeCategory.BOOLEAN): 0.10,
    (TypeCategory.BOOLEAN, TypeCategory.COMPLEX): 0.10,
    (TypeCategory.COMPLEX, TypeCategory.TEMPORAL): 0.15,
    (TypeCategory.TEMPORAL, TypeCategory.COMPLEX): 0.15,

    # NULL/UNKNOWN handling
    (TypeCategory.NULL, TypeCategory.STRING): 0.50,
    (TypeCategory.NULL, TypeCategory.NUMERIC): 0.45,
    (TypeCategory.NULL, TypeCategory.BOOLEAN): 0.45,
    (TypeCategory.NULL, TypeCategory.TEMPORAL): 0.45,
    (TypeCategory.NULL, TypeCategory.BINARY): 0.40,
    (TypeCategory.NULL, TypeCategory.COMPLEX): 0.40,
    (TypeCategory.STRING, TypeCategory.NULL): 0.50,
    (TypeCategory.NUMERIC, TypeCategory.NULL): 0.45,
    (TypeCategory.BOOLEAN, TypeCategory.NULL): 0.45,
    (TypeCategory.TEMPORAL, TypeCategory.NULL): 0.45,
    (TypeCategory.BINARY, TypeCategory.NULL): 0.40,
    (TypeCategory.COMPLEX, TypeCategory.NULL): 0.40,
}


# =============================================================================
# TYPE COMPATIBILITY SCORER
# =============================================================================


class TypeCompatibilityScorer:
    """
    Domain service for computing type compatibility scores.

    Uses a combination of explicit compatibility matrix and category-based
    fallback scoring to determine how compatible two data types are.

    Example:
        scorer = TypeCompatibilityScorer()
        score = scorer.score(DataType.INTEGER, DataType.LONG)  # 0.95
    """

    # Singleton instance
    _default_instance: ClassVar[TypeCompatibilityScorer | None] = None

    def __init__(
        self,
        unknown_score: float = 0.5,
        strict_mode: bool = False,
    ) -> None:
        """
        Initialize scorer.

        Args:
            unknown_score: Score for unknown type combinations
            strict_mode: If True, only exact matches score 1.0
        """
        self._unknown_score = unknown_score
        self._strict_mode = strict_mode

    @classmethod
    def default(cls) -> TypeCompatibilityScorer:
        """Get default singleton instance."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def reset_default(cls) -> None:
        """Reset singleton (for testing)."""
        cls._default_instance = None

    def score(self, source: DataType, target: DataType) -> float:
        """
        Get compatibility score between two types.

        Args:
            source: Source data type
            target: Target data type

        Returns:
            Score between 0.0 (incompatible) and 1.0 (exact match)
        """
        return self.get_compatibility(source, target).score

    def get_compatibility(
        self,
        source: DataType,
        target: DataType,
    ) -> TypeCompatibilityResult:
        """
        Get full compatibility result.

        Args:
            source: Source data type
            target: Target data type

        Returns:
            TypeCompatibilityResult with score, level, and reason
        """
        # Exact match
        if source == target:
            return TypeCompatibilityResult(
                source_type=source,
                target_type=target,
                score=1.0,
                level=CompatibilityLevel.EXACT,
                reason=f"Exact match: {source.value}",
            )

        # Check explicit mapping
        if (source, target) in COMPATIBILITY_MATRIX:
            score, reason = COMPATIBILITY_MATRIX[(source, target)]
            return TypeCompatibilityResult(
                source_type=source,
                target_type=target,
                score=score,
                level=CompatibilityLevel.from_score(score),
                reason=reason,
            )

        # Fall back to category-based scoring
        source_cat = TypeCategory.from_data_type(source)
        target_cat = TypeCategory.from_data_type(target)

        if (source_cat, target_cat) in CATEGORY_CROSS_SCORES:
            score = CATEGORY_CROSS_SCORES[(source_cat, target_cat)]
            reason = f"Category-based: {source_cat.value} → {target_cat.value}"
        else:
            score = self._unknown_score
            reason = f"Unknown conversion: {source.value} → {target.value}"

        return TypeCompatibilityResult(
            source_type=source,
            target_type=target,
            score=score,
            level=CompatibilityLevel.from_score(score),
            reason=reason,
        )

    def is_compatible(self, source: DataType, target: DataType) -> bool:
        """
        Check if types are compatible.

        Args:
            source: Source data type
            target: Target data type

        Returns:
            True if score >= 0.5
        """
        return self.score(source, target) >= 0.5

    def get_compatible_types(
        self,
        source: DataType,
        min_score: float = 0.5,
    ) -> list[DataType]:
        """
        Get list of types compatible with source.

        Args:
            source: Source data type
            min_score: Minimum compatibility score

        Returns:
            List of compatible DataTypes
        """
        compatible = []
        for target in DataType:
            if self.score(source, target) >= min_score:
                compatible.append(target)
        return compatible
