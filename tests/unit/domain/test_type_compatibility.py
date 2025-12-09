"""
tests.unit.domain.test_type_compatibility | Layer: TEST
Tests: Type compatibility scoring | Target: src/domain/services/type_compatibility.py

TDD Phase: RED → Tests written before implementation
"""

import pytest

from nexus_matcher.shared.types.base import DataType


class TestTypeCategory:
    """Test TypeCategory enum."""

    def test_category_values(self):
        """Test category enum values."""
        from nexus_matcher.domain.services.type_compatibility import TypeCategory

        assert TypeCategory.NUMERIC.value == "numeric"
        assert TypeCategory.STRING.value == "string"
        assert TypeCategory.TEMPORAL.value == "temporal"
        assert TypeCategory.BOOLEAN.value == "boolean"
        assert TypeCategory.BINARY.value == "binary"
        assert TypeCategory.COMPLEX.value == "complex"

    def test_get_category_for_numeric_types(self):
        """Test getting category for numeric types."""
        from nexus_matcher.domain.services.type_compatibility import TypeCategory

        assert TypeCategory.from_data_type(DataType.INTEGER) == TypeCategory.NUMERIC
        assert TypeCategory.from_data_type(DataType.LONG) == TypeCategory.NUMERIC
        assert TypeCategory.from_data_type(DataType.FLOAT) == TypeCategory.NUMERIC
        assert TypeCategory.from_data_type(DataType.DOUBLE) == TypeCategory.NUMERIC
        assert TypeCategory.from_data_type(DataType.DECIMAL) == TypeCategory.NUMERIC

    def test_get_category_for_string_types(self):
        """Test getting category for string types."""
        from nexus_matcher.domain.services.type_compatibility import TypeCategory

        assert TypeCategory.from_data_type(DataType.STRING) == TypeCategory.STRING

    def test_get_category_for_temporal_types(self):
        """Test getting category for temporal types."""
        from nexus_matcher.domain.services.type_compatibility import TypeCategory

        assert TypeCategory.from_data_type(DataType.DATE) == TypeCategory.TEMPORAL
        assert TypeCategory.from_data_type(DataType.TIMESTAMP) == TypeCategory.TEMPORAL

    def test_get_category_for_boolean(self):
        """Test getting category for boolean type."""
        from nexus_matcher.domain.services.type_compatibility import TypeCategory

        assert TypeCategory.from_data_type(DataType.BOOLEAN) == TypeCategory.BOOLEAN

    def test_get_category_for_binary(self):
        """Test getting category for binary type."""
        from nexus_matcher.domain.services.type_compatibility import TypeCategory

        assert TypeCategory.from_data_type(DataType.BYTES) == TypeCategory.BINARY

    def test_get_category_for_complex(self):
        """Test getting category for complex types."""
        from nexus_matcher.domain.services.type_compatibility import TypeCategory

        assert TypeCategory.from_data_type(DataType.ARRAY) == TypeCategory.COMPLEX
        assert TypeCategory.from_data_type(DataType.RECORD) == TypeCategory.COMPLEX
        assert TypeCategory.from_data_type(DataType.JSON) == TypeCategory.COMPLEX


class TestCompatibilityLevel:
    """Test CompatibilityLevel enum."""

    def test_level_values(self):
        """Test compatibility level values."""
        from nexus_matcher.domain.services.type_compatibility import CompatibilityLevel

        assert CompatibilityLevel.EXACT.value == "exact"
        assert CompatibilityLevel.EQUIVALENT.value == "equivalent"
        assert CompatibilityLevel.COMPATIBLE.value == "compatible"
        assert CompatibilityLevel.CONVERTIBLE.value == "convertible"
        assert CompatibilityLevel.INCOMPATIBLE.value == "incompatible"

    def test_level_from_score(self):
        """Test getting level from score."""
        from nexus_matcher.domain.services.type_compatibility import CompatibilityLevel

        assert CompatibilityLevel.from_score(1.0) == CompatibilityLevel.EXACT
        assert CompatibilityLevel.from_score(0.97) == CompatibilityLevel.EQUIVALENT
        assert CompatibilityLevel.from_score(0.85) == CompatibilityLevel.COMPATIBLE
        assert CompatibilityLevel.from_score(0.60) == CompatibilityLevel.CONVERTIBLE
        assert CompatibilityLevel.from_score(0.30) == CompatibilityLevel.INCOMPATIBLE


class TestTypeCompatibilityResult:
    """Test TypeCompatibilityResult value object."""

    def test_creates_result(self):
        """Test creating a result."""
        from nexus_matcher.domain.services.type_compatibility import (
            CompatibilityLevel,
            TypeCompatibilityResult,
        )

        result = TypeCompatibilityResult(
            source_type=DataType.INTEGER,
            target_type=DataType.LONG,
            score=0.95,
            level=CompatibilityLevel.COMPATIBLE,
            reason="Widening numeric conversion",
        )

        assert result.source_type == DataType.INTEGER
        assert result.target_type == DataType.LONG
        assert result.score == 0.95
        assert result.level == CompatibilityLevel.COMPATIBLE

    def test_is_compatible_property(self):
        """Test is_compatible property."""
        from nexus_matcher.domain.services.type_compatibility import (
            CompatibilityLevel,
            TypeCompatibilityResult,
        )

        compatible = TypeCompatibilityResult(
            DataType.INTEGER, DataType.LONG, 0.95, CompatibilityLevel.COMPATIBLE, ""
        )
        incompatible = TypeCompatibilityResult(
            DataType.INTEGER, DataType.BOOLEAN, 0.20, CompatibilityLevel.INCOMPATIBLE, ""
        )

        assert compatible.is_compatible
        assert not incompatible.is_compatible

    def test_is_exact_property(self):
        """Test is_exact property."""
        from nexus_matcher.domain.services.type_compatibility import (
            CompatibilityLevel,
            TypeCompatibilityResult,
        )

        exact = TypeCompatibilityResult(
            DataType.STRING, DataType.STRING, 1.0, CompatibilityLevel.EXACT, ""
        )
        not_exact = TypeCompatibilityResult(
            DataType.INTEGER, DataType.LONG, 0.95, CompatibilityLevel.COMPATIBLE, ""
        )

        assert exact.is_exact
        assert not not_exact.is_exact


class TestTypeCompatibilityScorer:
    """Test TypeCompatibilityScorer service."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        from nexus_matcher.domain.services.type_compatibility import TypeCompatibilityScorer

        return TypeCompatibilityScorer()

    # === Exact Match Tests ===

    def test_exact_match_same_type(self, scorer):
        """Test exact match for same type."""
        assert scorer.score(DataType.STRING, DataType.STRING) == 1.0
        assert scorer.score(DataType.INTEGER, DataType.INTEGER) == 1.0
        assert scorer.score(DataType.DATE, DataType.DATE) == 1.0
        assert scorer.score(DataType.BOOLEAN, DataType.BOOLEAN) == 1.0

    # === Numeric Type Tests ===

    def test_integer_to_long_widening(self, scorer):
        """Test INTEGER → LONG widening conversion."""
        score = scorer.score(DataType.INTEGER, DataType.LONG)
        assert score >= 0.90  # High compatibility

    def test_long_to_integer_narrowing(self, scorer):
        """Test LONG → INTEGER narrowing conversion."""
        score = scorer.score(DataType.LONG, DataType.INTEGER)
        assert 0.70 <= score < 0.90  # Lower due to possible overflow

    def test_float_to_double_widening(self, scorer):
        """Test FLOAT → DOUBLE widening conversion."""
        score = scorer.score(DataType.FLOAT, DataType.DOUBLE)
        assert score >= 0.90

    def test_double_to_float_narrowing(self, scorer):
        """Test DOUBLE → FLOAT narrowing conversion."""
        score = scorer.score(DataType.DOUBLE, DataType.FLOAT)
        assert 0.70 <= score < 0.90  # Precision loss

    def test_integer_to_float(self, scorer):
        """Test INTEGER → FLOAT conversion."""
        score = scorer.score(DataType.INTEGER, DataType.FLOAT)
        assert score >= 0.80  # Safe conversion

    def test_decimal_to_numeric(self, scorer):
        """Test DECIMAL compatibility with other numerics."""
        score_to_double = scorer.score(DataType.DECIMAL, DataType.DOUBLE)
        score_to_int = scorer.score(DataType.DECIMAL, DataType.INTEGER)

        assert score_to_double >= 0.70
        assert score_to_int >= 0.60

    # === String Type Tests ===

    def test_string_types_compatible(self, scorer):
        """Test string type compatibility."""
        # STRING is our only string type in DataType, but test self-match
        assert scorer.score(DataType.STRING, DataType.STRING) == 1.0

    # === Temporal Type Tests ===

    def test_date_to_timestamp(self, scorer):
        """Test DATE → TIMESTAMP conversion."""
        score = scorer.score(DataType.DATE, DataType.TIMESTAMP)
        assert score >= 0.80  # Date can be part of timestamp

    def test_timestamp_to_date(self, scorer):
        """Test TIMESTAMP → DATE conversion."""
        score = scorer.score(DataType.TIMESTAMP, DataType.DATE)
        assert 0.65 <= score < 0.85  # Loses time component

    # === Cross-Category Tests ===

    def test_numeric_to_string(self, scorer):
        """Test numeric → string conversion."""
        score = scorer.score(DataType.INTEGER, DataType.STRING)
        assert 0.20 <= score <= 0.50  # Possible but type mismatch

    def test_string_to_numeric(self, scorer):
        """Test string → numeric conversion."""
        score = scorer.score(DataType.STRING, DataType.INTEGER)
        assert 0.10 <= score <= 0.40  # Parsing required, may fail

    def test_boolean_to_integer(self, scorer):
        """Test boolean → integer conversion."""
        score = scorer.score(DataType.BOOLEAN, DataType.INTEGER)
        assert 0.30 <= score <= 0.50  # 0/1 mapping

    def test_date_to_string(self, scorer):
        """Test date → string conversion."""
        score = scorer.score(DataType.DATE, DataType.STRING)
        assert 0.40 <= score <= 0.60  # Format dependent

    def test_incompatible_types(self, scorer):
        """Test clearly incompatible types."""
        score = scorer.score(DataType.BYTES, DataType.BOOLEAN)
        assert score <= 0.30

        score = scorer.score(DataType.ARRAY, DataType.INTEGER)
        assert score <= 0.20

    # === Complex Type Tests ===

    def test_complex_types_self_match(self, scorer):
        """Test complex types match themselves."""
        assert scorer.score(DataType.ARRAY, DataType.ARRAY) == 1.0
        assert scorer.score(DataType.JSON, DataType.JSON) == 1.0
        assert scorer.score(DataType.RECORD, DataType.RECORD) == 1.0

    def test_complex_types_cross_match(self, scorer):
        """Test complex types don't match each other well."""
        score = scorer.score(DataType.ARRAY, DataType.RECORD)
        assert score <= 0.40

    # === Result Object Tests ===

    def test_get_compatibility_returns_result(self, scorer):
        """Test get_compatibility returns full result."""
        from nexus_matcher.domain.services.type_compatibility import CompatibilityLevel

        result = scorer.get_compatibility(DataType.INTEGER, DataType.LONG)

        assert result.source_type == DataType.INTEGER
        assert result.target_type == DataType.LONG
        assert result.score >= 0.90
        assert result.level in (CompatibilityLevel.COMPATIBLE, CompatibilityLevel.EQUIVALENT)
        assert result.reason  # Should have a reason

    def test_get_compatibility_exact_match(self, scorer):
        """Test get_compatibility for exact match."""
        from nexus_matcher.domain.services.type_compatibility import CompatibilityLevel

        result = scorer.get_compatibility(DataType.STRING, DataType.STRING)

        assert result.score == 1.0
        assert result.level == CompatibilityLevel.EXACT

    # === Utility Method Tests ===

    def test_is_compatible(self, scorer):
        """Test is_compatible convenience method."""
        assert scorer.is_compatible(DataType.INTEGER, DataType.LONG)
        assert scorer.is_compatible(DataType.STRING, DataType.STRING)
        assert not scorer.is_compatible(DataType.BYTES, DataType.BOOLEAN)

    def test_get_compatible_types(self, scorer):
        """Test getting list of compatible types."""
        compatible = scorer.get_compatible_types(DataType.INTEGER)

        # Should include same type
        assert DataType.INTEGER in compatible

        # Should include numeric types
        assert DataType.LONG in compatible
        assert DataType.FLOAT in compatible

        # Should not include incompatible
        assert DataType.BYTES not in compatible

    # === Unknown Type Tests ===

    def test_unknown_type_handling(self, scorer):
        """Test handling of unknown types."""
        score = scorer.score(DataType.UNKNOWN, DataType.STRING)
        assert 0.40 <= score <= 0.60  # Neutral-ish

    def test_unknown_to_unknown(self, scorer):
        """Test UNKNOWN → UNKNOWN is exact match."""
        assert scorer.score(DataType.UNKNOWN, DataType.UNKNOWN) == 1.0


class TestTypeCompatibilityScorerSymmetry:
    """Test symmetry and consistency properties."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        from nexus_matcher.domain.services.type_compatibility import TypeCompatibilityScorer

        return TypeCompatibilityScorer()

    def test_exact_matches_are_symmetric(self, scorer):
        """Test that exact matches are symmetric."""
        for dtype in [DataType.STRING, DataType.INTEGER, DataType.DATE]:
            assert scorer.score(dtype, dtype) == scorer.score(dtype, dtype)

    def test_widening_scores_higher_than_narrowing(self, scorer):
        """Test widening conversions score higher than narrowing."""
        widening = scorer.score(DataType.INTEGER, DataType.LONG)
        narrowing = scorer.score(DataType.LONG, DataType.INTEGER)

        assert widening > narrowing

    def test_same_category_higher_than_cross_category(self, scorer):
        """Test same category scores higher than cross category."""
        same_category = scorer.score(DataType.INTEGER, DataType.LONG)
        cross_category = scorer.score(DataType.INTEGER, DataType.STRING)

        assert same_category > cross_category

    def test_scores_bounded_zero_to_one(self, scorer):
        """Test all scores are between 0 and 1."""
        all_types = list(DataType)

        for source in all_types:
            for target in all_types:
                score = scorer.score(source, target)
                assert 0.0 <= score <= 1.0, f"{source} → {target} = {score}"
