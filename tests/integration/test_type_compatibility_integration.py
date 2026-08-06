"""
tests.integration.test_type_compatibility_integration | Layer: TEST
Integration tests for type compatibility with NexusMatcher.
"""

import pytest

from nexus_matcher.domain.services.type_compatibility import (
    CompatibilityLevel,
    TypeCategory,
    TypeCompatibilityScorer,
)
from nexus_matcher.shared.types.base import DataType


class TestTypeCompatibilityIntegration:
    """Integration tests for type compatibility scorer."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        return TypeCompatibilityScorer.default()

    def test_default_singleton(self):
        """Test default singleton pattern works."""
        scorer1 = TypeCompatibilityScorer.default()
        scorer2 = TypeCompatibilityScorer.default()
        assert scorer1 is scorer2

    def test_all_data_types_have_category(self, scorer):
        """Test all DataType values have a category mapping."""
        for dtype in DataType:
            category = TypeCategory.from_data_type(dtype)
            assert category is not None, f"No category for {dtype}"

    def test_all_data_types_score_with_themselves(self, scorer):
        """Test all types score 1.0 with themselves (exact match)."""
        for dtype in DataType:
            score = scorer.score(dtype, dtype)
            assert score == 1.0, f"{dtype} self-match should be 1.0, got {score}"

    def test_all_scores_in_valid_range(self, scorer):
        """Test all type combinations produce valid scores."""
        for source in DataType:
            for target in DataType:
                score = scorer.score(source, target)
                assert 0.0 <= score <= 1.0, f"{source}→{target} = {score}"

    def test_numeric_types_form_compatible_group(self, scorer):
        """Test numeric types have high compatibility within group."""
        numeric_types = [
            DataType.INTEGER,
            DataType.LONG,
            DataType.FLOAT,
            DataType.DOUBLE,
            DataType.DECIMAL,
        ]

        for source in numeric_types:
            for target in numeric_types:
                score = scorer.score(source, target)
                # Same type is 1.0, within category should be >= 0.65
                if source == target:
                    assert score == 1.0
                else:
                    assert score >= 0.65, f"{source}→{target} should be high"

    def test_string_types_compatible_with_uuid_enum(self, scorer):
        """Test STRING compatible with UUID and ENUM (both string-like)."""
        assert scorer.score(DataType.UUID, DataType.STRING) >= 0.80
        assert scorer.score(DataType.ENUM, DataType.STRING) >= 0.80

    def test_temporal_types_within_category(self, scorer):
        """Test temporal types have high compatibility."""
        score1 = scorer.score(DataType.DATE, DataType.TIMESTAMP)
        score2 = scorer.score(DataType.TIMESTAMP, DataType.DATE)

        assert score1 >= 0.70
        assert score2 >= 0.65  # Loses information

    def test_cross_category_low_scores(self, scorer):
        """Test cross-category conversions have low scores."""
        # Binary to Boolean should be very low
        score = scorer.score(DataType.BYTES, DataType.BOOLEAN)
        assert score <= 0.30

        # Complex to Numeric should be very low
        score = scorer.score(DataType.ARRAY, DataType.INTEGER)
        assert score <= 0.20


class TestTypeCompatibilityWithSchemaField:
    """Test type compatibility in context of schema matching."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        return TypeCompatibilityScorer.default()

    def test_field_to_entry_type_scoring(self, scorer):
        """Test scoring field type against entry type."""
        # Simulate: field is INTEGER, entry expects LONG
        field_type = DataType.INTEGER
        entry_type = DataType.LONG

        result = scorer.get_compatibility(field_type, entry_type)

        assert result.score >= 0.90
        assert result.is_compatible

    def test_incompatible_field_entry_types(self, scorer):
        """Test incompatible field and entry types."""
        # Field is BYTES, entry expects BOOLEAN
        result = scorer.get_compatibility(DataType.BYTES, DataType.BOOLEAN)

        assert result.score <= 0.30
        assert not result.is_compatible

    def test_unknown_field_type_neutral_score(self, scorer):
        """Test UNKNOWN field type gets neutral scores."""
        result = scorer.get_compatibility(DataType.UNKNOWN, DataType.STRING)

        assert 0.40 <= result.score <= 0.60


class TestTypeCompatibilityLevels:
    """Test compatibility level classification."""

    def test_level_boundaries(self):
        """Test level boundaries are correct."""
        assert CompatibilityLevel.from_score(1.0) == CompatibilityLevel.EXACT
        assert CompatibilityLevel.from_score(0.99) == CompatibilityLevel.EQUIVALENT
        assert CompatibilityLevel.from_score(0.95) == CompatibilityLevel.EQUIVALENT
        assert CompatibilityLevel.from_score(0.94) == CompatibilityLevel.COMPATIBLE
        assert CompatibilityLevel.from_score(0.80) == CompatibilityLevel.COMPATIBLE
        assert CompatibilityLevel.from_score(0.79) == CompatibilityLevel.CONVERTIBLE
        assert CompatibilityLevel.from_score(0.50) == CompatibilityLevel.CONVERTIBLE
        assert CompatibilityLevel.from_score(0.49) == CompatibilityLevel.INCOMPATIBLE
        assert CompatibilityLevel.from_score(0.0) == CompatibilityLevel.INCOMPATIBLE


class TestTypeCompatibilityEdgeCases:
    """Test edge cases and unusual scenarios."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        return TypeCompatibilityScorer()

    def test_custom_unknown_score(self):
        """Test custom unknown_score parameter."""
        scorer = TypeCompatibilityScorer(unknown_score=0.3)

        # For types not in matrix or category scores
        # The unknown score should be used
        result = scorer.get_compatibility(DataType.BOOLEAN, DataType.ARRAY)
        # Should use category cross score if available, else unknown_score
        assert result.score <= 0.5

    def test_get_compatible_types_filtering(self, scorer):
        """Test get_compatible_types with different thresholds."""
        # With default threshold (0.5)
        compatible = scorer.get_compatible_types(DataType.INTEGER)
        assert DataType.LONG in compatible
        assert DataType.FLOAT in compatible

        # With high threshold (0.9)
        very_compatible = scorer.get_compatible_types(DataType.INTEGER, min_score=0.9)
        assert DataType.INTEGER in very_compatible  # Self
        assert DataType.LONG in very_compatible  # 0.95

    def test_result_properties(self, scorer):
        """Test result object properties work correctly."""
        exact = scorer.get_compatibility(DataType.STRING, DataType.STRING)
        assert exact.is_exact
        assert exact.is_compatible

        incompatible = scorer.get_compatibility(DataType.BYTES, DataType.BOOLEAN)
        assert not incompatible.is_exact
        assert not incompatible.is_compatible
