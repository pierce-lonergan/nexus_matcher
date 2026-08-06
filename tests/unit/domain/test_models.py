"""
tests.unit.domain.test_models | Layer: TEST
Unit tests for domain models.

## Relationships
# TESTS → domain/models/entities :: SchemaField, DictionaryEntry, MatchResult
"""

import pytest

from nexus_matcher.domain.models import (
    DictionaryEntry,
    MatchResult,
    Schema,
    SchemaField,
)
from nexus_matcher.shared.types import DataType, MatchDecision, ProtectionLevel
from nexus_matcher.shared.types.base import PerformanceMetrics, ScoreBreakdown


class TestDataType:
    """Tests for DataType enum."""

    def test_from_string_basic_types(self):
        """Test basic type string normalization."""
        assert DataType.from_string("string") == DataType.STRING
        assert DataType.from_string("STRING") == DataType.STRING
        assert DataType.from_string("varchar") == DataType.STRING
        assert DataType.from_string("VARCHAR(255)") == DataType.STRING

    def test_from_string_numeric_types(self):
        """Test numeric type normalization."""
        assert DataType.from_string("int") == DataType.INTEGER
        assert DataType.from_string("integer") == DataType.INTEGER
        assert DataType.from_string("bigint") == DataType.LONG
        assert DataType.from_string("long") == DataType.LONG
        assert DataType.from_string("float") == DataType.FLOAT
        assert DataType.from_string("double") == DataType.DOUBLE
        assert DataType.from_string("decimal") == DataType.DOUBLE

    def test_from_string_temporal_types(self):
        """Test temporal type normalization."""
        assert DataType.from_string("date") == DataType.DATE
        assert DataType.from_string("timestamp") == DataType.TIMESTAMP
        assert DataType.from_string("datetime") == DataType.TIMESTAMP

    def test_from_string_unknown(self):
        """Test unknown type handling."""
        assert DataType.from_string("") == DataType.UNKNOWN
        assert DataType.from_string("xyz") == DataType.UNKNOWN


class TestSchemaField:
    """Tests for SchemaField model."""

    def test_create_basic_field(self):
        """Test basic field creation."""
        field = SchemaField(
            name="email",
            data_type=DataType.STRING,
            description="Customer email address",
        )

        assert field.name == "email"
        assert field.data_type == DataType.STRING
        assert field.full_path == "email"  # Auto-set if empty

    def test_create_nested_field(self):
        """Test nested field creation."""
        field = SchemaField(
            name="email",
            data_type=DataType.STRING,
            full_path="customer.contact.email",
            parent_path="customer.contact",
        )

        assert field.is_nested
        assert field.depth == 2
        assert field.root_name == "customer"

    def test_with_path(self):
        """Test path modification."""
        field = SchemaField(name="email", data_type=DataType.STRING)
        nested = field.with_path("customer.contact")

        assert nested.full_path == "customer.contact.email"
        assert nested.parent_path == "customer.contact"

    def test_to_searchable_text(self):
        """Test searchable text generation."""
        field = SchemaField(
            name="customer_email",
            data_type=DataType.STRING,
            description="Email address for customer",
        )

        text = field.to_searchable_text()
        assert "customer email" in text
        assert "Email address" in text


class TestDictionaryEntry:
    """Tests for DictionaryEntry model."""

    def test_create_entry(self):
        """Test basic entry creation."""
        entry = DictionaryEntry(
            id="field_001",
            business_name="Customer Email Address",
            logical_name="cust_email",
            definition="Primary email address for customer contact",
            data_type=DataType.STRING,
            protection_level=ProtectionLevel.PII,
        )

        assert entry.id == "field_001"
        assert entry.business_name == "Customer Email Address"
        assert entry.protection_level == ProtectionLevel.PII

    def test_content_hash_consistency(self):
        """Test that content hash is deterministic."""
        entry = DictionaryEntry(
            id="field_001",
            business_name="Test",
            logical_name="test",
            definition="Test definition",
            data_type=DataType.STRING,
        )

        hash1 = entry.content_hash
        hash2 = entry.content_hash

        assert hash1 == hash2
        assert len(hash1) == 32  # BLAKE2b with 16-byte digest = 32 hex chars

    def test_content_hash_difference(self):
        """Test that different content produces different hash."""
        entry1 = DictionaryEntry(
            id="1",
            business_name="Test A",
            logical_name="test",
            definition="",
            data_type=DataType.STRING,
        )

        entry2 = DictionaryEntry(
            id="1",
            business_name="Test B",
            logical_name="test",
            definition="",
            data_type=DataType.STRING,
        )

        assert entry1.content_hash != entry2.content_hash

    def test_type_compatibility_exact_match(self):
        """Test exact type match."""
        entry = DictionaryEntry(
            id="1",
            business_name="Test",
            logical_name="test",
            definition="",
            data_type=DataType.STRING,
        )

        assert entry.matches_type(DataType.STRING) == 1.0

    def test_type_compatibility_partial_match(self):
        """Test partial type compatibility."""
        entry = DictionaryEntry(
            id="1",
            business_name="Test",
            logical_name="test",
            definition="",
            data_type=DataType.INTEGER,
        )

        score = entry.matches_type(DataType.LONG)
        assert 0.5 < score < 1.0

    def test_validation_empty_id(self):
        """Test that empty ID raises error."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            DictionaryEntry(
                id="",
                business_name="Test",
                logical_name="test",
                definition="",
                data_type=DataType.STRING,
            )

    def test_validation_empty_business_name(self):
        """Test that empty business name raises error."""
        with pytest.raises(ValueError, match="business_name cannot be empty"):
            DictionaryEntry(
                id="1",
                business_name="",
                logical_name="test",
                definition="",
                data_type=DataType.STRING,
            )


class TestMatchResult:
    """Tests for MatchResult model."""

    def test_create_match_result(self):
        """Test match result creation."""
        field = SchemaField(name="email", data_type=DataType.STRING)
        entry = DictionaryEntry(
            id="1",
            business_name="Email",
            logical_name="email",
            definition="",
            data_type=DataType.STRING,
        )

        result = MatchResult(
            schema_field=field,
            dictionary_entry=entry,
            rank=1,
            final_confidence=0.95,
            score_breakdown=ScoreBreakdown(semantic_score=0.95),
            decision=MatchDecision.AUTO_APPROVE,
            performance=PerformanceMetrics(latency_ms=50.0),
        )

        assert result.rank == 1
        assert result.final_confidence == 0.95
        assert result.is_auto_approved
        assert not result.needs_review

    def test_validation_confidence_range(self):
        """Test confidence validation."""
        field = SchemaField(name="email", data_type=DataType.STRING)
        entry = DictionaryEntry(
            id="1",
            business_name="Email",
            logical_name="email",
            definition="",
            data_type=DataType.STRING,
        )

        with pytest.raises(ValueError, match="Confidence must be"):
            MatchResult(
                schema_field=field,
                dictionary_entry=entry,
                rank=1,
                final_confidence=1.5,  # Invalid
                score_breakdown=ScoreBreakdown(),
                decision=MatchDecision.AUTO_APPROVE,
                performance=PerformanceMetrics(latency_ms=50.0),
            )


class TestSchema:
    """Tests for Schema model."""

    def test_create_schema(self):
        """Test schema creation."""
        fields = (
            SchemaField(name="id", data_type=DataType.STRING, full_path="id"),
            SchemaField(name="email", data_type=DataType.STRING, full_path="email"),
        )

        schema = Schema(
            name="Customer",
            fields=fields,
            namespace="com.example",
            source_format="avro",
        )

        assert schema.name == "Customer"
        assert schema.field_count == 2
        assert "id" in schema.field_paths
        assert "email" in schema.field_paths

    def test_get_field(self):
        """Test field retrieval by path."""
        fields = (SchemaField(name="email", data_type=DataType.STRING, full_path="contact.email"),)
        schema = Schema(name="Test", fields=fields)

        field = schema.get_field("contact.email")
        assert field is not None
        assert field.name == "email"

        missing = schema.get_field("nonexistent")
        assert missing is None
