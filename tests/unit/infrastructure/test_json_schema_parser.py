"""
tests.unit.infrastructure.test_json_schema_parser | Layer: TEST
Tests: JSON Schema parser | Target: src/infrastructure/adapters/schema_parsers/json_schema.py

TDD Phase: RED → Tests written before implementation
"""

import json
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from nexus_matcher.shared.types.base import DataType


class TestJsonSchemaParserProperties:
    """Test basic parser properties."""

    def test_format_name(self):
        """Test format name is 'json_schema'."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        parser = JsonSchemaParser()
        assert parser.format_name == "json_schema"

    def test_file_extensions(self):
        """Test supported file extensions."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        parser = JsonSchemaParser()
        assert ".json" in parser.file_extensions
        assert ".schema.json" in parser.file_extensions


class TestJsonSchemaParserSimpleTypes:
    """Test parsing simple type schemas."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        return JsonSchemaParser()

    def test_parse_string_property(self, parser):
        """Test parsing string property."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        schema_obj = result.unwrap()
        assert len(schema_obj.fields) == 1
        assert schema_obj.fields[0].name == "name"
        assert schema_obj.fields[0].data_type == DataType.STRING

    def test_parse_integer_property(self, parser):
        """Test parsing integer property."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.data_type == DataType.INTEGER

    def test_parse_number_property(self, parser):
        """Test parsing number property (maps to DOUBLE)."""
        schema = {
            "type": "object",
            "properties": {
                "amount": {"type": "number"}
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.data_type == DataType.DOUBLE

    def test_parse_boolean_property(self, parser):
        """Test parsing boolean property."""
        schema = {
            "type": "object",
            "properties": {
                "active": {"type": "boolean"}
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.data_type == DataType.BOOLEAN

    def test_parse_null_type(self, parser):
        """Test parsing null type."""
        schema = {
            "type": "object",
            "properties": {
                "nothing": {"type": "null"}
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.data_type == DataType.UNKNOWN


class TestJsonSchemaParserFormatTypes:
    """Test parsing format-annotated types."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        return JsonSchemaParser()

    def test_parse_date_format(self, parser):
        """Test parsing date format."""
        schema = {
            "type": "object",
            "properties": {
                "birth_date": {"type": "string", "format": "date"}
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.data_type == DataType.DATE

    def test_parse_datetime_format(self, parser):
        """Test parsing date-time format."""
        schema = {
            "type": "object",
            "properties": {
                "created_at": {"type": "string", "format": "date-time"}
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.data_type == DataType.TIMESTAMP

    def test_parse_uuid_format(self, parser):
        """Test parsing uuid format."""
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "format": "uuid"}
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.data_type == DataType.UUID

    def test_parse_byte_format(self, parser):
        """Test parsing byte format (base64)."""
        schema = {
            "type": "object",
            "properties": {
                "data": {"type": "string", "format": "byte"}
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.data_type == DataType.BYTES


class TestJsonSchemaParserArrayTypes:
    """Test parsing array types."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        return JsonSchemaParser()

    def test_parse_array_of_strings(self, parser):
        """Test parsing array of strings."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.data_type == DataType.ARRAY
        assert field.is_array
        assert field.array_item_type == DataType.STRING

    def test_parse_array_of_integers(self, parser):
        """Test parsing array of integers."""
        schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.is_array
        assert field.array_item_type == DataType.INTEGER


class TestJsonSchemaParserNestedObjects:
    """Test parsing nested object types."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        return JsonSchemaParser()

    def test_parse_nested_object(self, parser):
        """Test parsing nested object."""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    }
                }
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        schema_obj = result.unwrap()

        # Should have the parent record plus nested fields
        field_paths = [f.full_path for f in schema_obj.fields]
        assert "address" in field_paths
        assert "address.street" in field_paths
        assert "address.city" in field_paths

    def test_nested_field_paths(self, parser):
        """Test nested field paths are correct."""
        schema = {
            "type": "object",
            "properties": {
                "customer": {
                    "type": "object",
                    "properties": {
                        "contact": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        schema_obj = result.unwrap()

        email_field = next(
            (f for f in schema_obj.fields if f.name == "email"), None
        )
        assert email_field is not None
        assert email_field.full_path == "customer.contact.email"
        assert email_field.parent_path == "customer.contact"


class TestJsonSchemaParserRequiredFields:
    """Test required field handling."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        return JsonSchemaParser()

    def test_required_field_not_nullable(self, parser):
        """Test required fields are not nullable."""
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"}
            },
            "required": ["id"]
        }

        result = parser.parse(schema)

        assert result.is_success
        schema_obj = result.unwrap()

        id_field = next((f for f in schema_obj.fields if f.name == "id"), None)
        name_field = next((f for f in schema_obj.fields if f.name == "name"), None)

        assert not id_field.is_nullable  # Required
        assert name_field.is_nullable  # Not required


class TestJsonSchemaParserNullableTypes:
    """Test nullable type handling."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        return JsonSchemaParser()

    def test_type_array_with_null(self, parser):
        """Test type array with null is nullable."""
        schema = {
            "type": "object",
            "properties": {
                "optional_name": {"type": ["string", "null"]}
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.is_nullable
        assert field.data_type == DataType.STRING


class TestJsonSchemaParserMetadata:
    """Test metadata extraction."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        return JsonSchemaParser()

    def test_extracts_description(self, parser):
        """Test description is extracted."""
        schema = {
            "type": "object",
            "properties": {
                "balance": {
                    "type": "number",
                    "description": "Account balance in USD"
                }
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.description == "Account balance in USD"

    def test_extracts_default_value(self, parser):
        """Test default value is extracted."""
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "default": "active"
                }
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.default_value == "active"

    def test_extracts_schema_title(self, parser):
        """Test schema title becomes schema name."""
        schema = {
            "title": "CustomerProfile",
            "type": "object",
            "properties": {
                "id": {"type": "string"}
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        schema_obj = result.unwrap()
        assert schema_obj.name == "CustomerProfile"


class TestJsonSchemaParserEnums:
    """Test enum handling."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        return JsonSchemaParser()

    def test_parse_string_enum(self, parser):
        """Test parsing string enum."""
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "pending"]
                }
            }
        }

        result = parser.parse(schema)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.data_type == DataType.ENUM


class TestJsonSchemaParserFileOperations:
    """Test file parsing operations."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        return JsonSchemaParser()

    def test_parse_file(self, parser):
        """Test parsing from file."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }

        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            f.flush()

            result = parser.parse_file(Path(f.name))

            assert result.is_success
            assert len(result.unwrap().fields) == 1

    def test_parse_file_not_found(self, parser):
        """Test error on file not found."""
        result = parser.parse_file(Path("/nonexistent/schema.json"))

        assert result.is_failure
        assert "FILE_NOT_FOUND" in (result.error_code or "")

    def test_parse_file_wrong_extension(self, parser):
        """Test error on wrong extension."""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{}")
            f.flush()

            result = parser.parse_file(Path(f.name))

            assert result.is_failure


class TestJsonSchemaParserCanParse:
    """Test can_parse detection."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        return JsonSchemaParser()

    def test_can_parse_valid_schema(self, parser):
        """Test can_parse returns True for valid schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }

        assert parser.can_parse(schema)

    def test_cannot_parse_avro_schema(self, parser):
        """Test can_parse returns False for Avro schema."""
        avro_schema = {
            "type": "record",
            "name": "Test",
            "fields": [{"name": "id", "type": "string"}]
        }

        assert not parser.can_parse(avro_schema)

    def test_cannot_parse_invalid_json(self, parser):
        """Test can_parse returns False for invalid JSON."""
        assert not parser.can_parse("not valid json {")


class TestJsonSchemaParserEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
            JsonSchemaParser,
        )

        return JsonSchemaParser()

    def test_empty_properties(self, parser):
        """Test schema with empty properties."""
        schema = {
            "type": "object",
            "properties": {}
        }

        result = parser.parse(schema)

        assert result.is_success
        assert len(result.unwrap().fields) == 0

    def test_schema_without_properties(self, parser):
        """Test schema without properties key."""
        schema = {
            "type": "object"
        }

        result = parser.parse(schema)

        assert result.is_success
        assert len(result.unwrap().fields) == 0

    def test_invalid_json_string(self, parser):
        """Test parsing invalid JSON string."""
        result = parser.parse("not valid json {")

        assert result.is_failure
        assert "PARSE_ERROR" in (result.error_code or "")

    def test_source_format_set(self, parser):
        """Test source_format is set correctly."""
        schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}}
        }

        result = parser.parse(schema)

        assert result.is_success
        assert result.unwrap().source_format == "json_schema"
