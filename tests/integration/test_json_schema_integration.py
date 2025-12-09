"""
tests.integration.test_json_schema_integration | Layer: TEST
Integration tests for JSON Schema parser with NexusMatcher.
"""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
import json

from nexus_matcher.infrastructure.adapters.schema_parsers import (
    JsonSchemaParser,
    AvroSchemaParser,
)
from nexus_matcher.domain.ports.schema_parser import SchemaParserRegistry
from nexus_matcher.shared.types.base import DataType


class TestJsonSchemaParserIntegration:
    """Integration tests for JSON Schema parser."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return JsonSchemaParser()

    def test_parse_realistic_customer_schema(self, parser):
        """Test parsing a realistic customer schema."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Customer",
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Unique customer identifier"
                },
                "first_name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 50
                },
                "last_name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 50
                },
                "email": {
                    "type": "string",
                    "format": "email"
                },
                "birth_date": {
                    "type": "string",
                    "format": "date"
                },
                "account_balance": {
                    "type": "number",
                    "description": "Current account balance in USD"
                },
                "is_active": {
                    "type": "boolean",
                    "default": True
                },
                "created_at": {
                    "type": "string",
                    "format": "date-time"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "zip_code": {"type": "string"},
                        "country": {"type": "string", "default": "USA"}
                    },
                    "required": ["street", "city"]
                }
            },
            "required": ["customer_id", "first_name", "last_name", "email"]
        }

        result = parser.parse(schema)

        assert result.is_success
        schema_obj = result.unwrap()

        # Check schema metadata
        assert schema_obj.name == "Customer"
        assert schema_obj.source_format == "json_schema"

        # Check field count (10 top-level + 5 nested address fields)
        assert len(schema_obj.fields) == 15

        # Verify specific field types
        fields_by_path = {f.full_path: f for f in schema_obj.fields}

        assert fields_by_path["customer_id"].data_type == DataType.UUID
        assert fields_by_path["birth_date"].data_type == DataType.DATE
        assert fields_by_path["account_balance"].data_type == DataType.DOUBLE
        assert fields_by_path["is_active"].data_type == DataType.BOOLEAN
        assert fields_by_path["created_at"].data_type == DataType.TIMESTAMP
        assert fields_by_path["tags"].data_type == DataType.ARRAY
        assert fields_by_path["tags"].array_item_type == DataType.STRING
        assert fields_by_path["address"].data_type == DataType.RECORD
        assert fields_by_path["address.city"].data_type == DataType.STRING

        # Check required fields are not nullable
        assert not fields_by_path["customer_id"].is_nullable
        assert not fields_by_path["first_name"].is_nullable

        # Check nested required
        assert not fields_by_path["address.street"].is_nullable
        assert not fields_by_path["address.city"].is_nullable
        assert fields_by_path["address.country"].is_nullable

    def test_parse_banking_transaction_schema(self, parser):
        """Test parsing a banking transaction schema."""
        schema = {
            "title": "Transaction",
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string", "format": "uuid"},
                "account_number": {"type": "string"},
                "transaction_type": {
                    "type": "string",
                    "enum": ["DEPOSIT", "WITHDRAWAL", "TRANSFER"]
                },
                "amount": {"type": "number"},
                "currency": {"type": "string", "default": "USD"},
                "timestamp": {"type": "string", "format": "date-time"},
                "description": {"type": ["string", "null"]},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string"},
                        "device_id": {"type": "string"}
                    }
                }
            },
            "required": ["transaction_id", "account_number", "transaction_type", "amount"]
        }

        result = parser.parse(schema)

        assert result.is_success
        schema_obj = result.unwrap()

        fields_by_path = {f.full_path: f for f in schema_obj.fields}

        # Check enum type
        assert fields_by_path["transaction_type"].data_type == DataType.ENUM

        # Check nullable type array
        assert fields_by_path["description"].is_nullable
        assert fields_by_path["description"].data_type == DataType.STRING


class TestSchemaParserRegistry:
    """Test schema parser registry with multiple parsers."""

    def test_registry_detects_json_schema(self):
        """Test registry can detect JSON Schema."""
        registry = SchemaParserRegistry()
        registry.register(AvroSchemaParser())
        registry.register(JsonSchemaParser())

        json_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }

        detected = registry.detect_parser(json_schema)

        assert detected is not None
        assert detected.format_name == "json_schema"

    def test_registry_detects_avro_schema(self):
        """Test registry correctly detects Avro over JSON Schema."""
        registry = SchemaParserRegistry()
        registry.register(AvroSchemaParser())
        registry.register(JsonSchemaParser())

        avro_schema = {
            "type": "record",
            "name": "Test",
            "fields": [{"name": "id", "type": "string"}]
        }

        detected = registry.detect_parser(avro_schema)

        assert detected is not None
        assert detected.format_name == "avro"

    def test_registry_get_parser_for_file(self):
        """Test getting parser by file extension."""
        registry = SchemaParserRegistry()
        registry.register(AvroSchemaParser())
        registry.register(JsonSchemaParser())

        json_parser = registry.get_parser_for_file(Path("schema.json"))
        avro_parser = registry.get_parser_for_file(Path("schema.avsc"))

        assert json_parser.format_name == "json_schema"
        assert avro_parser.format_name == "avro"

    def test_registry_lists_formats(self):
        """Test listing registered formats."""
        registry = SchemaParserRegistry()
        registry.register(AvroSchemaParser())
        registry.register(JsonSchemaParser())

        formats = registry.list_formats()

        assert "avro" in formats
        assert "json_schema" in formats


class TestJsonSchemaWithNexusMatcher:
    """Test JSON Schema parser integration with matching pipeline."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return JsonSchemaParser()

    def test_parsed_fields_searchable(self, parser):
        """Test parsed fields can be converted to searchable text."""
        schema = {
            "type": "object",
            "properties": {
                "cust_acct_bal": {
                    "type": "number",
                    "description": "Customer account balance"
                }
            }
        }

        result = parser.parse(schema)
        assert result.is_success

        field = result.unwrap().fields[0]
        searchable = field.to_searchable_text()

        # Searchable text normalizes underscores to spaces
        assert "cust" in searchable
        assert "acct" in searchable
        assert "bal" in searchable
        assert "Customer account balance" in searchable

    def test_deep_nesting(self, parser):
        """Test deeply nested objects are flattened correctly."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {
                                        "value": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        result = parser.parse(schema)
        assert result.is_success

        fields_by_path = {f.full_path: f for f in result.unwrap().fields}

        assert "level1" in fields_by_path
        assert "level1.level2" in fields_by_path
        assert "level1.level2.level3" in fields_by_path
        assert "level1.level2.level3.value" in fields_by_path

        # Check parent paths
        value_field = fields_by_path["level1.level2.level3.value"]
        assert value_field.parent_path == "level1.level2.level3"
