"""
nexus_matcher.infrastructure.adapters.schema_parsers.avro | Layer: INFRASTRUCTURE
Avro schema parser implementation.

## Relationships
# IMPLEMENTS → domain/ports/schema_parser :: SchemaParser protocol
# DEPENDS_ON → fastavro :: Avro parsing library
# USED_BY    → application/use_cases/match_schema :: schema parsing

## Attributes
# Security: Validates schema structure to prevent malformed input
# Performance: O(n) parsing where n = number of fields
# Reliability: Handles nested records, unions, arrays recursively
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nexus_matcher.domain.models.entities import Schema, SchemaField
from nexus_matcher.domain.ports.schema_parser import BaseSchemaParser
from nexus_matcher.shared.types.base import DataType, Result


class AvroSchemaParser(BaseSchemaParser):
    """
    Parser for Apache Avro schemas (.avsc files).

    Supports:
    - Primitive types (string, int, long, float, double, boolean, bytes)
    - Complex types (record, array, map, enum, fixed)
    - Union types (nullable fields)
    - Nested records (flattened to dot-notation paths)
    - Logical types (date, timestamp, decimal, uuid)

    Example:
        parser = AvroSchemaParser()
        result = parser.parse_file(Path("customer.avsc"))
        if result.is_success:
            schema = result.unwrap()
            for field in schema.fields:
                print(f"{field.full_path}: {field.data_type}")
    """

    # Avro type to DataType mapping
    TYPE_MAP: dict[str, DataType] = {
        "string": DataType.STRING,
        "int": DataType.INTEGER,
        "long": DataType.LONG,
        "float": DataType.FLOAT,
        "double": DataType.DOUBLE,
        "boolean": DataType.BOOLEAN,
        "bytes": DataType.BYTES,
        "null": DataType.UNKNOWN,
    }

    # Logical type mapping
    LOGICAL_TYPE_MAP: dict[str, DataType] = {
        "date": DataType.DATE,
        "time-millis": DataType.TIMESTAMP,
        "time-micros": DataType.TIMESTAMP,
        "timestamp-millis": DataType.TIMESTAMP,
        "timestamp-micros": DataType.TIMESTAMP,
        "decimal": DataType.DECIMAL,
        "uuid": DataType.UUID,
    }

    @property
    def format_name(self) -> str:
        """Format identifier."""
        return "avro"

    @property
    def file_extensions(self) -> frozenset[str]:
        """Supported file extensions."""
        return frozenset({".avsc", ".avro"})

    def _parse_content(self, content: dict[str, Any]) -> Schema:
        """
        Parse Avro schema from dictionary.

        Args:
            content: Parsed Avro schema dictionary

        Returns:
            Schema domain model

        Raises:
            ValueError: If schema is invalid
        """
        # Validate required fields
        if "type" not in content:
            raise ValueError("Avro schema must have 'type' field")

        if content["type"] != "record":
            raise ValueError(f"Top-level type must be 'record', got '{content['type']}'")

        if "name" not in content:
            raise ValueError("Avro schema must have 'name' field")

        if "fields" not in content:
            raise ValueError("Avro record must have 'fields' array")

        # Extract schema metadata
        name = content["name"]
        namespace = content.get("namespace", "")
        doc = content.get("doc", "")

        # Parse all fields recursively
        fields = self._parse_fields(content["fields"], parent_path="")

        return Schema(
            name=name,
            fields=tuple(fields),
            namespace=namespace,
            source_format="avro",
            source_metadata={
                "doc": doc,
                "aliases": content.get("aliases", []),
            },
        )

    def _parse_fields(
        self,
        fields: list[dict[str, Any]],
        parent_path: str,
    ) -> list[SchemaField]:
        """
        Recursively parse Avro fields.

        Args:
            fields: List of Avro field definitions
            parent_path: Dot-separated path to parent

        Returns:
            List of SchemaField instances
        """
        result: list[SchemaField] = []

        for field_def in fields:
            field_name = field_def.get("name", "")
            if not field_name:
                continue

            full_path = f"{parent_path}.{field_name}" if parent_path else field_name
            field_type = field_def.get("type")
            doc = field_def.get("doc", "")
            default = field_def.get("default")

            # Parse the type (may be nested)
            parsed = self._parse_type(field_type, full_path)

            # Create the schema field
            schema_field = SchemaField(
                name=field_name,
                data_type=parsed["data_type"],
                full_path=full_path,
                parent_path=parent_path,
                description=doc,
                is_nullable=parsed["is_nullable"],
                is_array=parsed["is_array"],
                array_item_type=parsed.get("array_item_type"),
                default_value=default,
                source_metadata={
                    "avro_type": field_type,
                    "order": field_def.get("order"),
                    "aliases": field_def.get("aliases", []),
                },
            )

            result.append(schema_field)

            # If this is a record type, recursively parse nested fields
            if parsed.get("nested_fields"):
                result.extend(parsed["nested_fields"])

        return result

    def _parse_type(
        self,
        avro_type: Any,
        current_path: str,
    ) -> dict[str, Any]:
        """
        Parse an Avro type definition.

        Args:
            avro_type: Avro type (string, dict, or list)
            current_path: Current field path

        Returns:
            Dictionary with parsed type info
        """
        result: dict[str, Any] = {
            "data_type": DataType.UNKNOWN,
            "is_nullable": False,
            "is_array": False,
            "array_item_type": None,
            "nested_fields": None,
        }

        # Case 1: Simple string type
        if isinstance(avro_type, str):
            result["data_type"] = self.TYPE_MAP.get(avro_type, DataType.UNKNOWN)
            return result

        # Case 2: Union type (nullable or multi-type)
        if isinstance(avro_type, list):
            return self._parse_union(avro_type, current_path)

        # Case 3: Complex type (dict)
        if isinstance(avro_type, dict):
            return self._parse_complex_type(avro_type, current_path)

        return result

    def _parse_union(
        self,
        union_types: list[Any],
        current_path: str,
    ) -> dict[str, Any]:
        """
        Parse Avro union type.

        Common patterns:
        - ["null", "string"] → nullable string
        - ["null", "int", "string"] → nullable, first non-null wins

        Args:
            union_types: List of types in union
            current_path: Current field path

        Returns:
            Parsed type info
        """
        result: dict[str, Any] = {
            "data_type": DataType.UNKNOWN,
            "is_nullable": False,
            "is_array": False,
            "array_item_type": None,
            "nested_fields": None,
        }

        # Check for null in union (nullable field)
        non_null_types = [t for t in union_types if t != "null"]
        result["is_nullable"] = len(non_null_types) < len(union_types)

        if not non_null_types:
            return result

        # Use first non-null type
        first_type = non_null_types[0]
        parsed = self._parse_type(first_type, current_path)

        result["data_type"] = parsed["data_type"]
        result["is_array"] = parsed["is_array"]
        result["array_item_type"] = parsed.get("array_item_type")
        result["nested_fields"] = parsed.get("nested_fields")

        return result

    def _parse_complex_type(
        self,
        type_def: dict[str, Any],
        current_path: str,
    ) -> dict[str, Any]:
        """
        Parse complex Avro type (record, array, map, enum, fixed).

        Args:
            type_def: Type definition dictionary
            current_path: Current field path

        Returns:
            Parsed type info
        """
        result: dict[str, Any] = {
            "data_type": DataType.UNKNOWN,
            "is_nullable": False,
            "is_array": False,
            "array_item_type": None,
            "nested_fields": None,
        }

        type_name = type_def.get("type", "")

        # Check for logical type first
        logical_type = type_def.get("logicalType")
        if logical_type:
            result["data_type"] = self.LOGICAL_TYPE_MAP.get(logical_type, DataType.UNKNOWN)
            return result

        # Record type (nested fields)
        if type_name == "record":
            result["data_type"] = DataType.RECORD
            nested_fields = type_def.get("fields", [])
            if nested_fields:
                result["nested_fields"] = self._parse_fields(nested_fields, current_path)
            return result

        # Array type
        if type_name == "array":
            result["data_type"] = DataType.ARRAY
            result["is_array"] = True
            items = type_def.get("items")
            if items:
                item_parsed = self._parse_type(items, current_path)
                result["array_item_type"] = item_parsed["data_type"]
            return result

        # Map type (treated as record)
        if type_name == "map":
            result["data_type"] = DataType.RECORD
            return result

        # Enum type
        if type_name == "enum":
            result["data_type"] = DataType.ENUM
            return result

        # Fixed type (bytes)
        if type_name == "fixed":
            result["data_type"] = DataType.BYTES
            return result

        # Fallback to type map
        result["data_type"] = self.TYPE_MAP.get(type_name, DataType.UNKNOWN)
        return result

    def can_parse(self, content: str | dict[str, Any]) -> bool:
        """
        Check if content is valid Avro schema.

        Args:
            content: Content to check

        Returns:
            True if this parser can handle the content
        """
        try:
            if isinstance(content, str):
                parsed = json.loads(content)
            else:
                parsed = content

            # Check for Avro-specific markers
            if not isinstance(parsed, dict):
                return False

            # Must have type = record and fields
            if parsed.get("type") != "record":
                return False

            if "fields" not in parsed:
                return False

            return True

        except (json.JSONDecodeError, TypeError):
            return False
