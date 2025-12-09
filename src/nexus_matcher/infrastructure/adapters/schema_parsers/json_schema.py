"""
nexus_matcher.infrastructure.adapters.schema_parsers.json_schema | Layer: INFRASTRUCTURE
JSON Schema parser implementation.

## Relationships
# IMPLEMENTS → domain/ports/schema_parser :: SchemaParser protocol
# DEPENDS_ON → json :: JSON parsing (stdlib)
# USED_BY    → application/use_cases/match_schema :: schema parsing

## Attributes
# Security: Validates schema structure to prevent malformed input
# Performance: O(n) parsing where n = number of fields
# Reliability: Handles nested objects, arrays, formats recursively
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nexus_matcher.domain.models.entities import Schema, SchemaField
from nexus_matcher.domain.ports.schema_parser import BaseSchemaParser
from nexus_matcher.shared.types.base import DataType, Result


class JsonSchemaParser(BaseSchemaParser):
    """
    Parser for JSON Schema (draft-04/06/07).

    Supports:
    - Primitive types (string, integer, number, boolean, null)
    - Complex types (object, array)
    - Format annotations (date, date-time, uuid, byte, binary)
    - Nullable types via ["type", "null"] arrays
    - Nested objects (flattened to dot-notation paths)
    - Enum types

    Example:
        parser = JsonSchemaParser()
        result = parser.parse_file(Path("schema.json"))
        if result.is_success:
            schema = result.unwrap()
            for field in schema.fields:
                print(f"{field.full_path}: {field.data_type}")
    """

    # JSON Schema type to DataType mapping
    TYPE_MAP: dict[str, DataType] = {
        "string": DataType.STRING,
        "integer": DataType.INTEGER,
        "number": DataType.DOUBLE,
        "boolean": DataType.BOOLEAN,
        "array": DataType.ARRAY,
        "object": DataType.RECORD,
        "null": DataType.UNKNOWN,
    }

    # Format to DataType mapping
    FORMAT_MAP: dict[str, DataType] = {
        "date": DataType.DATE,
        "date-time": DataType.TIMESTAMP,
        "time": DataType.TIMESTAMP,
        "uuid": DataType.UUID,
        "byte": DataType.BYTES,
        "binary": DataType.BYTES,
        # These remain as STRING
        "email": DataType.STRING,
        "uri": DataType.STRING,
        "uri-reference": DataType.STRING,
        "hostname": DataType.STRING,
        "ipv4": DataType.STRING,
        "ipv6": DataType.STRING,
    }

    @property
    def format_name(self) -> str:
        """Format identifier."""
        return "json_schema"

    @property
    def file_extensions(self) -> frozenset[str]:
        """Supported file extensions."""
        return frozenset({".json", ".schema.json"})

    def _parse_content(self, content: dict[str, Any]) -> Schema:
        """
        Parse JSON Schema from dictionary.

        Args:
            content: Parsed JSON Schema dictionary

        Returns:
            Schema domain model

        Raises:
            ValueError: If schema is invalid
        """
        # Extract schema metadata
        title = content.get("title", "Untitled")
        description = content.get("description", "")

        # Get the type - must be object for root
        schema_type = content.get("type", "object")

        # Handle type array (like ["object", "null"])
        if isinstance(schema_type, list):
            non_null_types = [t for t in schema_type if t != "null"]
            schema_type = non_null_types[0] if non_null_types else "object"

        # Get properties
        properties = content.get("properties", {})
        required = set(content.get("required", []))

        # Parse all properties recursively
        fields = self._parse_properties(properties, required, parent_path="")

        return Schema(
            name=title,
            fields=tuple(fields),
            namespace="",
            source_format="json_schema",
            source_metadata={
                "description": description,
                "$schema": content.get("$schema", ""),
                "$id": content.get("$id", ""),
            },
        )

    def _parse_properties(
        self,
        properties: dict[str, Any],
        required: set[str],
        parent_path: str,
    ) -> list[SchemaField]:
        """
        Recursively parse JSON Schema properties.

        Args:
            properties: Property definitions
            required: Set of required property names
            parent_path: Dot-separated path to parent

        Returns:
            List of SchemaField instances
        """
        result: list[SchemaField] = []

        for prop_name, prop_def in properties.items():
            full_path = f"{parent_path}.{prop_name}" if parent_path else prop_name

            # Parse the property type
            parsed = self._parse_property(prop_def, full_path)

            # Determine nullability
            is_nullable = parsed["is_nullable"]
            if prop_name not in required:
                is_nullable = True

            # Create the schema field
            schema_field = SchemaField(
                name=prop_name,
                data_type=parsed["data_type"],
                full_path=full_path,
                parent_path=parent_path,
                description=prop_def.get("description", ""),
                is_nullable=is_nullable,
                is_array=parsed["is_array"],
                array_item_type=parsed.get("array_item_type"),
                default_value=prop_def.get("default"),
                source_metadata={
                    "json_schema_type": prop_def.get("type"),
                    "format": prop_def.get("format"),
                    "enum": prop_def.get("enum"),
                    "pattern": prop_def.get("pattern"),
                    "minLength": prop_def.get("minLength"),
                    "maxLength": prop_def.get("maxLength"),
                    "minimum": prop_def.get("minimum"),
                    "maximum": prop_def.get("maximum"),
                },
            )

            result.append(schema_field)

            # If this is an object type, recursively parse nested properties
            if parsed.get("nested_fields"):
                result.extend(parsed["nested_fields"])

        return result

    def _parse_property(
        self,
        prop_def: dict[str, Any],
        current_path: str,
    ) -> dict[str, Any]:
        """
        Parse a JSON Schema property definition.

        Args:
            prop_def: Property definition dictionary
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

        prop_type = prop_def.get("type")

        # Handle type array (e.g., ["string", "null"])
        if isinstance(prop_type, list):
            non_null_types = [t for t in prop_type if t != "null"]
            result["is_nullable"] = len(non_null_types) < len(prop_type)
            prop_type = non_null_types[0] if non_null_types else "null"

        # Check for enum first
        if "enum" in prop_def:
            result["data_type"] = DataType.ENUM
            return result

        # Check for format annotation (overrides base type)
        fmt = prop_def.get("format")
        if fmt and fmt in self.FORMAT_MAP:
            result["data_type"] = self.FORMAT_MAP[fmt]
            return result

        # Handle different types
        if prop_type == "array":
            result["data_type"] = DataType.ARRAY
            result["is_array"] = True
            items = prop_def.get("items", {})
            if items:
                item_parsed = self._parse_property(items, current_path)
                result["array_item_type"] = item_parsed["data_type"]
            return result

        if prop_type == "object":
            result["data_type"] = DataType.RECORD
            nested_props = prop_def.get("properties", {})
            nested_required = set(prop_def.get("required", []))
            if nested_props:
                result["nested_fields"] = self._parse_properties(
                    nested_props, nested_required, current_path
                )
            return result

        # Simple type mapping
        if prop_type:
            result["data_type"] = self.TYPE_MAP.get(prop_type, DataType.UNKNOWN)

        return result

    def can_parse(self, content: str | dict[str, Any]) -> bool:
        """
        Check if content is valid JSON Schema.

        JSON Schema is identified by:
        - Has "type" = "object" (or array of types including object)
        - Has "properties" key (for object schemas)
        - Does NOT have "fields" key (that's Avro)
        - Does NOT have "type" = "record" (that's Avro)

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

            if not isinstance(parsed, dict):
                return False

            # Reject Avro schemas
            if parsed.get("type") == "record" and "fields" in parsed:
                return False

            # Accept if it has properties OR is explicitly type: object
            has_properties = "properties" in parsed
            is_object_type = parsed.get("type") == "object"

            # Also accept type arrays containing "object"
            type_val = parsed.get("type")
            if isinstance(type_val, list) and "object" in type_val:
                is_object_type = True

            return has_properties or is_object_type

        except (json.JSONDecodeError, TypeError):
            return False
