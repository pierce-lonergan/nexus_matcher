"""
nexus_matcher.infrastructure.adapters.schema_parsers.avro | Layer: INFRASTRUCTURE
Avro schema parser implementation.

## Relationships
# IMPLEMENTS → domain/ports/schema_parser :: SchemaParser protocol
# DEPENDS_ON → json :: JSON parsing (stdlib)
# USED_BY    → application/use_cases/match_schema :: schema parsing

## Attributes
# Security: Validates schema structure to prevent malformed input
# Performance: O(n) parsing where n = number of resolved fields
# Reliability: Handles nested records, unions, arrays, maps, named type
#              references and recursive schemas (cycle-guarded)
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from nexus_matcher.domain.models.entities import Schema, SchemaField
from nexus_matcher.domain.ports.schema_parser import BaseSchemaParser
from nexus_matcher.shared.types.base import DataType


class AvroSchemaParser(BaseSchemaParser):
    """
    Parser for Apache Avro schemas (.avsc files).

    Supports:
    - Primitive types (string, int, long, float, double, boolean, bytes)
    - Complex types (record, array, map, enum, fixed)
    - Union types (nullable fields)
    - Nested records (flattened to dot-notation paths)
    - Records inside arrays ("field[].child") and maps ("field{}.child")
    - Named type references: a record/enum/fixed defined once and referred to
      later by name, including recursive references
    - Logical types (date, timestamp, decimal, uuid)

    Two gaps this parser used to have, both of which silently shrank the output
    rather than failing:

    1. `array` items were reduced to a scalar DataType, so `history:
       array<record{visit_date, provider_id}>` produced one ARRAY field and none
       of the fields a user actually wants to map. Arrays of records are the
       normal way Avro models repeated groups.
    2. Named type references resolved to DataType.UNKNOWN. Avro requires a named
       type to be defined once; the second and later uses are a bare string. So
       in a schema with `home_address` and `work_address` both of type
       `Address`, the first expanded and the second vanished.

    Example:
        parser = AvroSchemaParser()
        result = parser.parse_file(Path("customer.avsc"))
        if result.is_success:
            schema = result.unwrap()
            for field in schema.fields:
                print(f"{field.full_path}: {field.data_type}")
    """

    # Avro type to DataType mapping
    TYPE_MAP: ClassVar[dict[str, DataType]] = {
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
    LOGICAL_TYPE_MAP: ClassVar[dict[str, DataType]] = {
        "date": DataType.DATE,
        "time-millis": DataType.TIMESTAMP,
        "time-micros": DataType.TIMESTAMP,
        "timestamp-millis": DataType.TIMESTAMP,
        "timestamp-micros": DataType.TIMESTAMP,
        "local-timestamp-millis": DataType.TIMESTAMP,
        "local-timestamp-micros": DataType.TIMESTAMP,
        "decimal": DataType.DECIMAL,
        "uuid": DataType.UUID,
    }

    _NAMED_TYPE_KINDS = ("record", "error", "enum", "fixed")

    def __init__(self) -> None:
        # Named types registered during a parse, keyed by both the fullname
        # ("com.example.Address") and the short name ("Address"), which is how
        # Avro allows them to be referenced from the same namespace.
        self._named_types: dict[str, dict[str, Any]] = {}

    @property
    def format_name(self) -> str:
        """Format identifier."""
        return "avro"

    @property
    def file_extensions(self) -> frozenset[str]:
        """Supported file extensions."""
        return frozenset({".avsc", ".avro"})

    # =========================================================================
    # NAMED TYPE REGISTRY
    # =========================================================================

    def _register_named_type(
        self,
        type_def: dict[str, Any],
        enclosing_namespace: str,
    ) -> str:
        """
        Register a named type so later references to it resolve.

        Returns:
            The type's fullname, or "" if it is not a named type.
        """
        kind = type_def.get("type")
        name = type_def.get("name")
        if kind not in self._NAMED_TYPE_KINDS or not isinstance(name, str):
            return ""

        namespace = type_def.get("namespace", enclosing_namespace) or ""
        fullname = f"{namespace}.{name}" if namespace and "." not in name else name

        self._named_types.setdefault(fullname, type_def)
        self._named_types.setdefault(name, type_def)
        return fullname

    def _lookup_named_type(self, name: str) -> dict[str, Any] | None:
        """Resolve a bare type name against the registry."""
        if name in self._named_types:
            return self._named_types[name]
        # A reference may be a fullname whose short form was registered.
        short = name.rsplit(".", 1)[-1]
        return self._named_types.get(short)

    # =========================================================================
    # PARSING
    # =========================================================================

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
        if not isinstance(content, dict):
            raise ValueError(f"Avro schema must be an object, got {type(content)}")

        if "type" not in content:
            raise ValueError("Avro schema must have 'type' field")

        if content["type"] != "record":
            raise ValueError(f"Top-level type must be 'record', got '{content['type']}'")

        if "name" not in content:
            raise ValueError("Avro schema must have 'name' field")

        if "fields" not in content:
            raise ValueError("Avro record must have 'fields' array")

        name = content["name"]
        namespace = content.get("namespace", "")
        doc = content.get("doc", "")

        # Fresh registry per parse so schemas cannot leak into one another.
        self._named_types = {}
        self._register_named_type(content, namespace)

        fields = self._parse_fields(
            content["fields"],
            parent_path="",
            namespace=namespace,
            expanding=(name,),
        )

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
        namespace: str,
        expanding: tuple[str, ...],
    ) -> list[SchemaField]:
        """
        Recursively parse Avro fields.

        Args:
            fields: List of Avro field definitions
            parent_path: Dot-separated path to parent
            namespace: Enclosing namespace for named-type resolution
            expanding: Named types currently being expanded (cycle guard)

        Returns:
            List of SchemaField instances
        """
        result: list[SchemaField] = []

        if not isinstance(fields, list):
            return result

        for field_def in fields:
            if not isinstance(field_def, dict):
                continue

            field_name = field_def.get("name", "")
            if not field_name:
                continue

            full_path = f"{parent_path}.{field_name}" if parent_path else field_name
            field_type = field_def.get("type")
            doc = field_def.get("doc", "")
            default = field_def.get("default")

            parsed = self._parse_type(field_type, full_path, namespace, expanding)

            metadata: dict[str, Any] = {
                "avro_type": field_type,
                "order": field_def.get("order"),
                "aliases": field_def.get("aliases", []),
            }
            if parsed.get("symbols"):
                metadata["symbols"] = parsed["symbols"]
            if parsed.get("named_type"):
                metadata["named_type"] = parsed["named_type"]
            if parsed.get("recursive"):
                metadata["recursive"] = True
            if parsed.get("logical_type"):
                metadata["logicalType"] = parsed["logical_type"]

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
                source_metadata=metadata,
            )

            result.append(schema_field)

            if parsed.get("nested_fields"):
                result.extend(parsed["nested_fields"])

        return result

    @staticmethod
    def _blank_result() -> dict[str, Any]:
        return {
            "data_type": DataType.UNKNOWN,
            "is_nullable": False,
            "is_array": False,
            "array_item_type": None,
            "nested_fields": None,
            "symbols": None,
            "named_type": None,
            "recursive": False,
            "logical_type": None,
        }

    def _parse_type(
        self,
        avro_type: Any,
        current_path: str,
        namespace: str,
        expanding: tuple[str, ...],
    ) -> dict[str, Any]:
        """
        Parse an Avro type definition.

        Args:
            avro_type: Avro type (string, dict, or list)
            current_path: Current field path
            namespace: Enclosing namespace
            expanding: Named types currently being expanded (cycle guard)

        Returns:
            Dictionary with parsed type info
        """
        result = self._blank_result()

        # Case 1: string -- either a primitive or a reference to a named type.
        if isinstance(avro_type, str):
            if avro_type in self.TYPE_MAP:
                result["data_type"] = self.TYPE_MAP[avro_type]
                return result

            referenced = self._lookup_named_type(avro_type)
            if referenced is None:
                # An unknown bare type name is a malformed schema: Avro requires
                # named types to be defined before use. Returning UNKNOWN here
                # is what made whole subtrees disappear silently.
                raise ValueError(
                    f"Unknown Avro type {avro_type!r} at {current_path or '<root>'}. "
                    f"It is neither a primitive nor a named type defined earlier "
                    f"in this schema. Known named types: "
                    f"{sorted(set(self._named_types)) or 'none'}"
                )

            result["named_type"] = avro_type

            short = avro_type.rsplit(".", 1)[-1]
            if short in expanding or avro_type in expanding:
                # Recursive reference (e.g. a LinkedList node). Emit the field,
                # stop expanding.
                result["recursive"] = True
                result["data_type"] = DataType.RECORD
                return result

            expanded = self._parse_complex_type(
                referenced, current_path, namespace, (*expanding, short)
            )
            expanded["named_type"] = avro_type
            return expanded

        # Case 2: Union type (nullable or multi-type)
        if isinstance(avro_type, list):
            return self._parse_union(avro_type, current_path, namespace, expanding)

        # Case 3: Complex type (dict)
        if isinstance(avro_type, dict):
            return self._parse_complex_type(avro_type, current_path, namespace, expanding)

        raise ValueError(
            f"Unsupported Avro type node at {current_path or '<root>'}: {type(avro_type).__name__}"
        )

    def _parse_union(
        self,
        union_types: list[Any],
        current_path: str,
        namespace: str,
        expanding: tuple[str, ...],
    ) -> dict[str, Any]:
        """
        Parse Avro union type.

        Common patterns:
        - ["null", "string"] → nullable string
        - ["null", {"type": "array", ...}] → nullable array, items expanded
        - ["null", "Address"] → nullable named-type reference

        The first non-null branch determines the field's type, and its nested
        fields are carried through. That matters because nullable records and
        nullable arrays-of-records are the overwhelmingly common shape in
        real Avro, and dropping their children empties the schema.
        """
        result = self._blank_result()

        non_null_types = [t for t in union_types if t != "null"]
        result["is_nullable"] = len(non_null_types) < len(union_types)

        if not non_null_types:
            return result

        parsed = self._parse_type(non_null_types[0], current_path, namespace, expanding)

        parsed["is_nullable"] = result["is_nullable"] or parsed["is_nullable"]
        return parsed

    def _parse_complex_type(
        self,
        type_def: dict[str, Any],
        current_path: str,
        namespace: str,
        expanding: tuple[str, ...],
    ) -> dict[str, Any]:
        """
        Parse complex Avro type (record, array, map, enum, fixed).

        Args:
            type_def: Type definition dictionary
            current_path: Current field path
            namespace: Enclosing namespace
            expanding: Named types currently being expanded (cycle guard)

        Returns:
            Parsed type info
        """
        result = self._blank_result()

        type_name = type_def.get("type", "")
        inner_namespace = type_def.get("namespace", namespace) or namespace

        # Register before recursing so a record can refer to itself.
        self._register_named_type(type_def, namespace)

        # Logical types annotate a primitive/fixed carrier type.
        logical_type = type_def.get("logicalType")
        if logical_type:
            result["logical_type"] = logical_type
            mapped = self.LOGICAL_TYPE_MAP.get(logical_type)
            if mapped is not None:
                result["data_type"] = mapped
                return result
            # Unknown logical type: fall through to the carrier type rather than
            # reporting UNKNOWN, so e.g. an unrecognised logicalType on a string
            # still parses as a string.

        # Record type (nested fields)
        if type_name in ("record", "error"):
            result["data_type"] = DataType.RECORD
            record_name = type_def.get("name")
            child_expanding = (
                (*expanding, record_name) if isinstance(record_name, str) else expanding
            )
            nested_fields = type_def.get("fields", [])
            if nested_fields:
                result["nested_fields"] = self._parse_fields(
                    nested_fields, current_path, inner_namespace, child_expanding
                )
            return result

        # Array type -- expand the item schema's own fields.
        if type_name == "array":
            result["data_type"] = DataType.ARRAY
            result["is_array"] = True
            items = type_def.get("items")
            if items is not None:
                item_path = f"{current_path}[]"
                item_parsed = self._parse_type(items, item_path, inner_namespace, expanding)
                result["array_item_type"] = item_parsed["data_type"]
                result["nested_fields"] = item_parsed.get("nested_fields")
                if item_parsed.get("recursive"):
                    result["recursive"] = True
            return result

        # Map type -- values may themselves be records.
        if type_name == "map":
            result["data_type"] = DataType.RECORD
            values = type_def.get("values")
            if values is not None:
                value_path = f"{current_path}{{}}"
                value_parsed = self._parse_type(values, value_path, inner_namespace, expanding)
                result["nested_fields"] = value_parsed.get("nested_fields")
                if value_parsed.get("recursive"):
                    result["recursive"] = True
            return result

        # Enum type
        if type_name == "enum":
            result["data_type"] = DataType.ENUM
            result["symbols"] = type_def.get("symbols", [])
            return result

        # Fixed type (bytes)
        if type_name == "fixed":
            result["data_type"] = DataType.BYTES
            return result

        # A nested union expressed as {"type": [...]}
        if isinstance(type_name, list):
            return self._parse_union(type_name, current_path, namespace, expanding)

        # A carrier primitive, possibly under an unknown logicalType.
        if isinstance(type_name, str) and type_name in self.TYPE_MAP:
            result["data_type"] = self.TYPE_MAP[type_name]
            return result

        # {"type": {...}} nesting
        if isinstance(type_name, dict):
            return self._parse_type(type_name, current_path, namespace, expanding)

        if isinstance(type_name, str) and type_name:
            return self._parse_type(type_name, current_path, namespace, expanding)

        raise ValueError(
            f"Avro type definition at {current_path or '<root>'} has no usable "
            f"'type' key: {type_def!r}"
        )

    def can_parse(self, content: str | dict[str, Any]) -> bool:
        """
        Check if content is valid Avro schema.

        Args:
            content: Content to check

        Returns:
            True if this parser can handle the content
        """
        try:
            parsed = json.loads(content) if isinstance(content, str) else content

            if not isinstance(parsed, dict):
                return False

            if parsed.get("type") != "record":
                return False

            return "fields" in parsed

        except (json.JSONDecodeError, TypeError):
            return False
