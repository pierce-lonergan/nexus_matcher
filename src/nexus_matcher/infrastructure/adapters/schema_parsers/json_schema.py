"""
nexus_matcher.infrastructure.adapters.schema_parsers.json_schema | Layer: INFRASTRUCTURE
JSON Schema parser implementation.

## Relationships
# IMPLEMENTS → domain/ports/schema_parser :: SchemaParser protocol
# DEPENDS_ON → json :: JSON parsing (stdlib)
# USED_BY    → application/use_cases/match_schema :: schema parsing

## Attributes
# Security: Local ($ref) references only -- external/remote refs are rejected,
#           never fetched, so a schema cannot make the parser issue requests.
# Performance: O(n) parsing where n = number of resolved fields
# Reliability: Handles nested objects, arrays, $ref/$defs, allOf/anyOf/oneOf,
#              and recursive schemas (cycle-guarded)
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from nexus_matcher.domain.models.entities import Schema, SchemaField
from nexus_matcher.domain.ports.schema_parser import BaseSchemaParser
from nexus_matcher.shared.types.base import DataType


class JsonSchemaParser(BaseSchemaParser):
    """
    Parser for JSON Schema (draft-04 through 2020-12).

    Supports:
    - Primitive types (string, integer, number, boolean, null)
    - Complex types (object, array)
    - Format annotations (date, date-time, uuid, byte, binary)
    - Nullable types via ["type", "null"] arrays
    - Nested objects (flattened to dot-notation paths)
    - Arrays of objects (flattened to "field[].child" paths)
    - Enum types
    - $ref against local $defs / definitions / any in-document JSON pointer
    - allOf (merged), anyOf / oneOf (branches merged, nullable if a null branch
      is present)
    - Recursive schemas, via cycle detection

    Real-world JSON Schemas are mostly $ref plus allOf. A parser that ignores
    those keywords silently emits a field list missing most of the schema, which
    is worse than failing outright. Every field reachable through them is now
    materialised.

    Example:
        parser = JsonSchemaParser()
        result = parser.parse_file(Path("schema.json"))
        if result.is_success:
            schema = result.unwrap()
            for field in schema.fields:
                print(f"{field.full_path}: {field.data_type}")
    """

    # JSON Schema type to DataType mapping
    TYPE_MAP: ClassVar[dict[str, DataType]] = {
        "string": DataType.STRING,
        "integer": DataType.INTEGER,
        "number": DataType.DOUBLE,
        "boolean": DataType.BOOLEAN,
        "array": DataType.ARRAY,
        "object": DataType.RECORD,
        "null": DataType.UNKNOWN,
    }

    # Format to DataType mapping
    FORMAT_MAP: ClassVar[dict[str, DataType]] = {
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

    # Keywords that combine subschemas
    _COMPOSITION_KEYWORDS = ("allOf", "anyOf", "oneOf")

    def __init__(self) -> None:
        # Root document, set per parse, used to resolve in-document $ref.
        self._root: dict[str, Any] = {}

    @property
    def format_name(self) -> str:
        """Format identifier."""
        return "json_schema"

    @property
    def file_extensions(self) -> frozenset[str]:
        """Supported file extensions."""
        return frozenset({".json", ".schema.json"})

    # =========================================================================
    # $ref RESOLUTION
    # =========================================================================

    def _resolve_pointer(self, ref: str) -> dict[str, Any]:
        """
        Resolve an in-document JSON pointer such as "#/$defs/Address".

        Args:
            ref: The $ref string

        Returns:
            The referenced subschema

        Raises:
            ValueError: For external refs, or refs that do not resolve. Both are
                raised rather than skipped: a $ref the parser cannot follow means
                the emitted field list is incomplete, and silently returning a
                short list is exactly the failure mode this parser had before.
        """
        if not ref.startswith("#"):
            raise ValueError(
                f"External $ref is not supported: {ref!r}. Only in-document "
                "references (starting with '#') are resolved; the parser never "
                "fetches remote schemas. Inline the definition or pre-bundle the "
                "schema."
            )

        pointer = ref[1:].lstrip("/")
        if not pointer:
            return self._root

        node: Any = self._root
        for raw_token in pointer.split("/"):
            token = raw_token.replace("~1", "/").replace("~0", "~")
            if isinstance(node, list):
                try:
                    node = node[int(token)]
                    continue
                except (ValueError, IndexError) as e:
                    raise ValueError(f"$ref {ref!r} does not resolve: {e}") from e
            if not isinstance(node, dict) or token not in node:
                raise ValueError(f"$ref {ref!r} does not resolve: no {token!r} in document")
            node = node[token]

        if not isinstance(node, dict):
            raise ValueError(f"$ref {ref!r} resolves to a non-object: {type(node)}")

        return node

    def _deref(
        self,
        schema: dict[str, Any],
        ref_stack: tuple[str, ...],
    ) -> tuple[dict[str, Any], tuple[str, ...], bool]:
        """
        Follow $ref (possibly chained) and merge sibling keywords.

        Returns:
            (resolved_schema, updated_ref_stack, cycle_detected)

            cycle_detected is True when the ref is already being expanded higher
            up the recursion. Recursive schemas (a Node whose children are Nodes)
            are legal and common; expansion stops at the cycle instead of
            recursing forever.
        """
        seen_here: list[str] = []
        current = schema

        while isinstance(current, dict) and "$ref" in current:
            ref = current["$ref"]
            if not isinstance(ref, str):
                raise ValueError(f"$ref must be a string, got {type(ref)}")

            if ref in ref_stack or ref in seen_here:
                return current, ref_stack, True

            seen_here.append(ref)
            target = self._resolve_pointer(ref)

            # Draft 2019-09+ allows keywords alongside $ref; siblings win.
            siblings = {k: v for k, v in current.items() if k != "$ref"}
            current = {**target, **siblings} if siblings else dict(target)

        return current, ref_stack + tuple(seen_here), False

    # =========================================================================
    # COMPOSITION (allOf / anyOf / oneOf)
    # =========================================================================

    def _flatten_composition(
        self,
        schema: dict[str, Any],
        ref_stack: tuple[str, ...],
    ) -> tuple[dict[str, Any], tuple[str, ...], bool]:
        """
        Collapse allOf / anyOf / oneOf into a single effective schema.

        allOf  -- intersection: merge every branch's properties, and union their
                  `required` lists.
        anyOf  -- a value may match any branch. Every branch's properties are
                  merged so no field disappears from the flattened output. A
                  {"type": "null"} branch makes the field nullable.
        oneOf  -- treated like anyOf. Exactly-one-of semantics cannot be encoded
                  in a flat field list; merging keeps every candidate field
                  visible to the matcher, which is what callers need. The branch
                  count is recorded in source_metadata so the ambiguity is not
                  lost. Branch-local `required` is ignored for anyOf/oneOf: a
                  field required by only one alternative is not required overall.
        """
        schema, ref_stack, cycle = self._deref(schema, ref_stack)
        if cycle:
            return schema, ref_stack, True

        if not any(k in schema for k in self._COMPOSITION_KEYWORDS):
            return schema, ref_stack, False

        merged: dict[str, Any] = {
            k: v for k, v in schema.items() if k not in self._COMPOSITION_KEYWORDS
        }
        merged_props: dict[str, Any] = dict(merged.get("properties", {}))
        merged_required: list[str] = list(merged.get("required", []))
        nullable_branch = False
        any_cycle = False

        for keyword in self._COMPOSITION_KEYWORDS:
            branches = schema.get(keyword)
            if not isinstance(branches, list):
                continue

            for branch in branches:
                if not isinstance(branch, dict):
                    continue

                if branch.get("type") == "null":
                    nullable_branch = True
                    continue

                resolved, ref_stack, branch_cycle = self._flatten_composition(branch, ref_stack)
                if branch_cycle:
                    any_cycle = True
                    continue

                for key, value in resolved.items():
                    if key == "properties":
                        for prop_name, prop_def in value.items():
                            merged_props.setdefault(prop_name, prop_def)
                    elif key == "required":
                        if keyword == "allOf":
                            merged_required.extend(value)
                    elif key not in merged:
                        merged[key] = value

            if keyword in ("anyOf", "oneOf") and branches:
                merged.setdefault("_composition", keyword)
                merged.setdefault("_composition_branches", len(branches))

        if merged_props:
            merged["properties"] = merged_props
            merged.setdefault("type", "object")
        if merged_required:
            merged["required"] = sorted(set(merged_required))
        if nullable_branch:
            merged["_nullable_branch"] = True

        return merged, ref_stack, any_cycle

    # =========================================================================
    # PARSING
    # =========================================================================

    def _parse_content(self, content: dict[str, Any]) -> Schema:
        """
        Parse JSON Schema from dictionary.

        Args:
            content: Parsed JSON Schema dictionary

        Returns:
            Schema domain model

        Raises:
            ValueError: If schema is invalid or contains an unresolvable $ref
        """
        if not isinstance(content, dict):
            raise ValueError(f"JSON Schema must be an object, got {type(content)}")

        self._root = content

        title = content.get("title", "Untitled")
        description = content.get("description", "")

        # The root may itself be a $ref or an allOf composition.
        effective, ref_stack, _ = self._flatten_composition(content, ())

        properties = effective.get("properties", {})
        required = set(effective.get("required", []))

        fields = self._parse_properties(properties, required, "", ref_stack)

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
        ref_stack: tuple[str, ...],
    ) -> list[SchemaField]:
        """
        Recursively parse JSON Schema properties.

        Args:
            properties: Property definitions
            required: Set of required property names
            parent_path: Dot-separated path to parent
            ref_stack: $refs currently being expanded (cycle guard)

        Returns:
            List of SchemaField instances
        """
        result: list[SchemaField] = []

        if not isinstance(properties, dict):
            return result

        for prop_name, prop_def in properties.items():
            if not isinstance(prop_def, dict):
                continue

            full_path = f"{parent_path}.{prop_name}" if parent_path else prop_name

            parsed = self._parse_property(prop_def, full_path, ref_stack)
            effective = parsed["effective"]

            is_nullable = parsed["is_nullable"]
            if prop_name not in required:
                is_nullable = True

            metadata: dict[str, Any] = {
                "json_schema_type": effective.get("type", prop_def.get("type")),
                "format": effective.get("format"),
                "enum": effective.get("enum"),
                "pattern": effective.get("pattern"),
                "minLength": effective.get("minLength"),
                "maxLength": effective.get("maxLength"),
                "minimum": effective.get("minimum"),
                "maximum": effective.get("maximum"),
            }
            if "$ref" in prop_def:
                metadata["$ref"] = prop_def["$ref"]
            if effective.get("_composition"):
                metadata["composition"] = effective["_composition"]
                metadata["composition_branches"] = effective.get("_composition_branches")
            if parsed.get("recursive"):
                metadata["recursive"] = True

            schema_field = SchemaField(
                name=prop_name,
                data_type=parsed["data_type"],
                full_path=full_path,
                parent_path=parent_path,
                description=effective.get("description", prop_def.get("description", "")),
                is_nullable=is_nullable,
                is_array=parsed["is_array"],
                array_item_type=parsed.get("array_item_type"),
                default_value=effective.get("default", prop_def.get("default")),
                source_metadata=metadata,
            )

            result.append(schema_field)

            if parsed.get("nested_fields"):
                result.extend(parsed["nested_fields"])

        return result

    def _parse_property(
        self,
        prop_def: dict[str, Any],
        current_path: str,
        ref_stack: tuple[str, ...],
    ) -> dict[str, Any]:
        """
        Parse a JSON Schema property definition.

        Args:
            prop_def: Property definition dictionary
            current_path: Current field path
            ref_stack: $refs currently being expanded (cycle guard)

        Returns:
            Dictionary with parsed type info, including the dereferenced
            "effective" schema so callers read annotations from the right place.
        """
        result: dict[str, Any] = {
            "data_type": DataType.UNKNOWN,
            "is_nullable": False,
            "is_array": False,
            "array_item_type": None,
            "nested_fields": None,
            "recursive": False,
            "effective": prop_def,
        }

        effective, ref_stack, cycle = self._flatten_composition(prop_def, ref_stack)
        result["effective"] = effective

        if cycle:
            # Recursive schema. Emit the field, do not expand it further.
            result["recursive"] = True
            result["data_type"] = DataType.RECORD
            return result

        if effective.get("_nullable_branch"):
            result["is_nullable"] = True

        prop_type = effective.get("type")

        # Handle type array (e.g., ["string", "null"])
        if isinstance(prop_type, list):
            non_null_types = [t for t in prop_type if t != "null"]
            if len(non_null_types) < len(prop_type):
                result["is_nullable"] = True
            prop_type = non_null_types[0] if non_null_types else "null"

        # Objects and arrays are often declared implicitly, by carrying
        # "properties"/"items" without a "type". Treating those as UNKNOWN
        # dropped every nested field underneath them.
        if prop_type is None and "properties" in effective:
            prop_type = "object"
        if prop_type is None and "items" in effective:
            prop_type = "array"

        # Enum wins over the base type
        if "enum" in effective:
            result["data_type"] = DataType.ENUM
            enum_values = effective["enum"]
            if isinstance(enum_values, list) and None in enum_values:
                result["is_nullable"] = True
            return result

        # Format annotation overrides the base type
        fmt = effective.get("format")
        if fmt and fmt in self.FORMAT_MAP:
            result["data_type"] = self.FORMAT_MAP[fmt]
            return result

        if prop_type == "array":
            return self._parse_array(effective, current_path, ref_stack, result)

        if prop_type == "object":
            result["data_type"] = DataType.RECORD
            nested_props = effective.get("properties", {})
            nested_required = set(effective.get("required", []))
            if nested_props:
                result["nested_fields"] = self._parse_properties(
                    nested_props, nested_required, current_path, ref_stack
                )
            return result

        if prop_type:
            result["data_type"] = self.TYPE_MAP.get(prop_type, DataType.UNKNOWN)

        return result

    def _parse_array(
        self,
        effective: dict[str, Any],
        current_path: str,
        ref_stack: tuple[str, ...],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Parse an array property, including the item schema's own fields.

        The old implementation read only the scalar data_type of `items` and
        discarded everything else, so an array of objects -- `orders: [{id,
        total}]`, the single most common shape in real payload schemas --
        contributed one ARRAY field and zero matchable ones. Item fields are now
        emitted under a "field[].child" path, which keeps them distinguishable
        from a plain nested object at "field.child".
        """
        result["data_type"] = DataType.ARRAY
        result["is_array"] = True

        items = effective.get("items")

        # Tuple validation: "items": [ {...}, {...} ]. Expand each positionally.
        if isinstance(items, list):
            nested: list[SchemaField] = []
            for position, item_def in enumerate(items):
                if not isinstance(item_def, dict):
                    continue
                item_path = f"{current_path}[{position}]"
                item_parsed = self._parse_property(item_def, item_path, ref_stack)
                if position == 0:
                    result["array_item_type"] = item_parsed["data_type"]
                nested.extend(self._array_item_fields(item_parsed, item_path, ref_stack))
            result["nested_fields"] = nested or None
            return result

        if not isinstance(items, dict):
            return result

        item_path = f"{current_path}[]"
        item_parsed = self._parse_property(items, item_path, ref_stack)
        result["array_item_type"] = item_parsed["data_type"]
        result["nested_fields"] = self._array_item_fields(item_parsed, item_path, ref_stack) or None
        return result

    def _array_item_fields(
        self,
        item_parsed: dict[str, Any],
        item_path: str,
        ref_stack: tuple[str, ...],
    ) -> list[SchemaField]:
        """Materialise the fields carried by an array's item schema."""
        nested = list(item_parsed.get("nested_fields") or [])
        if nested:
            return nested

        item_effective = item_parsed["effective"]
        item_props = item_effective.get("properties")
        if isinstance(item_props, dict) and item_props:
            return self._parse_properties(
                item_props,
                set(item_effective.get("required", [])),
                item_path,
                ref_stack,
            )

        return []

    def can_parse(self, content: str | dict[str, Any]) -> bool:
        """
        Check if content is valid JSON Schema.

        JSON Schema is identified by:
        - Has "type" = "object" (or array of types including object)
        - Has "properties" key (for object schemas)
        - Has $ref / $defs / definitions / allOf / anyOf / oneOf at the root
        - Does NOT have "fields" key (that's Avro)
        - Does NOT have "type" = "record" (that's Avro)

        Args:
            content: Content to check

        Returns:
            True if this parser can handle the content
        """
        try:
            parsed = json.loads(content) if isinstance(content, str) else content

            if not isinstance(parsed, dict):
                return False

            # Reject Avro schemas
            if parsed.get("type") == "record" and "fields" in parsed:
                return False

            has_properties = "properties" in parsed
            is_object_type = parsed.get("type") == "object"

            type_val = parsed.get("type")
            if isinstance(type_val, list) and "object" in type_val:
                is_object_type = True

            # A root that is only a $ref or a composition is still JSON Schema.
            has_structure = any(
                key in parsed for key in ("$ref", "$defs", "definitions", "allOf", "anyOf", "oneOf")
            )

            return has_properties or is_object_type or has_structure

        except (json.JSONDecodeError, TypeError):
            return False
