"""
tests.unit.infrastructure.test_json_schema_refs_and_arrays | Layer: TEST
Regression guards for JsonSchemaParser structural keyword support.

The parser used to read only `type`, `properties`, `items` and `format`. It had
no handling for $ref, $defs/definitions, allOf, anyOf or oneOf, and it kept only
the scalar data_type of an array's `items`. Real-world JSON Schemas are built
almost entirely out of those keywords, so the parser returned a field list that
was missing most of the schema -- silently, with is_success True.

Every test below asserts a field that the old parser did not emit at all.
"""

from __future__ import annotations

import pytest

from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import (
    JsonSchemaParser,
)
from nexus_matcher.shared.types.base import DataType


@pytest.fixture
def parser() -> JsonSchemaParser:
    return JsonSchemaParser()


def _paths(schema) -> list[str]:
    return [f.full_path for f in schema.fields]


def _by_path(schema) -> dict:
    return {f.full_path: f for f in schema.fields}


# =============================================================================
# $ref / $defs
# =============================================================================


class TestRefResolution:
    def test_ref_to_defs_expands_nested_properties(self, parser):
        schema_def = {
            "title": "Order",
            "type": "object",
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "postcode": {"type": "string"},
                    },
                }
            },
            "properties": {
                "billing": {"$ref": "#/$defs/Address"},
            },
        }

        result = parser.parse(schema_def)

        assert result.is_success, result.error
        paths = _paths(result.unwrap())
        assert "billing.street" in paths, f"$ref not expanded; got {paths}"
        assert "billing.postcode" in paths

    def test_ref_to_legacy_definitions_expands(self, parser):
        """draft-07 uses "definitions" rather than "$defs"."""
        schema_def = {
            "type": "object",
            "definitions": {
                "Money": {
                    "type": "object",
                    "properties": {"amount": {"type": "number"}},
                }
            },
            "properties": {"total": {"$ref": "#/definitions/Money"}},
        }

        paths = _paths(parser.parse(schema_def).unwrap())

        assert "total.amount" in paths

    def test_same_ref_used_twice_expands_both_times(self, parser):
        schema_def = {
            "type": "object",
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            },
            "properties": {
                "home": {"$ref": "#/$defs/Address"},
                "work": {"$ref": "#/$defs/Address"},
            },
        }

        paths = _paths(parser.parse(schema_def).unwrap())

        assert "home.city" in paths
        assert "work.city" in paths

    def test_ref_chain_resolves(self, parser):
        schema_def = {
            "type": "object",
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"type": "object", "properties": {"leaf": {"type": "string"}}},
            },
            "properties": {"x": {"$ref": "#/$defs/A"}},
        }

        assert "x.leaf" in _paths(parser.parse(schema_def).unwrap())

    def test_recursive_ref_terminates(self, parser):
        """A self-referential $ref must not hang or blow the stack."""
        schema_def = {
            "type": "object",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "child": {"$ref": "#/$defs/Node"},
                    },
                }
            },
            "properties": {"tree": {"$ref": "#/$defs/Node"}},
        }

        result = parser.parse(schema_def)

        assert result.is_success, result.error
        fields = _by_path(result.unwrap())
        assert "tree.label" in fields
        assert fields["tree.child"].source_metadata.get("recursive") is True

    def test_unresolvable_ref_fails_loudly(self, parser):
        """A dangling $ref means the output is incomplete -- say so."""
        schema_def = {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Missing"}},
        }

        result = parser.parse(schema_def)

        assert result.is_failure
        assert "Missing" in (result.error or "")

    def test_external_ref_is_rejected_not_fetched(self, parser):
        """The parser must never reach out to the network for a schema."""
        schema_def = {
            "type": "object",
            "properties": {"x": {"$ref": "https://example.com/schemas/address.json"}},
        }

        result = parser.parse(schema_def)

        assert result.is_failure
        assert "External $ref" in (result.error or "")

    def test_ref_recorded_in_metadata(self, parser):
        schema_def = {
            "type": "object",
            "$defs": {"A": {"type": "string"}},
            "properties": {"x": {"$ref": "#/$defs/A"}},
        }

        field = _by_path(parser.parse(schema_def).unwrap())["x"]

        assert field.source_metadata["$ref"] == "#/$defs/A"
        assert field.data_type == DataType.STRING


# =============================================================================
# allOf / anyOf / oneOf
# =============================================================================


class TestComposition:
    def test_all_of_merges_properties(self, parser):
        schema_def = {
            "type": "object",
            "properties": {
                "audit": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"created_at": {"type": "string"}},
                        },
                        {
                            "type": "object",
                            "properties": {"created_by": {"type": "string"}},
                        },
                    ]
                }
            },
        }

        paths = _paths(parser.parse(schema_def).unwrap())

        assert "audit.created_at" in paths, f"allOf not merged; got {paths}"
        assert "audit.created_by" in paths

    def test_root_level_all_of_merges(self, parser):
        schema_def = {
            "title": "Merged",
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "integer"}}},
            ],
        }

        paths = _paths(parser.parse(schema_def).unwrap())

        assert set(paths) == {"a", "b"}

    def test_all_of_with_refs_merges(self, parser):
        schema_def = {
            "type": "object",
            "$defs": {
                "Timestamps": {
                    "type": "object",
                    "properties": {"updated_at": {"type": "string"}},
                }
            },
            "properties": {
                "record": {
                    "allOf": [
                        {"$ref": "#/$defs/Timestamps"},
                        {"type": "object", "properties": {"id": {"type": "integer"}}},
                    ]
                }
            },
        }

        paths = _paths(parser.parse(schema_def).unwrap())

        assert "record.updated_at" in paths
        assert "record.id" in paths

    def test_all_of_unions_required(self, parser):
        schema_def = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                },
                {
                    "type": "object",
                    "properties": {"b": {"type": "string"}},
                    "required": ["b"],
                },
            ]
        }

        fields = _by_path(parser.parse(schema_def).unwrap())

        assert fields["a"].is_nullable is False
        assert fields["b"].is_nullable is False

    def test_any_of_with_null_branch_is_nullable(self, parser):
        schema_def = {
            "type": "object",
            "properties": {
                "contact": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"email": {"type": "string"}},
                        },
                        {"type": "null"},
                    ]
                }
            },
            "required": ["contact"],
        }

        schema = parser.parse(schema_def).unwrap()
        fields = _by_path(schema)

        assert fields["contact"].is_nullable is True
        assert "contact.email" in fields

    def test_one_of_branches_are_all_visible(self, parser):
        """
        oneOf cannot be represented exactly in a flat field list. Merging keeps
        every candidate field matchable and records the ambiguity in metadata.
        """
        schema_def = {
            "type": "object",
            "properties": {
                "payment": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {"card_number": {"type": "string"}},
                        },
                        {
                            "type": "object",
                            "properties": {"iban": {"type": "string"}},
                        },
                    ]
                }
            },
        }

        schema = parser.parse(schema_def).unwrap()
        fields = _by_path(schema)

        assert "payment.card_number" in fields
        assert "payment.iban" in fields
        assert fields["payment"].source_metadata["composition"] == "oneOf"
        assert fields["payment"].source_metadata["composition_branches"] == 2

    def test_any_of_branch_required_does_not_make_field_required(self, parser):
        """A field required by only one alternative is not required overall."""
        schema_def = {
            "type": "object",
            "properties": {
                "x": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"a": {"type": "string"}},
                            "required": ["a"],
                        },
                        {
                            "type": "object",
                            "properties": {"b": {"type": "string"}},
                        },
                    ]
                }
            },
        }

        fields = _by_path(parser.parse(schema_def).unwrap())

        assert fields["x.a"].is_nullable is True


# =============================================================================
# ARRAY ITEMS
# =============================================================================


class TestArrayItems:
    def test_array_of_objects_emits_item_fields(self, parser):
        schema_def = {
            "type": "object",
            "properties": {
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "quantity": {"type": "integer"},
                        },
                    },
                }
            },
        }

        result = parser.parse(schema_def)

        assert result.is_success, result.error
        paths = _paths(result.unwrap())
        assert "line_items[].sku" in paths, f"array item fields dropped; got {paths}"
        assert "line_items[].quantity" in paths

    def test_array_item_paths_are_distinct_from_object_paths(self, parser):
        """ "orders[].id" must not collide with a plain nested "orders.id"."""
        schema_def = {
            "type": "object",
            "properties": {
                "orders": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                    },
                },
                "meta": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                },
            },
        }

        paths = _paths(parser.parse(schema_def).unwrap())

        assert "orders[].id" in paths
        assert "meta.id" in paths

    def test_array_of_ref_expands(self, parser):
        schema_def = {
            "type": "object",
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
            "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
        }

        paths = _paths(parser.parse(schema_def).unwrap())

        assert "items[].name" in paths

    def test_nested_array_of_objects(self, parser):
        schema_def = {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "members": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"email": {"type": "string"}},
                                },
                            }
                        },
                    },
                }
            },
        }

        paths = _paths(parser.parse(schema_def).unwrap())

        assert "groups[].members[].email" in paths, f"got {paths}"

    def test_array_of_primitives_unchanged(self, parser):
        schema_def = {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        }

        field = _by_path(parser.parse(schema_def).unwrap())["tags"]

        assert field.is_array is True
        assert field.array_item_type == DataType.STRING


# =============================================================================
# IMPLICIT TYPES
# =============================================================================


class TestImplicitTypes:
    def test_object_without_explicit_type_still_expands(self, parser):
        """
        "type" is optional in JSON Schema. A subschema carrying "properties" is
        an object; treating it as UNKNOWN dropped every field underneath.
        """
        schema_def = {
            "type": "object",
            "properties": {"settings": {"properties": {"theme": {"type": "string"}}}},
        }

        schema = parser.parse(schema_def).unwrap()
        fields = _by_path(schema)

        assert fields["settings"].data_type == DataType.RECORD
        assert "settings.theme" in fields

    def test_array_without_explicit_type_still_expands(self, parser):
        schema_def = {
            "type": "object",
            "properties": {
                "rows": {"items": {"type": "object", "properties": {"v": {"type": "string"}}}}
            },
        }

        schema = parser.parse(schema_def).unwrap()
        fields = _by_path(schema)

        assert fields["rows"].is_array is True
        assert "rows[].v" in fields


# =============================================================================
# DETECTION
# =============================================================================


class TestCanParse:
    def test_ref_only_root_is_recognised(self, parser):
        assert parser.can_parse({"$ref": "#/$defs/X", "$defs": {"X": {}}})

    def test_all_of_root_is_recognised(self, parser):
        assert parser.can_parse({"allOf": [{"type": "object"}]})

    def test_avro_still_rejected(self, parser):
        assert not parser.can_parse(
            {"type": "record", "name": "R", "fields": [{"name": "a", "type": "string"}]}
        )
