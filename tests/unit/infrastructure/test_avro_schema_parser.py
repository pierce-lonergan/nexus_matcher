"""
tests.unit.infrastructure.test_avro_schema_parser | Layer: TEST
Tests for AvroSchemaParser, focused on the two gaps that silently shrank output.

Gap 1 -- arrays of records. `_parse_complex_type` kept only
`items`' scalar data_type and dropped `nested_fields`, so
`array<record{...}>` contributed one ARRAY field and none of its children.
Arrays of records are how Avro models repeated groups, so this removed most of
a typical schema.

Gap 2 -- named type references. Avro requires a named type to be defined once;
later uses are a bare string. `_parse_type` mapped any unrecognised string to
DataType.UNKNOWN, so in a schema where `home_address` and `work_address` are
both of type `Address`, the first expanded and the second vanished.

Neither failed. Both just returned a shorter field list.
"""

from __future__ import annotations

import pytest

from nexus_matcher.infrastructure.adapters.schema_parsers.avro import AvroSchemaParser
from nexus_matcher.shared.types.base import DataType


@pytest.fixture
def parser() -> AvroSchemaParser:
    return AvroSchemaParser()


def _paths(schema) -> list[str]:
    return [f.full_path for f in schema.fields]


def _by_path(schema) -> dict:
    return {f.full_path: f for f in schema.fields}


# =============================================================================
# GAP 1: ARRAYS OF RECORDS
# =============================================================================


class TestArraysOfRecords:
    """Fields inside an array's item record must be emitted."""

    def test_array_of_records_emits_item_fields(self, parser):
        schema_def = {
            "type": "record",
            "name": "Patient",
            "fields": [
                {"name": "patient_id", "type": "long"},
                {
                    "name": "visits",
                    "type": {
                        "type": "array",
                        "items": {
                            "type": "record",
                            "name": "Visit",
                            "fields": [
                                {"name": "visit_date", "type": "int"},
                                {"name": "provider_id", "type": "long"},
                            ],
                        },
                    },
                },
            ],
        }

        result = parser.parse(schema_def)

        assert result.is_success, result.error
        paths = _paths(result.unwrap())
        assert "visits[].visit_date" in paths, f"array item fields were dropped; got {paths}"
        assert "visits[].provider_id" in paths

    def test_array_field_records_item_type(self, parser):
        schema_def = {
            "type": "record",
            "name": "R",
            "fields": [
                {
                    "name": "rows",
                    "type": {
                        "type": "array",
                        "items": {
                            "type": "record",
                            "name": "Row",
                            "fields": [{"name": "v", "type": "string"}],
                        },
                    },
                }
            ],
        }

        fields = _by_path(parser.parse(schema_def).unwrap())

        assert fields["rows"].is_array is True
        assert fields["rows"].array_item_type == DataType.RECORD
        assert fields["rows[].v"].data_type == DataType.STRING

    def test_array_of_primitives_still_works(self, parser):
        """The existing scalar behaviour must not regress."""
        schema_def = {
            "type": "record",
            "name": "R",
            "fields": [{"name": "tags", "type": {"type": "array", "items": "string"}}],
        }

        fields = _by_path(parser.parse(schema_def).unwrap())

        assert fields["tags"].is_array is True
        assert fields["tags"].array_item_type == DataType.STRING

    def test_nullable_array_of_records_expands(self, parser):
        """["null", {array of record}] -- the most common real shape."""
        schema_def = {
            "type": "record",
            "name": "R",
            "fields": [
                {
                    "name": "orders",
                    "type": [
                        "null",
                        {
                            "type": "array",
                            "items": {
                                "type": "record",
                                "name": "Order",
                                "fields": [{"name": "total", "type": "double"}],
                            },
                        },
                    ],
                }
            ],
        }

        schema = parser.parse(schema_def).unwrap()
        fields = _by_path(schema)

        assert fields["orders"].is_nullable is True
        assert fields["orders"].is_array is True
        assert "orders[].total" in fields

    def test_map_of_records_expands(self, parser):
        """Map values were dropped entirely; a map of records is common."""
        schema_def = {
            "type": "record",
            "name": "R",
            "fields": [
                {
                    "name": "attributes",
                    "type": {
                        "type": "map",
                        "values": {
                            "type": "record",
                            "name": "Attr",
                            "fields": [
                                {"name": "label", "type": "string"},
                                {"name": "weight", "type": "double"},
                            ],
                        },
                    },
                }
            ],
        }

        paths = _paths(parser.parse(schema_def).unwrap())

        assert "attributes{}.label" in paths, f"map values dropped; got {paths}"
        assert "attributes{}.weight" in paths


# =============================================================================
# GAP 2: NAMED TYPE REFERENCES
# =============================================================================


class TestNamedTypeReferences:
    """A named type used a second time must expand, not become UNKNOWN."""

    def test_second_use_of_named_record_expands(self, parser):
        schema_def = {
            "type": "record",
            "name": "Customer",
            "namespace": "com.example",
            "fields": [
                {
                    "name": "home_address",
                    "type": {
                        "type": "record",
                        "name": "Address",
                        "fields": [
                            {"name": "street", "type": "string"},
                            {"name": "postcode", "type": "string"},
                        ],
                    },
                },
                {"name": "work_address", "type": "Address"},
            ],
        }

        schema = parser.parse(schema_def).unwrap()
        fields = _by_path(schema)

        assert fields["work_address"].data_type == DataType.RECORD, (
            f"reused named type resolved to {fields['work_address'].data_type} instead of RECORD"
        )
        assert "work_address.street" in fields, (
            f"reused named type was not expanded; got {sorted(fields)}"
        )
        assert "work_address.postcode" in fields

    def test_named_type_reference_by_fullname(self, parser):
        schema_def = {
            "type": "record",
            "name": "Customer",
            "namespace": "com.example",
            "fields": [
                {
                    "name": "primary",
                    "type": {
                        "type": "record",
                        "name": "Address",
                        "namespace": "com.example",
                        "fields": [{"name": "city", "type": "string"}],
                    },
                },
                {"name": "secondary", "type": "com.example.Address"},
            ],
        }

        paths = _paths(parser.parse(schema_def).unwrap())

        assert "secondary.city" in paths, f"fullname ref not resolved; got {paths}"

    def test_named_enum_reference_resolves(self, parser):
        schema_def = {
            "type": "record",
            "name": "R",
            "fields": [
                {
                    "name": "status",
                    "type": {
                        "type": "enum",
                        "name": "Status",
                        "symbols": ["ACTIVE", "CLOSED"],
                    },
                },
                {"name": "prev_status", "type": "Status"},
            ],
        }

        fields = _by_path(parser.parse(schema_def).unwrap())

        assert fields["status"].data_type == DataType.ENUM
        assert fields["prev_status"].data_type == DataType.ENUM

    def test_unknown_type_name_fails_loudly(self, parser):
        """
        A bare name that is neither a primitive nor a defined named type is a
        malformed schema. Returning UNKNOWN is what hid the missing subtree.
        """
        schema_def = {
            "type": "record",
            "name": "R",
            "fields": [{"name": "x", "type": "NeverDefined"}],
        }

        result = parser.parse(schema_def)

        assert result.is_failure
        assert "NeverDefined" in (result.error or "")

    def test_recursive_type_terminates(self, parser):
        """A self-referential record must not recurse forever."""
        schema_def = {
            "type": "record",
            "name": "Node",
            "fields": [
                {"name": "label", "type": "string"},
                {"name": "next", "type": ["null", "Node"]},
            ],
        }

        result = parser.parse(schema_def)

        assert result.is_success, result.error
        fields = _by_path(result.unwrap())
        assert fields["next"].data_type == DataType.RECORD
        assert fields["next"].is_nullable is True
        assert fields["next"].source_metadata.get("recursive") is True


# =============================================================================
# BASELINE BEHAVIOUR (must not regress)
# =============================================================================


class TestAvroBaseline:
    """Behaviour the parser already had."""

    def test_format_and_extensions(self, parser):
        assert parser.format_name == "avro"
        assert ".avsc" in parser.file_extensions

    def test_primitives(self, parser):
        schema_def = {
            "type": "record",
            "name": "R",
            "fields": [
                {"name": "a", "type": "string"},
                {"name": "b", "type": "int"},
                {"name": "c", "type": "long"},
                {"name": "d", "type": "double"},
                {"name": "e", "type": "boolean"},
                {"name": "f", "type": "bytes"},
            ],
        }

        fields = _by_path(parser.parse(schema_def).unwrap())

        assert fields["a"].data_type == DataType.STRING
        assert fields["b"].data_type == DataType.INTEGER
        assert fields["c"].data_type == DataType.LONG
        assert fields["d"].data_type == DataType.DOUBLE
        assert fields["e"].data_type == DataType.BOOLEAN
        assert fields["f"].data_type == DataType.BYTES

    def test_nullable_union(self, parser):
        schema_def = {
            "type": "record",
            "name": "R",
            "fields": [{"name": "x", "type": ["null", "string"]}],
        }

        field = _by_path(parser.parse(schema_def).unwrap())["x"]

        assert field.data_type == DataType.STRING
        assert field.is_nullable is True

    def test_logical_types(self, parser):
        schema_def = {
            "type": "record",
            "name": "R",
            "fields": [
                {"name": "d", "type": {"type": "int", "logicalType": "date"}},
                {
                    "name": "ts",
                    "type": {"type": "long", "logicalType": "timestamp-millis"},
                },
                {"name": "u", "type": {"type": "string", "logicalType": "uuid"}},
            ],
        }

        fields = _by_path(parser.parse(schema_def).unwrap())

        assert fields["d"].data_type == DataType.DATE
        assert fields["ts"].data_type == DataType.TIMESTAMP
        assert fields["u"].data_type == DataType.UUID

    def test_nested_record_paths(self, parser):
        schema_def = {
            "type": "record",
            "name": "R",
            "fields": [
                {
                    "name": "outer",
                    "type": {
                        "type": "record",
                        "name": "Inner",
                        "fields": [{"name": "leaf", "type": "string"}],
                    },
                }
            ],
        }

        fields = _by_path(parser.parse(schema_def).unwrap())

        assert fields["outer.leaf"].parent_path == "outer"
        assert fields["outer.leaf"].name == "leaf"

    def test_doc_becomes_description(self, parser):
        schema_def = {
            "type": "record",
            "name": "R",
            "fields": [{"name": "x", "type": "string", "doc": "The customer email"}],
        }

        field = _by_path(parser.parse(schema_def).unwrap())["x"]

        assert field.description == "The customer email"

    def test_rejects_non_record_root(self, parser):
        result = parser.parse({"type": "enum", "name": "E", "symbols": ["A"]})
        assert result.is_failure

    def test_can_parse_detects_avro(self, parser):
        assert parser.can_parse({"type": "record", "name": "R", "fields": []})
        assert not parser.can_parse({"type": "object", "properties": {}})

    def test_parsers_do_not_leak_named_types_between_schemas(self, parser):
        """
        The registry must be per-parse. A name defined in schema A must not
        resolve while parsing schema B.
        """
        first = {
            "type": "record",
            "name": "A",
            "fields": [
                {
                    "name": "addr",
                    "type": {
                        "type": "record",
                        "name": "Address",
                        "fields": [{"name": "city", "type": "string"}],
                    },
                }
            ],
        }
        second = {
            "type": "record",
            "name": "B",
            "fields": [{"name": "addr", "type": "Address"}],
        }

        assert parser.parse(first).is_success
        assert parser.parse(second).is_failure, "named type leaked from a previous parse"
