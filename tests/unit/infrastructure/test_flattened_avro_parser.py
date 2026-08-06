"""
tests.unit.infrastructure.test_flattened_avro_parser | Layer: TEST
Flattened Avro parsing, and the hierarchy reconstruction that carries the accuracy.

The load-bearing behaviour is not "does it parse" -- it is whether the PARENT PATH is
recovered from the joined identifier. Supplying parent context separately from the leaf
name is worth +19.3 points of P@1 on the labelled benchmark, the single largest factor in
the pipeline. A flattened name already contains the path; if the parser hands the matcher
one opaque token, that entire gain is thrown away silently and nothing else fails.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from nexus_matcher.domain.services.context_enricher import ContextEnricher
from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import (
    FlattenedAvroParser,
    field_from_flattened,
    flatten_avro_schema,
    map_data_type,
    split_flattened_name,
)
from nexus_matcher.shared.types.base import DataType


class TestNameSplitting:
    @pytest.mark.parametrize(
        "flattened,expected",
        [
            ("customer_addresses__street_name", ["customer", "addresses", "street", "name"]),
            ("cust_id", ["cust", "id"]),
            ("orders__items__sku", ["orders", "items", "sku"]),
            ("streetName", ["street", "Name"]),
            ("customer.address.city", ["customer", "address", "city"]),
        ],
    )
    def test_splits_into_path_segments(self, flattened, expected):
        segments, _ = split_flattened_name(flattened)
        assert segments == expected

    def test_double_underscore_marks_an_array(self):
        _, touches_array = split_flattened_name("orders__items__sku")
        assert touches_array is True

    def test_single_underscore_is_not_an_array(self):
        _, touches_array = split_flattened_name("customer_id")
        assert touches_array is False

    def test_empty_name(self):
        assert split_flattened_name("") == ([], False)


class TestHierarchyReconstruction:
    """The whole point: parent path must be recovered, not discarded."""

    def test_leaf_and_parent_are_separated(self):
        f = field_from_flattened("customer_addresses__street")
        assert f.name == "street"
        assert f.parent_path == "customer.addresses"
        assert f.full_path == "customer.addresses.street"

    def test_parent_context_reaches_the_query_text(self):
        f = field_from_flattened("customer_addresses__street", doc="Street line")
        enriched = ContextEnricher().enrich(f)
        for token in ("customer", "addresses", "street"):
            assert token in enriched.lower(), f"{token!r} missing from {enriched!r}"

    def test_doc_is_carried_into_the_query_text(self):
        f = field_from_flattened("cust_bal", doc="Current account balance in USD")
        assert "balance" in ContextEnricher().enrich(f).lower()

    def test_explicit_path_overrides_name_inference(self):
        """A dotted path from the flattener is authoritative; '_' is ambiguous."""
        f = field_from_flattened("customer_street_name", explicit_path="customer.street_name")
        assert f.name == "street_name"
        assert f.parent_path == "customer"

    def test_flattened_name_is_preserved_for_round_tripping(self):
        f = field_from_flattened("a_b__c")
        assert f.source_metadata["flattened_name"] == "a_b__c"

    def test_array_boundary_sets_is_array(self):
        assert field_from_flattened("orders__sku").is_array is True
        assert field_from_flattened("customer_id").is_array is False


class TestTypeMapping:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("string", DataType.STRING),
            ("LONG", DataType.LONG),
            ("int", DataType.INTEGER),
            ("boolean", DataType.BOOLEAN),
            ("varchar(255)", DataType.STRING),
            ("decimal(10,2)", DataType.DECIMAL),
            (None, DataType.UNKNOWN),
            (["null", "string"], DataType.STRING),
            ({"type": "long", "logicalType": "timestamp-millis"}, DataType.TIMESTAMP),
        ],
    )
    def test_maps_types(self, raw, expected):
        assert map_data_type(raw) == expected

    def test_unknown_type_does_not_raise(self):
        assert map_data_type("some_exotic_type") == DataType.UNKNOWN


class TestParserInputShapes:
    """Real flatteners emit several shapes; all of them must just work."""

    def _names(self, schema):
        return {f.source_metadata["flattened_name"] for f in schema.fields}

    def test_dict_keyed_by_flattened_name(self):
        rows = {"cust_id": {"dataType": "LONG"}, "cust_addr__city": {"dataType": "STRING"}}
        s = FlattenedAvroParser().parse(rows).unwrap()
        assert self._names(s) == {"cust_id", "cust_addr__city"}

    def test_list_of_row_objects(self):
        rows = [
            {"flattenedName": "cust_id", "dataType": "LONG", "nullable": False},
            {"flattenedName": "cust_nm", "dataType": "STRING", "doc": "Full legal name"},
        ]
        s = FlattenedAvroParser().parse(rows).unwrap()
        assert self._names(s) == {"cust_id", "cust_nm"}
        assert any("legal name" in f.description for f in s.fields)

    def test_plain_list_of_names(self):
        s = FlattenedAvroParser().parse(["a_b", "c__d"]).unwrap()
        assert self._names(s) == {"a_b", "c__d"}

    def test_wrapped_payload(self):
        s = FlattenedAvroParser().parse({"fields": [{"name": "cust_id"}]}).unwrap()
        assert self._names(s) == {"cust_id"}

    def test_alternate_key_spellings(self):
        rows = [{"column": "x_y", "description": "a definition", "type": "string"}]
        f = FlattenedAvroParser().parse(rows).unwrap().fields[0]
        assert f.description == "a definition"
        assert f.data_type == DataType.STRING

    def test_unknown_columns_are_kept_as_metadata(self):
        rows = [{"flattenedName": "a", "governance_status": "PII"}]
        f = FlattenedAvroParser().parse(rows).unwrap().fields[0]
        assert f.source_metadata["governance_status"] == "PII"

    def test_empty_input_is_a_failure_not_a_crash(self):
        assert FlattenedAvroParser().parse([]).is_failure

    def test_raw_avro_is_declined(self):
        """Raw Avro must fall through to the real Avro parser."""
        assert FlattenedAvroParser().can_parse({"type": "record", "fields": []}) is False

    def test_parse_file_json(self, tmp_path):
        p = tmp_path / "flat.json"
        p.write_text(json.dumps({"cust_id": {"dataType": "LONG"}}), encoding="utf-8")
        assert FlattenedAvroParser().parse_file(p).unwrap().fields[0].name == "id"

    def test_parse_file_csv(self, tmp_path):
        p = tmp_path / "flat.csv"
        p.write_text("flattenedName,dataType,doc\ncust_id,LONG,The identifier\n", encoding="utf-8")
        f = FlattenedAvroParser().parse_file(p).unwrap().fields[0]
        assert f.description == "The identifier"

    def test_missing_file_is_a_failure(self, tmp_path):
        assert FlattenedAvroParser().parse_file(tmp_path / "nope.json").is_failure


class TestRawAvroFlattening:
    """Flattening from the .avsc directly -- exact hierarchy, and doc preserved."""

    SCHEMA: ClassVar[dict] = {
        "type": "record",
        "name": "Customer",
        "fields": [
            {"name": "id", "type": "long", "doc": "Unique customer identifier"},
            {
                "name": "address",
                "doc": "Postal address",
                "type": {
                    "type": "record",
                    "name": "Addr",
                    "fields": [
                        {"name": "street", "type": ["null", "string"], "doc": "Street line"},
                        {"name": "city", "type": "string"},
                    ],
                },
            },
            {
                "name": "orders",
                "type": {
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "Order",
                        "fields": [{"name": "sku", "type": "string", "doc": "Stock keeping unit"}],
                    },
                },
            },
            {"name": "tags", "type": {"type": "array", "items": "string"}, "doc": "Free tags"},
        ],
    }

    def _by_flat(self):
        return {f.source_metadata["flattened_name"]: f for f in flatten_avro_schema(self.SCHEMA)}

    def test_nested_records_are_joined_with_single_underscore(self):
        assert "address_street" in self._by_flat()

    def test_array_of_records_uses_the_double_underscore_boundary(self):
        assert "orders__sku" in self._by_flat()

    def test_array_of_primitives_stays_one_column(self):
        f = self._by_flat()["tags"]
        assert f.data_type == DataType.ARRAY
        assert f.source_metadata.get("array_serialized") is True

    def test_union_is_unwrapped_and_marked_nullable(self):
        f = self._by_flat()["address_street"]
        assert f.data_type == DataType.STRING
        assert f.is_nullable is True

    def test_doc_is_propagated(self):
        """The Groovy flattener drops doc; this must not."""
        assert self._by_flat()["orders__sku"].description == "Stock keeping unit"

    def test_leaf_without_doc_inherits_the_nearest_ancestor(self):
        assert self._by_flat()["address_city"].description == "Postal address"

    def test_inheritance_can_be_disabled(self):
        fields = {
            f.source_metadata["flattened_name"]: f
            for f in flatten_avro_schema(self.SCHEMA, inherit_doc=False)
        }
        assert fields["address_city"].description == ""

    def test_dotted_paths_are_exact(self):
        assert self._by_flat()["address_street"].full_path == "address.street"
        assert self._by_flat()["address_street"].parent_path == "address"
