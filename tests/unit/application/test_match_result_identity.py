"""
tests.unit.application.test_match_result_identity | Layer: TEST
Every field handed to the matcher must come back out, under the name the caller used.

This library exists so that an Avro column inherits the protection level of the glossary
entry it matches. A column missing from the result dict inherits nothing, and nobody is
told: `match_schema` returned a dict keyed by `full_path`, and the flattened parser maps
several distinct columns onto one dotted path. `contact__email` and `contact_email` are
two legal fields of one Avro record; both became `contact.email` and the second silently
overwrote the first. The caller got a shorter dict, no exception, and one ungoverned
column.

The same key was unaddressable from the other direction: the parser rewrites `_` into `.`
to recover the parent path (worth +19.3 P@1, so the rewrite stays), which meant a caller
looking up their own column name got a KeyError from a result set that did contain it.

Both are pinned here: the count invariant, and the round-trip of the caller's own names.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application.use_cases.match_schema import (
    MatchingConfig,
    NexusMatcher,
    field_result_key,
)
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import FlattenedAvroParser
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import DataType, ProtectionLevel, Result


class _StubProvider:
    """Model-free encoder: identical unit vectors, so no download and no network."""

    dimension = 8
    model_name = "stub"

    def embed(self, texts):
        rows = np.tile(np.eye(1, 8, 0, dtype=np.float32), (len(list(texts)), 1))

        class _Batch:
            embeddings = rows

        return Result.success(_Batch())

    def embed_single(self, text):
        return Result.success(np.eye(1, 8, 0, dtype=np.float32)[0])


@pytest.fixture
def matcher() -> NexusMatcher:
    """A matcher with a two-entry glossary, one of which is PII."""
    m = NexusMatcher(
        embedding_provider=_StubProvider(),
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=8)
        ),
        schema_parser_registry={"flattened_avro": FlattenedAvroParser()},
        config=MatchingConfig(results_per_field=3),
    )
    m._index_dictionary(
        [
            DictionaryEntry(
                id="e1",
                business_name="Contact Email Address",
                logical_name="contact_email",
                definition="Email address of a contact",
                data_type=DataType.STRING,
                protection_level=ProtectionLevel.PII,
                domain="CONTACT",
            ),
            DictionaryEntry(
                id="e2",
                business_name="Customer Identifier",
                logical_name="cust_id",
                definition="Unique identifier of the customer",
                data_type=DataType.LONG,
                protection_level=ProtectionLevel.INTERNAL,
                domain="CUSTOMER",
            ),
        ]
    )
    return m


# One Avro record may legally hold BOTH an array<Contact> and a scalar contact_email.
# The flattener emits them as two distinct physical columns that split to one dotted path.
COLLIDING = {
    "cust_id": {"dataType": "LONG"},
    "contact__email": {"dataType": "STRING", "doc": "Email of each listed contact"},
    "contact_email": {"dataType": "STRING", "doc": "Primary contact email"},
}


class TestNoFieldIsLost:
    """The count invariant: fewer results than fields means a column nobody governs."""

    def test_every_input_field_produces_a_result_entry(self, matcher):
        results = matcher.match_schema(COLLIDING, schema_format="flattened_avro")
        assert len(results) == len(COLLIDING), (
            f"{len(COLLIDING)} columns in, {len(results)} out: {sorted(results)}"
        )

    def test_colliding_columns_stay_separate_fields(self, matcher):
        """
        Both entries must describe their OWN column. Overwriting left one key whose
        results belonged to the other field entirely -- a wrong classification rather
        than a missing one.
        """
        results = matcher.match_schema(COLLIDING, schema_format="flattened_avro")
        matched = {
            m[0].schema_field.source_metadata["flattened_name"] for m in results.values() if m
        }
        assert matched == set(COLLIDING)

    def test_the_lost_column_still_inherits_a_protection_level(self, matcher):
        """The whole purpose: a vanished column inherits no governance classification."""
        results = matcher.match_schema(COLLIDING, schema_format="flattened_avro")
        for column in ("contact__email", "contact_email"):
            assert results[column], f"{column} matched nothing"
            assert results[column][0].dictionary_entry.protection_level is ProtectionLevel.PII

    def test_session_counts_every_field_it_parsed(self, matcher):
        session = matcher.match_schema_session(COLLIDING, schema_format="flattened_avro")
        assert session.field_count == session.schema.field_count

    def test_duplicate_identifiers_do_not_displace_each_other(self, matcher):
        """
        A flattened export can list the same column twice. Neither row may be dropped,
        so the second is parked under a visibly synthetic suffix rather than silently
        replacing the first.
        """
        rows = [{"flattenedName": "cust_id"}, {"flattenedName": "cust_id"}]
        schema = FlattenedAvroParser().parse(rows).unwrap()
        results = matcher._match_fields(schema.fields)
        assert len(results) == 2
        assert "cust_id" in results
        assert "cust_id#2" in results


class TestCallerAddressableKeys:
    """A result the caller cannot look up is no better than a missing one."""

    def test_keys_are_the_names_the_caller_fed_in(self, matcher):
        results = matcher.match_schema(COLLIDING, schema_format="flattened_avro")
        assert set(results) == set(COLLIDING)

    def test_keys_keep_input_order(self, matcher):
        results = matcher.match_schema(COLLIDING, schema_format="flattened_avro")
        assert list(results) == list(COLLIDING)

    def test_dotted_paths_are_not_invented_for_flattened_input(self, matcher):
        """`cust_id` is one column; `cust.id` is a path the caller never mentioned."""
        results = matcher.match_schema(COLLIDING, schema_format="flattened_avro")
        assert "cust.id" not in results

    def test_other_parsers_keep_their_dotted_path_keys(self, matcher):
        """
        Backward compatibility. Raw Avro, JSON Schema and SQL DDL carry no flattened
        name, and their callers have always addressed results by `full_path`.
        """
        fields = [
            SchemaField(
                name="email",
                data_type=DataType.STRING,
                full_path="customer.contact.email",
                parent_path="customer.contact",
            ),
            SchemaField(
                name="id",
                data_type=DataType.LONG,
                full_path="customer.id",
                parent_path="customer",
            ),
        ]
        results = matcher._match_fields(fields)
        assert set(results) == {"customer.contact.email", "customer.id"}


class TestFieldResultKey:
    """The identity rule itself, so a caller can predict the key from a field."""

    def test_flattened_name_wins_when_present(self):
        field = SchemaField(
            name="email",
            data_type=DataType.STRING,
            full_path="contact.email",
            source_metadata={"flattened_name": "contact__email"},
        )
        assert field_result_key(field) == "contact__email"

    def test_falls_back_to_full_path(self):
        field = SchemaField(name="email", data_type=DataType.STRING, full_path="contact.email")
        assert field_result_key(field) == "contact.email"

    def test_blank_flattened_name_is_not_used_as_a_key(self):
        """An empty key would collide with every other empty one."""
        field = SchemaField(
            name="value",
            data_type=DataType.STRING,
            source_metadata={"flattened_name": ""},
        )
        assert field_result_key(field) == "value"
