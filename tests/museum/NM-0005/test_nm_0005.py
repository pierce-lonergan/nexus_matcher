"""
NM-0005 -- a matched field disappeared from the results, taking its governance with it.

Results were keyed by `SchemaField.full_path`. That is NOT unique: the flattened parser
maps `contact__email` (an array of contacts) and `contact_email` (a scalar column) onto the
same reconstructed path `contact.email`, because "_" is both the separator and a legal
character inside an Avro field name. The second field overwrote the first in the results
dict, so match_schema returned FEWER entries than it was given, with no error.

The consequence is the one this library exists to prevent: the vanished column is never
matched, so it inherits no protection level, and nothing anywhere says so.

This asserts the OBSERVABLE SYMPTOM -- output count and per-field addressability -- rather
than the keying strategy, so it stays valid if the identity scheme changes again.
"""

from __future__ import annotations

import pytest

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.shared.types.base import DataType


def _dictionary():
    return [
        DictionaryEntry(
            id=f"d-{i}",
            business_name=name,
            logical_name="",
            definition=f"The {name.lower()} of the customer record.",
            data_type=DataType.STRING,
            domain="customer",
        )
        for i, name in enumerate(
            ["Contact Email Address", "Customer Full Name", "Shipping Street Line"]
        )
    ]


def _colliding_fields():
    """
    Two DISTINCT flattened Avro columns that reconstruct to the same dotted path.

    `contact__email` is the email of an element of a repeated `contact` record;
    `contact_email` is a flat scalar column. Both are legal fields of one schema and both
    must be matched.
    """
    return [
        SchemaField(
            name="email",
            data_type=DataType.STRING,
            full_path="contact.email",
            parent_path="contact",
            description="Email of a contact in the repeated contact block",
            source_metadata={"flattened_name": "contact__email"},
        ),
        SchemaField(
            name="email",
            data_type=DataType.STRING,
            full_path="contact.email",
            parent_path="contact",
            description="Flat contact email column",
            source_metadata={"flattened_name": "contact_email"},
        ),
    ]


@pytest.fixture(scope="module")
def results():
    matcher = NexusMatcher.from_config(MatchingConfig())
    matcher._index_dictionary(_dictionary())
    return matcher._match_fields(_colliding_fields())


def test_every_input_field_produces_a_result(results):
    """The conservation law. Two fields in, two results out -- always."""
    assert len(results) == 2, (
        f"2 fields were submitted but {len(results)} came back. A column was dropped and "
        f"inherits no governance classification."
    )


def test_the_two_fields_did_not_share_a_key(results):
    assert len(set(results)) == 2, "the two fields collapsed onto one key"


def test_each_result_is_populated(results):
    """A surviving-but-empty entry would satisfy a naive count check."""
    for key, matches in results.items():
        assert matches, f"{key!r} came back with no candidates at all"
