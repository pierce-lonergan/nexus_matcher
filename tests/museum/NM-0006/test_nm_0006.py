"""
NM-0006 -- the results came back under names the caller's schema never used.

The flattened parser deliberately rewrites `cust_addr__city` into the dotted path
`cust.addr.city`, because recovering the parent path is worth +19.3 points of P@1 -- the
largest single factor in the pipeline. That rewrite is correct and must stay.

What was wrong is that the RESULT DICT was keyed by the rewritten path. So a caller who
handed in their own flattened column names got back a mapping whose keys they had never
seen:

    results = matcher.match_schema("customer_flat.json")
    results["cust_addr__city"]        -> KeyError
    list(results)                     -> ['cust.id', 'cust.addr.city', ...]

The field was present, matched, and scored. It was simply not addressable, and iterating
the keys produced names that appear in no schema, no export and no downstream table. The
count was right, which is why the conservation check that catches NM-0005 cannot see this.

Deliberately driven through the real parser rather than hand-built SchemaFields: the
defect only exists because the parser derives a name, so a test that supplies the derived
name itself cannot fail. And the assertions are on ADDRESSABILITY -- the caller's string
resolves -- not on any particular keying scheme, so the identity strategy may change again
without invalidating them.
"""

from __future__ import annotations

import pytest

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.models.entities import DictionaryEntry
from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import (
    FlattenedAvroParser,
)
from nexus_matcher.shared.types.base import DataType

# The shape a schema flattener emits: one joined identifier per column, `__` at an array
# boundary. None of these survives the split unchanged, which is the point.
FLATTENED_COLUMNS = [
    "cust_id",
    "cust_addr__city",
    "cust_addr__postal_code",
    "cust_contact__email_address",
]


def _dictionary():
    return [
        DictionaryEntry(
            id=f"d-{i}",
            business_name=name,
            logical_name="",
            definition=f"The {name.lower()} held on the customer record.",
            data_type=DataType.STRING,
            domain="customer",
        )
        for i, name in enumerate(
            [
                "Customer Identifier",
                "Customer City",
                "Customer Postal Code",
                "Customer Email Address",
            ]
        )
    ]


@pytest.fixture(scope="module")
def parsed_fields():
    schema = FlattenedAvroParser().parse(FLATTENED_COLUMNS, schema_name="customer_flat")
    assert schema.is_success, schema.error
    return schema.unwrap().fields


@pytest.fixture(scope="module")
def results(parsed_fields):
    matcher = NexusMatcher.from_config(MatchingConfig())
    matcher._index_dictionary(_dictionary())
    return matcher._match_fields(parsed_fields)


def test_the_parser_really_does_rewrite_these_names(parsed_fields):
    """
    Guards the guard. If the parser stopped reconstructing the hierarchy, every assertion
    below would hold trivially and this entry would silently become a hole -- while the
    repo quietly lost the +19.3 P@1 the rewrite buys.
    """
    derived = [f.full_path for f in parsed_fields]
    assert derived != FLATTENED_COLUMNS, (
        "the flattened parser is no longer reconstructing a dotted path, so there is no "
        "longer any difference between the caller's name and the derived one"
    )


@pytest.mark.parametrize("column", FLATTENED_COLUMNS)
def test_every_column_is_addressable_by_the_name_the_caller_supplied(results, column):
    """
    The symptom exactly as a user meets it: `results["cust_addr__city"]`.

    A result set you cannot index by the identifiers you put in is a result set you have
    to re-derive the library's internal path rules to read.
    """
    assert column in results, (
        f"{column!r} is missing from the results. The field was matched -- it came back "
        f"under a name this caller never used. Keys were: {sorted(results)}"
    )


def test_no_result_arrives_under_a_name_the_caller_never_used(results):
    """
    The other direction, and the one an `in` check cannot catch.

    Adding the flattened name ALONGSIDE the derived path would satisfy every assertion
    above while still making `for key in results` emit phantom columns.
    """
    unexpected = sorted(set(results) - set(FLATTENED_COLUMNS))
    assert not unexpected, (
        f"these keys correspond to no submitted column: {unexpected}. Iterating the "
        f"results yields names that appear in no schema and no downstream table."
    )


def test_each_addressed_result_is_populated(results):
    """A present-but-empty entry would satisfy a naive membership check."""
    for column in FLATTENED_COLUMNS:
        assert results[column], f"{column!r} came back with no candidates at all"
