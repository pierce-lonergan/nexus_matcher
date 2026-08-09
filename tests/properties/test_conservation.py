"""
tests.properties.test_conservation | Layer: TEST
THE CONSERVATION LAW: every field in, exactly one result out, under a name the caller can
look up.

This is the governance-loss class. `match_schema` returns a dict, and a dict silently
absorbs a collision: two distinct columns landing on one key leaves the caller a shorter
result set, no exception, and one column that inherits no protection level at all -- in a
library whose entire purpose is making a column inherit one. That is NM-0005, and its only
visible symptom was a count nobody had reason to check.

`tests/unit/application/test_match_result_identity.py` pins the one pair of names that
caused it. This file asserts the law itself over generated names, because the flattening
rewrite that produced that collision (`_` -> `.`, worth +19.3 P@1 and therefore staying)
maps a whole FAMILY of legal column names onto shared paths. Three examples the parser
still collapses today, all found by generation rather than by hand:

    "a.b"  "a_b"  "a__b"   -> full_path "a.b"      three columns, one path
    ".a"   "a."             -> full_path "a"        two columns, one path

The law has three parts, and each is the negation of a way a column has actually been lost:

  COUNT      len(results) == len(fields)          -- nothing evaporated
  IDENTITY   results[i] describes fields[i]       -- nothing got another field's matches
  ADDRESS    the key is the caller's own name     -- nothing became unreachable
"""

from __future__ import annotations

import re

from hypothesis import assume, given
from hypothesis import strategies as st

from nexus_matcher.domain.models.entities import SchemaField
from nexus_matcher.shared.types.base import DataType
from tests.properties._support import PROPERTY_SETTINGS, build_matcher, glossary

# Names chosen because each one has broken something, here or in a system like it.
#
# The separator family is the NM-0005 mechanism: the flattener joins path segments with
# "_" and marks array boundaries with "__", so "a.b", "a_b" and "a__b" are three legal,
# distinct columns of one record that reconstruct to a single dotted path.
ADVERSARIAL_NAMES: tuple[str, ...] = (
    # differ only by separator -- the collision that cost a column its classification
    "a.b",
    "a_b",
    "a__b",
    "a-b",
    "a b",
    "cust.addr.city",
    "cust_addr_city",
    "cust__addr__city",
    # leading and trailing dots: ".a" and "a." both reconstruct to "a"
    ".a",
    "a.",
    ".",
    "..",
    "a..b",
    # empty and whitespace -- full_path falls back to the name, so both keys are ""
    "",
    " ",
    "\t",
    # unicode, including a zero-width space that makes two names look identical
    "café",
    "naïve",
    "Ünïcødé",
    "日本語",
    "customer\u200b_id",
    "customer_id",
    # case-only differences
    "CUSTOMER_ID",
    "CustomerId",
    # the annotations real exports carry
    "[deprecated] customer_id",
    "[deprecated] customer.id",
    # 255 characters, and 255 characters that differ in the last one
    "A" * 255,
    "A" * 254 + "B",
    # names that look like the synthetic suffix `_unique_result_key` hands out, so a
    # generated name cannot be mistaken for a disambiguated one
    "#2",
    "x#2",
    "x",
)


def _flattened_field(name: str) -> SchemaField:
    """
    A field shaped the way `FlattenedAvroParser` shapes one.

    `full_path` is the LOSSY reconstruction -- separators collapsed to dots -- and
    `flattened_name` is the caller's exact string. Building fields this way is what
    reproduces the collision: several names, one path.
    """
    path = name.replace("__", ".").replace("_", ".").replace("-", ".").replace(" ", ".")
    parent, _, leaf = path.rpartition(".")
    return SchemaField(
        name=leaf or path,
        data_type=DataType.STRING,
        full_path=path,
        parent_path=parent,
        source_metadata={"flattened_name": name},
    )


def _plain_field(name: str) -> SchemaField:
    """A field from a parser that has no flattened name: raw Avro, JSON Schema, SQL DDL."""
    return SchemaField(name=name, data_type=DataType.STRING)


field_shape = st.sampled_from((_flattened_field, _plain_field))
adversarial_names = st.lists(st.sampled_from(ADVERSARIAL_NAMES), min_size=1, max_size=12)


@PROPERTY_SETTINGS
@given(names=adversarial_names, shapes=st.data(), entries=glossary())
def test_every_field_produces_exactly_one_result(names, shapes, entries):
    """
    COUNT and IDENTITY, together, over generated names.

    The two halves have to be asserted together. A count that holds while the mapping is
    wrong is the more expensive failure of the two: the caller sees the right number of
    rows and one of them carries another column's matches, so a column is not missing a
    classification, it has acquired somebody else's.

    `is` rather than `==`: SchemaField is a frozen dataclass, so two distinct fields with
    the same name compare EQUAL, and an equality check here would pass while the results
    for a duplicated column were served from the wrong one of the pair. Identity is what
    the property actually means.
    """
    fields = [shapes.draw(field_shape)(name) for name in names]
    results = build_matcher(entries)._match_fields(fields)

    assert len(results) == len(fields), (
        f"{len(fields)} fields in, {len(results)} results out. A field with no entry in "
        f"the result dict inherits no protection level and nothing raised. "
        f"Names: {[f.source_metadata.get('flattened_name', f.name) for f in fields]!r}"
    )
    assert len(set(results)) == len(fields), "two fields share one output key"

    for (key, matches), field in zip(results.items(), fields, strict=True):
        for match in matches:
            assert match.schema_field is field, (
                f"key {key!r} carries results computed for a DIFFERENT field: "
                f"{match.schema_field.full_path!r} not {field.full_path!r}"
            )


@PROPERTY_SETTINGS
@given(names=adversarial_names, shapes=st.data(), entries=glossary())
def test_every_key_is_the_callers_own_name(names, shapes, entries):
    """
    ADDRESS. A result the caller cannot look up is no better than a missing one.

    The rewrite that recovers the parent path means a caller asking for their own column
    name once got a KeyError from a result set that did contain their field. So every key
    must be either the caller's exact string or that string plus the visibly synthetic
    `#n` suffix that a genuine duplicate takes -- never a path we invented.

    The expected key is the GENERATED NAME, not `field_result_key(field)`. Deriving it
    from the function under test is the H-004 failure exactly: an oracle that shares the
    error it is looking for. Written that way, this test stayed green while
    `field_result_key` was reverted to `full_path` -- the NM-0005 defect itself -- because
    both sides moved together. Both field shapes here are constructed so the caller's own
    string IS the answer: a flattened field carries it in `flattened_name`, and a plain
    field's `full_path` falls back to its name.
    """
    fields = [shapes.draw(field_shape)(name) for name in names]
    results = build_matcher(entries)._match_fields(fields)

    for key, name in zip(results, names, strict=True):
        assert key == name or re.fullmatch(re.escape(name) + r"#[0-9]+", key), (
            f"key {key!r} is neither the caller's name {name!r} nor a `#n` disambiguation of it"
        )


@PROPERTY_SETTINGS
@given(name=st.sampled_from(ADVERSARIAL_NAMES), copies=st.integers(2, 6), entries=glossary())
def test_repeated_columns_all_survive(name, copies, entries):
    """
    The same column listed N times yields N results, not one.

    A flattened export really does repeat columns, and the first version of the key rule
    let the last occurrence overwrite every earlier one. The suffixes are deliberately
    ugly: `#` occurs in neither an Avro name nor a dotted path, so a suffixed key reads as
    synthetic rather than as a column the source actually had.

    The expected keys are pinned to the generated name rather than read back out of
    `field_result_key`, for the reason in `test_every_key_is_the_callers_own_name`.
    """
    fields = [_flattened_field(name) for _ in range(copies)]
    results = build_matcher(entries)._match_fields(fields)

    assert len(results) == copies
    assert list(results) == [name, *(f"{name}#{n}" for n in range(2, copies + 1))]


@PROPERTY_SETTINGS
@given(
    names=st.lists(st.sampled_from(ADVERSARIAL_NAMES), min_size=1, max_size=12, unique=True),
    entries=glossary(),
)
def test_public_match_schema_conserves_every_parsed_column(names, entries):
    """
    The same law through the PUBLIC entry point, parser included.

    `_match_fields` is where the key is chosen, but a caller reaches it through
    `match_schema(source, "flattened_avro")`, and the parser is what manufactures the
    colliding paths in the first place. Asserting only against hand-built SchemaFields
    would leave the two halves untested together -- the cross-lane gap that has bitten
    this repo three times.

    Scope note: the parser drops a column whose name is empty, so the count is asserted
    against the fields the parser PRODUCED. That drop is the parser's decision and is
    recorded here rather than smuggled into a passing assertion; conservation is claimed
    from the matcher's input onwards.
    """
    parsable = [n for n in names if n.strip()]
    assume(parsable)

    source = {name: {"dataType": "STRING"} for name in parsable}
    matcher = build_matcher(entries)
    schema = matcher._parse_schema(source, "flattened_avro")
    results = matcher.match_schema(source, schema_format="flattened_avro")

    assert len(results) == len(schema.fields), (
        f"{len(schema.fields)} parsed fields, {len(results)} results: {sorted(set(results))!r}"
    )
    assert set(results) == {f.source_metadata["flattened_name"] for f in schema.fields}
