"""
tests.properties.test_incremental_work | Layer: TEST
What `sync` is allowed to RECOMPUTE -- pinned absolutely, never against its own hash.

## The hole this file exists to close

`test_sync_state_machine` asserts that an incremental index equals a full rebuild. That is
a real invariant and it is blind to the two failures that matter most here, for the same
reason in both cases: **a wrong amount of work produces a right answer.**

  * Widening `content_hash` to cover `protection_level` -- so that every governance edit
    forces a re-embed -- survived the entire property suite. `sync` decides with
    `content_hash` and the state machine's rebuild oracle re-derives with `content_hash`,
    so both sides moved together and the change was invisible. That is **H-004 (a
    differential test blind to the error both sides share) recurring inside the code
    written to prevent H-004**, and the remedy is the one H-004 already prescribes: stop
    comparing an implementation against itself and pin absolute, hand-verified behaviour.
  * Replacing `sync` with a full re-embed of every row also survived, because the FINAL
    STATE of a correct full rebuild is exactly what the invariant demands.

So nothing here is derived from `content_hash`, and nothing here is asserted about the
resulting index alone. The assertions are over the LITERAL TEXTS the embedding provider
was handed, compared against strings written out by hand in this file.

## The contract, and what it costs when it breaks

`content_hash` covers the EMBEDDED TEXT ONLY -- see the module docstring of
`nexus_matcher.application.ingest`. Editing a governance or audit column must NOT
invalidate a vector. If it does, the first time anybody touches a `Classification` or a
`last_reviewed_by` column the daily sync stops being incremental and becomes a full
re-embed of the whole glossary: at ~600 texts/sec that is milliseconds turning into about
three minutes per 100k entries, which is the entire cost `sync()` exists to avoid.

Work is counted as TEXTS ENCODED, never as elapsed time -- see `_support.CountingProvider`.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nexus_matcher.application import ingest
from nexus_matcher.domain.models.entities import DictionaryEntry
from nexus_matcher.shared.types.base import DataType, ProtectionLevel
from tests.properties._support import CountingProvider, encode

# The same explicit mapping `test_sync_state_machine` pins, and for the same reason: an
# INFERRED mapping over a dict source walks a set comprehension, so under a different
# PYTHONHASHSEED it can pick a different column. Duplicated rather than imported so this
# file states its own fixture completely.
COLUMNS = {
    "business_name": "Term",
    "logical_name": "Technical Name",
    "definition": "Business Definition",
    "protection_level": "Classification",
}


def rows() -> list[dict]:
    """Three unambiguous rows: distinct terms, distinct technical names, no id column."""
    return [
        {
            "Term": "Customer Email",
            "Technical Name": "cust_email",
            "Business Definition": "the email address of a customer",
            "Classification": "Internal",
        },
        {
            "Term": "Account Balance",
            "Technical Name": "acct_bal",
            "Business Definition": "current balance of the account",
            "Classification": "Internal",
        },
        {
            "Term": "Posting Date",
            "Technical Name": "post_dt",
            "Business Definition": "when the transaction posted",
            "Classification": "Internal",
        },
    ]


# HAND-WRITTEN, not computed. `DictionaryEntry.to_searchable_text()` joins business name,
# logical name with underscores turned into spaces, and definition, in that order. Writing
# the three strings out is what makes every assertion below independent of the code that
# produces them -- read `_embed_documents`'s argument list against this and you are reading
# an absolute expectation, not an oracle that shares whatever the implementation believes.
SEARCHABLE = (
    "Customer Email cust email the email address of a customer",
    "Account Balance acct bal current balance of the account",
    "Posting Date post dt when the transaction posted",
)


def _fresh() -> tuple[ingest.GlossaryIndex, CountingProvider, list[dict]]:
    """A built index with the initial build's work already drained off the counter."""
    source = rows()
    provider = CountingProvider()
    index = ingest.build_index(source, provider=provider, columns=COLUMNS)
    provider.take()
    return index, provider, source


def _id_of(index: ingest.GlossaryIndex, term: str) -> str:
    """The derived id for a row, read back rather than recomputed from the id rule."""
    matches = [e.id for e in index.entries.values() if e.business_name == term]
    assert len(matches) == 1, f"expected exactly one entry named {term!r}, got {matches!r}"
    return matches[0]


def _vector_bytes(index: ingest.GlossaryIndex, entry_id: str) -> bytes:
    return index.vectors[index.order.index(entry_id)].tobytes()


# =============================================================================
# WHAT GETS EMBEDDED AT ALL
# =============================================================================


def test_the_first_build_embeds_each_row_once_under_its_hand_written_text():
    """
    The premise every other test in this file rests on, asserted rather than assumed.

    If the embedded text were not these three strings, "nothing was re-embedded" below
    would be a statement about something other than the glossary, and the file would be
    reporting nothing while looking like coverage.
    """
    provider = CountingProvider()
    ingest.build_index(rows(), provider=provider, columns=COLUMNS)

    assert provider.take() == list(SEARCHABLE)


# =============================================================================
# THE CONTRACT: A GOVERNANCE EDIT MUST NOT INVALIDATE A VECTOR
# =============================================================================


def test_editing_only_the_classification_re_embeds_nothing():
    """
    THE gate the widened-`content_hash` mutation escaped.

    Zero texts encoded, asserted directly against the provider. `report.embedded` is
    asserted too, but it is the weaker of the two: it is computed from the same hash the
    decision was made with, so it moves whenever the hash's definition moves and can
    therefore never contradict it. The provider count can.

    The last two assertions are the non-vacuity half. Without them a `sync` that ignored
    its source entirely -- the cheapest possible way to encode nothing -- would pass. The
    edit must LAND (the entry now says RESTRICTED, and the raw governance string with it)
    while the vector stays byte-identical.
    """
    index, provider, source = _fresh()
    entry_id = _id_of(index, "Customer Email")
    before = _vector_bytes(index, entry_id)

    source[0]["Classification"] = "Restricted"
    report = ingest.sync(index, source, columns=COLUMNS)

    assert provider.take() == [], (
        "editing a governance column re-embedded rows. `content_hash` covers the embedded "
        "text only, so a classification edit must reuse every vector -- otherwise the "
        "first edit to any audit column turns a millisecond daily sync into a full "
        "re-embed of the whole glossary"
    )
    assert report.embedded == 0
    assert report.updated == []

    assert index.entries[entry_id].protection_level is ProtectionLevel.RESTRICTED, (
        "the classification edit was not applied, so encoding nothing proves nothing"
    )
    assert index.entries[entry_id].source_metadata["governance_raw"] == "Restricted"
    assert _vector_bytes(index, entry_id) == before


def test_reclassifying_every_row_at_once_still_re_embeds_nothing():
    """
    The shape of the real event: a governance review that re-labels the whole glossary.

    One row is the easy case; a bulk reclassification is where a hash that covers
    governance costs a full re-embed of every entry in one sync.
    """
    index, provider, source = _fresh()
    for row in source:
        row["Classification"] = "PII"

    report = ingest.sync(index, source, columns=COLUMNS)

    assert provider.take() == [], "a bulk reclassification re-embedded the entire glossary"
    assert report.embedded == 0
    assert {e.protection_level for e in index.entries.values()} == {ProtectionLevel.PII}


def test_editing_the_definition_re_embeds_exactly_that_row():
    """
    The other side of the contract: text that IS embedded must invalidate its vector.

    A test that only asserted "governance edits are free" would be satisfied by a `sync`
    that never re-embedded anything at all -- which is the worse defect of the two, because
    the index then answers from vectors that no longer describe the glossary.

    The expected text is written out by hand for the same reason `SEARCHABLE` is, and the
    stored vector is checked against `encode` of that literal string rather than against
    whatever `sync` chose to encode.
    """
    index, provider, source = _fresh()
    edited = _id_of(index, "Customer Email")
    untouched = _id_of(index, "Posting Date")
    untouched_before = _vector_bytes(index, untouched)

    source[0]["Business Definition"] = "the postal address of a customer"
    expected_text = "Customer Email cust email the postal address of a customer"
    report = ingest.sync(index, source, columns=COLUMNS)

    assert provider.take() == [expected_text], (
        "a definition edit must re-embed that row and only that row"
    )
    assert report.embedded == 1
    assert report.updated == [edited]
    assert _vector_bytes(index, edited) == encode(expected_text).tobytes(), (
        "the row was re-embedded but the index kept the old vector, so every search "
        "against it answers from text the glossary no longer contains"
    )
    assert _vector_bytes(index, untouched) == untouched_before


# =============================================================================
# THE WORK A MIXED SYNC IS ALLOWED TO DO
# =============================================================================


def test_a_mixed_edit_embeds_only_the_rows_whose_text_moved():
    """
    An add, a definition edit, a governance edit and a delete in one sync: 2 texts, not 4.

    This is the deterministic form of the state machine's new work assertion, and it is
    what a `sync` replaced by a full re-embed fails on: a full rebuild of the surviving
    three rows encodes three texts and lands in exactly the right final state. Only the
    count separates the two.
    """
    index, provider, source = _fresh()
    source[0]["Business Definition"] = "the postal address of a customer"
    source[1]["Classification"] = "Restricted"
    del source[2]
    source.append(
        {
            "Term": "Merchant Id",
            "Technical Name": "merch_id",
            "Business Definition": "identifier of the merchant",
            "Classification": "Internal",
        }
    )

    report = ingest.sync(index, source, columns=COLUMNS)

    assert sorted(provider.take()) == sorted(
        [
            "Customer Email cust email the postal address of a customer",
            "Merchant Id merch id identifier of the merchant",
        ]
    ), "sync encoded texts it already had vectors for -- it is no longer incremental"
    assert report.embedded == 2
    assert len(index.entries) == 3


def test_a_sync_with_no_edits_at_all_embeds_nothing():
    """
    The case a daily sync actually hits, and the one a full re-embed cannot fake.

    Same source in, same source out. Any encoding here is pure waste, and it is waste
    proportional to the size of the glossary.
    """
    index, provider, source = _fresh()

    report = ingest.sync(index, source, columns=COLUMNS)

    assert provider.take() == [], "a no-op sync re-embedded the glossary"
    assert report.embedded == 0
    assert report.unchanged == 3


# =============================================================================
# THE HASH ITSELF, STATED AS A CONTRACT RATHER THAN COMPARED TO ITSELF
# =============================================================================


def _entry(**overrides) -> DictionaryEntry:
    base = {
        "id": "e0",
        "business_name": "Customer Email",
        "logical_name": "cust_email",
        "definition": "the email address of a customer",
        "data_type": DataType.STRING,
    }
    return DictionaryEntry(**{**base, **overrides})


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("protection_level", ProtectionLevel.RESTRICTED),
        ("domain", "CUSTOMER"),
        ("data_type", DataType.LONG),
        ("parent_table", "customers"),
        ("source_metadata", {"last_reviewed_by": "someone else"}),
    ],
)
def test_a_field_that_is_not_embedded_does_not_move_the_content_hash(attribute, value):
    """
    Named, one attribute at a time, so a widening cannot hide behind a passing suite.

    Every attribute here is absent from `to_searchable_text`, so a vector computed before
    the change is still the right vector after it. A hash that moved would order a re-embed
    that cannot change the result -- the definition of wasted work.
    """
    before = _entry()
    after = replace(before, **{attribute: value})

    assert ingest.content_hash(before) == ingest.content_hash(after), (
        f"changing {attribute!r} moved the content hash, so every edit to it forces a "
        f"re-embed of that row for a vector that cannot differ"
    )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("business_name", "Customer Address"),
        ("logical_name", "cust_addr"),
        ("definition", "the postal address of a customer"),
    ],
)
def test_every_embedded_field_does_move_the_content_hash(attribute, value):
    """
    The complement, and the reason the test above is not simply "the hash never moves".

    These three are exactly what `to_searchable_text` joins. A hash blind to one of them
    leaves a stale vector in the index behind an entry that looks freshly synced.
    """
    before = _entry()
    after = replace(before, **{attribute: value})

    assert before.to_searchable_text() != after.to_searchable_text(), (
        "fixture error: this change was supposed to alter the embedded text"
    )
    assert ingest.content_hash(before) != ingest.content_hash(after), (
        f"changing {attribute!r} changes the text that gets embedded but not the hash, so "
        f"sync leaves the old vector in place and searches answer from stale text"
    )
