"""
NM-0024 -- a vector went stale while sync reported the row unchanged.

`content_hash` decides what gets re-embedded. It hashed a hand-written list of three
fields; `DictionaryEntry.to_searchable_text()` embedded those three AND `synonyms`. So
editing an entry's synonyms changed the text that got encoded while leaving the hash
untouched: sync saw no change, skipped the row, and the stored vector silently stopped
matching the entry it belonged to.

Nothing errors in that state. The report says "unchanged", the index looks healthy, and
the entry just quietly stops matching what it claims to match.

Why it escaped: the property suite compared incremental sync against a full rebuild, and
BOTH SIDES used the same content_hash. A divergence in the hash is invisible to an oracle
built on the hash -- H-004, the hazard that says differential tests cannot see errors both
sides share.

The fix is structural rather than "add synonyms to the list": the hash is now derived FROM
the embedded text, so the two cannot drift. Two hand-maintained lists of "fields that
matter" are kept in step by discipline, and discipline is what failed here.
"""

from __future__ import annotations

import pytest

from nexus_matcher.application.ingest import content_hash
from nexus_matcher.domain.models.entities import DictionaryEntry
from nexus_matcher.shared.types.base import DataType, ProtectionLevel


def _entry(**overrides) -> DictionaryEntry:
    base = {
        "id": "d-1",
        "business_name": "Customer Email Address",
        "logical_name": "cust_email",
        "definition": "The email address used to contact a customer.",
        "data_type": DataType.STRING,
    }
    base.update(overrides)
    return DictionaryEntry(**base)


def test_the_hash_covers_everything_that_gets_embedded():
    """
    The invariant, stated directly: if two entries embed different text, they must hash
    differently. Anything else means a vector can go stale without sync noticing.

    Asserted over the embedded STRING rather than over a list of field names, so it stays
    true when somebody adds a field to to_searchable_text and forgets this file exists --
    which is exactly how the defect arose.
    """
    a = _entry(synonyms=frozenset({"mail"}))
    b = _entry(synonyms=frozenset({"mail", "electronic mail", "contact address"}))
    assert a.to_searchable_text() != b.to_searchable_text(), "fixture no longer varies the text"
    assert content_hash(a) != content_hash(b), (
        "two entries that embed DIFFERENT text share a content hash, so sync will skip "
        "one of them and leave its vector stale while reporting the row unchanged"
    )


@pytest.mark.parametrize(
    "field",
    ["business_name", "logical_name", "definition"],
)
def test_each_embedded_field_still_moves_the_hash(field):
    """Guards the opposite mistake: a hash so narrow it misses an obvious edit."""
    assert content_hash(_entry()) != content_hash(_entry(**{field: "something else"}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protection_level", ProtectionLevel.RESTRICTED),
        ("domain", "finance"),
        ("parent_table", "dbo.customer"),
        ("source_metadata", {"last_reviewed_by": "someone"}),
    ],
)
def test_audit_fields_still_do_not_move_the_hash(field, value):
    """
    The guarantee the narrow hash existed to provide, which the fix must not cost.

    If an audit column invalidated the vector, the first person to touch `last_reviewed_by`
    would turn every incremental sync into a full re-embed -- minutes instead of
    milliseconds on a large glossary, which is the entire reason sync() exists.
    """
    assert content_hash(_entry()) == content_hash(_entry(**{field: value}))
