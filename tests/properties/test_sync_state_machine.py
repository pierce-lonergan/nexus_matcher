"""
tests.properties.test_sync_state_machine | Layer: TEST
An incremental index must be indistinguishable from one built from scratch.

`sync()` is the only place in this library that decides what NOT to recompute. Everything
else can be checked by looking at one call; this cannot, because its bugs are historical.
An entry re-embedded into the wrong row, a vector left behind after a delete, a hash
written for a row that was not embedded -- none of them raise, none of them show up in the
next report, and all of them leave an index that keeps answering queries from vectors that
no longer describe the glossary. The library's whole job is telling a column which
governance classification it inherits, and it will do that confidently from a stale vector.

So the invariant is the strongest one available: after ANY sequence of edits and syncs,
the incremental index must equal a full rebuild from the same source. Bit-identical
vectors, identical content hashes, identical entry objects.

## Why a state machine and not a table of cases

The bugs live in the INTERACTIONS. `sync` removes ids, then re-embeds, then appends new
rows, then rewrites `order` and filters `vectors` -- and the positional map it uses to
scatter re-embedded vectors is built before the append and consumed before the filter.
Every one of those steps is individually simple and jointly ordering-sensitive. A
delete-then-add in one sync exercises a different path from a delete, sync, add, sync, and
neither is the case anyone writes by hand.

The rules deliberately include the two edits that are easy to conflate:

  * `edit_definition` changes the EMBEDDED text, so the content hash moves and the row
    must be re-embedded;
  * `edit_classification` changes only governance, so the hash does NOT move and the row
    must NOT be re-embedded -- but the entry object still has to be refreshed. Refreshing
    entries only inside the re-embed branch meant a governance edit was applied or dropped
    depending on whether some unrelated row happened to change in the same sync.

`rename_term` and `rename_technical` change the derived id, so they arrive as an add plus
a removal rather than as an update -- the case that renumbering ids by row position used
to turn into a full rebuild.

## Scope note: there are no postings to compare

`GlossaryIndex` holds entries, vectors, order and hashes. It holds no sparse index, and
`sync()` never touches one: the BM25 postings live in `NexusMatcher._index_dictionary`,
which rebuilds them wholesale from a fresh entry list. So "postings identical" has nothing
to compare here, and a sparse retriever paired with a synced `GlossaryIndex` would simply
be stale. That is reported rather than papered over.
"""

from __future__ import annotations

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from nexus_matcher.application import ingest
from tests.properties._support import BagOfTokensProvider

# An explicit mapping rather than the inferred one.
#
# `read_source` derives the column list for a dict source with a SET comprehension, and
# `map_columns` then lets the last colliding normalised key win -- so an inferred mapping
# over a JSON/dict glossary can pick a different column under a different PYTHONHASHSEED.
# Pinning the mapping keeps this test measuring `sync`, not that. (The hash-seed
# sensitivity is real and is reported separately; it is not this file's subject.)
COLUMNS = {
    "business_name": "Term",
    "logical_name": "Technical Name",
    "definition": "Business Definition",
    "protection_level": "Classification",
}

# Small pools on purpose: collisions are the interesting inputs. Two rows sharing a term
# AND a technical name derive the same id and must be disambiguated by order of
# appearance, in the incremental path exactly as in the rebuild.
TERMS = st.sampled_from(["Customer Email", "Account Balance", "Posting Date", "Merchant Id"])
TECHNICAL = st.sampled_from(["cust_email", "acct_bal", "post_dt", "merch_id"])
DEFINITIONS = st.sampled_from(["", "the email of a customer", "current balance", "when posted"])
CLASSIFICATIONS = st.sampled_from(["PII", "Internal", "Restricted", "Non-Public", ""])
POSITION = st.integers(min_value=0, max_value=63)


def _row(term: str, technical: str, definition: str, classification: str) -> dict:
    return {
        "Term": term,
        "Technical Name": technical,
        "Business Definition": definition,
        "Classification": classification,
    }


class GlossarySyncMachine(RuleBasedStateMachine):
    """Edit a glossary any way you like; the index must never disagree with a rebuild."""

    def __init__(self) -> None:
        super().__init__()
        self.provider = BagOfTokensProvider()
        self.rows: list[dict] = [_row("Customer Email", "cust_email", "the email", "PII")]
        self.index = ingest.build_index(list(self.rows), provider=self.provider, columns=COLUMNS)

    # -- edits -------------------------------------------------------------------------

    @rule(term=TERMS, technical=TECHNICAL, definition=DEFINITIONS, classification=CLASSIFICATIONS)
    def add_row(self, term, technical, definition, classification):
        self.rows.append(_row(term, technical, definition, classification))

    @precondition(lambda self: self.rows)
    @rule(at=POSITION, definition=DEFINITIONS)
    def edit_definition(self, at, definition):
        """Moves the content hash: this row MUST be re-embedded."""
        self.rows[at % len(self.rows)]["Business Definition"] = definition

    @precondition(lambda self: self.rows)
    @rule(at=POSITION, classification=CLASSIFICATIONS)
    def edit_classification(self, at, classification):
        """
        Leaves the content hash alone: this row must NOT be re-embedded, and its entry
        object must still be refreshed. Getting that wrong is how an index reports a stale
        PII classification while looking perfectly healthy.
        """
        self.rows[at % len(self.rows)]["Classification"] = classification

    @precondition(lambda self: self.rows)
    @rule(at=POSITION, term=TERMS)
    def rename_term(self, at, term):
        """Changes the DERIVED id, so this arrives as an addition plus a removal."""
        self.rows[at % len(self.rows)]["Term"] = term

    @precondition(lambda self: self.rows)
    @rule(at=POSITION, technical=TECHNICAL)
    def rename_technical(self, at, technical):
        self.rows[at % len(self.rows)]["Technical Name"] = technical

    @precondition(lambda self: self.rows)
    @rule(at=POSITION)
    def delete_row(self, at):
        del self.rows[at % len(self.rows)]

    @precondition(lambda self: self.rows)
    @rule(at=POSITION)
    def duplicate_row(self, at):
        """Two rows with one derived id. The `-2` suffix must be assigned identically in
        both paths, or the incremental index and the rebuild disagree about identity."""
        self.rows.append(dict(self.rows[at % len(self.rows)]))

    @precondition(lambda self: len(self.rows) > 1)
    @rule(at=POSITION)
    def move_row(self, at):
        """Re-ordering the SOURCE, which renumbers nothing but does move suffix
        assignment. The rebuild sees the new order; the incremental index keeps its own
        `order` list and appends, so the two must be compared per id, never positionally."""
        index = at % len(self.rows)
        self.rows.insert(0, self.rows.pop(index))

    # -- the sync itself ---------------------------------------------------------------

    @rule()
    def sync_and_compare(self):
        report = ingest.sync(self.index, list(self.rows), columns=COLUMNS)
        rebuilt = ingest.build_index(list(self.rows), provider=self.provider, columns=COLUMNS)
        _assert_indistinguishable(self.index, rebuilt)
        _assert_report_adds_up(report, self.index)

    # -- always true, sync or no sync --------------------------------------------------

    @invariant()
    def order_vectors_and_entries_stay_aligned(self):
        """
        The alignment `sync` has to maintain by hand, checked after every single step.

        `order` is what maps a row of `vectors` to an id. If it grows a duplicate, or
        drifts out of step with the matrix, every lookup after the break silently returns
        another entry's vector -- and nothing raises, because the shapes still work.
        """
        index = self.index
        assert len(index.order) == len(set(index.order)), (
            f"`order` contains a duplicate id: {index.order}"
        )
        assert set(index.order) == set(index.entries), (
            f"`order` and `entries` disagree about which ids exist: "
            f"{set(index.order) ^ set(index.entries)}"
        )
        assert set(index.hashes) == set(index.entries), (
            f"`hashes` and `entries` disagree: {set(index.hashes) ^ set(index.entries)}"
        )
        assert index.vectors.shape[0] == len(index.order), (
            f"{index.vectors.shape[0]} vectors for {len(index.order)} ids -- every lookup "
            f"past the break returns another entry's vector"
        )
        assert index.vectors.dtype.name == "float32"

    def teardown(self):
        """One last sync, so a run that ended on an edit still gets compared."""
        ingest.sync(self.index, list(self.rows), columns=COLUMNS)
        rebuilt = ingest.build_index(list(self.rows), provider=self.provider, columns=COLUMNS)
        _assert_indistinguishable(self.index, rebuilt)


def _assert_indistinguishable(incremental, rebuilt) -> None:
    """
    Incremental == full rebuild, on everything a reader of the index can observe.

    Comparison is BY ID, never by position: the rebuild lists ids in source order while an
    incremental index appends new ones at the end, and both are correct. Vectors are
    compared as BYTES rather than with `==` or `allclose`, because this is the one place a
    tolerance would be indefensible -- `sync`'s entire contract is that it reuses a vector
    unchanged or recomputes it from the same text with the same encoder. Anything other
    than bit-identical means it did neither.
    """
    assert incremental.hashes == rebuilt.hashes, (
        f"content hashes diverged: {set(incremental.hashes.items()) ^ set(rebuilt.hashes.items())}"
    )
    assert set(incremental.order) == set(rebuilt.order)
    assert incremental.vectors.shape == rebuilt.vectors.shape

    incremental_vectors = dict(zip(incremental.order, incremental.vectors, strict=True))
    rebuilt_vectors = dict(zip(rebuilt.order, rebuilt.vectors, strict=True))
    for entry_id, vector in rebuilt_vectors.items():
        assert incremental_vectors[entry_id].tobytes() == vector.tobytes(), (
            f"vector for {entry_id!r} is not the one a rebuild produces -- searches "
            f"against this index answer from a stale embedding"
        )

    assert set(incremental.entries) == set(rebuilt.entries)
    for entry_id, entry in rebuilt.entries.items():
        assert incremental.entries[entry_id] == entry, (
            f"entry {entry_id!r} is stale. Incremental: "
            f"{incremental.entries[entry_id].protection_level} / "
            f"{incremental.entries[entry_id].definition!r}; rebuild: "
            f"{entry.protection_level} / {entry.definition!r}"
        )


def _assert_report_adds_up(report, index) -> None:
    """
    The report is what a caller decides on, so it has to describe the index it produced.

    `embedded` is the number the whole feature exists to keep small; a report that
    under-counts it hides a full rebuild, and one that over-counts it hides a sync that
    skipped work it owed.
    """
    assert report.embedded == len(report.added) + len(report.updated)
    assert len(index.entries) == report.unchanged + len(report.added) + len(report.updated)
    assert not (set(report.added) & set(report.removed))
    assert not (set(report.updated) & set(report.removed))
    for entry_id in report.removed:
        assert entry_id not in index.entries, f"{entry_id!r} reported removed, still indexed"
    for entry_id in (*report.added, *report.updated):
        assert entry_id in index.entries, f"{entry_id!r} reported synced, absent from the index"


GlossarySyncMachine.TestCase.settings = settings(
    max_examples=40,
    stateful_step_count=30,
    # See tests/properties/_support.py: hypothesis's deadline is a timing assertion, and
    # H-007 measured a 30.6% throughput band on identical code under load. Nothing here is
    # timed; everything here is exact.
    deadline=None,
)

TestGlossarySync = GlossarySyncMachine.TestCase
