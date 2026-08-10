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

## The second dimension: WORK DONE

"Incremental == full rebuild" is, on its own, an invariant that a full rebuild satisfies
perfectly. Replacing `sync` with a re-embed of every row on every call survived this
machine in its first form, and it had to: the final state of a correct full rebuild IS the
final state the invariant demands. Correct, and useless -- a glossary sync that re-embeds
100k rows to apply one edit costs about three minutes instead of milliseconds, which is
the whole reason `sync` exists.

So every sync here also asserts WHAT WAS ENCODED, against the texts of the rows whose
embedded text actually moved. The expectation is built from `to_searchable_text` -- the
text `sync` really hands the provider -- and never from `content_hash`, because `sync`
decides with `content_hash` and an oracle that re-derives with the same hash is blind to
any change in what that hash covers. That blindness is H-004, and it is what let a
`content_hash` widened to cover `protection_level` pass this file unnoticed. See
`test_incremental_work.py` for the absolute, hand-written pins of the same contract.

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

from collections import Counter

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from nexus_matcher.application import ingest
from tests.properties._support import BagOfTokensProvider, CountingProvider

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
        # Two providers, on purpose. The index gets the counting one so that everything it
        # encodes is attributable to `sync`; the rebuild oracle gets a plain one, because a
        # rebuild legitimately encodes every row and would otherwise drown the count. Both
        # wrap the same `encode`, so the vectors stay bit-comparable.
        self.provider = CountingProvider()
        self.rebuild_provider = BagOfTokensProvider()
        self.rows: list[dict] = [_row("Customer Email", "cust_email", "the email", "PII")]
        self.index = ingest.build_index(list(self.rows), provider=self.provider, columns=COLUMNS)
        # id -> the text last embedded for it. The model of what the index's vectors were
        # computed from, and the only basis on which "this sync owed N embeddings" can be
        # stated without asking `content_hash`, which is the thing under test.
        self.embedded: dict[str, str] = {}
        self._record_work_done()

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

    # -- work accounting ---------------------------------------------------------------

    def _current_entries(self):
        """The entries `sync` is about to see, loaded exactly the way `sync` loads them."""
        return ingest.load_entries(list(self.rows), columns=COLUMNS)

    def _record_work_done(self) -> None:
        """Reset the model to the state a correct sync has just left the index in."""
        self.provider.take()
        self.embedded = {e.id: e.to_searchable_text() for e in self._current_entries()}

    def _sync_and_check_work(self):
        """
        One sync, plus the assertion the final-state invariant structurally cannot make.

        `owed` is the set of texts whose EMBEDDED FORM moved since the last sync -- derived
        from `to_searchable_text`, which is the string the provider is actually handed, and
        never from `content_hash`. That distinction is the whole point: `sync` decides what
        to recompute using `content_hash`, so an expectation re-derived from `content_hash`
        agrees with `sync` by construction and cannot notice the hash growing to cover a
        governance column. It did not notice, which is how a widened hash shipped past this
        file.
        """
        entries = self._current_entries()
        owed = sorted(
            entry.to_searchable_text()
            for entry in entries
            if self.embedded.get(entry.id) != entry.to_searchable_text()
        )
        assert self.provider.take() == [], "something other than `sync` encoded text"

        report = ingest.sync(self.index, list(self.rows), columns=COLUMNS)
        done = sorted(self.provider.take())

        assert done == owed, (
            f"`sync` encoded {len(done)} texts for {len(owed)} changed rows. "
            f"Encoded but not owed: {_multiset_difference(done, owed)!r}; "
            f"owed but not encoded: {_multiset_difference(owed, done)!r}. "
            f"Encoding a row whose text did not move is the full-rebuild cost `sync` "
            f"exists to avoid -- minutes instead of milliseconds on every daily sync; "
            f"failing to encode one that did move leaves a stale vector behind an entry "
            f"that looks freshly synced."
        )
        assert report.embedded == len(done), (
            f"the report says {report.embedded} embeddings, {len(done)} were computed. A "
            f"caller reading this number to decide the sync was cheap cannot see a sync "
            f"that quietly rebuilt."
        )

        self.embedded = {entry.id: entry.to_searchable_text() for entry in entries}
        return report

    # -- the sync itself ---------------------------------------------------------------

    @rule()
    def sync_and_compare(self):
        report = self._sync_and_check_work()
        rebuilt = ingest.build_index(
            list(self.rows), provider=self.rebuild_provider, columns=COLUMNS
        )
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
        """One last sync, so a run that ended on an edit still gets compared -- and
        counted, since a run whose final step was an edit is exactly where an over-eager
        re-embed would otherwise go unobserved."""
        self._sync_and_check_work()
        rebuilt = ingest.build_index(
            list(self.rows), provider=self.rebuild_provider, columns=COLUMNS
        )
        _assert_indistinguishable(self.index, rebuilt)


def _multiset_difference(a: list[str], b: list[str]) -> list[str]:
    """`a` minus `b`, counting duplicates -- two rows can legitimately share a text."""
    return sorted((Counter(a) - Counter(b)).elements())


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
