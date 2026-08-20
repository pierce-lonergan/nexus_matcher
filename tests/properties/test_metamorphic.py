"""
tests.properties.test_metamorphic | Layer: TEST
Relations between two runs, when we cannot say what either run's answer should be.

There is no oracle for "which glossary entry does this column mean". There IS an oracle
for "these two runs must agree", and that is what a metamorphic relation buys: a
transformation of the input whose effect on the output is known exactly, even though the
output itself is not.

**Every relation below states its exact-vs-tolerance decision and the measurement behind
it.** Identities and ORDER are asserted exactly; SCORES carry a 1e-6 tolerance measured
from float32 reassociation (see `FLOAT32_REASSOCIATION`); and the sparse arm's scores are
deliberately not asserted at all, because appending documents changes IDF by construction.
Asserting score equality there would produce a gate that goes red on correct code, and a
gate that cries wolf is worse than no gate: it teaches people to re-run red until it
passes. Two of the four relations here were caught claiming more than was true, by a
six-seed soak and by hypothesis itself; both corrections are recorded where they happened.

## What these relations do NOT cover, measured

Written down because an unlisted hole is indistinguishable from a closed one, and because
both numbers below were produced by this file's own fixtures while it was being built:

  * Adding ONE glossary entry with zero token overlap and EXACTLY zero cosine to a column
    changes that column's top-1 match in **4% of cases with the sparse arm off (16/400)
    and 11% with it on**. Retrieval order is untouched -- see
    `TestIrrelevantAddition` -- so the flip comes entirely from `_fuse_results`
    min-max-normalizing over the candidate set: a new low scorer lowers the floor, every
    other candidate's normalized semantic score rises toward 1, the semantic signal loses
    resolution, and the remaining 30% of weight decides. Observed flipping a top-1 whose
    lead was 0.116 of confidence.
  * Appending a duplicate row of an existing entry changes which OTHER entry wins in
    **10% of cases with the sparse arm on (60/600)** and never with it off, because BM25
    document frequency is a corpus statistic.
  * **The dense tie-break is not stable under corpus growth.** `_top_k_indices` sorts with
    `np.argsort(-scores)` and selects with `np.argpartition`, neither of which is stable,
    so among EQUAL cosines the order is whatever the partitioning happened to produce and
    it re-shuffles when the corpus changes size. Found by
    `TestDuplicateInsertion` on clean code; reproduction and verified fix in this lane's
    report. `fuse_linear_ids` documents the tie-break as "dense retrieval order", and
    H-005's gate proves it does not depend on `PYTHONHASHSEED` -- true, and not the same
    thing as stable. With the measured margin at 0.0024, exact ties are the normal regime,
    and two tied entries can carry different protection levels.

All of these are corpus-coupling of the ANSWER, not of retrieval quality, and each means a
column's inherited protection level can change because of an edit to an unrelated row.
They are reported rather than asserted: this lane owns tests, not `_fuse_results` or the
vector store.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from nexus_matcher.application.use_cases.match_schema import _tokenize_identifier
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.ports.retrieval import SparseDocument
from nexus_matcher.domain.ports.vector_store import SearchResult
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
from nexus_matcher.shared.types.base import DataType
from tests.properties._support import (
    PROPERTY_SETTINGS,
    build_matcher,
    confidences,
    disjoint_entries,
    encode,
    glossary,
    ranked_ids,
    schema_fields,
    strictly_separated_ranks,
)

# Float32 reassociation tolerance.
#
# Every transformation these relations apply changes the SHAPE or the ROW ORDER of a
# matrix BLAS multiplies, and BLAS packs its panels accordingly, so the same dot products
# accumulate in a different order and land a few ULPs apart. The store's own docstring
# records the same effect between `search_batch` and the per-query loop. This is
# arithmetic, not a defect, and it is why none of the score comparisons here can be
# bit-exact -- a fact this file learned the hard way, having first shipped the glossary
# permutation relation as bit-exact and watched a six-seed soak redden it on the second
# seed at 7.1e-08.
#
# MEASURED, over hypothesis-drawn inputs:
#     glossary permutation   3.9e-08 over 42,320 confidence comparisons  (7.1e-08 worst
#                                                                         run observed)
#     field permutation      5.4e-08 over 42,320 confidence comparisons
#     corpus growth          3.0e-08 over 51,404 cosine comparisons
#
# 1e-6 is ~14x the largest movement ever seen here and orders of magnitude below any
# scoring change that could reorder a pair that was not already tied. RATCHET: it may be
# tightened, it may never be loosened.
#
# CORRECTION, 2026-08-20. The three figures above are UNOBSERVED MAXIMA, not measured
# ones, and the difference matters: hypothesis found a glossary permutation that moves a
# rank-1 confidence by 7.0e-02 -- 1.8 MILLION times the 3.9e-08 stated above. The cause is
# not float32 reassociation either. `fusedRetrieval` itself moved 0.90 -> 1.00, and 0.90 is
# exactly `fusion_alpha`: an arm's MIN-MAX POSITION changed with corpus order, which is a
# tie-break question, not an arithmetic one. This library already has a museum entry for
# order-dependent ranking (NM-0020, PYTHONHASHSEED), so the class is known.
#
# The tolerance is NOT loosened -- the ratchet holds and the property is right to be red.
# What is corrected is the claim: a search that did not find something is not a measurement
# that it does not exist. A later random search of 450 shuffles across 150 glossaries also
# found nothing, so the construction is narrow; "narrow" is not "absent".
#
# RESOLVED, 2026-08-20, as NM-0034 -- and the diagnosis above is itself half wrong, so it
# is left standing rather than quietly rewritten. The min-max POSITION did move, exactly as
# stated; what moved it was arithmetic after all, just not in the arm anyone was watching.
# BM25 floors a negative IDF at `epsilon * average_idf`, and the average was taken with
# `ndarray.sum()` over an array laid out in first-seen term order -- i.e. in GLOSSARY ROW
# ORDER. On the falsifying corpus the raw IDFs cancel exactly, so the true total is 0.0 and
# the pairwise total is +2.220446e-16 in one row order and -2.220446e-16 in the other. The
# floor changes SIGN, and `BM25Retriever.search` keeps a document only when its score is
# > 0, so the sparse arm returned three documents in one order and NOTHING in the other.
# Min-max is scale-free, so an arm made entirely of 1e-17 dust was stretched over the full
# [0, 1] range: rank 1's lexical component went 0.0 -> 1.0, its fused score 0.90 -> 1.00,
# and its confidence 0.88625 -> 0.95625. The repair is `math.fsum`, whose result is the
# correctly-rounded exact sum and therefore identical for every permutation of its input.
# `TestTheOrderDependentIdfFloor` below pins the reduced example, because the hypothesis
# database that found it is gitignored and does not travel.
#
# THE TOLERANCE IS NOT TIGHTENED EITHER, and that is a measurement rather than caution.
# What sets the floor here is the dense arm's float32 GEMM, which NM-0034 does not touch.
# Soaked after the fix, two 20,000-example runs at different seeds:
#     seed 1   238,073 confidence comparisons   worst 6.389e-08
#     seed 7   234,531 confidence comparisons   worst 8.198e-08
# So the band is if anything WIDER than the 7.1e-08 that forced this constant in the first
# place -- 40,000 examples see more of it than six seeds did. 1e-6 is ~12x the largest
# movement now observed, which is the same margin it was chosen with. It stays. Anyone who
# can bound the GEMM band tighter may tighten it; it may never be loosened.
FLOAT32_REASSOCIATION = 1e-6


def _assert_order_matches_where_separated(ids_before, scores_before, ids_after, message):
    """
    Compare two orderings only at the positions the scores alone decide.

    A position sitting inside a run of equal scores is placed by the tie-break, and this
    codebase's dense tie-break is an unstable numpy sort whose output legitimately changes
    when the array changes length. Comparing those positions would make this test red on
    correct code roughly whenever a query ties against several entries -- which, at the
    0.0024 margin H-005 measured, is often.
    """
    assert len(ids_before) == len(ids_after), message
    for position in strictly_separated_ranks(
        scores_before, tolerance=FLOAT32_REASSOCIATION, truncated=False
    ):
        assert ids_before[position] == ids_after[position], (
            f"{message}: position {position} has a unique score "
            f"{scores_before[position]!r} yet moved from {ids_before[position]!r} to "
            f"{ids_after[position]!r}"
        )


def _by_field(results):
    """
    Index results by the FIELD OBJECT rather than by the result key.

    Two fields can legitimately share a key -- the same column listed twice takes a `#2`
    suffix, and which of the pair gets the suffix is positional. Comparing by key across a
    permutation would then report a difference that is the documented suffix rule doing
    its job, not a ranking change. `schema_field` is the field the result was computed
    for, so it is the honest join key.
    """
    indexed = {}
    for matches in results.values():
        assert matches, "a field matched nothing against a non-empty glossary"
        indexed[id(matches[0].schema_field)] = matches
    return indexed


class TestPermutationInvariance:
    """
    Re-ordering the inputs must not re-order the answer.

    H-005 measured a 0.0024 margin between the gold entry and the nearest wrong one, and
    ranking depended on `PYTHONHASHSEED` until the deterministic tie-break landed. At that
    separation any order that leaks into ranking changes which glossary entry a column
    matches, and therefore which protection level it inherits.
    """

    @PROPERTY_SETTINGS
    @given(entries=glossary(), fields=schema_fields(), seed=st.integers(0, 2**32 - 1))
    def test_shuffling_the_glossary_leaves_every_score_where_it_was(self, entries, fields, seed):
        """
        TOLERANCE 1e-6, and the tolerance was earned rather than assumed.

        This was written asserting bit-identity, on the reasoning that a cosine is a dot
        product against one row and BM25 weights are order-independent corpus statistics,
        so the multiset of confidences must be invariant. That reasoning is right about the
        MATHS and wrong about the arithmetic: permuting the corpus changes the row order of
        the matrix BLAS packs into its panels, so the same dot product accumulates in a
        different order. A six-seed soak found it in the second seed -- confidences
        0.38547333158065733 against 0.3854732606036857, a difference of 7.1e-08.

        Had it shipped bit-exact it would have reddened on correct code roughly one run in
        six, which is the definition of the gate this package refuses to write. Measured
        maximum across 42,320 confidence comparisons drawn by hypothesis: 3.9e-08, and
        7.1e-08 for the run that caught it -- but see the CORRECTION above the constant:
        a 7.0e-02 movement was later found, so those are unobserved maxima rather than
        measured ones. The tolerance is 1e-6, ~14x the largest
        movement ever observed here and far below any scoring change that could reorder a
        pair that was not already tied.
        """
        shuffled = list(entries)
        np.random.default_rng(seed).shuffle(shuffled)

        before = _by_field(build_matcher(entries)._match_fields(fields))
        after = _by_field(build_matcher(shuffled)._match_fields(fields))

        assert set(before) == set(after)
        for key in before:
            a, b = confidences(before[key]), confidences(after[key])
            assert len(a) == len(b)
            for rank, (x, y) in enumerate(zip(a, b, strict=True)):
                assert abs(x - y) <= FLOAT32_REASSOCIATION, (
                    f"rank {rank} confidence moved {abs(x - y):.3e} when the glossary was "
                    f"merely re-ordered -- far beyond float32 reassociation, so a score "
                    f"depends on where an entry sits in the corpus"
                )

    @PROPERTY_SETTINGS
    @given(entries=glossary(), fields=schema_fields(), seed=st.integers(0, 2**32 - 1))
    def test_shuffling_the_glossary_leaves_every_untied_rank_identical(self, entries, fields, seed):
        """
        EXACT ids, at every rank the scores alone determine by more than float32 noise.

        Ranks held by a UNIQUE confidence are asserted; tied ranks are not, and that is the
        property rather than a softening of it. Equal-scoring candidates are ordered by the
        dense tie-break, which is an unstable numpy sort over a matrix whose rows ARE the
        glossary in order -- so a shuffle is entitled to swap two entries that scored
        identically. This fires readily rather than theoretically: a query whose tokens
        appear in no entry gives every candidate the same confidence, which is a normal
        case here.

        `strictly_separated_ranks` explains why a rank bounded above and below cannot be
        shared by any candidate, including the ones the top-k truncation hid. The
        separation must exceed the reassociation tolerance for the same reason the sibling
        test carries one.
        """
        shuffled = list(entries)
        np.random.default_rng(seed).shuffle(shuffled)

        before = _by_field(build_matcher(entries)._match_fields(fields))
        after = _by_field(build_matcher(shuffled)._match_fields(fields))

        for key in before:
            scores = confidences(before[key])
            ids_before, ids_after = ranked_ids(before[key]), ranked_ids(after[key])
            for rank in strictly_separated_ranks(scores, tolerance=FLOAT32_REASSOCIATION):
                assert ids_before[rank] == ids_after[rank], (
                    f"rank {rank} has a unique confidence {scores[rank]!r} yet changed "
                    f"from {ids_before[rank]!r} to {ids_after[rank]!r} when only the "
                    f"glossary ORDER changed"
                )

    @PROPERTY_SETTINGS
    @given(entries=glossary(), fields=schema_fields(min_size=2), seed=st.integers(0, 2**32 - 1))
    def test_shuffling_the_fields_changes_nothing_a_caller_can_see(self, entries, fields, seed):
        """
        TOLERANCE 1e-6 on scores, EXACT on ranks. The tolerance is not a concession to
        sloppiness, it is the batched retrieval path being honest about float32.

        Fields are encoded and retrieved in ONE batched call, and `search_batch` chunks
        the query block. Permuting the fields changes chunk membership, so BLAS sums the
        same products in a different order and a cosine moves by a few ULPs. Measured max
        6.7e-08 across 4118 result windows; the tolerance is 1e-6.

        Ranks are still asserted EXACTLY wherever the confidences either side of a rank
        differ by more than that tolerance -- which is to say, wherever a rank is not
        decided by a difference smaller than the arithmetic's own noise. Asserting ranks
        unconditionally would make this test flake on genuine ties, and asserting nothing
        would let a scatter-back off-by-one through.
        """
        permuted = list(fields)
        np.random.default_rng(seed).shuffle(permuted)

        matcher = build_matcher(entries)
        before = _by_field(matcher._match_fields(fields))
        after = _by_field(matcher._match_fields(permuted))

        assert set(before) == set(after), "a field disappeared when the fields were re-ordered"
        for key in before:
            a, b = confidences(before[key]), confidences(after[key])
            assert len(a) == len(b)
            for rank, (x, y) in enumerate(zip(a, b, strict=True)):
                assert abs(x - y) <= FLOAT32_REASSOCIATION, (
                    f"rank {rank} confidence moved {abs(x - y):.3e} on field order alone, "
                    f"far beyond float32 reassociation"
                )
            ids_before, ids_after = ranked_ids(before[key]), ranked_ids(after[key])
            for rank in range(len(a) - 1):
                above_clear = rank == 0 or a[rank - 1] - a[rank] > FLOAT32_REASSOCIATION
                if above_clear and a[rank] - a[rank + 1] > FLOAT32_REASSOCIATION:
                    assert ids_before[rank] == ids_after[rank], (
                        f"rank {rank} is decided by a clear {a[rank] - a[rank + 1]:.3e} "
                        f"margin yet changed with field order: {ids_before[rank]!r} -> "
                        f"{ids_after[rank]!r}"
                    )


class TestTheOrderDependentIdfFloor:
    """
    NM-0034, pinned as a literal so it survives without the hypothesis database.

    The example below is hypothesis's own shrink of the counterexample that reddened
    `test_shuffling_the_glossary_leaves_every_score_where_it_was`: four entries, one field,
    and `numpy.random.default_rng(0)` for the permutation. On the broken code
    `customer.customer` came back at confidence 0.8862500000000001 when the glossary was
    listed e0 e1 e2 e3 and 0.95625 when it was listed e2 e0 e1 e3 -- 0.07 of published
    confidence, from re-ordering rows.

    THIS IS PINNED RATHER THAN LEFT TO THE PROPERTY BECAUSE THE PROPERTY DOES NOT
    RELIABLY FIND IT. A random search of 450 shuffles across 150 glossaries found nothing;
    hypothesis reached it by construction, and a fresh 8-run sweep of the property against
    the BROKEN code went 0/8 because `.hypothesis/examples/` is gitignored and a fresh
    clone inherits no counterexample. A property that only fails on the machine that
    already knows the answer is not a gate anywhere else, so the answer is written down
    here.

    Both levels are asserted, for the same reason NM-0020 asserts both: the confidence is
    the symptom a caller meets, and the BM25 output is where the order actually leaked, so
    a future regression is reported against the stage that caused it.
    """

    ENTRIES = (
        ("e0", "Customer", "account", ""),
        ("e1", "Address", "balance", "email"),
        ("e2", "Customer Address Account", "email", ""),
        ("e3", "Transaction Identifier Opened", "account_balance", "customer email address amount"),
    )
    PERMUTATION_SEED = 0

    @classmethod
    def _entries(cls) -> list[DictionaryEntry]:
        return [
            DictionaryEntry(
                id=entry_id,
                business_name=business_name,
                logical_name=logical_name,
                definition=definition,
                data_type=DataType.STRING,
                domain="CUSTOMER",
            )
            for entry_id, business_name, logical_name, definition in cls.ENTRIES
        ]

    @staticmethod
    def _field() -> SchemaField:
        return SchemaField(
            name="customer",
            data_type=DataType.STRING,
            full_path="customer.customer",
            parent_path="customer",
            description="",
        )

    @classmethod
    def _shuffled(cls) -> list[DictionaryEntry]:
        shuffled = cls._entries()
        np.random.default_rng(cls.PERMUTATION_SEED).shuffle(shuffled)
        return shuffled

    def test_the_permutation_really_is_a_permutation(self):
        """
        Guards the guard.

        `default_rng(0).shuffle` on four elements is entitled to return them unchanged, and
        a future numpy could change the stream. If it ever does, every assertion below
        would hold by comparing a run against itself.
        """
        before = [e.id for e in self._entries()]
        after = [e.id for e in self._shuffled()]
        assert sorted(before) == sorted(after)
        assert before != after, (
            f"the permutation seed no longer permutes anything ({after}), so this class "
            f"compares a run against itself and proves nothing"
        )

    def test_the_corpus_still_sits_on_the_knife_edge_that_caused_it(self):
        """
        Guards the guard, at the cause.

        The sign flip needed raw IDFs that cancel EXACTLY: five terms at -0.847298 and four
        at +0.847298, total 0.0, which a pairwise float64 reduction returns as
        +/-2.220446e-16 depending on the row order it walked. Edit any of the four
        definitions above and that cancellation is gone -- the floor becomes an ordinary
        number, both orders agree for a reason that has nothing to do with the fix, and
        this class goes on printing PASS while covering nothing.
        """
        retriever = BM25Retriever()
        documents = [
            SparseDocument(id=entry.id, text=entry.to_searchable_text())
            for entry in self._entries()
        ]
        assert retriever.index(documents).is_success

        n_docs = len(documents)
        tokenized = [retriever._tokenize(doc.text) for doc in documents]
        raw = [
            math.log(n_docs - df + 0.5) - math.log(df + 0.5)
            for df in (
                sum(1 for tokens in tokenized if term in tokens) for term in retriever._vocab
            )
        ]

        assert any(value < 0 for value in raw), (
            "no term appears in more than half this corpus, so the epsilon floor is never "
            "applied and the defect cannot be reproduced by this fixture"
        )
        assert math.fsum(raw) == 0.0, (
            f"the raw IDFs no longer cancel (exact total {math.fsum(raw)!r}), so the "
            f"average is no longer a quantity whose SIGN a reduction order can decide"
        )

    def test_bm25_returns_the_same_documents_whatever_order_they_were_indexed_in(self):
        """
        EXACT, ids and scores both, and exact is the right assertion here: a BM25 score is
        a function of the corpus as a SET, and every input to it -- document frequency,
        average document length, the epsilon floor -- is a statistic of that set.

        This is the assertion that fails first when the floor goes back to an
        order-dependent reduction, and it fails at full strength rather than as a rounding
        difference: the arm returns three documents in one order and none in the other.
        """
        query = "customer customer"
        runs = []
        for order in (self._entries(), self._shuffled()):
            retriever = BM25Retriever()
            assert retriever.index(
                [SparseDocument(id=e.id, text=e.to_searchable_text()) for e in order]
            ).is_success
            found = retriever.search(query, top_k=100)
            assert found.is_success
            runs.append([(hit.id, hit.score) for hit in found.unwrap()])

        assert runs[0] == runs[1], (
            f"BM25 returned {runs[0]} for one glossary order and {runs[1]} for a "
            f"permutation of the same rows, so a lexical score depends on where an entry "
            f"sits in the corpus rather than on what it says"
        )

    def test_the_published_confidence_does_not_move_when_the_rows_are_re_ordered(self):
        """
        The symptom a caller meets, asserted at the tolerance the property itself uses.

        1e-6 rather than bit-identity for the reason `FLOAT32_REASSOCIATION` exists: the
        dense arm's cosines come out of a float32 GEMM whose panels are packed from the
        corpus rows, so permuting the rows is entitled to move a confidence by a few ULPs.
        The measured band is 8.2e-08 over two 20,000-example soaks; the defect moved this
        confidence by 7.0e-02, a million times that. There is no tolerance question here,
        only a defect question.

        ONE `SchemaField` INSTANCE, reused. `_by_field` joins on the field OBJECT, which
        is the honest key across a permutation -- see its docstring -- so handing the two
        runs two equal-but-distinct fields would make the two result maps share no key and
        the loop below iterate zero times.
        """
        field = self._field()
        before = _by_field(build_matcher(self._entries())._match_fields([field]))
        after = _by_field(build_matcher(self._shuffled())._match_fields([field]))

        assert set(before) == set(after)
        assert before, "the matcher returned nothing, so every comparison below is vacuous"
        for key in before:
            ranked_before, ranked_after = ranked_ids(before[key]), ranked_ids(after[key])
            scores_before, scores_after = confidences(before[key]), confidences(after[key])
            assert ranked_before == ranked_after, (
                f"re-ordering the glossary rows changed the ranking from {ranked_before} "
                f"to {ranked_after}"
            )
            for rank, (was, now) in enumerate(zip(scores_before, scores_after, strict=True)):
                assert abs(was - now) <= FLOAT32_REASSOCIATION, (
                    f"rank {rank} confidence moved {abs(was - now):.3e} on row order "
                    f"alone: {was!r} -> {now!r}. This is NM-0034; the sparse arm's "
                    f"epsilon IDF floor is being derived from an order-dependent sum "
                    f"again."
                )

    def test_the_rank_one_confidence_is_the_one_the_defect_moved(self):
        """
        Names the number, so a future reader can check the story rather than trust it.

        0.88625 is what both orders produce once the floor is exact, and it is the LOWER of
        the two values the defect produced -- the sparse arm is genuinely empty for this
        query, so rank 1's fused score is `fusion_alpha` = 0.90 and nothing else. The
        broken code's 0.95625 was 0.90 plus a full lexical 1.0 that min-max manufactured
        out of scores of 1.6e-17.
        """
        for label, order in (("as listed", self._entries()), ("permuted", self._shuffled())):
            matches = next(iter(build_matcher(order)._match_fields([self._field()]).values()))
            top = matches[0]
            assert top.dictionary_entry.id == "e0", f"{label}: rank 1 is {top.dictionary_entry.id}"
            assert top.final_confidence == pytest.approx(0.88625, abs=1e-9), (
                f"{label}: rank 1 confidence is {top.final_confidence!r}, not the 0.88625 "
                f"this defect was measured against"
            )
            assert top.score_breakdown.fused_retrieval_score == pytest.approx(0.90, abs=1e-9), (
                f"{label}: fused retrieval score is "
                f"{top.score_breakdown.fused_retrieval_score!r}; the defect showed as this "
                f"number moving to 1.00, so it is the one worth naming"
            )


class TestDuplicateInsertion:
    """
    A glossary listing the same term twice must not change what a column means.

    Real exports duplicate rows constantly -- two systems contributing the same term, a
    join fanning out, a merged cell read back twice. The duplicate is allowed to WIN,
    because it is textually identical to the entry it copies and the tie-break has to pick
    one; what it must not do is change which CONTENT wins.
    """

    @PROPERTY_SETTINGS
    @given(entries=glossary(), fields=schema_fields(), pick=st.integers(0, 2**16))
    def test_an_exact_duplicate_cannot_change_which_content_wins(self, entries, fields, pick):
        """
        EXACT, on the winning entry's searchable text, with the sparse arm OFF, and only
        where the winner was strictly ahead of the runner-up.

        Identity is deliberately not asserted: the duplicate and its original produce the
        same vector and the same confidence, so which id surfaces is settled by the
        tie-break and either answer is correct. TEXT is what a caller cares about.

        THE STRICT-SEPARATION GUARD IS A FINDING, NOT A CONCESSION. Written without it,
        this property is FALSE on clean code, and hypothesis produced the counterexample
        in a few hundred examples: eight entries, seven of them scoring an identical
        0.7071068 against `account.customer`; inserting a duplicate of an unrelated entry
        re-shuffled that tied block from (e1, e2, e3, e4, e5, e6) to
        (e2, e1, e4, e5, e6, e3) and the winning entry changed from one carrying
        definition '' to one carrying definition 'customer', at identical confidence
        0.845595941524288. The cause is `_top_k_indices` sorting with an unstable
        `np.argsort`; making it `kind="stable"` (and sorting the `argpartition`
        candidates back into corpus order first) removes the counterexample. Until that
        lands, a tied rank has no defined answer to assert, so this property is stated
        over the ranks that the scores alone decide.

        The sparse arm is off because with it on this is FALSE for a second, independent
        reason: BM25 document frequency is a corpus statistic, so a duplicate row shifts
        the IDF of its own terms and changes which UNRELATED entry wins, measured at
        60/600. Also reported rather than asserted -- a test that holds 90% of the time is
        not a gate.
        """
        original = entries[pick % len(entries)]
        duplicate = DictionaryEntry(
            id="duplicate-of-" + original.id,
            business_name=original.business_name,
            logical_name=original.logical_name,
            definition=original.definition,
            data_type=original.data_type,
            domain=original.domain,
        )

        before = _by_field(build_matcher(entries, sparse=False)._match_fields(fields))
        after = _by_field(build_matcher([*entries, duplicate], sparse=False)._match_fields(fields))

        for key in before:
            scores = confidences(before[key])
            if len(scores) > 1 and scores[0] - scores[1] <= FLOAT32_REASSOCIATION:
                continue  # a tied rank-1 has no answer the scores determine; see above
            won_before = before[key][0].dictionary_entry
            won_after = after[key][0].dictionary_entry
            assert won_before.to_searchable_text() == won_after.to_searchable_text(), (
                f"duplicating {original.business_name!r} changed the winning ENTRY from "
                f"{won_before.id!r} to {won_after.id!r}, and they are different terms"
            )
            assert won_after.id in (won_before.id, duplicate.id), (
                "the winner changed to an entry that is neither the previous winner nor "
                "the inserted duplicate"
            )


class TestIrrelevantAddition:
    """
    An entry sharing no vocabulary with a column must not move that column's retrieval.

    The precondition is CONSTRUCTED, not assumed: `DISJOINT_WORDS` own reserved encoder
    dimensions, so an entry built from them is exactly orthogonal to any query built from
    the business vocabulary, and both halves of "zero token overlap and cosine below 0.2"
    are asserted below rather than hoped for.
    """

    @PROPERTY_SETTINGS
    @given(fields=schema_fields(), noise=disjoint_entries(max_size=3))
    def test_the_fixture_really_is_irrelevant(self, fields, noise):
        """
        The precondition itself, asserted. A metamorphic test whose "irrelevant" input is
        only probably irrelevant proves nothing on the runs where it was not.
        """
        matcher = build_matcher(noise)
        for field in fields:
            query = matcher._build_query_text(field)
            query_tokens = _tokenize_identifier(field.name)
            for entry in noise:
                entry_tokens = _tokenize_identifier(entry.logical_name) | _tokenize_identifier(
                    entry.business_name
                )
                assert not (query_tokens & entry_tokens), "fixture vocabularies overlap"
                cosine = float(encode(query) @ encode(entry.to_searchable_text()))
                assert cosine == 0.0, f"expected exact orthogonality, measured {cosine!r}"
                assert cosine < 0.2

    @PROPERTY_SETTINGS
    @given(entries=glossary(), fields=schema_fields(), noise=disjoint_entries(max_size=3))
    def test_it_does_not_reorder_retrieval_over_the_entries_that_were_there(
        self, entries, fields, noise
    ):
        """
        EXACT on every pre-existing entry's cosine (to 1e-6), and on the ORDER at every
        position the scores alone decide.

        A cosine is a property of one entry and one query, so adding a row cannot move any
        other row's score; and min-max normalization is affine with a positive slope, so it
        cannot reorder what it rescales. Both stages are asserted because they fail
        differently: a corpus-dependent transform on the index (mean-centering the matrix,
        say) breaks the score map, and a fusion that renormalizes non-monotonically breaks
        the fused order.

        Tied positions are excluded for the reason recorded in `TestDuplicateInsertion`:
        the dense tie-break is an unstable sort, so tied entries legitimately re-shuffle
        when the corpus grows. The per-id score map is asserted over EVERY entry including
        tied ones -- it is tie-immune, and it is the stronger of the two claims.

        This is the relation the brief asked for, restricted to what is actually true. The
        unrestricted form -- "never changes an existing field's top-1" -- is FALSE, at 4%
        with the sparse arm off, and the mechanism is in this module's docstring. That is
        the finding; this is the gate.
        """
        grown = [*entries, *noise]
        original_ids = {e.id for e in entries}

        base = build_matcher(entries, sparse=False)
        after = build_matcher(grown, sparse=False)

        for field in fields:
            query = encode(base._build_query_text(field))
            dense_before = base._vector_store.search(query, top_k=len(grown) + 1).unwrap()
            dense_after = [
                r
                for r in after._vector_store.search(query, top_k=len(grown) + 1).unwrap()
                if r.id in original_ids
            ]

            scores_before = {r.id: r.score for r in dense_before}
            scores_after = {r.id: r.score for r in dense_after}
            assert set(scores_before) == set(scores_after), (
                "an irrelevant entry changed WHICH pre-existing entries are retrievable"
            )
            for entry_id, was in scores_before.items():
                assert abs(was - scores_after[entry_id]) <= FLOAT32_REASSOCIATION, (
                    f"cosine for {entry_id!r} moved {abs(was - scores_after[entry_id]):.3e} "
                    f"because an unrelated entry was indexed"
                )

            _assert_order_matches_where_separated(
                [r.id for r in dense_before],
                [r.score for r in dense_before],
                [r.id for r in dense_after],
                "an irrelevant entry re-ordered dense retrieval over the entries already there",
            )

            fused_before = base._fuse_results(dense_before, {})

            # NON-VACUITY, and it was missing. Everything below compares one fused list
            # against another fused list, so it is satisfied by ANY function of the dense
            # list that preserves order -- including `return [(r.id, r.score) for r in
            # dense]`, which is fusion deleted outright. VERIFIED: this test stayed green
            # under exactly that one-line replacement before this assertion landed, while
            # its docstring claimed to cover fused ordering.
            #
            # With the lexical arm empty, min-max maps the best cosine to 1.0 and the worst
            # to 0.0, so the fused endpoints are `fusion_alpha` and exactly 0.0 -- values a
            # pass-through cannot produce, because they no longer depend on the cosines.
            # Guarded on two distinct cosines: a constant arm normalizes to all-zeros by
            # design, and that case is pinned in TestFusionActuallyFuses instead.
            if len({r.score for r in dense_before}) > 1:
                alpha = base._config.fusion_alpha
                assert abs(fused_before[0][1] - alpha) <= FLOAT32_REASSOCIATION, (
                    f"the top fused score is {fused_before[0][1]!r}, not the "
                    f"{alpha!r} min-max normalization guarantees -- the candidate list "
                    f"reaching confidence scoring was never fused"
                )
                assert fused_before[-1][1] == 0.0, (
                    f"the last fused score is {fused_before[-1][1]!r}, not 0.0 -- the "
                    f"worst candidate did not normalize to the floor, so these scores are "
                    f"raw cosines wearing a fused list's shape"
                )

            fused_after = [
                (i, s) for i, s in after._fuse_results(dense_after, {}) if i in original_ids
            ]
            _assert_order_matches_where_separated(
                [i for i, _ in fused_before],
                [s for _, s in fused_before],
                [i for i, _ in fused_after],
                "an irrelevant entry re-ordered the fused candidate list",
            )


def _tiny_matcher():
    """A matcher built only so `_fuse_results` can be called; its glossary is irrelevant."""
    return build_matcher(
        [
            DictionaryEntry(
                id="d0",
                business_name="Customer Email",
                logical_name="cust_email",
                definition="",
                data_type=DataType.STRING,
            ),
            DictionaryEntry(
                id="d1",
                business_name="Account Balance",
                logical_name="acct_bal",
                definition="",
                data_type=DataType.DOUBLE,
            ),
        ]
    )


class TestFusionActuallyFuses:
    """
    An ABSOLUTE pin on `_fuse_results`, because no metamorphic relation can supply one.

    A relation compares two runs, so every relation in this file is satisfied by any
    implementation that is a consistent function of its input. Fusion deleted -- `return
    [(r.id, r.score) for r in dense]` -- is exactly that, and it survived the whole file.
    The fix is not a better relation; a relation structurally cannot catch it. It is a
    hand-computed expected value, which is what H-004 prescribes.

    The arithmetic, from `core.fusion`: min-max normalize each arm over its own candidates,
    weight the dense arm by `fusion_alpha` and the lexical arm by the remainder, then sort
    descending. `alpha` is read from config rather than written as 0.9 because it is a
    TUNED retrieval parameter -- re-sweeping it is a legitimate change and must not redden
    a test about fusion's shape.
    """

    def test_it_sorts_by_the_fused_score_instead_of_passing_dense_order_through(self):
        """
        Dense input handed over deliberately OUT of score order.

        In production the store returns candidates already sorted, so a pass-through and a
        real fusion are indistinguishable from their ordering alone -- which is precisely
        why every relation missed this. Feeding an unsorted list separates them: sorting is
        `fuse_linear_ids`'s documented job, not an accident of its caller.
        """
        matcher = _tiny_matcher()
        alpha = matcher._config.fusion_alpha
        dense = [
            SearchResult(id="low", score=0.4),
            SearchResult(id="high", score=0.8),
            SearchResult(id="mid", score=0.6),
        ]

        fused = matcher._fuse_results(dense, {})

        assert [i for i, _ in fused] == ["high", "mid", "low"], (
            f"fused order is {[i for i, _ in fused]!r}: the candidate list was returned in "
            f"the order it arrived, so nothing fused it and the confidence scoring below "
            f"is reading raw retrieval scores"
        )
        # min 0.4, max 0.8, span 0.4 -> normalized 1.0, 0.5, 0.0 -> weighted by alpha.
        for (entry_id, score), expected in zip(fused, (alpha, alpha * 0.5, 0.0), strict=True):
            assert abs(score - expected) <= FLOAT32_REASSOCIATION, (
                f"fused score for {entry_id!r} is {score!r}, not the {expected!r} that "
                f"min-max normalization at alpha={alpha!r} gives"
            )

    def test_a_candidate_only_the_lexical_arm_found_still_reaches_the_list(self):
        """
        The sparse arm is CONSULTED, not decorative.

        `_fuse_results` receiving a sparse map and dropping it costs the 0.542-P@1 lexical
        arm outright, and silently: dense-only answers still look like answers. A
        lexical-only candidate is the one observation that cannot be produced without
        reading the sparse map at all.
        """
        matcher = _tiny_matcher()
        alpha = matcher._config.fusion_alpha
        dense = [SearchResult(id="high", score=0.8), SearchResult(id="low", score=0.4)]

        fused = matcher._fuse_results(dense, {"lexical_only": 10.0, "low": 5.0})

        # Lexical min-max: low -> 0.0, lexical_only -> 1.0. So high = alpha, low = 0.0,
        # lexical_only = 1 - alpha, which orders it above `low` and below `high`.
        assert [i for i, _ in fused] == ["high", "lexical_only", "low"], (
            f"fused order is {[i for i, _ in fused]!r}: a candidate the lexical arm found "
            f"was dropped or misplaced, so the sparse retriever's contribution is lost"
        )
        assert abs(dict(fused)["lexical_only"] - (1.0 - alpha)) <= FLOAT32_REASSOCIATION

    def test_equal_scores_are_emitted_in_dense_retrieval_order(self):
        """
        The tie-break `fuse_linear_ids` documents, pinned in both directions.

        H-005 was ranking that moved with `PYTHONHASHSEED`, and the fix was to walk the
        dense list in rank order rather than a set. Asserting BOTH input orders is what
        makes this a statement about the tie-break rather than about one lucky arrangement.
        """
        matcher = _tiny_matcher()
        for order in (("a", "b", "c"), ("c", "b", "a")):
            dense = [SearchResult(id=i, score=0.5) for i in order]
            fused = matcher._fuse_results(dense, {})
            assert [i for i, _ in fused] == list(order), (
                f"tied candidates came back as {[i for i, _ in fused]!r} from dense order "
                f"{list(order)!r} -- a tie is being settled by something other than dense "
                f"rank, which is where H-005's hash-seed dependence lived"
            )


class TestCorpusGrowth:
    """
    Appending a shard: what must hold, and what provably must not be asserted.

    Appending documents changes N, changes every term's IDF and changes the average
    document length BM25 normalizes by. Scores therefore MOVE, by construction and
    correctly. Measured here: 3512 of 3516 queries saw a BM25 score change after a
    disjoint shard was appended, and the BM25 ranking of the pre-existing documents
    changed for 2411 of them -- 69%. So neither score equality nor sparse rank stability
    is available, and writing either would produce a gate that reddens on correct code.
    """

    @PROPERTY_SETTINGS
    @given(entries=glossary(), fields=schema_fields(), shard=disjoint_entries())
    def test_the_shard_moves_sparse_scores_and_leaves_dense_ranking_alone(
        self, entries, fields, shard
    ):
        """
        NON-VACUITY first, then the invariant.

        The first assertion is the important one and is easy to leave out: it proves the
        shard actually perturbed the system. Without it, every remaining assertion in this
        test would still pass against a `sync` that ignored the shard entirely, and the
        test would be reporting nothing while looking like coverage.

        Then: dense ranking over the pre-existing entries is EXACT, and dense scores carry
        the 1e-6 float32 tolerance (measured max 6.0e-08 across 6764 queries -- a bigger
        corpus matrix is blocked differently by BLAS, so a cosine can move by an ULP even
        though its inputs did not).
        """
        grown = [*entries, *shard]
        original_ids = {e.id for e in entries}

        base = build_matcher(entries)
        after = build_matcher(grown)

        moved = False
        compared = 0
        for field in fields:
            query_text = base._build_query_text(field)
            sparse_before = {
                r.id: r.score for r in base._sparse_retriever.search(query_text, 100).unwrap()
            }
            sparse_after = {
                r.id: r.score for r in after._sparse_retriever.search(query_text, 100).unwrap()
            }
            for doc_id, score in sparse_before.items():
                if doc_id in sparse_after:
                    compared += 1
                    moved = moved or sparse_after[doc_id] != score

            query = encode(query_text)
            dense_before = base._vector_store.search(query, top_k=len(grown) + 1).unwrap()
            kept = [
                r
                for r in after._vector_store.search(query, top_k=len(grown) + 1).unwrap()
                if r.id in original_ids
            ]

            scores_after = {r.id: r.score for r in kept}
            assert {r.id for r in dense_before} == set(scores_after), (
                "appending a shard changed WHICH pre-existing entries are retrievable"
            )
            for was in dense_before:
                assert abs(was.score - scores_after[was.id]) <= FLOAT32_REASSOCIATION, (
                    f"cosine for {was.id!r} moved "
                    f"{abs(was.score - scores_after[was.id]):.3e} because OTHER entries "
                    f"were added"
                )
            _assert_order_matches_where_separated(
                [r.id for r in dense_before],
                [r.score for r in dense_before],
                [r.id for r in kept],
                "appending a shard re-ordered dense retrieval over the entries that were "
                "already indexed",
            )

        assume(compared)
        assert moved, (
            "no BM25 score changed after appending a shard, so this test proved nothing. "
            "Appending documents must change N and the average document length."
        )

    @PROPERTY_SETTINGS
    @given(entries=glossary(), fields=schema_fields(), shard=disjoint_entries())
    def test_a_shard_entry_never_becomes_a_columns_match(self, entries, fields, shard):
        """
        EXACT. A column must never inherit its classification from a shard that shares no
        vocabulary with it.

        This is arithmetic, not luck. Once at least two distinct cosines are present, the
        best entry normalizes to 1.0, its fused score is at least `fusion_alpha` = 0.90 and
        its confidence is therefore at least 0.7 * 0.90 = 0.63; a zero-cosine shard entry
        normalizes to 0.0 and can reach at most the 0.30 of weight the lexical, edit, type
        and domain signals carry between them. 0.30 < 0.63, with no dependence on the data.

        The `all candidates share one cosine` case is excluded because there the semantic
        signal carries no information at all -- min-max maps a constant map to all-zeros --
        and the answer is decided entirely by the other 30%. That degenerate case is real
        (a column whose tokens appear in no entry) and it deserves its own decision about
        what SHOULD happen; asserting a guess about it here would be inventing a
        requirement.
        """
        grown = [*entries, *shard]
        shard_ids = {e.id for e in shard}
        matcher = build_matcher(grown)

        results = matcher._match_fields(fields)
        for matches in results.values():
            field = matches[0].schema_field
            cosines = [
                r.score
                for r in matcher._vector_store.search(
                    encode(matcher._build_query_text(field)), top_k=len(grown)
                ).unwrap()
            ]
            if min(cosines) == max(cosines):
                continue
            assert matches[0].dictionary_entry.id not in shard_ids, (
                f"column {field.full_path!r} matched shard entry "
                f"{matches[0].dictionary_entry.id!r}, which shares no token with it"
            )
