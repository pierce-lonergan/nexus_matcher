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

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from nexus_matcher.application.use_cases.match_schema import _tokenize_identifier
from nexus_matcher.domain.models.entities import DictionaryEntry
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
        7.1e-08 for the run that caught it. The tolerance is 1e-6, ~14x the largest
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
            fused_after = [
                (i, s) for i, s in after._fuse_results(dense_after, {}) if i in original_ids
            ]
            _assert_order_matches_where_separated(
                [i for i, _ in fused_before],
                [s for _, s in fused_before],
                [i for i, _ in fused_after],
                "an irrelevant entry re-ordered the fused candidate list",
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
