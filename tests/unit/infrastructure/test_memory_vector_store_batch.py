"""
tests.unit.infrastructure.test_memory_vector_store_batch | Layer: TEST
Behavioural tests for InMemoryVectorStore.search_batch and the index rebuild.

`search_batch` is an OPTIMIZATION, so almost every test here is a differential test: it
asserts the batch path agrees with the per-query loop that it replaces. The one thing it
deliberately does not assert is bit-identity -- BLAS accumulates a matrix-matrix product
in a different order than a matrix-vector one, so tied entries can swap. The tests use
well-separated vectors so that ordering IS determinate, and the tie behaviour gets its own
test rather than being papered over with a loose tolerance everywhere.

Differential tests have a BLIND SPOT, and it is worth naming because the method has already
been broken in exactly that spot once. Everything above passes identically whichever way
round the GEMM is written (`queries @ corpus.T` or `corpus @ queries.T`) and whether the
batch runs as one GEMM or as a per-query loop -- all of those return the same results, and
differ only in cost and in the layout of the intermediate. So the results are pinned by the
differential tests, and the STRUCTURE that makes them fast is pinned separately by
test_search_batch_scores_each_query_as_a_contiguous_row, which asserts on the score block
itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.domain.ports.vector_store import VectorDocument, VectorStoreConfig
from nexus_matcher.infrastructure.adapters.vector_stores import memory as memory_module
from nexus_matcher.infrastructure.adapters.vector_stores.memory import (
    _MAX_CHUNK,
    InMemoryVectorStore,
)

DIM = 64


def make_store(n: int, dim: int = DIM, seed: int = 7, payload_groups: int = 4):
    """A store of `n` well-separated unit vectors, plus the raw matrix."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store = InMemoryVectorStore(VectorStoreConfig(collection_name="c", dimension=dim))
    store.upsert(
        [
            VectorDocument(id=f"d{i}", embedding=vecs[i], payload={"g": f"g{i % payload_groups}"})
            for i in range(n)
        ]
    )
    return store, vecs


def queries(n: int, dim: int = DIM, seed: int = 99):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, dim)).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def ids_of(results):
    return [[r.id for r in row] for row in results]


# Two corpus sizes, several orders of magnitude apart. `search_batch` takes the SAME path
# for both -- there is no size threshold -- but the corpus width is what sets the chunk
# size, so running the small one too keeps a chunking bug that only shows up on short
# corpora from hiding.
_SMALL_N = 200
_LARGE_N = 16_884


@pytest.mark.parametrize("n", [_SMALL_N, _LARGE_N], ids=["small_corpus", "large_corpus"])
def test_search_batch_agrees_with_per_query_loop(n):
    """
    Pins the whole point of search_batch: it must return what the loop it replaces returns.

    Run at both corpus sizes: the chunk width is derived from the number of entries, so a
    single size could satisfy this while the other chunked wrongly.
    """
    store, _ = make_store(n)
    qs = queries(40)

    loop = [store.search(q, top_k=10).unwrap() for q in qs]
    batch = store.search_batch(qs, top_k=10).unwrap()

    assert ids_of(batch) == ids_of(loop)
    for a, b in zip(loop, batch, strict=True):
        for ra, rb in zip(a, b, strict=True):
            assert ra.score == pytest.approx(rb.score, abs=1e-5)


def test_search_batch_returns_one_row_per_query_in_input_order():
    """
    Pins the output contract callers rely on to zip results back onto their fields.

    A chunked implementation that dropped or reordered a chunk would still return
    plausible-looking results, so this compares against per-query search rather than just
    counting rows.
    """
    store, _ = make_store(_LARGE_N)
    qs = queries(_MAX_CHUNK * 2 + 5)  # deliberately spans several chunks, unevenly

    batch = store.search_batch(qs, top_k=5).unwrap()

    assert len(batch) == len(qs)
    for i, q in enumerate(qs):
        assert [r.id for r in batch[i]] == [r.id for r in store.search(q, top_k=5).unwrap()]


def test_search_batch_scores_are_sorted_descending_per_row():
    """Pins that the per-query top-k selection survives being run row-wise over a block."""
    store, _ = make_store(_LARGE_N)
    for row in store.search_batch(queries(20), top_k=15).unwrap():
        scores = [r.score for r in row]
        assert scores == sorted(scores, reverse=True)


def test_search_batch_finds_the_exact_nearest_neighbour():
    """
    Pins that batching stayed EXACT rather than quietly becoming approximate.

    Querying with a corpus vector itself must return that vector first, at score ~1.0.
    """
    store, vecs = make_store(_LARGE_N)
    probe = [17, 999, len(vecs) - 1]

    batch = store.search_batch(vecs[probe], top_k=3).unwrap()

    for want, row in zip(probe, batch, strict=True):
        assert row[0].id == f"d{want}"
        assert row[0].score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("n", [_SMALL_N, _LARGE_N], ids=["small_corpus", "large_corpus"])
def test_search_batch_applies_the_filter_to_every_query(n):
    """
    Pins that the batch-wide mask is applied to all queries, not just the first chunk.

    The mask is built once for the whole batch and broadcast across the score block; a
    broadcast along the wrong axis would filter per-query-position instead of per-entry
    and still return the right NUMBER of results.
    """
    store, _ = make_store(n)
    qs = queries(_MAX_CHUNK + 3)

    batch = store.search_batch(qs, top_k=5, filter={"g": "g2"}).unwrap()

    assert len(batch) == len(qs)
    assert all(row for row in batch)
    for row in batch:
        assert all(r.payload["g"] == "g2" for r in row)


def test_search_batch_filter_agrees_with_per_query_filtered_search():
    """Pins the filtered batch path against the filtered loop it replaces."""
    store, _ = make_store(_LARGE_N)
    qs = queries(20)

    loop = [store.search(q, top_k=8, filter={"g": "g1"}).unwrap() for q in qs]
    batch = store.search_batch(qs, top_k=8, filter={"g": "g1"}).unwrap()

    assert ids_of(batch) == ids_of(loop)


def test_search_batch_top_k_larger_than_corpus():
    """Pins the k >= N branch of top-k selection, where argpartition is not usable."""
    store, _ = make_store(5, dim=8)
    batch = store.search_batch(queries(3, dim=8), top_k=100).unwrap()
    assert [len(row) for row in batch] == [5, 5, 5]


def test_search_batch_on_empty_store_returns_a_row_per_query():
    """
    Pins that an empty collection yields one empty list per query, not a single empty list.

    Callers zip this against their inputs, so collapsing the shape would silently
    misalign every field with the wrong result.
    """
    store = InMemoryVectorStore(VectorStoreConfig(collection_name="c", dimension=DIM))
    assert store.search_batch(queries(4), top_k=5).unwrap() == [[], [], [], []]


def test_search_batch_with_no_queries_returns_no_rows():
    """Pins the degenerate empty-batch case against an IndexError in the chunk loop."""
    store, _ = make_store(_SMALL_N)
    assert store.search_batch([], top_k=5).unwrap() == []


def test_search_batch_rejects_wrong_query_dimension():
    """
    Pins that a dimension mismatch is REPORTED rather than silently reshaped.

    Queries arrive as one flat block, so a wrong width is exactly the case where a
    reshape could invent a plausible batch out of the wrong numbers.
    """
    store, _ = make_store(_SMALL_N)
    result = store.search_batch(np.zeros((3, DIM + 1), dtype=np.float32), top_k=5)
    assert result.is_failure
    assert "dimension" in result.error.lower()


def test_search_batch_normalizes_queries():
    """
    Pins cosine semantics: scaling a query must not change its ranking or its scores.

    The corpus is pre-normalized at index time and only the query is normalized at search
    time, so a batch path that forgot that step would still return sorted, plausible
    results with inflated scores.
    """
    store, _ = make_store(_LARGE_N)
    qs = queries(10)

    unit = store.search_batch(qs, top_k=10).unwrap()
    scaled = store.search_batch(qs * 37.0, top_k=10).unwrap()

    assert ids_of(scaled) == ids_of(unit)
    for a, b in zip(unit, scaled, strict=True):
        for ra, rb in zip(a, b, strict=True):
            assert ra.score == pytest.approx(rb.score, abs=1e-5)


def test_search_batch_differs_from_the_loop_only_on_ties():
    """
    Documents the one way batching is NOT identical to the loop, so it is not mistaken
    for a bug later.

    A GEMM sums in a different order than a GEMV, so scores move by a few float32 ULPs and
    entries that are exactly tied can swap. This builds a corpus of deliberate duplicates
    and asserts the damage is bounded: scores still agree, and the SET of returned ids is
    unchanged, even where the order within a tie group is not.
    """
    n = _LARGE_N
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((n, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs[1::2] = vecs[::2][: len(vecs[1::2])]  # every odd entry duplicates an even one

    store = InMemoryVectorStore(VectorStoreConfig(collection_name="c", dimension=DIM))
    store.upsert([VectorDocument(id=f"d{i}", embedding=vecs[i], payload={}) for i in range(n)])

    qs = queries(30)

    loop = [store.search(q, top_k=20).unwrap() for q in qs]
    batch = store.search_batch(qs, top_k=20).unwrap()

    for a, b in zip(loop, batch, strict=True):
        assert [r.score for r in a] == pytest.approx([r.score for r in b], abs=1e-5)
        assert {r.id for r in a} == {r.id for r in b}


def test_search_batch_is_deterministic():
    """Pins that repeated batches agree; a reused scratch buffer would show up here."""
    store, _ = make_store(_LARGE_N)
    qs = queries(50)
    first = ids_of(store.search_batch(qs, top_k=10).unwrap())
    for _ in range(3):
        assert ids_of(store.search_batch(qs, top_k=10).unwrap()) == first


# =============================================================================
# Score-block structure
#
# Everything above is differential, and differential tests cannot see the shape of the
# work. These assert on the intermediate instead.
# =============================================================================


def _spy_on_top_k(monkeypatch) -> list[np.ndarray]:
    """Capture every score array search_batch hands to the top-k selector."""
    captured: list[np.ndarray] = []
    real = memory_module._top_k_indices

    def spy(scores, k):
        captured.append(scores)
        return real(scores, k)

    monkeypatch.setattr(memory_module, "_top_k_indices", spy)
    return captured


@pytest.mark.parametrize("payload_filter", [None, {"g": "g2"}], ids=["unfiltered", "filtered"])
def test_search_batch_scores_each_query_as_a_contiguous_row(monkeypatch, payload_filter):
    """
    Pins the GEMM orientation: `queries @ corpus.T`, NOT `corpus @ queries.T`.

    The two are the same GEMM with the same FLOPs and they return identical results, so
    EVERY other test in this file passes under either one -- which is exactly how the
    flipped version got written and shipped once already. What differs is the layout of the
    intermediate: queries-on-the-left produces a (queries x entries) block in which one
    query's scores are one contiguous row; corpus-on-the-left produces its transpose,
    (entries x queries), and every per-query top-k then has to stride down a column of it.

    So this asserts on the block rather than on the results. It fails if the orientation is
    flipped back. It also fails if the batch is quietly replaced by a per-query loop -- a
    matrix-VECTOR product yields a freshly allocated 1-D array with no block behind it --
    which pins the other half of the same decision: there is no small-corpus fallback, and
    this corpus is deliberately small enough that a size threshold would have caught it.
    """
    n = 300
    n_queries = 40
    store, _ = make_store(n)
    assert store._collections["c"].embeddings.nbytes < 1 << 20, "corpus must stay small"

    captured = _spy_on_top_k(monkeypatch)
    store.search_batch(queries(n_queries), top_k=5, filter=payload_filter).unwrap()

    assert len(captured) == n_queries
    for row in captured:
        # One score per corpus entry...
        assert row.shape == (n,)
        # ...laid out contiguously, so the top-k walks it in order...
        assert row.flags["C_CONTIGUOUS"]
        # ...as a row of a 2-D score block whose LAST axis is the corpus. Under
        # `corpus @ queries.T` the block is (entries x queries) and this is the query axis.
        assert row.base is not None, "scores were not a view into a batched score block"
        assert row.base.ndim == 2
        assert row.base.shape[1] == n
        assert row.base.shape[0] <= n_queries


# =============================================================================
# Index rebuild
# =============================================================================


def test_rebuild_index_stores_unit_norm_rows():
    """
    Pins the invariant the whole search path is built on: the corpus matrix is normalized
    ONCE at index time.

    Search normalizes only the query and treats the corpus dot product as a cosine. If a
    rebuild stopped normalizing, every score would silently become an unnormalized dot
    product -- still sorted, still plausible, quietly wrong.
    """
    store, _ = make_store(500)
    matrix = store._collections["c"].embeddings

    assert matrix.dtype == np.float32
    assert matrix.flags["C_CONTIGUOUS"]
    assert np.linalg.norm(matrix, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_rebuild_index_survives_a_zero_vector():
    """
    Pins the 1e-9 norm floor: a zero embedding must not divide by zero and poison the
    matrix with NaN, which would make EVERY subsequent search return garbage, not just
    searches touching that entry.
    """
    store = InMemoryVectorStore(VectorStoreConfig(collection_name="c", dimension=8))
    store.upsert(
        [
            VectorDocument(id="zero", embedding=np.zeros(8, dtype=np.float32)),
            VectorDocument(id="unit", embedding=np.eye(8, dtype=np.float32)[0]),
        ]
    )

    assert np.isfinite(store._collections["c"].embeddings).all()
    results = store.search(np.eye(8, dtype=np.float32)[0], top_k=2).unwrap()
    assert results[0].id == "unit"


def test_rebuild_index_matches_an_explicit_normalization():
    """
    Pins the einsum row-norms against the textbook formula they replaced.

    einsum is used to avoid materializing a second full-size temporary; it must agree
    with np.linalg.norm to float32 precision or ranking could drift.
    """
    store, vecs = make_store(300)
    # Re-upsert the same directions at wildly different lengths; normalization has to
    # remove the scale, so the stored matrix must come back identical to the unit vectors.
    scales = (np.arange(300, dtype=np.float32) % 7 + 0.5)[:, None]
    unnormalized = vecs * scales
    store.upsert([VectorDocument(id=f"d{i}", embedding=unnormalized[i]) for i in range(300)])

    got = store._collections["c"].embeddings
    want = unnormalized / np.linalg.norm(unnormalized, axis=1, keepdims=True)
    assert got == pytest.approx(want, abs=1e-6)


def test_upsert_then_delete_keeps_ids_and_matrix_aligned():
    """
    Pins that a rebuild after deletion keeps row i of the matrix pointing at ids[i].

    The matrix is rebuilt from a dict, so a rebuild that reused a stale id list would
    return real documents with the wrong scores -- results that look entirely normal.
    """
    store, _ = make_store(400)
    store.delete([f"d{i}" for i in range(0, 400, 3)])

    coll = store._collections["c"]
    assert len(coll.ids) == len(coll.documents) == coll.embeddings.shape[0]
    for row, doc_id in enumerate(coll.ids):
        want = coll.documents[doc_id].embedding
        want = want / np.linalg.norm(want)
        assert coll.embeddings[row] == pytest.approx(want, abs=1e-6)

    # and the surviving entries still retrieve themselves
    survivor = coll.ids[5]
    top = store.search(coll.documents[survivor].embedding, top_k=1).unwrap()
    assert top[0].id == survivor
