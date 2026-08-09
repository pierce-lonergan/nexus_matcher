"""
H-004 -- differential tests are invariant to errors both sides share.

A GEMM orientation flip passed 19 of 19 tests. Those tests compared the batched search
path against the per-query loop; the transpose was applied consistently, so both sides
moved together, agreement was perfect, and both were wrong.

An implementation-vs-implementation oracle detects DIVERGENCE and nothing else. It is
structurally blind to any error the two implementations share -- which includes every
error inherited from the assumption they were both built from.

The control is twofold: pin ABSOLUTE values on a hand-verified fixture, and use inputs
whose shape makes a transpose impossible to hide. Square or symmetric inputs let a
transposed matrix multiply produce the right answer by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.domain.ports.vector_store import VectorDocument, VectorStoreConfig
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore

# Deliberately NON-SQUARE (5 documents, 3 dimensions) and non-symmetric. With 3 queries
# against 3 documents in 3 dimensions, a transposed product returns a plausible matrix and
# every shape assertion still holds.
DIM = 3
DOCS = [
    ("d0", [1.0, 0.0, 0.0]),
    ("d1", [0.0, 1.0, 0.0]),
    ("d2", [0.0, 0.0, 1.0]),
    ("d3", [0.6, 0.8, 0.0]),
    ("d4", [0.0, 0.6, 0.8]),
]


@pytest.fixture
def store():
    s = InMemoryVectorStore(VectorStoreConfig(collection_name="h004", dimension=DIM))
    s.upsert(
        [VectorDocument(id=i, embedding=np.array(v, dtype=np.float32), payload={}) for i, v in DOCS]
    )
    return s


def test_absolute_nearest_neighbour_is_hand_verifiable(store):
    """
    Not "the two paths agree" -- the actual right answer, computed by hand.

    Query [1,0,0] is exactly d0. Its cosine to d3 = [0.6,0.8,0] is 0.6, to d1/d2/d4 it is
    0. So the ranking is forced: d0, then d3, then the rest.
    """
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2).unwrap()
    assert [r.id for r in results] == ["d0", "d3"]
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[1].score == pytest.approx(0.6, abs=1e-5)


def test_batched_path_gives_the_same_absolute_answers(store):
    """
    The differential check is still worth having -- it just cannot be the ONLY one. Here
    it runs against queries whose correct answers are already pinned above.
    """
    queries = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]
    rows = store.search_batch(queries, top_k=1).unwrap()
    assert [r[0].id for r in rows] == ["d0", "d1", "d2"]


def test_query_count_and_document_count_differ_so_a_transpose_cannot_hide(store):
    """
    The specific structural control for NM-0017.

    Two queries against five documents. A transposed product yields a 5x2 where a 2x5 is
    required; with equal counts it would yield a plausible square and slip through.
    """
    queries = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]
    rows = store.search_batch(queries, top_k=len(DOCS)).unwrap()
    assert len(rows) == len(queries), "one result row per QUERY, not per document"
    assert all(len(r) == len(DOCS) for r in rows)
    assert rows[0][0].id == "d0"
    assert rows[1][0].id == "d2"


def test_asymmetric_scores_are_not_accidentally_symmetric(store):
    """
    A transpose is invisible when score(q_i, d_j) == score(q_j, d_i). These vectors are
    chosen so that is false, making the orientation observable in the VALUES.
    """
    q_a = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    q_b = np.array([0.0, 0.6, 0.8], dtype=np.float32)
    rows = store.search_batch([q_a, q_b], top_k=len(DOCS)).unwrap()
    score_a_d1 = next(r.score for r in rows[0] if r.id == "d1")
    score_b_d0 = next(r.score for r in rows[1] if r.id == "d0")
    assert score_a_d1 == pytest.approx(0.8, abs=1e-5)
    assert score_b_d0 == pytest.approx(0.0, abs=1e-5)
    assert score_a_d1 != pytest.approx(score_b_d0, abs=1e-3), (
        "these probes became symmetric, so they no longer detect a transpose"
    )
