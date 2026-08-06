"""
tests.unit.infrastructure.test_hnsw_vector_store | Layer: TEST
Behavioural tests for the HNSW (usearch) vector store.

The store is APPROXIMATE, so these tests assert on recall against the exact store rather
than on an exact ordering. Vectors are generated CLUSTERED, not i.i.d. Gaussian: random
high-dimensional vectors are nearly equidistant, which destroys the neighbourhood
structure every graph index depends on and would make these tests fail for reasons that
say nothing about the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.domain.ports.vector_store import VectorDocument, VectorStoreConfig
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore

usearch = pytest.importorskip("usearch", reason="usearch not installed")

from nexus_matcher.infrastructure.adapters.vector_stores.hnsw import (  # noqa: E402
    HnswVectorStore,
)

DIM = 64
N = 2000


@pytest.fixture(scope="module")
def corpus():
    """Clustered unit vectors, plus queries drawn near real points."""
    rng = np.random.default_rng(7)
    n_clusters = 40
    centers = rng.standard_normal((n_clusters, DIM)).astype(np.float32)
    vecs = (
        centers[rng.integers(0, n_clusters, N)]
        + rng.standard_normal((N, DIM)).astype(np.float32) * 0.35
    )
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    queries = (
        vecs[rng.integers(0, N, 100)] + rng.standard_normal((100, DIM)).astype(np.float32) * 0.1
    )
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return vecs, queries


@pytest.fixture(scope="module")
def docs(corpus):
    vecs, _ = corpus
    return [
        VectorDocument(id=f"d{i}", embedding=vecs[i], payload={"group": "a" if i % 2 else "b"})
        for i in range(N)
    ]


@pytest.fixture(scope="module")
def hnsw(docs):
    store = HnswVectorStore(
        VectorStoreConfig(collection_name="dictionary", dimension=DIM), build_threads=1
    )
    store.upsert(docs)
    return store


@pytest.fixture(scope="module")
def exact(docs):
    store = InMemoryVectorStore(VectorStoreConfig(collection_name="dictionary", dimension=DIM))
    store.upsert(docs)
    return store


class TestHnswBasics:
    def test_store_type(self, hnsw):
        assert hnsw.store_type == "hnsw"

    def test_reports_document_count(self, hnsw):
        assert hnsw.get_collection_info("dictionary").unwrap().count == N

    def test_returns_requested_number_of_results(self, hnsw, corpus):
        _, queries = corpus
        assert len(hnsw.search(queries[0], top_k=10).unwrap()) == 10

    def test_scores_are_descending(self, hnsw, corpus):
        _, queries = corpus
        scores = [r.score for r in hnsw.search(queries[0], top_k=20).unwrap()]
        assert scores == sorted(scores, reverse=True)

    def test_get_by_id(self, hnsw):
        assert hnsw.get_by_id("d5").unwrap().id == "d5"

    def test_dimension_mismatch_is_rejected(self, hnsw):
        bad = VectorDocument(id="bad", embedding=np.zeros(DIM + 1, dtype=np.float32), payload={})
        assert hnsw.upsert([bad]).is_failure


class TestHnswRecall:
    """Approximate results must still agree with exact search almost always."""

    def test_recall_at_10_is_high(self, hnsw, exact, corpus):
        _, queries = corpus
        recalls = []
        for q in queries:
            approx = {r.id for r in hnsw.search(q, top_k=10).unwrap()}
            truth = {r.id for r in exact.search(q, top_k=10).unwrap()}
            recalls.append(len(approx & truth) / 10)
        mean_recall = float(np.mean(recalls))
        assert mean_recall >= 0.95, f"recall@10 collapsed to {mean_recall:.3f}"

    def _top1_agreement(self, hnsw, exact, queries) -> float:
        agree = 0
        for q in queries:
            a = hnsw.search(q, top_k=1).unwrap()
            b = exact.search(q, top_k=1).unwrap()
            if a and b and a[0].id == b[0].id:
                agree += 1
        return agree / len(queries)

    def test_top_1_usually_matches_exact(self, hnsw, exact, corpus):
        """
        The fixture builds with build_threads=1, which makes graph construction
        deterministic, so this measures 1.00 here. The bar is still 0.90 rather than 1.00
        because HNSW is approximate by construction and a change to connectivity or
        expansion_add legitimately costs a few percent; the test is meant to catch a
        collapse in index quality, not to pin an exact value.
        """
        _, queries = corpus
        agreement = self._top1_agreement(hnsw, exact, queries)
        assert agreement >= 0.90, f"top-1 agreement collapsed to {agreement:.3f}"

    def test_raising_expansion_search_increases_agreement(self, docs, exact, corpus):
        """
        The recall/latency dial must actually work: spending more search effort has to
        move results toward exact. This is the property that makes the approximation
        tunable rather than merely lossy.

        Uses a PRIVATE index because it mutates `expansion_search`, which the shared
        module-scoped fixture is also read by -- that coupling made the test order- and
        state-dependent.

        The index is built with build_threads=1. Parallel construction interleaves
        neighbour selection nondeterministically, which moved top-1 agreement between
        0.94 and 0.98 across builds and flaked this assertion at roughly 20%. Building
        single-threaded is exactly reproducible.
        """
        _, queries = corpus
        store = HnswVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=DIM), build_threads=1
        )
        store.upsert(docs)

        store.expansion_search = 16
        low = self._top1_agreement(store, exact, queries)
        store.expansion_search = 512
        high = self._top1_agreement(store, exact, queries)

        assert high >= low, f"more search effort did not help: {low:.3f} -> {high:.3f}"
        assert high >= 0.95, f"even at ef_search=512 agreement is only {high:.3f}"


class TestHnswFiltering:
    def test_filter_is_respected(self, hnsw, corpus):
        _, queries = corpus
        results = hnsw.search(queries[0], top_k=10, filter={"group": "a"}).unwrap()
        assert results
        assert all(r.payload["group"] == "a" for r in results)

    def test_filter_still_fills_top_k(self, hnsw, corpus):
        """Filtering happens after graph search, so the store must over-fetch."""
        _, queries = corpus
        results = hnsw.search(queries[1], top_k=10, filter={"group": "b"}).unwrap()
        assert len(results) == 10


class TestHnswMutation:
    def test_deleted_documents_are_not_returned(self, corpus, docs):
        store = HnswVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=DIM), build_threads=1
        )
        store.upsert(docs[:500])
        vecs, _ = corpus

        removed = ["d0", "d1", "d2"]
        assert store.delete(removed).unwrap() == 3
        assert store.get_collection_info("dictionary").unwrap().count == 497

        for target in removed:
            hits = {r.id for r in store.search(vecs[0], top_k=25).unwrap()}
            assert target not in hits

    def test_upsert_updates_existing_id_without_duplicating(self, docs):
        store = HnswVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=DIM), build_threads=1
        )
        store.upsert(docs[:100])
        before = store.get_collection_info("dictionary").unwrap().count

        replacement = VectorDocument(
            id="d0",
            embedding=np.ones(DIM, dtype=np.float32) / np.sqrt(DIM),
            payload={"group": "changed"},
        )
        store.upsert([replacement])

        assert store.get_collection_info("dictionary").unwrap().count == before
        assert store.get_by_id("d0").unwrap().payload["group"] == "changed"

    def test_delete_collection_empties_the_store(self, docs):
        store = HnswVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=DIM), build_threads=1
        )
        store.upsert(docs[:100])
        assert store.delete_collection("dictionary").unwrap() is True
        assert store.get_collection_info("dictionary").unwrap().count == 0


class TestHnswTuning:
    def test_expansion_search_is_adjustable(self, hnsw):
        original = hnsw.expansion_search
        try:
            hnsw.expansion_search = 64
            assert hnsw.expansion_search == 64
        finally:
            hnsw.expansion_search = original

    def test_unsupported_metric_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported distance_metric"):
            HnswVectorStore(
                VectorStoreConfig(
                    collection_name="dictionary", dimension=DIM, distance_metric="hamming"
                )
            )
