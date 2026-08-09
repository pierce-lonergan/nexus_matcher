"""
nexus_matcher.infrastructure.adapters.vector_stores.memory | Layer: INFRASTRUCTURE
In-memory vector store implementation for testing and development.

## Relationships
# IMPLEMENTS → domain/ports/vector_store :: VectorStore protocol
# USED_BY    → tests/* :: test fixtures
# USED_BY    → application/* :: development mode

## Attributes
# Security: No persistence, data lost on restart
# Performance: O(n) EXACT search (brute force). Per query it is bandwidth-bound, not
#              compute-bound -- the cost is streaming the corpus matrix, so callers with
#              several queries in hand should use `search_batch`, which reads it once
#              instead of once per query (2.8x at 10k entries, 5.6x at 100k).
# Reliability: Thread-safe with locks
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nexus_matcher.domain.ports.vector_store import (
    BaseVectorStore,
    CollectionInfo,
    SearchResult,
    VectorDocument,
    VectorStoreConfig,
)
from nexus_matcher.shared.types.base import DocumentId, EmbeddingVector, Result

# Ceiling on the (chunk x N) similarity block `search_batch` allocates. It is the only
# allocation that grows with BOTH the batch and the corpus: 688 queries against 100k
# entries in one shot would be 275 MB. 32 MB keeps that bounded while leaving each GEMM
# wide enough for batching to be worth it (at 100k entries this is a chunk of 83).
_SCORE_BLOCK_BYTES = 32 << 20

# Never chunk below this, even for a corpus large enough that the byte budget says to.
# A chunk of 1 degenerates back to the per-query matvec this exists to avoid.
_MIN_CHUNK = 16

# ...and never above this, even for a corpus small enough that the byte budget allows it.
# The whole (chunk x N) block is materialized before any of it is consumed, so an unbounded
# chunk would trade the per-query loop's repeated corpus reads for one oversized
# allocation. At 100k entries the byte budget binds first anyway (a chunk of 83); this only
# bites for smaller corpora.
_MAX_CHUNK = 128


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Indices of the `k` highest entries of a 1-D score array, best first.

    argpartition is O(N) and only the k survivors get sorted, O(k log k); a full argsort
    of the corpus would be O(N log N) for a top-k that is typically 100 out of 100,000.
    """
    n = scores.shape[0]
    if k >= n:
        return np.argsort(-scores)
    candidates = np.argpartition(-scores, k - 1)[:k]
    return candidates[np.argsort(-scores[candidates])]


@dataclass
class InMemoryCollection:
    """In-memory storage for a single collection."""

    name: str
    dimension: int
    distance_metric: str = "cosine"
    documents: dict[DocumentId, VectorDocument] = field(default_factory=dict)
    embeddings: np.ndarray | None = None
    ids: list[DocumentId] = field(default_factory=list)

    def rebuild_index(self) -> None:
        """
        Rebuild the embedding matrix for search.

        The matrix is stored L2-NORMALIZED and C-contiguous so that a cosine search is
        a single BLAS matrix-vector product. Normalizing here (once per index build)
        rather than per query is what makes search O(N*d) FLOPs with no allocation
        instead of O(N*d) FLOPs plus a full-size temporary array on every call.
        """
        if not self.documents:
            self.embeddings = None
            self.ids = []
            return

        self.ids = list(self.documents.keys())
        vectors = [self.documents[doc_id].embedding for doc_id in self.ids]

        # `concatenate` + reshape rather than `vstack`: vstack runs atleast_2d on every
        # row, which is one Python-level call per entry. Row length is already checked
        # against `dimension` by the caller, so the reshape is exact.
        matrix = (
            np.concatenate(vectors)
            .reshape(len(self.ids), self.dimension)
            .astype(np.float32, copy=False)
        )

        # einsum accumulates the row norms in a single pass. np.linalg.norm instead
        # materializes `matrix * matrix` -- a SECOND full-size array, 153 MB at
        # 100k x 384 -- and that temporary was about half the cost of a rebuild. Rebuilds
        # are not rare: every upsert and every delete triggers one, so a bulk load of
        # 100k entries and a one-document delete both paid it. Together with the
        # concatenate above, 2.26x-2.55x over 10k-100k entries (A/A noise 1.01x-1.08x).
        norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
        np.maximum(norms, 1e-9, out=norms)
        matrix /= norms[:, None]

        self.embeddings = np.ascontiguousarray(matrix, dtype=np.float32)


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store implementation.

    Uses brute-force cosine similarity search. Suitable for:
    - Testing and development
    - Small datasets (<10,000 vectors)
    - Quick prototyping

    Example:
        config = VectorStoreConfig(collection_name="test", dimension=768)
        store = InMemoryVectorStore(config)

        store.upsert([VectorDocument(id="1", embedding=vec, payload={"name": "test"})])
        results = store.search(query_vec, top_k=5)
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        super().__init__(config)
        self._collections: dict[str, InMemoryCollection] = {}
        self._lock = threading.RLock()

        # Create default collection
        self._create_collection_internal(config)

    @property
    def store_type(self) -> str:
        """Store type identifier."""
        return "memory"

    def _create_collection_internal(self, config: VectorStoreConfig) -> InMemoryCollection:
        """Create collection without locking."""
        collection = InMemoryCollection(
            name=config.collection_name,
            dimension=config.dimension,
            distance_metric=config.distance_metric,
        )
        self._collections[config.collection_name] = collection
        return collection

    def create_collection(self, config: VectorStoreConfig) -> Result[CollectionInfo]:
        """Create a new collection."""
        with self._lock:
            if config.collection_name in self._collections:
                return Result.failure(f"Collection '{config.collection_name}' already exists")

            collection = self._create_collection_internal(config)

            return Result.success(
                CollectionInfo(
                    name=collection.name,
                    dimension=collection.dimension,
                    count=0,
                    index_type="brute_force",
                    distance_metric=collection.distance_metric,
                )
            )

    def delete_collection(self, name: str) -> Result[bool]:
        """Delete a collection."""
        with self._lock:
            if name not in self._collections:
                return Result.failure(f"Collection '{name}' not found")

            del self._collections[name]
            return Result.success(True)

    def get_collection_info(self, name: str) -> Result[CollectionInfo]:
        """Get collection information."""
        with self._lock:
            if name not in self._collections:
                return Result.failure(f"Collection '{name}' not found")

            collection = self._collections[name]
            return Result.success(
                CollectionInfo(
                    name=collection.name,
                    dimension=collection.dimension,
                    count=len(collection.documents),
                    index_type="brute_force",
                    distance_metric=collection.distance_metric,
                )
            )

    def _upsert_internal(
        self,
        documents: Sequence[VectorDocument],
        collection: str,
    ) -> int:
        """Internal upsert implementation."""
        with self._lock:
            if collection not in self._collections:
                raise ValueError(f"Collection '{collection}' not found")

            coll = self._collections[collection]

            for doc in documents:
                # Validate dimension
                if len(doc.embedding) != coll.dimension:
                    raise ValueError(
                        f"Embedding dimension {len(doc.embedding)} "
                        f"doesn't match collection dimension {coll.dimension}"
                    )
                coll.documents[doc.id] = doc

            # Rebuild search index
            coll.rebuild_index()

            return len(documents)

    def _delete_internal(
        self,
        ids: Sequence[DocumentId],
        collection: str,
    ) -> int:
        """Internal delete implementation."""
        with self._lock:
            if collection not in self._collections:
                raise ValueError(f"Collection '{collection}' not found")

            coll = self._collections[collection]
            deleted = 0

            for doc_id in ids:
                if doc_id in coll.documents:
                    del coll.documents[doc_id]
                    deleted += 1

            # Rebuild search index
            coll.rebuild_index()

            return deleted

    def _snapshot(
        self,
        collection: str,
    ) -> tuple[np.ndarray | None, list[DocumentId], dict[DocumentId, VectorDocument]]:
        """
        Take a consistent view of a collection's index, holding the lock only for that.

        `rebuild_index` REPLACES `embeddings`/`ids` rather than mutating them in place,
        so the snapshot stays internally consistent once taken and the numpy work can run
        outside the lock. Holding an RLock across the matrix product serialized every
        concurrent caller and made the ThreadPoolExecutor in BatchProcessor useless.
        """
        with self._lock:
            if collection not in self._collections:
                raise ValueError(f"Collection '{collection}' not found")

            coll = self._collections[collection]
            return coll.embeddings, coll.ids, coll.documents

    @staticmethod
    def _filter_mask(
        ids: list[DocumentId],
        documents: dict[DocumentId, VectorDocument],
        filter: dict[str, Any],
    ) -> np.ndarray:
        """Boolean mask over `ids` of the documents whose payload matches `filter`."""
        mask = np.ones(len(ids), dtype=bool)
        for i, doc_id in enumerate(ids):
            doc = documents.get(doc_id)
            if doc is None:
                mask[i] = False
                continue
            for key, value in filter.items():
                if doc.payload.get(key) != value:
                    mask[i] = False
                    break
        return mask

    @staticmethod
    def _results_from_indices(
        top_indices: np.ndarray,
        similarities: np.ndarray,
        ids: list[DocumentId],
        documents: dict[DocumentId, VectorDocument],
        include_embeddings: bool,
    ) -> list[SearchResult]:
        """Resolve ranked row indices back to SearchResults, dropping filtered-out rows."""
        results: list[SearchResult] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score == -np.inf:
                continue

            doc_id = ids[idx]
            doc = documents.get(doc_id)
            if doc is None:
                continue

            results.append(
                SearchResult(
                    id=doc_id,
                    score=score,
                    payload=doc.payload,
                    embedding=doc.embedding if include_embeddings else None,
                )
            )

        return results

    def _search_one(
        self,
        embeddings: np.ndarray,
        ids: list[DocumentId],
        documents: dict[DocumentId, VectorDocument],
        query_embedding: EmbeddingVector,
        k: int,
        mask: np.ndarray | None,
        include_embeddings: bool,
    ) -> list[SearchResult]:
        """Score one query against an already-snapshotted index."""
        # Normalize query only; the corpus matrix is already normalized.
        query = np.asarray(query_embedding, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-9)

        similarities = embeddings @ query

        if mask is not None:
            similarities = np.where(mask, similarities, -np.inf)

        return self._results_from_indices(
            _top_k_indices(similarities, k), similarities, ids, documents, include_embeddings
        )

    def _search_internal(
        self,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str,
        filter: dict[str, Any] | None,
        include_embeddings: bool,
    ) -> list[SearchResult]:
        """Internal search implementation using cosine similarity."""
        embeddings, ids, documents = self._snapshot(collection)

        if embeddings is None or len(ids) == 0:
            return []

        mask = self._filter_mask(ids, documents, filter) if filter else None
        return self._search_one(
            embeddings,
            ids,
            documents,
            query_embedding,
            min(top_k, len(ids)),
            mask,
            include_embeddings,
        )

    def search_batch(
        self,
        query_embeddings: Sequence[EmbeddingVector],
        top_k: int = 10,
        collection: str | None = None,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> Result[list[list[SearchResult]]]:
        """
        Search many queries at once. Returns one result list per query, in input order.

        `search` scores one query with a matrix-VECTOR product, which streams the entire
        corpus matrix out of RAM -- 153 MB at 100k x 384 -- and is bandwidth-bound rather
        than compute-bound. F separate calls therefore read that 153 MB F times over.
        Batching reads it once per chunk and turns the work into a GEMM, which is what the
        hardware is actually good at.

        Measured on a 32-core CPU, top_k=100, min of 8 interleaved in both orders. The
        same code A/B'd against itself through this harness lands within 1.01x-1.12x, so
        treat anything in that band as noise rather than as a result:

            entries   queries   per-query loop   search_batch   speedup
              1,000        64           6.3 ms         6.4 ms     0.98x
             10,000        64          26.2 ms         9.5 ms     2.76x
             30,000        64          62.1 ms        17.3 ms     3.60x
            100,000        64         248.5 ms        58.7 ms     4.23x
            100,000       688        2391.7 ms       427.9 ms     5.59x

        On the real FHIR dictionary (4,598 entries, 1,556 queries, BGE embeddings):
        564 ms -> 189 ms, 2.98x, or 363 us -> 122 us per query.

        There is deliberately NO small-corpus fallback to the per-query loop. An earlier
        version had one, on the theory that a corpus already in cache has no memory traffic
        to amortize; the apparent loss it was guarding (0.75x at 793 entries) turned out to
        be a threading artifact of measuring on a saturated box. Sweeping OpenBLAS threads
        at 793 entries, identical FLOPs: 1 thread, loop 14.7 ms vs GEMM 7.4 ms (GEMM 2.00x
        FASTER); 4 threads, GEMM 2.51x faster; 24 threads, GEMM 176.1 ms (0.09x). Under the
        cache explanation the single-threaded run would show the loss too. It shows a gain,
        so the loss was OpenBLAS spreading 24 threads across a tiny GEMM on a busy machine,
        not the corpus fitting in cache. The GEMM path always runs.

        This is still the EXACT scan -- no approximation, every entry is scored. It is not
        bit-identical to the per-query loop, though: BLAS accumulates a matrix-matrix
        product in a different order than a matrix-vector one, so scores can move by a few
        float32 ULPs and entries that were already tied can swap. Measured on the FHIR
        dictionary (4,598 entries, 1,556 queries, real BGE embeddings): scores agreed to
        1.0e-06, 123 of 1,556 top-100 lists reordered somewhere, 4 differed in membership
        at the rank-100 boundary, 2 had a different top-1 -- all of them exact ties (median
        score gap at the swap 0.0). P@1, MRR@10 and Recall@10 were unchanged to four
        decimals. Each path is deterministic run to run.

        Callers holding every query already (a whole schema's fields, say) should prefer
        this; there is no reason to use it for a single query.

        Args:
            query_embeddings: Query vectors, one row per query.
            top_k: Number of results per query.
            collection: Collection name (uses default if None).
            filter: Metadata filter, applied to every query in the batch.
            include_embeddings: Include vectors in results.

        Returns:
            Result containing one list of SearchResults per input query.
        """
        coll = collection or self._default_collection

        try:
            return Result.success(
                self._search_batch_internal(
                    query_embeddings, top_k, coll, filter, include_embeddings
                )
            )
        except Exception as e:
            return Result.failure(f"Batch search failed: {e}")

    def _search_batch_internal(
        self,
        query_embeddings: Sequence[EmbeddingVector],
        top_k: int,
        collection: str,
        filter: dict[str, Any] | None,
        include_embeddings: bool,
    ) -> list[list[SearchResult]]:
        """Internal batched search; see `search_batch` for the measurements."""
        embeddings, ids, documents = self._snapshot(collection)

        n_queries = len(query_embeddings)
        if embeddings is None or len(ids) == 0 or n_queries == 0:
            return [[] for _ in range(n_queries)]

        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32).reshape(n_queries, -1)
        if queries.shape[1] != embeddings.shape[1]:
            raise ValueError(
                f"Query dimension {queries.shape[1]} doesn't match collection "
                f"dimension {embeddings.shape[1]}"
            )

        # One mask for the whole batch. Building it walks every id in Python, which costs
        # 30 ms at 100k entries -- more than ten times the matrix product it guards. Per
        # query that is the dominant cost of a filtered search; here it is paid once.
        mask = self._filter_mask(ids, documents, filter) if filter else None

        n_entries = len(ids)
        k = min(top_k, n_entries)

        norms = np.sqrt(np.einsum("ij,ij->i", queries, queries))
        np.maximum(norms, 1e-9, out=norms)
        queries = queries / norms[:, None]

        chunk = min(_MAX_CHUNK, max(_MIN_CHUNK, _SCORE_BLOCK_BYTES // (n_entries * 4)))

        out: list[list[SearchResult]] = []
        for start in range(0, n_queries, chunk):
            block = queries[start : start + chunk]

            # QUERIES ON THE LEFT. The score block comes out (chunk x N): one row per
            # query, the corpus on the fastest-varying axis, so each query's scores are a
            # CONTIGUOUS row that the top-k below can walk in order. The transposed form
            # (`embeddings @ block.T`) computes exactly the same numbers -- it is the same
            # GEMM -- but lands them (N x chunk), which forces every top-k to stride down a
            # column of the block. A rewrite to that orientation was tried and reverted: it
            # did not reproduce a win at any scale and was a net loss at the largest one.
            # Note that a differential test against the per-query loop CANNOT catch a flip
            # here, since both orientations return the same results; see
            # test_search_batch_scores_each_query_as_a_contiguous_row.
            scores = block @ embeddings.T

            if mask is not None:
                # The mask is per-ENTRY and entries are the last axis, so it broadcasts
                # across every query row unchanged.
                scores = np.where(mask, scores, -np.inf)

            # One row per query, in input order.
            for row in scores:
                out.append(
                    self._results_from_indices(
                        _top_k_indices(row, k), row, ids, documents, include_embeddings
                    )
                )

        return out

    def _get_by_id_internal(
        self,
        id: DocumentId,
        collection: str,
        include_embedding: bool,
    ) -> VectorDocument | None:
        """Internal get by ID implementation."""
        with self._lock:
            if collection not in self._collections:
                raise ValueError(f"Collection '{collection}' not found")

            coll = self._collections[collection]
            doc = coll.documents.get(id)

            if doc is None:
                return None

            if not include_embedding:
                return VectorDocument(
                    id=doc.id,
                    embedding=np.array([]),
                    payload=doc.payload,
                )

            return doc

    def clear_all(self) -> int:
        """Clear all collections."""
        with self._lock:
            total = sum(len(c.documents) for c in self._collections.values())
            for collection in self._collections.values():
                collection.documents.clear()
                collection.rebuild_index()
            return total
