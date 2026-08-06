"""
nexus_matcher.infrastructure.adapters.vector_stores.hnsw | Layer: INFRASTRUCTURE
Approximate nearest-neighbour vector store backed by usearch (HNSW).

## Relationships
# IMPLEMENTS  -> domain/ports/vector_store :: VectorStore protocol
# DEPENDS_ON  -> usearch :: HNSW index
# USED_BY     -> application/use_cases/match_schema :: dense retrieval at scale
# ALTERNATIVE -> vector_stores/memory :: exact brute force, FASTER below ~50k entries

## Attributes
# Security: In-process only, no network calls
# Performance: ~1.65x end-to-end vs brute force at 100k entries; see the tables below
# Reliability: Approximate by construction -- recall is a tuned parameter, not a given

## Why this exists

The in-memory store does an exact O(N*d) scan. That is genuinely the right choice for a
small dictionary and it is what the defaults still use. But the cost is linear in
dictionary size on every query, so a large enterprise glossary makes dense retrieval the
bottleneck.

Two sets of numbers, and the difference between them is the point.

RAW INDEX ONLY (384-dim, 100k clustered vectors, CPU, no port layer):

    backend                       ms/query      qps    recall@10
    brute force                      2.92      342       1.000 (exact)
    HNSW 32 / 200 / 256              0.101    9853       0.997
    HNSW 48 / 300 / 512              0.193    5181       1.000

END TO END through this class (benchmarks/exp_scale.py, 100k entries, top_k=10):

    backend                       ms/query      qps      P@1
    InMemoryVectorStore              2.469     405      0.5889
    HnswVectorStore                  1.494     670      0.5914

So the realistic gain is about **1.65x**, not the 29x the raw index suggests. Once the
search drops to ~0.1 ms, the per-query cost is dominated by this class's own Python work
-- resolving keys to documents and constructing SearchResult/Result objects -- not by
the distance computation. Quote the end-to-end row, not the raw row.

The gap widens as the corpus grows, because the brute-force term is linear in N while
the Python overhead is constant. Below roughly 50k entries the exact store is simply the
better choice: it is faster in wall-clock AND exact.

## A warning about benchmarking this

An earlier version of this benchmark used i.i.d. Gaussian random vectors and measured
recall@50 of 0.087 -- which would have condemned HNSW entirely. That number was an
artifact. Random high-dimensional vectors are nearly equidistant, so there is no
neighbourhood structure for a graph index to exploit, and *every* ANN method degenerates
on them. Real text embeddings are strongly clustered and behave like the table above.
If you re-tune these parameters, benchmark against real embeddings from your own
dictionary, never against `np.random.standard_normal`.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Sequence
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

# usearch metric names keyed by the config's distance_metric.
_METRICS = {
    "cosine": "cos",
    "cos": "cos",
    "dot": "ip",
    "ip": "ip",
    "inner_product": "ip",
    "euclidean": "l2sq",
    "l2": "l2sq",
    "l2sq": "l2sq",
}


def _require_usearch():
    try:
        from usearch.index import Index
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "usearch is required for HnswVectorStore. Install with: pip install usearch"
        ) from exc
    return Index


class HnswVectorStore(BaseVectorStore):
    """
    Approximate vector store using a usearch HNSW index.

    Drop-in replacement for InMemoryVectorStore when the dictionary is large enough that
    an exact scan dominates query latency (roughly >50k entries).

    Args:
        config: Collection configuration (dimension, distance_metric).
        connectivity: HNSW graph degree (M). Higher = better recall, slower build,
            more memory.
        expansion_add: Build-time candidate list size (ef_construction).
        expansion_search: Query-time candidate list size (ef_search). This is the main
            recall/latency dial and can be changed after the index is built.

    Example:
        store = HnswVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=384)
        )
        store.upsert(docs)
        results = store.search(query_vec, top_k=10).unwrap()

    Note:
        Results are APPROXIMATE. If you need exact guarantees (e.g. a regression test
        asserting a specific ordering), use InMemoryVectorStore instead.
    """

    def __init__(
        self,
        config: VectorStoreConfig,
        connectivity: int = 32,
        expansion_add: int = 200,
        expansion_search: int = 256,
        build_threads: int = 0,
    ) -> None:
        super().__init__(config)
        self._connectivity = connectivity
        self._expansion_add = expansion_add
        self._expansion_search = expansion_search
        # 0 = use all cores. Parallel construction makes the resulting graph
        # NONDETERMINISTIC: neighbour selection depends on insertion interleaving, so
        # two builds over identical data give slightly different graphs (measured top-1
        # agreement against exact search varied 0.94-0.98 across five builds, versus
        # exactly 1.00 with a single thread). That is fine in production and worth the
        # ~30x faster build, but it makes any test asserting a recall floor flaky.
        # Pass build_threads=1 when reproducibility matters more than build speed.
        self._build_threads = build_threads

        self._lock = threading.RLock()
        self._documents: dict[DocumentId, VectorDocument] = {}
        # usearch addresses vectors by integer key, so we keep a bijection to doc ids.
        self._key_to_id: dict[int, DocumentId] = {}
        self._id_to_key: dict[DocumentId, int] = {}
        self._next_key = 0
        self._index = self._new_index()

    def _new_index(self):
        Index = _require_usearch()
        metric = _METRICS.get((self._config.distance_metric or "cosine").lower())
        if metric is None:
            raise ValueError(
                f"Unsupported distance_metric {self._config.distance_metric!r}; "
                f"expected one of {sorted(set(_METRICS))}"
            )
        return Index(
            ndim=self._config.dimension,
            metric=metric,
            dtype="f32",
            connectivity=self._connectivity,
            expansion_add=self._expansion_add,
            expansion_search=self._expansion_search,
        )

    @property
    def store_type(self) -> str:
        return "hnsw"

    # -- collection management ------------------------------------------------

    def create_collection(self, config: VectorStoreConfig) -> Result[CollectionInfo]:
        """This store holds exactly one collection, fixed at construction."""
        if config.collection_name != self._config.collection_name:
            return Result.failure(
                f"HnswVectorStore holds a single collection "
                f"({self._config.collection_name!r}); cannot create "
                f"{config.collection_name!r}"
            )
        return self.get_collection_info(config.collection_name)

    def delete_collection(self, name: str) -> Result[bool]:
        with self._lock:
            if name != self._config.collection_name:
                return Result.failure(f"Collection '{name}' not found")
            self._documents.clear()
            self._key_to_id.clear()
            self._id_to_key.clear()
            self._next_key = 0
            self._index = self._new_index()
            return Result.success(True)

    def get_collection_info(self, name: str) -> Result[CollectionInfo]:
        with self._lock:
            if name != self._config.collection_name:
                return Result.failure(f"Collection '{name}' not found")
            return Result.success(
                CollectionInfo(
                    name=name,
                    dimension=self._config.dimension,
                    count=len(self._documents),
                    index_type="hnsw",
                    distance_metric=self._config.distance_metric,
                )
            )

    # -- data -----------------------------------------------------------------

    def _upsert_internal(
        self,
        documents: Sequence[VectorDocument],
        collection: str,
    ) -> int:
        with self._lock:
            if collection != self._config.collection_name:
                raise ValueError(f"Collection '{collection}' not found")
            if not documents:
                return 0

            vectors = np.empty((len(documents), self._config.dimension), dtype=np.float32)
            keys = np.empty(len(documents), dtype=np.int64)

            for i, doc in enumerate(documents):
                if len(doc.embedding) != self._config.dimension:
                    raise ValueError(
                        f"Embedding dimension {len(doc.embedding)} doesn't match "
                        f"collection dimension {self._config.dimension}"
                    )

                key = self._id_to_key.get(doc.id)
                if key is None:
                    key = self._next_key
                    self._next_key += 1
                    self._id_to_key[doc.id] = key
                    self._key_to_id[key] = doc.id

                keys[i] = key
                vectors[i] = doc.embedding
                self._documents[doc.id] = doc

            # usearch handles re-adding an existing key as an update.
            self._index.add(keys, vectors, log=False, threads=self._build_threads)
            return len(documents)

    def _delete_internal(
        self,
        ids: Sequence[DocumentId],
        collection: str,
    ) -> int:
        with self._lock:
            if collection != self._config.collection_name:
                raise ValueError(f"Collection '{collection}' not found")

            deleted = 0
            for doc_id in ids:
                key = self._id_to_key.pop(doc_id, None)
                if key is None:
                    continue
                self._key_to_id.pop(key, None)
                self._documents.pop(doc_id, None)
                # Older usearch builds cannot remove; the id mapping has already
                # been dropped so the vector becomes unreachable either way and
                # is filtered out at search time.
                with contextlib.suppress(Exception):
                    self._index.remove(key)
                deleted += 1
            return deleted

    def _search_internal(
        self,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str,
        filter: dict[str, Any] | None,
        include_embeddings: bool,
    ) -> list[SearchResult]:
        with self._lock:
            if collection != self._config.collection_name:
                raise ValueError(f"Collection '{collection}' not found")
            if not self._documents:
                return []
            # Bind REFERENCES, never copies. Copying these dicts per query is O(N) and
            # silently reintroduces the linear cost the index exists to remove -- at
            # 100k entries it made HNSW measurably SLOWER than the exact brute-force
            # store (4.62 ms/query vs 2.60 ms) despite the search itself taking 0.1 ms.
            #
            # Safe because the lookups below are point `.get()` calls, never iteration.
            # A concurrent upsert or delete may add or remove a key mid-search, in which
            # case that one candidate resolves to None and is skipped; it cannot corrupt
            # the scan the way mutation-during-iteration would.
            index = self._index
            key_to_id = self._key_to_id
            documents = self._documents

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

        # A metadata filter rejects candidates AFTER the graph search, so ask for extra
        # neighbours up front; otherwise a selective filter can empty the result set.
        fetch = top_k if not filter else min(top_k * 10, len(documents))
        matches = index.search(query, fetch, log=False)

        results: list[SearchResult] = []
        for key, distance in zip(matches.keys, matches.distances, strict=False):
            doc_id = key_to_id.get(int(key))
            if doc_id is None:
                continue
            doc = documents.get(doc_id)
            if doc is None:
                continue

            if filter and any(doc.payload.get(k) != v for k, v in filter.items()):
                continue

            # usearch returns a DISTANCE; convert to a similarity so scores are
            # directionally consistent with InMemoryVectorStore (higher is better).
            results.append(
                SearchResult(
                    id=doc_id,
                    score=float(1.0 - distance),
                    payload=doc.payload,
                    embedding=doc.embedding if include_embeddings else None,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def _get_by_id_internal(
        self,
        id: DocumentId,
        collection: str,
        include_embedding: bool,
    ) -> VectorDocument | None:
        with self._lock:
            if collection != self._config.collection_name:
                raise ValueError(f"Collection '{collection}' not found")
            doc = self._documents.get(id)
            if doc is None:
                return None
            if not include_embedding:
                return VectorDocument(id=doc.id, embedding=np.array([]), payload=doc.payload)
            return doc

    # -- tuning ---------------------------------------------------------------

    @property
    def expansion_search(self) -> int:
        """Query-time candidate list size (ef_search)."""
        return self._expansion_search

    @expansion_search.setter
    def expansion_search(self, value: int) -> None:
        """
        Trade recall against latency without rebuilding the index.

        Measured at 100k x 384: 64 -> recall@10 0.710, 128 -> 0.899, 256 -> 0.997,
        512 -> 1.000.
        """
        with self._lock:
            self._expansion_search = value
            self._index.expansion_search = value
