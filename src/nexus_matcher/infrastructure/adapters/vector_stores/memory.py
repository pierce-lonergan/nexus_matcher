"""
nexus_matcher.infrastructure.adapters.vector_stores.memory | Layer: INFRASTRUCTURE
In-memory vector store implementation for testing and development.

## Relationships
# IMPLEMENTS → domain/ports/vector_store :: VectorStore protocol
# USED_BY    → tests/* :: test fixtures
# USED_BY    → application/* :: development mode

## Attributes
# Security: No persistence, data lost on restart
# Performance: O(n) search (brute force), suitable for small datasets
# Reliability: Thread-safe with locks
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from nexus_matcher.domain.ports.vector_store import (
    BaseVectorStore,
    CollectionInfo,
    SearchResult,
    VectorDocument,
    VectorStoreConfig,
)
from nexus_matcher.shared.types.base import DocumentId, EmbeddingVector, Result


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
        """Rebuild the embedding matrix for search."""
        if not self.documents:
            self.embeddings = None
            self.ids = []
            return

        self.ids = list(self.documents.keys())
        vectors = [self.documents[doc_id].embedding for doc_id in self.ids]
        self.embeddings = np.vstack(vectors).astype(np.float32)


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

            return Result.success(CollectionInfo(
                name=collection.name,
                dimension=collection.dimension,
                count=0,
                index_type="brute_force",
                distance_metric=collection.distance_metric,
            ))

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
            return Result.success(CollectionInfo(
                name=collection.name,
                dimension=collection.dimension,
                count=len(collection.documents),
                index_type="brute_force",
                distance_metric=collection.distance_metric,
            ))

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

    def _search_internal(
        self,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str,
        filter: dict[str, Any] | None,
        include_embeddings: bool,
    ) -> list[SearchResult]:
        """Internal search implementation using cosine similarity."""
        with self._lock:
            if collection not in self._collections:
                raise ValueError(f"Collection '{collection}' not found")

            coll = self._collections[collection]

            if coll.embeddings is None or len(coll.ids) == 0:
                return []

            # Normalize query
            query = query_embedding.astype(np.float32)
            query_norm = query / (np.linalg.norm(query) + 1e-9)

            # Normalize collection embeddings
            norms = np.linalg.norm(coll.embeddings, axis=1, keepdims=True) + 1e-9
            normalized = coll.embeddings / norms

            # Compute cosine similarities
            similarities = np.dot(normalized, query_norm)

            # Apply filter if provided
            if filter:
                mask = np.ones(len(coll.ids), dtype=bool)
                for i, doc_id in enumerate(coll.ids):
                    doc = coll.documents[doc_id]
                    for key, value in filter.items():
                        if doc.payload.get(key) != value:
                            mask[i] = False
                            break
                similarities = np.where(mask, similarities, -np.inf)

            # Get top-k indices
            k = min(top_k, len(coll.ids))
            top_indices = np.argsort(similarities)[-k:][::-1]

            # Build results
            results: list[SearchResult] = []
            for idx in top_indices:
                if similarities[idx] == -np.inf:
                    continue

                doc_id = coll.ids[idx]
                doc = coll.documents[doc_id]

                results.append(SearchResult(
                    id=doc_id,
                    score=float(similarities[idx]),
                    payload=doc.payload,
                    embedding=doc.embedding if include_embeddings else None,
                ))

            return results

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
