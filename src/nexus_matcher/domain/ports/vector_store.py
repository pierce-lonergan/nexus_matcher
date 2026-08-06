"""
nexus_matcher.domain.ports.vector_store | Layer: DOMAIN
Port interface for vector storage and similarity search.

## Relationships
# USED_BY    → domain/services/search_service :: dense retrieval
# USED_BY    → infrastructure/adapters/vector_stores/* :: implementations
# DEPENDS_ON → shared/types/base :: EmbeddingVector, DocumentId

## Attributes
# Security: No PII in vectors, but IDs may reference sensitive data
# Performance: Should support ANN search with configurable parameters
# Reliability: Should handle concurrent access safely

## Extension Points
# Plugin: nexus_matcher.vector_stores entry point
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from nexus_matcher.shared.types.base import DocumentId, EmbeddingVector, Result, Score

# =============================================================================
# VECTOR STORE TYPES
# =============================================================================


@dataclass(frozen=True)
class VectorDocument:
    """Document with embedding for storage."""

    id: DocumentId
    embedding: EmbeddingVector
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Result from vector similarity search."""

    id: DocumentId
    score: Score
    payload: dict[str, Any] = field(default_factory=dict)
    embedding: EmbeddingVector | None = None


@dataclass(frozen=True)
class CollectionInfo:
    """Information about a vector collection."""

    name: str
    dimension: int
    count: int
    index_type: str = "unknown"
    distance_metric: str = "cosine"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for vector store operations."""

    collection_name: str = "default"
    dimension: int = 768
    distance_metric: str = "cosine"  # cosine, euclidean, dot
    index_params: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# VECTOR STORE PROTOCOL
# =============================================================================


@runtime_checkable
class VectorStore(Protocol):
    """
    Protocol for vector stores.

    Implementations provide vector storage and similarity search
    (Qdrant, FAISS, Pinecone, in-memory, etc.).

    Example usage:
        store = QdrantVectorStore(config)
        store.upsert([VectorDocument(id="1", embedding=emb, payload={"name": "test"})])

        results = store.search(query_embedding, top_k=10)
        for result in results:
            print(f"{result.id}: {result.score}")
    """

    @property
    def store_type(self) -> str:
        """Get store type identifier."""
        ...

    def create_collection(
        self,
        config: VectorStoreConfig,
    ) -> Result[CollectionInfo]:
        """
        Create a new collection.

        Args:
            config: Collection configuration

        Returns:
            Result containing CollectionInfo on success
        """
        ...

    def delete_collection(self, name: str) -> Result[bool]:
        """
        Delete a collection.

        Args:
            name: Collection name

        Returns:
            Result containing True on success
        """
        ...

    def get_collection_info(self, name: str) -> Result[CollectionInfo]:
        """
        Get collection information.

        Args:
            name: Collection name

        Returns:
            Result containing CollectionInfo on success
        """
        ...

    def upsert(
        self,
        documents: Sequence[VectorDocument],
        collection: str | None = None,
    ) -> Result[int]:
        """
        Insert or update documents.

        Args:
            documents: Documents to upsert
            collection: Collection name (uses default if None)

        Returns:
            Result containing number of documents upserted
        """
        ...

    def delete(
        self,
        ids: Sequence[DocumentId],
        collection: str | None = None,
    ) -> Result[int]:
        """
        Delete documents by ID.

        Args:
            ids: Document IDs to delete
            collection: Collection name

        Returns:
            Result containing number of documents deleted
        """
        ...

    def search(
        self,
        query_embedding: EmbeddingVector,
        top_k: int = 10,
        collection: str | None = None,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> Result[list[SearchResult]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            collection: Collection name
            filter: Metadata filter
            include_embeddings: Include vectors in results

        Returns:
            Result containing list of SearchResults
        """
        ...

    def get_by_id(
        self,
        id: DocumentId,
        collection: str | None = None,
        include_embedding: bool = True,
    ) -> Result[VectorDocument | None]:
        """
        Get document by ID.

        Args:
            id: Document ID
            collection: Collection name
            include_embedding: Include vector in result

        Returns:
            Result containing VectorDocument or None if not found
        """
        ...


# =============================================================================
# BASE IMPLEMENTATION
# =============================================================================


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.

    Provides common functionality and validation.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        self._config = config
        self._default_collection = config.collection_name

    @property
    @abstractmethod
    def store_type(self) -> str:
        """Store type identifier."""
        ...

    @abstractmethod
    def create_collection(self, config: VectorStoreConfig) -> Result[CollectionInfo]:
        """Create collection."""
        ...

    @abstractmethod
    def delete_collection(self, name: str) -> Result[bool]:
        """Delete collection."""
        ...

    @abstractmethod
    def get_collection_info(self, name: str) -> Result[CollectionInfo]:
        """Get collection info."""
        ...

    @abstractmethod
    def _upsert_internal(
        self,
        documents: Sequence[VectorDocument],
        collection: str,
    ) -> int:
        """Internal upsert implementation."""
        ...

    @abstractmethod
    def _delete_internal(
        self,
        ids: Sequence[DocumentId],
        collection: str,
    ) -> int:
        """Internal delete implementation."""
        ...

    @abstractmethod
    def _search_internal(
        self,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str,
        filter: dict[str, Any] | None,
        include_embeddings: bool,
    ) -> list[SearchResult]:
        """Internal search implementation."""
        ...

    @abstractmethod
    def _get_by_id_internal(
        self,
        id: DocumentId,
        collection: str,
        include_embedding: bool,
    ) -> VectorDocument | None:
        """Internal get by ID implementation."""
        ...

    def upsert(
        self,
        documents: Sequence[VectorDocument],
        collection: str | None = None,
    ) -> Result[int]:
        """Upsert documents."""
        if not documents:
            return Result.success(0)

        coll = collection or self._default_collection

        try:
            count = self._upsert_internal(documents, coll)
            return Result.success(count)
        except Exception as e:
            return Result.failure(f"Upsert failed: {e}")

    def delete(
        self,
        ids: Sequence[DocumentId],
        collection: str | None = None,
    ) -> Result[int]:
        """Delete documents."""
        if not ids:
            return Result.success(0)

        coll = collection or self._default_collection

        try:
            count = self._delete_internal(ids, coll)
            return Result.success(count)
        except Exception as e:
            return Result.failure(f"Delete failed: {e}")

    def search(
        self,
        query_embedding: EmbeddingVector,
        top_k: int = 10,
        collection: str | None = None,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> Result[list[SearchResult]]:
        """Search for similar vectors."""
        coll = collection or self._default_collection

        try:
            results = self._search_internal(
                query_embedding, top_k, coll, filter, include_embeddings
            )
            return Result.success(results)
        except Exception as e:
            return Result.failure(f"Search failed: {e}")

    def get_by_id(
        self,
        id: DocumentId,
        collection: str | None = None,
        include_embedding: bool = True,
    ) -> Result[VectorDocument | None]:
        """Get document by ID."""
        coll = collection or self._default_collection

        try:
            doc = self._get_by_id_internal(id, coll, include_embedding)
            return Result.success(doc)
        except Exception as e:
            return Result.failure(f"Get failed: {e}")


# =============================================================================
# REGISTRY
# =============================================================================


class VectorStoreRegistry:
    """Registry for vector store implementations."""

    def __init__(self) -> None:
        self._factories: dict[str, type[VectorStore]] = {}

    def register(self, store_type: str, factory: type[VectorStore]) -> None:
        """Register a vector store factory."""
        self._factories[store_type] = factory

    def create(
        self,
        store_type: str,
        config: VectorStoreConfig,
        **kwargs,
    ) -> VectorStore | None:
        """Create a vector store instance."""
        factory = self._factories.get(store_type)
        if factory:
            return factory(config, **kwargs)
        return None

    def list_stores(self) -> list[str]:
        """Get list of registered store types."""
        return list(self._factories.keys())

    def load_entry_points(self) -> None:
        """Load stores from entry points."""
        try:
            from importlib.metadata import entry_points

            eps = entry_points(group="nexus_matcher.vector_stores")
            for ep in eps:
                try:
                    store_class = ep.load()
                    self.register(ep.name, store_class)
                except Exception as e:
                    import logging

                    logging.warning(f"Failed to load vector store {ep.name}: {e}")
        except ImportError:
            pass
