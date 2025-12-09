"""
nexus_matcher.infrastructure.adapters.vector_stores.qdrant | Layer: INFRASTRUCTURE
Qdrant vector store implementation for production use.

## Relationships
# IMPLEMENTS → domain/ports/vector_store :: VectorStore protocol
# DEPENDS_ON → qdrant-client :: Qdrant Python client
# USED_BY    → application/use_cases/match_schema :: vector search

## Attributes
# Security: API key authentication for cloud instances
# Performance: HNSW index for O(log n) ANN search
# Reliability: Connection pooling, retry logic
"""

from __future__ import annotations

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


# Lazy import to avoid hard dependency
def _get_qdrant_client():
    """Get QdrantClient class with lazy import."""
    try:
        from qdrant_client import QdrantClient
        return QdrantClient
    except ImportError as e:
        raise ImportError(
            "qdrant-client is required for QdrantVectorStore. "
            "Install with: pip install qdrant-client"
        ) from e


def _get_qdrant_models():
    """Get Qdrant models with lazy import."""
    try:
        from qdrant_client import models
        return models
    except ImportError as e:
        raise ImportError(
            "qdrant-client is required for QdrantVectorStore. "
            "Install with: pip install qdrant-client"
        ) from e


def _map_distance_metric(metric: str):
    """Map distance metric string to Qdrant Distance enum."""
    models = _get_qdrant_models()
    
    mapping = {
        "cosine": models.Distance.COSINE,
        "euclidean": models.Distance.EUCLID,
        "euclid": models.Distance.EUCLID,
        "dot": models.Distance.DOT,
    }
    return mapping.get(metric.lower(), models.Distance.COSINE)


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store implementation.

    Production-grade vector store using Qdrant for:
    - Large-scale deployments (millions of vectors)
    - High-throughput search
    - Filtered vector search
    - Distributed deployments

    Example:
        # Local instance
        config = VectorStoreConfig(collection_name="entries", dimension=768)
        store = QdrantVectorStore(config, host="localhost", port=6333)

        # Cloud instance
        store = QdrantVectorStore(
            config,
            url="https://xxx.qdrant.io",
            api_key="your_api_key"
        )

        # In-memory (for testing)
        store = QdrantVectorStore(config, location=":memory:")

        # Use
        store.upsert([VectorDocument(id="1", embedding=vec, payload={})])
        results = store.search(query_vec, top_k=10)
    """

    def __init__(
        self,
        config: VectorStoreConfig,
        host: str | None = None,
        port: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
        prefer_grpc: bool = False,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize Qdrant vector store.

        Args:
            config: Vector store configuration
            host: Qdrant host for local connection
            port: Qdrant port for local connection
            url: Full URL for Qdrant Cloud
            api_key: API key for authentication
            location: Location for in-memory (":memory:") or path
            prefer_grpc: Use gRPC instead of HTTP
            timeout: Request timeout in seconds
        """
        super().__init__(config)

        QdrantClient = _get_qdrant_client()

        # Determine connection mode
        if location == ":memory:":
            self._client = QdrantClient(location=":memory:")
        elif url:
            self._client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )
        elif host:
            self._client = QdrantClient(
                host=host,
                port=port or 6333,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )
        else:
            # Default to localhost
            self._client = QdrantClient(
                host="localhost",
                port=6333,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )

        self._timeout = timeout

    @property
    def store_type(self) -> str:
        """Store type identifier."""
        return "qdrant"

    def create_collection(self, config: VectorStoreConfig) -> Result[CollectionInfo]:
        """Create a new collection."""
        try:
            models = _get_qdrant_models()

            # Check if collection already exists
            if self._client.collection_exists(config.collection_name):
                return Result.failure(
                    f"Collection '{config.collection_name}' already exists"
                )

            # Create collection
            self._client.create_collection(
                collection_name=config.collection_name,
                vectors_config=models.VectorParams(
                    size=config.dimension,
                    distance=_map_distance_metric(config.distance_metric),
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=config.index_params.get("m", 16),
                    ef_construct=config.index_params.get("ef_construct", 100),
                ),
            )

            return Result.success(CollectionInfo(
                name=config.collection_name,
                dimension=config.dimension,
                count=0,
                index_type="hnsw",
                distance_metric=config.distance_metric,
            ))

        except Exception as e:
            return Result.failure(f"Failed to create collection: {e}")

    def delete_collection(self, name: str) -> Result[bool]:
        """Delete a collection."""
        try:
            if not self._client.collection_exists(name):
                return Result.failure(f"Collection '{name}' not found")

            self._client.delete_collection(name)
            return Result.success(True)

        except Exception as e:
            return Result.failure(f"Failed to delete collection: {e}")

    def get_collection_info(self, name: str) -> Result[CollectionInfo]:
        """Get collection information."""
        try:
            info = self._client.get_collection(name)

            # Access dimension from config.params.vectors.size
            dimension = info.config.params.vectors.size

            return Result.success(CollectionInfo(
                name=name,
                dimension=dimension,
                count=info.points_count or 0,
                index_type="hnsw",
                distance_metric=str(info.config.params.vectors.distance.value).lower(),
            ))

        except Exception as e:
            return Result.failure(f"Failed to get collection info: {e}")

    def _upsert_internal(
        self,
        documents: Sequence[VectorDocument],
        collection: str,
    ) -> int:
        """Internal upsert implementation."""
        models = _get_qdrant_models()

        points = [
            models.PointStruct(
                id=doc.id,
                vector=doc.embedding.tolist(),
                payload=doc.payload,
            )
            for doc in documents
        ]

        self._client.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )

        return len(documents)

    def _delete_internal(
        self,
        ids: Sequence[DocumentId],
        collection: str,
    ) -> int:
        """Internal delete implementation."""
        models = _get_qdrant_models()

        self._client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=list(ids)),
            wait=True,
        )

        return len(ids)

    def _search_internal(
        self,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str,
        filter: dict[str, Any] | None,
        include_embeddings: bool,
    ) -> list[SearchResult]:
        """Internal search implementation."""
        models = _get_qdrant_models()

        # Build filter if provided
        qdrant_filter = None
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
            qdrant_filter = models.Filter(must=conditions)

        # Execute search using query_points (new API)
        results = self._client.query_points(
            collection_name=collection,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_vectors=include_embeddings,
            with_payload=True,
        )

        # Convert to SearchResult - new API returns QueryResponse with points
        return [
            SearchResult(
                id=str(point.id),
                score=point.score,
                payload=point.payload or {},
                embedding=np.array(point.vector, dtype=np.float32) if point.vector else None,
            )
            for point in results.points
        ]

    def _get_by_id_internal(
        self,
        id: DocumentId,
        collection: str,
        include_embedding: bool,
    ) -> VectorDocument | None:
        """Internal get by ID implementation."""
        results = self._client.retrieve(
            collection_name=collection,
            ids=[id],
            with_vectors=include_embedding,
            with_payload=True,
        )

        if not results:
            return None

        point = results[0]
        return VectorDocument(
            id=str(point.id),
            embedding=np.array(point.vector, dtype=np.float32) if point.vector else np.array([]),
            payload=point.payload or {},
        )

    def count(self, collection: str | None = None) -> Result[int]:
        """Get document count in collection."""
        coll = collection or self._default_collection
        try:
            info = self._client.get_collection(coll)
            return Result.success(info.vectors_count or 0)
        except Exception as e:
            return Result.failure(f"Failed to get count: {e}")

    def scroll(
        self,
        collection: str | None = None,
        limit: int = 100,
        offset: str | None = None,
        filter: dict[str, Any] | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Result[tuple[list[VectorDocument], str | None]]:
        """
        Scroll through collection.

        Returns:
            Tuple of (documents, next_offset)
        """
        models = _get_qdrant_models()
        coll = collection or self._default_collection

        try:
            # Build filter
            qdrant_filter = None
            if filter:
                conditions = [
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter.items()
                ]
                qdrant_filter = models.Filter(must=conditions)

            points, next_offset = self._client.scroll(
                collection_name=coll,
                limit=limit,
                offset=offset,
                scroll_filter=qdrant_filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            docs = [
                VectorDocument(
                    id=str(p.id),
                    embedding=np.array(p.vector, dtype=np.float32) if p.vector else np.array([]),
                    payload=p.payload or {},
                )
                for p in points
            ]

            return Result.success((docs, next_offset))

        except Exception as e:
            return Result.failure(f"Scroll failed: {e}")
