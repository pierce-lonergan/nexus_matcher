"""
tests.unit.infrastructure.test_qdrant_vector_store | Layer: TEST
Tests: Qdrant vector store | Target: src/infrastructure/adapters/vector_stores/qdrant.py

TDD Phase: RED → Tests written before implementation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from nexus_matcher.domain.ports.vector_store import (
    VectorStoreConfig,
    VectorDocument,
    CollectionInfo,
    SearchResult,
)
from nexus_matcher.shared.types.base import DataType

# Check if qdrant-client is available
try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

pytestmark = pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")


class TestQdrantVectorStoreProperties:
    """Test basic store properties."""

    def test_store_type(self):
        """Test store type is 'qdrant'."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            QdrantVectorStore,
        )

        config = VectorStoreConfig(collection_name="test", dimension=768)
        store = QdrantVectorStore(config, location=":memory:")
        assert store.store_type == "qdrant"

    def test_creates_with_config(self):
        """Test store accepts configuration."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            QdrantVectorStore,
        )

        config = VectorStoreConfig(
            collection_name="test_collection",
            dimension=384,
            distance_metric="cosine"
        )

        store = QdrantVectorStore(config, location=":memory:")
        assert store._config.collection_name == "test_collection"
        assert store._config.dimension == 384


class TestQdrantVectorStoreInMemory:
    """Test Qdrant with in-memory storage (real client)."""

    def test_create_and_use_collection(self):
        """Test complete workflow: create, upsert, search."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            QdrantVectorStore,
        )
        import uuid

        config = VectorStoreConfig(collection_name="workflow_test", dimension=4)
        store = QdrantVectorStore(config, location=":memory:")
        
        # Create collection
        result = store.create_collection(config)
        assert result.is_success

        # Upsert documents with UUID-style IDs
        doc1_id = str(uuid.uuid4())
        doc2_id = str(uuid.uuid4())
        docs = [
            VectorDocument(
                id=doc1_id,
                embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                payload={"name": "first"}
            ),
            VectorDocument(
                id=doc2_id,
                embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                payload={"name": "second"}
            ),
        ]
        upsert_result = store.upsert(docs, collection="workflow_test")
        assert upsert_result.is_success
        assert upsert_result.unwrap() == 2

        # Search
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        search_result = store.search(query, top_k=2, collection="workflow_test")
        assert search_result.is_success
        results = search_result.unwrap()
        assert len(results) >= 1
        # doc1 should be closest to query
        assert results[0].id == doc1_id

    def test_create_collection_already_exists(self):
        """Test creating collection that already exists."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            QdrantVectorStore,
        )

        config = VectorStoreConfig(collection_name="dupe_test", dimension=4)
        store = QdrantVectorStore(config, location=":memory:")
        
        # Create first time
        store.create_collection(config)
        
        # Try to create again
        result = store.create_collection(config)
        assert result.is_failure
        assert "already exists" in result.error.lower()

    def test_get_collection_info(self):
        """Test getting collection info."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            QdrantVectorStore,
        )

        config = VectorStoreConfig(collection_name="info_test", dimension=8)
        store = QdrantVectorStore(config, location=":memory:")
        store.create_collection(config)

        result = store.get_collection_info("info_test")

        assert result.is_success
        info = result.unwrap()
        assert info.dimension == 8

    def test_upsert_empty_list(self):
        """Test upserting empty list returns 0."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            QdrantVectorStore,
        )

        config = VectorStoreConfig(collection_name="empty_test", dimension=4)
        store = QdrantVectorStore(config, location=":memory:")

        result = store.upsert([])

        assert result.is_success
        assert result.unwrap() == 0

    def test_delete_empty_list(self):
        """Test deleting empty list returns 0."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            QdrantVectorStore,
        )

        config = VectorStoreConfig(collection_name="del_empty_test", dimension=4)
        store = QdrantVectorStore(config, location=":memory:")

        result = store.delete([])

        assert result.is_success
        assert result.unwrap() == 0

    def test_get_nonexistent_document(self):
        """Test getting non-existent document."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            QdrantVectorStore,
        )
        import uuid

        config = VectorStoreConfig(collection_name="get_test", dimension=4)
        store = QdrantVectorStore(config, location=":memory:")
        store.create_collection(config)

        result = store.get_by_id(str(uuid.uuid4()), collection="get_test")

        assert result.is_success
        assert result.unwrap() is None

    def test_delete_collection(self):
        """Test deleting a collection."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            QdrantVectorStore,
        )

        config = VectorStoreConfig(collection_name="to_delete", dimension=4)
        store = QdrantVectorStore(config, location=":memory:")
        store.create_collection(config)

        result = store.delete_collection("to_delete")

        assert result.is_success

    def test_search_with_filter(self):
        """Test search with metadata filter."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            QdrantVectorStore,
        )
        import uuid

        config = VectorStoreConfig(collection_name="filter_test", dimension=4)
        store = QdrantVectorStore(config, location=":memory:")
        store.create_collection(config)

        # Insert documents with different domains
        cust_id = str(uuid.uuid4())
        acct_id = str(uuid.uuid4())
        docs = [
            VectorDocument(
                id=cust_id,
                embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                payload={"domain": "customer"}
            ),
            VectorDocument(
                id=acct_id,
                embedding=np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
                payload={"domain": "account"}
            ),
        ]
        store.upsert(docs, collection="filter_test")

        # Search with filter - should only get customer domain
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = store.search(
            query, 
            top_k=2, 
            filter={"domain": "customer"}, 
            collection="filter_test"
        )

        assert result.is_success
        results = result.unwrap()
        # Should only have customer domain results
        for r in results:
            assert r.payload.get("domain") == "customer"


class TestQdrantVectorStoreDistanceMetrics:
    """Test distance metric handling."""

    def test_cosine_distance(self):
        """Test cosine distance metric mapping."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            _map_distance_metric,
        )
        from qdrant_client import models

        assert _map_distance_metric("cosine") == models.Distance.COSINE

    def test_euclidean_distance(self):
        """Test euclidean distance metric mapping."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            _map_distance_metric,
        )
        from qdrant_client import models

        assert _map_distance_metric("euclidean") == models.Distance.EUCLID

    def test_dot_distance(self):
        """Test dot product distance metric mapping."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            _map_distance_metric,
        )
        from qdrant_client import models

        assert _map_distance_metric("dot") == models.Distance.DOT


class TestQdrantVectorStoreErrorHandling:
    """Test error handling."""

    def test_get_collection_info_not_found(self):
        """Test getting info for non-existent collection."""
        from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import (
            QdrantVectorStore,
        )

        config = VectorStoreConfig(collection_name="test", dimension=4)
        store = QdrantVectorStore(config, location=":memory:")

        result = store.get_collection_info("nonexistent")

        assert result.is_failure
