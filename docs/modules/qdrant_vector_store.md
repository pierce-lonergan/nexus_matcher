# Module: Qdrant Vector Store Adapter

## Purpose

Production-grade vector store adapter for Qdrant. Supports both local and cloud Qdrant instances with optimized ANN (Approximate Nearest Neighbor) search using HNSW indexing.

## Domain Model

### Entities

- **QdrantVectorStore**: Production vector store implementation
  - Invariants: Collection must exist before operations, dimension must match
  - States: Connected, Disconnected
  - Events: N/A (infrastructure adapter)

### Value Objects

- **VectorDocument**: Document with embedding for storage
- **SearchResult**: Result from similarity search
- **CollectionInfo**: Metadata about a collection
- **VectorStoreConfig**: Configuration for store operations

### Domain Services

- Implements `VectorStore` protocol from domain/ports

## Qdrant Configuration

### Connection Options

```python
# Local instance
QdrantVectorStore(config, host="localhost", port=6333)

# Qdrant Cloud
QdrantVectorStore(config, url="https://xxx.qdrant.io", api_key="...")

# In-memory (for testing)
QdrantVectorStore(config, location=":memory:")
```

### Index Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| m | 16 | HNSW connections per node |
| ef_construct | 100 | HNSW construction search width |
| full_scan_threshold | 10000 | Switch to brute force below this |

### Distance Metrics

| Metric | Description |
|--------|-------------|
| Cosine | Normalized dot product (default) |
| Euclid | L2 distance |
| Dot | Inner product |

## Planned Implementation

- [x] Domain analysis complete
- [ ] QdrantVectorStore class
- [ ] Connection management
- [ ] Collection CRUD operations
- [ ] Vector upsert/delete
- [ ] Similarity search with filters
- [ ] Batch operations
- [ ] Unit tests (TDD)
- [ ] Integration tests

## Dependencies

```toml
[project.optional-dependencies]
qdrant = ["qdrant-client>=1.7.0"]
```

## Usage Example

```python
from nexus_matcher.infrastructure.adapters.vector_stores.qdrant import QdrantVectorStore
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig, VectorDocument

# Configure
config = VectorStoreConfig(
    collection_name="dictionary_entries",
    dimension=768,
    distance_metric="cosine"
)

# Connect to local Qdrant
store = QdrantVectorStore(config, host="localhost", port=6333)

# Or use in-memory for testing
store = QdrantVectorStore(config, location=":memory:")

# Upsert vectors
docs = [
    VectorDocument(
        id="entry_1",
        embedding=embedding_vector,
        payload={"name": "customer_id", "domain": "customer"}
    )
]
store.upsert(docs)

# Search
results = store.search(query_embedding, top_k=10)
for result in results.unwrap():
    print(f"{result.id}: {result.score}")
```

## File Structure

```
src/nexus_matcher/infrastructure/adapters/vector_stores/
├── __init__.py
├── memory.py        # Existing in-memory store
└── qdrant.py        # Qdrant production store (new)
```
