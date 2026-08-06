"""
nexus_matcher.domain.ports | Layer: DOMAIN
Port interfaces (abstractions) for external dependencies.

This module defines the contracts that infrastructure adapters must implement.
Following hexagonal architecture, the domain layer depends only on these
abstractions, not on concrete implementations.

## Port Categories

### Data Input Ports
- SchemaParser: Parse schemas from various formats (Avro, JSON Schema, SQL DDL)
- DictionaryLoader: Load data dictionaries from various sources (Excel, CSV, DB)

### ML/AI Ports
- EmbeddingProvider: Generate vector embeddings from text

### Storage Ports
- VectorStore: Store and search vector embeddings
- SparseRetriever: Sparse/lexical search (BM25)
- Cache: Result caching with multiple levels

### Processing Ports
- Reranker: Neural reranking of search results

## Usage

```python
from nexus_matcher.domain.ports import (
    SchemaParser,
    DictionaryLoader,
    EmbeddingProvider,
    VectorStore,
    SparseRetriever,
    Reranker,
    Cache,
)

# Type hint with protocols
def match_schema(
    parser: SchemaParser,
    embedder: EmbeddingProvider,
    store: VectorStore,
) -> list[MatchResult]:
    ...
```
"""

from nexus_matcher.domain.ports.cache import (
    BaseCache,
    Cache,
    CacheConfig,
    CacheRegistry,
    CacheStats,
    HierarchicalCache,
    HierarchicalCacheStats,
    SemanticCache,
    SemanticCacheConfig,
)
from nexus_matcher.domain.ports.dictionary_loader import (
    BaseDictionaryLoader,
    ColumnMapping,
    DictionaryLoader,
    DictionaryLoaderRegistry,
    LoadStatistics,
)
from nexus_matcher.domain.ports.embedding_provider import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingProviderRegistry,
    EmbeddingResult,
)
from nexus_matcher.domain.ports.retrieval import (
    BaseReranker,
    BaseSparseRetriever,
    RerankCandidate,
    Reranker,
    RerankerRegistry,
    RerankResult,
    SparseDocument,
    SparseRetriever,
    SparseRetrieverRegistry,
    SparseSearchResult,
)
from nexus_matcher.domain.ports.schema_parser import (
    BaseSchemaParser,
    SchemaParser,
    SchemaParserRegistry,
)
from nexus_matcher.domain.ports.vector_store import (
    BaseVectorStore,
    CollectionInfo,
    SearchResult,
    VectorDocument,
    VectorStore,
    VectorStoreConfig,
    VectorStoreRegistry,
)

__all__ = [
    "BaseCache",
    "BaseDictionaryLoader",
    "BaseEmbeddingProvider",
    "BaseReranker",
    "BaseSchemaParser",
    "BaseSparseRetriever",
    "BaseVectorStore",
    # Cache
    "Cache",
    "CacheConfig",
    "CacheRegistry",
    "CacheStats",
    "CollectionInfo",
    "ColumnMapping",
    # Dictionary Loader
    "DictionaryLoader",
    "DictionaryLoaderRegistry",
    "EmbeddingConfig",
    # Embedding Provider
    "EmbeddingProvider",
    "EmbeddingProviderRegistry",
    "EmbeddingResult",
    "HierarchicalCache",
    "HierarchicalCacheStats",
    "LoadStatistics",
    "RerankCandidate",
    "RerankResult",
    # Reranker
    "Reranker",
    "RerankerRegistry",
    # Schema Parser
    "SchemaParser",
    "SchemaParserRegistry",
    "SearchResult",
    "SemanticCache",
    "SemanticCacheConfig",
    "SparseDocument",
    # Sparse Retriever
    "SparseRetriever",
    "SparseRetrieverRegistry",
    "SparseSearchResult",
    "VectorDocument",
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "VectorStoreRegistry",
]
