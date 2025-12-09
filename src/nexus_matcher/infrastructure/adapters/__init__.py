"""
nexus_matcher.infrastructure.adapters | Layer: INFRASTRUCTURE
Concrete implementations of domain ports.

Contains:
- schema_parsers/: Avro, JSON Schema, SQL DDL, CSV parsers
- dictionary_loaders/: Excel, CSV, Database loaders
- embedding_providers/: SentenceTransformers, OpenAI providers
- vector_stores/: Qdrant, FAISS, In-Memory stores
- sparse_retrievers/: BM25, Elasticsearch retrievers
- rerankers/: ColBERT, CrossEncoder rerankers
- caches/: Memory, Redis, Hierarchical caches
"""
