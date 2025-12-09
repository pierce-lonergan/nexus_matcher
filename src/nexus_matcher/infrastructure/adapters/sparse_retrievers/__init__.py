"""
nexus_matcher.infrastructure.adapters.sparse_retrievers | Layer: INFRASTRUCTURE
Sparse retrieval adapter implementations (BM25, etc.).
"""

from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever

__all__ = ["BM25Retriever"]
