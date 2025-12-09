"""
nexus_matcher.domain.ports.retrieval | Layer: DOMAIN
Port interfaces for sparse retrieval (BM25) and reranking.

## Relationships
# USED_BY    → domain/services/search_service :: hybrid retrieval
# USED_BY    → infrastructure/adapters/sparse_retrievers/* :: BM25 implementations
# USED_BY    → infrastructure/adapters/rerankers/* :: reranker implementations
# DEPENDS_ON → shared/types/base :: DocumentId, Score

## Attributes
# Security: Document content may be sensitive
# Performance: BM25 should be O(n) for index build, O(log n) for search
# Reliability: Should handle empty indices gracefully
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from nexus_matcher.shared.types.base import DocumentId, Result, Score


# =============================================================================
# SPARSE RETRIEVAL TYPES
# =============================================================================


@dataclass(frozen=True)
class SparseDocument:
    """Document for sparse indexing."""

    id: DocumentId
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SparseSearchResult:
    """Result from sparse search."""

    id: DocumentId
    score: Score
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SPARSE RETRIEVER PROTOCOL
# =============================================================================


@runtime_checkable
class SparseRetriever(Protocol):
    """
    Protocol for sparse retrieval (BM25, Elasticsearch, etc.).

    Example usage:
        retriever = BM25Retriever()
        retriever.index([SparseDocument(id="1", text="customer email address")])
        results = retriever.search("email", top_k=10)
    """

    @property
    def retriever_type(self) -> str:
        """Get retriever type identifier."""
        ...

    def index(self, documents: Sequence[SparseDocument]) -> Result[int]:
        """
        Build or update the index.

        Args:
            documents: Documents to index

        Returns:
            Result containing number of documents indexed
        """
        ...

    def add(self, documents: Sequence[SparseDocument]) -> Result[int]:
        """
        Add documents to existing index.

        Args:
            documents: Documents to add

        Returns:
            Result containing number of documents added
        """
        ...

    def remove(self, ids: Sequence[DocumentId]) -> Result[int]:
        """
        Remove documents from index.

        Args:
            ids: Document IDs to remove

        Returns:
            Result containing number of documents removed
        """
        ...

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> Result[list[SparseSearchResult]]:
        """
        Search the index.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Result containing list of search results
        """
        ...

    def save(self, path: str) -> Result[bool]:
        """Save index to disk."""
        ...

    def load(self, path: str) -> Result[bool]:
        """Load index from disk."""
        ...


# =============================================================================
# BASE SPARSE RETRIEVER
# =============================================================================


class BaseSparseRetriever(ABC):
    """Abstract base class for sparse retrievers."""

    @property
    @abstractmethod
    def retriever_type(self) -> str:
        """Retriever type identifier."""
        ...

    @abstractmethod
    def index(self, documents: Sequence[SparseDocument]) -> Result[int]:
        """Build index."""
        ...

    @abstractmethod
    def add(self, documents: Sequence[SparseDocument]) -> Result[int]:
        """Add to index."""
        ...

    @abstractmethod
    def remove(self, ids: Sequence[DocumentId]) -> Result[int]:
        """Remove from index."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> Result[list[SparseSearchResult]]:
        """Search index."""
        ...

    def save(self, path: str) -> Result[bool]:
        """Default save (may not be supported)."""
        return Result.failure("Save not supported by this retriever")

    def load(self, path: str) -> Result[bool]:
        """Default load (may not be supported)."""
        return Result.failure("Load not supported by this retriever")


# =============================================================================
# RERANKER TYPES
# =============================================================================


@dataclass(frozen=True)
class RerankCandidate:
    """Candidate for reranking."""

    id: DocumentId
    text: str
    initial_score: Score = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankResult:
    """Result from reranking."""

    id: DocumentId
    score: Score
    rank: int
    original_rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# RERANKER PROTOCOL
# =============================================================================


@runtime_checkable
class Reranker(Protocol):
    """
    Protocol for rerankers (ColBERT, CrossEncoder, etc.).

    Example usage:
        reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
        results = reranker.rerank(
            query="customer email",
            candidates=[RerankCandidate(id="1", text="email address")],
            top_k=5
        )
    """

    @property
    def reranker_type(self) -> str:
        """Get reranker type identifier."""
        ...

    @property
    def model_name(self) -> str:
        """Get model name."""
        ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        top_k: int | None = None,
    ) -> Result[list[RerankResult]]:
        """
        Rerank candidates for a query.

        Args:
            query: Query text
            candidates: Candidates to rerank
            top_k: Return top K (all if None)

        Returns:
            Result containing reranked results
        """
        ...

    def score_pair(
        self,
        query: str,
        document: str,
    ) -> Result[Score]:
        """
        Score a single query-document pair.

        Args:
            query: Query text
            document: Document text

        Returns:
            Result containing relevance score
        """
        ...


# =============================================================================
# BASE RERANKER
# =============================================================================


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @property
    @abstractmethod
    def reranker_type(self) -> str:
        """Reranker type identifier."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name."""
        ...

    @abstractmethod
    def _score_pairs(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[Score]:
        """Internal scoring of query-document pairs."""
        ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        top_k: int | None = None,
    ) -> Result[list[RerankResult]]:
        """Rerank candidates."""
        if not candidates:
            return Result.success([])

        try:
            # Score all pairs
            texts = [c.text for c in candidates]
            scores = self._score_pairs(query, texts)

            # Create results with original ranks
            results = []
            for i, (candidate, score) in enumerate(zip(candidates, scores)):
                results.append(RerankResult(
                    id=candidate.id,
                    score=score,
                    rank=0,  # Will be set after sorting
                    original_rank=i + 1,
                    metadata=candidate.metadata,
                ))

            # Sort by score descending
            results.sort(key=lambda x: x.score, reverse=True)

            # Assign new ranks
            final_results = []
            for i, r in enumerate(results):
                final_results.append(RerankResult(
                    id=r.id,
                    score=r.score,
                    rank=i + 1,
                    original_rank=r.original_rank,
                    metadata=r.metadata,
                ))

            # Apply top_k
            if top_k is not None:
                final_results = final_results[:top_k]

            return Result.success(final_results)

        except Exception as e:
            return Result.failure(f"Reranking failed: {e}")

    def score_pair(
        self,
        query: str,
        document: str,
    ) -> Result[Score]:
        """Score a single pair."""
        try:
            scores = self._score_pairs(query, [document])
            return Result.success(scores[0])
        except Exception as e:
            return Result.failure(f"Scoring failed: {e}")


# =============================================================================
# REGISTRIES
# =============================================================================


class SparseRetrieverRegistry:
    """Registry for sparse retriever implementations."""

    def __init__(self) -> None:
        self._factories: dict[str, type[SparseRetriever]] = {}

    def register(self, retriever_type: str, factory: type[SparseRetriever]) -> None:
        """Register a sparse retriever factory."""
        self._factories[retriever_type] = factory

    def create(self, retriever_type: str, **kwargs) -> SparseRetriever | None:
        """Create a retriever instance."""
        factory = self._factories.get(retriever_type)
        if factory:
            return factory(**kwargs)
        return None

    def list_retrievers(self) -> list[str]:
        """Get list of registered retriever types."""
        return list(self._factories.keys())


class RerankerRegistry:
    """Registry for reranker implementations."""

    def __init__(self) -> None:
        self._factories: dict[str, type[Reranker]] = {}

    def register(self, reranker_type: str, factory: type[Reranker]) -> None:
        """Register a reranker factory."""
        self._factories[reranker_type] = factory

    def create(self, reranker_type: str, **kwargs) -> Reranker | None:
        """Create a reranker instance."""
        factory = self._factories.get(reranker_type)
        if factory:
            return factory(**kwargs)
        return None

    def list_rerankers(self) -> list[str]:
        """Get list of registered reranker types."""
        return list(self._factories.keys())
