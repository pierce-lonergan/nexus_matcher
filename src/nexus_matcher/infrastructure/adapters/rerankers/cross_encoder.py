"""
nexus_matcher.infrastructure.adapters.rerankers.cross_encoder | Layer: INFRASTRUCTURE
CrossEncoder reranker implementation using sentence-transformers.

## Relationships
# IMPLEMENTS → domain/ports/retrieval :: Reranker protocol
# DEPENDS_ON → sentence-transformers :: CrossEncoder models
# USED_BY    → application/services :: reranking stage

## Attributes
# Security: Model inputs may contain sensitive data
# Performance: GPU recommended for batch processing
# Reliability: Handles empty inputs gracefully
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from nexus_matcher.domain.ports.retrieval import (
    RerankCandidate,
    RerankResult,
)
from nexus_matcher.shared.types.base import Result, Score

logger = logging.getLogger(__name__)


# Lazy import to avoid hard dependency
def _get_cross_encoder():
    """Get CrossEncoder class with lazy import."""
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for CrossEncoderReranker. "
            "Install with: pip install sentence-transformers"
        ) from e


class CrossEncoderReranker:
    """
    CrossEncoder reranker implementation.

    Uses sentence-transformers CrossEncoder models for neural reranking.
    These models jointly encode query-document pairs to produce relevance scores.

    Recommended models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, good quality
    - cross-encoder/ms-marco-TinyBERT-L-2-v2: Fastest, lower quality
    - BAAI/bge-reranker-base: High quality, multilingual
    - BAAI/bge-reranker-large: Best quality, slower

    Example:
        reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

        candidates = [
            RerankCandidate(id="1", text="customer email address"),
            RerankCandidate(id="2", text="account balance"),
        ]

        result = reranker.rerank("find email field", candidates)
        for r in result.unwrap():
            print(f"{r.rank}. {r.id}: {r.score}")
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        """
        Initialize CrossEncoder reranker.

        Args:
            model_name: HuggingFace model name or path
            batch_size: Batch size for inference
            device: Device to use (None for auto-detect)
            max_length: Maximum sequence length
        """
        CrossEncoder = _get_cross_encoder()

        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length

        # Initialize model
        self._model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )

        logger.info(f"Loaded CrossEncoder model: {model_name}")

    @property
    def reranker_type(self) -> str:
        """Get reranker type identifier."""
        return "cross_encoder"

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        top_k: int | None = None,
    ) -> Result[list[RerankResult]]:
        """
        Rerank candidates using CrossEncoder scoring.

        Args:
            query: Query text
            candidates: Candidates to rerank
            top_k: Return top K (all if None)

        Returns:
            Result containing reranked results ordered by score descending
        """
        if not candidates:
            return Result.success([])

        try:
            # Create query-document pairs
            pairs = [[query, c.text] for c in candidates]

            # Get scores from model
            scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )

            # Combine with candidates and track original ranks
            scored = [
                (c, float(s), idx + 1)  # original_rank is 1-indexed
                for idx, (c, s) in enumerate(zip(candidates, scores, strict=False))
            ]

            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Apply top_k limit
            if top_k is not None:
                scored = scored[:top_k]

            # Build results with new ranks
            results = [
                RerankResult(
                    id=candidate.id,
                    score=score,
                    rank=new_rank,
                    original_rank=original_rank,
                    metadata=candidate.metadata,
                )
                for new_rank, (candidate, score, original_rank) in enumerate(scored, start=1)
            ]

            return Result.success(results)

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return Result.failure(f"Reranking failed: {e}")

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
        try:
            scores = self._model.predict(
                [[query, document]],
                show_progress_bar=False,
            )
            return Result.success(float(scores[0]))

        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return Result.failure(f"Scoring failed: {e}")

    def rerank_batch(
        self,
        queries: Sequence[str],
        candidates_per_query: Sequence[Sequence[RerankCandidate]],
        top_k: int | None = None,
    ) -> Result[list[list[RerankResult]]]:
        """
        Rerank multiple queries efficiently.

        Args:
            queries: List of query texts
            candidates_per_query: Candidates for each query
            top_k: Return top K per query

        Returns:
            Result containing list of reranked results per query
        """
        if len(queries) != len(candidates_per_query):
            return Result.failure("Number of queries must match number of candidate lists")

        results = []
        for query, candidates in zip(queries, candidates_per_query, strict=False):
            result = self.rerank(query, candidates, top_k)
            if result.is_failure:
                return Result.failure(result.error)
            results.append(result.unwrap())

        return Result.success(results)
