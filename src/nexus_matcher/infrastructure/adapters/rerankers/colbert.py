"""
infrastructure.adapters.rerankers.colbert | Layer: INFRASTRUCTURE
Implements: GAP-001 ColBERT MaxSim Reranker

Research Reference: README_RESEARCH_3.md, Lines 5-8
Research Quote: "The correct approach uses token-level embeddings where each
query and document becomes multiple 128-dimensional vectors, then computes
the sum of maximum similarities across query tokens."

Key Points:
- Current bi-encoder approach is WRONG (pools tokens into single vectors)
- Need proper MaxSim late interaction
- Model: answerai-colbert-small-v1 (33M params, CPU-optimized)
- Target: 60ms for top-100 reranking
- Impact: +10-20% accuracy improvement

Components:
- ColBERTMaxSimReranker: Production reranker using RAGatouille
- MockColBERTReranker: Testing mock with simulated MaxSim behavior
- MaxSimConfig: Configuration dataclass
- ColBERTStatistics: Performance tracking
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from nexus_matcher.domain.ports.retrieval import (
    RerankCandidate,
    RerankResult,
)
from nexus_matcher.shared.types.base import Result

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MaxSimConfig:
    """
    Configuration for ColBERT MaxSim reranker.

    Research-aligned defaults:
    - embedding_dim: 128 (ColBERT standard)
    - model_name: answerai-colbert-small-v1 (33M params, CPU-optimized)
    - device: cpu (for CPU-only infrastructure)

    Attributes:
        model_name: HuggingFace model name or path
        embedding_dim: Dimension of token embeddings (128 for ColBERT)
        device: Device for inference ("cpu", "cuda", "auto")
        batch_size: Batch size for encoding
        max_query_length: Maximum query token length
        max_doc_length: Maximum document token length
        default_top_k: Default number of results to return
    """

    model_name: str = "answerai-colbert-small-v1"
    embedding_dim: int = 128  # ColBERT standard
    device: str = "cpu"
    batch_size: int = 32
    max_query_length: int = 32
    max_doc_length: int = 180
    default_top_k: int | None = None

    # Performance tuning
    use_fp16: bool = False  # FP16 for GPU
    num_threads: int = 0  # 0 = auto-detect


# =============================================================================
# STATISTICS
# =============================================================================


@dataclass
class ColBERTStatistics:
    """
    Statistics tracking for ColBERT reranking.

    Attributes:
        total_reranks: Number of rerank operations performed
        total_candidates: Total candidates processed
        avg_latency_ms: Average reranking latency in milliseconds
        p95_latency_ms: 95th percentile latency
        p99_latency_ms: 99th percentile latency
    """

    total_reranks: int = 0
    total_candidates: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Internal tracking
    _latencies: list[float] = field(default_factory=list, repr=False)

    def record_rerank(self, num_candidates: int, latency_ms: float) -> None:
        """Record a reranking operation."""
        self.total_reranks += 1
        self.total_candidates += num_candidates
        self._latencies.append(latency_ms)

        # Update averages
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

        if len(self._latencies) >= 20:
            sorted_latencies = sorted(self._latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            self.p95_latency_ms = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
            self.p99_latency_ms = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_reranks = 0
        self.total_candidates = 0
        self.avg_latency_ms = 0.0
        self.p95_latency_ms = 0.0
        self.p99_latency_ms = 0.0
        self._latencies = []


# =============================================================================
# MOCK COLBERT RERANKER (For Testing)
# =============================================================================


class MockColBERTReranker:
    """
    Mock ColBERT MaxSim reranker for testing without model dependencies.

    Simulates MaxSim behavior:
    - Token-level scoring (not bi-encoder pooling)
    - Scores based on text overlap simulation
    - Realistic latency simulation

    Example:
        reranker = MockColBERTReranker()
        candidates = [
            RerankCandidate(id="1", text="customer email"),
            RerankCandidate(id="2", text="account balance"),
        ]
        result = reranker.rerank("email", candidates)
    """

    def __init__(
        self,
        config: MaxSimConfig | None = None,
        simulated_latency_ms: float = 0.5,  # Per candidate
    ) -> None:
        """
        Initialize mock reranker.

        Args:
            config: MaxSim configuration
            simulated_latency_ms: Simulated latency per candidate
        """
        self._config = config or MaxSimConfig()
        self._simulated_latency_ms = simulated_latency_ms
        self._statistics = ColBERTStatistics()

    @property
    def reranker_type(self) -> str:
        """Get reranker type identifier."""
        return "colbert_maxsim"

    @property
    def model_name(self) -> str:
        """Get model name."""
        return f"mock-{self._config.model_name}"

    @property
    def uses_token_embeddings(self) -> bool:
        """Whether this reranker uses token-level embeddings (not pooled)."""
        return True  # ColBERT uses token-level embeddings

    @property
    def is_bi_encoder(self) -> bool:
        """Whether this is a bi-encoder (WRONG for ColBERT)."""
        return False  # ColBERT MaxSim is NOT a bi-encoder

    def _compute_mock_maxsim_score(self, query: str, document: str) -> float:
        """
        Compute mock MaxSim score based on token overlap.

        This simulates the MaxSim computation:
        - Split into tokens
        - For each query token, find max similarity to any doc token
        - Sum all max similarities
        - Normalize to [0, 1]
        """
        query_tokens = set(query.lower().split())
        doc_tokens = set(document.lower().split())

        if not query_tokens:
            return 0.0

        # Simulate MaxSim: for each query token, find best match
        total_score = 0.0
        for q_token in query_tokens:
            best_match = 0.0
            for d_token in doc_tokens:
                # Simple Jaccard-like similarity
                if q_token == d_token:
                    best_match = 1.0
                    break
                elif q_token in d_token or d_token in q_token:
                    best_match = max(best_match, 0.7)
                elif len(set(q_token) & set(d_token)) / len(set(q_token) | set(d_token)) > 0.5:
                    best_match = max(best_match, 0.4)
            total_score += best_match

        # Normalize by query length
        normalized = total_score / len(query_tokens)

        # Add some randomness to simulate model behavior
        import random

        noise = random.uniform(-0.1, 0.1)

        return max(0.0, min(1.0, normalized + noise))

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        top_k: int | None = None,
    ) -> Result[list[RerankResult]]:
        """
        Rerank candidates using simulated MaxSim scoring.

        Args:
            query: Query text
            candidates: Candidates to rerank
            top_k: Return top K (all if None)

        Returns:
            Result containing reranked results ordered by score descending
        """
        if not candidates:
            return Result.success([])

        start_time = time.perf_counter()

        try:
            # Simulate latency
            time.sleep(len(candidates) * self._simulated_latency_ms / 1000)

            # Score all candidates
            scored = []
            for idx, candidate in enumerate(candidates):
                score = self._compute_mock_maxsim_score(query, candidate.text)
                scored.append((candidate, score, idx + 1))

            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Apply top_k limit
            if top_k is not None:
                scored = scored[:top_k]

            # Build results
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

            # Record statistics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._statistics.record_rerank(len(candidates), elapsed_ms)

            return Result.success(results)

        except Exception as e:
            logger.error(f"Mock reranking failed: {e}")
            return Result.failure(f"Reranking failed: {e}")

    def score_pair(self, query: str, document: str) -> Result[float]:
        """
        Score a single query-document pair.

        Args:
            query: Query text
            document: Document text

        Returns:
            Result containing MaxSim score
        """
        try:
            score = self._compute_mock_maxsim_score(query, document)
            return Result.success(score)
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

    def get_statistics(self) -> ColBERTStatistics:
        """Get reranking statistics."""
        return self._statistics

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._statistics.reset()


# =============================================================================
# PRODUCTION COLBERT MAXSIM RERANKER
# =============================================================================


def _get_ragatouille():
    """Lazy import of RAGatouille."""
    try:
        from ragatouille import RAGPretrainedModel

        return RAGPretrainedModel
    except ImportError as e:
        raise ImportError(
            "RAGatouille is required for ColBERT MaxSim reranking. "
            "Install with: pip install ragatouille"
        ) from e


class ColBERTMaxSimReranker:
    """
    Production ColBERT MaxSim reranker using RAGatouille.

    Implements proper MaxSim late interaction (NOT bi-encoder pooling).

    Research Context:
    - Uses token-level embeddings (128-dim vectors per token)
    - Computes sum of maximum similarities across query tokens
    - Model: answerai-colbert-small-v1 (33M params, CPU-optimized)
    - Target: 60ms for top-100 reranking

    Example:
        reranker = ColBERTMaxSimReranker()
        candidates = [
            RerankCandidate(id="1", text="customer email address"),
            RerankCandidate(id="2", text="account balance"),
        ]
        result = reranker.rerank("find email field", candidates)

    Note:
        Requires RAGatouille: pip install ragatouille
    """

    def __init__(
        self,
        config: MaxSimConfig | None = None,
        model_name: str | None = None,
    ) -> None:
        """
        Initialize ColBERT MaxSim reranker.

        Args:
            config: MaxSim configuration
            model_name: Override model name (default from config)
        """
        self._config = config or MaxSimConfig()
        self._model_name = model_name or self._config.model_name
        self._model = None
        self._statistics = ColBERTStatistics()

        # Lazy loading - model loaded on first use
        logger.info(f"ColBERT MaxSim reranker initialized (model: {self._model_name})")

    def _load_model(self) -> None:
        """Load the ColBERT model (lazy loading)."""
        if self._model is not None:
            return

        RAGPretrainedModel = _get_ragatouille()

        logger.info(f"Loading ColBERT model: {self._model_name}")
        self._model = RAGPretrainedModel.from_pretrained(self._model_name)
        logger.info("ColBERT model loaded successfully")

    @property
    def reranker_type(self) -> str:
        """Get reranker type identifier."""
        return "colbert_maxsim"

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def uses_token_embeddings(self) -> bool:
        """Whether this reranker uses token-level embeddings (not pooled)."""
        return True

    @property
    def is_bi_encoder(self) -> bool:
        """Whether this is a bi-encoder (WRONG for ColBERT)."""
        return False

    @staticmethod
    def _map_reranked_to_candidates(
        reranked: Sequence[dict[str, Any]],
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankResult]:
        """
        Join reranker output back onto the input candidates by POSITION.

        RAGatouille returns a list of dicts describing the reranked documents,
        each carrying a ``result_index`` that is the offset of that document in
        the ``documents`` list handed to ``rerank()``. That index is the only
        safe join key.

        Joining on ``content`` (the previous behaviour) is wrong because
        candidate texts are not unique. Two candidates legitimately share text
        all the time in this product: the same column description reused across
        two tables, a repeated enum label, or several columns with an empty
        description. A ``{text: candidate}`` dict collapses those duplicates to
        one entry, so every duplicate but the last is silently dropped from the
        output and the survivor is attributed the wrong candidate id, the wrong
        metadata, and the wrong ``original_rank``. Callers then map a schema
        field onto a dictionary entry it was never scored against.

        Args:
            reranked: Reranker output dicts
            candidates: The candidates originally passed to the reranker,
                in the order they were passed

        Returns:
            RerankResult list, ranked 1..N in reranker order

        Raises:
            ValueError: If an item cannot be attributed to exactly one
                candidate. Failing here is deliberate: a wrong-but-plausible
                mapping is worse than a loud error.
        """
        results: list[RerankResult] = []

        # Fallback join key for reranker builds that omit result_index: consume
        # positions per distinct text in order, so N duplicates map to the N
        # distinct candidates that produced them rather than collapsing to one.
        unused_by_text: dict[str, list[int]] = {}
        for idx, cand in enumerate(candidates):
            unused_by_text.setdefault(cand.text, []).append(idx)

        for new_rank, item in enumerate(reranked, start=1):
            score = item.get("score", item.get("relevance_score", 0.0))

            idx = item.get("result_index")
            if idx is None:
                text = item.get("content", item.get("text"))
                positions = unused_by_text.get(text) if text is not None else None
                if not positions:
                    raise ValueError(
                        "ColBERT returned a document that does not correspond to "
                        "any unconsumed input candidate (no 'result_index' and no "
                        "matching text). Refusing to guess an attribution."
                    )
                idx = positions.pop(0)
            else:
                idx = int(idx)
                # Keep the fallback bookkeeping consistent when the two paths mix.
                positions = (
                    unused_by_text.get(candidates[idx].text) if 0 <= idx < len(candidates) else None
                )
                if positions and idx in positions:
                    positions.remove(idx)

            if not 0 <= idx < len(candidates):
                raise ValueError(
                    f"ColBERT returned result_index {idx} which is outside the "
                    f"{len(candidates)} candidates that were submitted."
                )

            candidate = candidates[idx]
            results.append(
                RerankResult(
                    id=candidate.id,
                    score=float(score),
                    rank=new_rank,
                    original_rank=idx + 1,
                    metadata=candidate.metadata,
                )
            )

        return results

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        top_k: int | None = None,
    ) -> Result[list[RerankResult]]:
        """
        Rerank candidates using ColBERT MaxSim scoring.

        Args:
            query: Query text
            candidates: Candidates to rerank
            top_k: Return top K (all if None)

        Returns:
            Result containing reranked results ordered by score descending
        """
        if not candidates:
            return Result.success([])

        start_time = time.perf_counter()

        try:
            self._load_model()

            # Extract texts for reranking
            texts = [c.text for c in candidates]

            # Use RAGatouille's rerank method
            k = top_k or len(candidates)
            reranked = self._model.rerank(
                query=query,
                documents=texts,
                k=k,
            )

            # Map results back to candidates BY POSITION, never by text.
            results = self._map_reranked_to_candidates(reranked, candidates)

            # Record statistics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._statistics.record_rerank(len(candidates), elapsed_ms)

            return Result.success(results)

        except Exception as e:
            logger.error(f"ColBERT reranking failed: {e}")
            return Result.failure(f"Reranking failed: {e}")

    def score_pair(self, query: str, document: str) -> Result[float]:
        """
        Score a single query-document pair.

        Args:
            query: Query text
            document: Document text

        Returns:
            Result containing MaxSim score
        """
        result = self.rerank(query, [RerankCandidate(id="temp", text=document)])
        if result.is_failure:
            return Result.failure(result.error)

        results = result.unwrap()
        if results:
            return Result.success(results[0].score)
        return Result.success(0.0)

    def rerank_batch(
        self,
        queries: Sequence[str],
        candidates_per_query: Sequence[Sequence[RerankCandidate]],
        top_k: int | None = None,
    ) -> Result[list[list[RerankResult]]]:
        """
        Rerank multiple queries.

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

    def get_statistics(self) -> ColBERTStatistics:
        """Get reranking statistics."""
        return self._statistics

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._statistics.reset()


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Main classes
    "ColBERTMaxSimReranker",
    # Statistics
    "ColBERTStatistics",
    # Configuration
    "MaxSimConfig",
    "MockColBERTReranker",
]
