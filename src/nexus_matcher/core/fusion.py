"""
nexus_matcher.core.fusion | Layer: CORE
Score fusion algorithms for hybrid retrieval combining semantic and lexical scores.

## Relationships
# IMPLEMENTS → N/A :: Core fusion algorithms
# DEPENDS_ON → N/A :: Pure Python/NumPy
# USED_BY    → application/services/match_schema :: hybrid ranking
# USED_BY    → domain/services/search_service :: result fusion

## Attributes
# Security: No external calls, pure computation
# Performance: O(n) fusion, O(n log n) ranking
# Reliability: Handles missing scores gracefully

## Research Reference
# Reciprocal Rank Fusion: Cormack et al., SIGIR 2009
# "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
# CE-007: RRF Hybrid Fusion for exact-match improvement
# Target: +2-3% MRR improvement on exact keyword matches
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")  # Document/item identifier type


# =============================================================================
# ENUMERATIONS
# =============================================================================


class FusionMethod(str, Enum):
    """Available score fusion methods."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    LINEAR = "linear"  # Weighted linear combination
    COMBSUM = "combsum"  # Sum of normalized scores
    COMBMNZ = "combmnz"  # CombSUM × number of lists containing item
    MAX_SCORE = "max_score"  # Maximum score across lists


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class FusionConfig:
    """
    Configuration for score fusion.

    Attributes:
        method: Fusion method to use
        rrf_k: RRF smoothing constant (default 60, per original paper)
        semantic_weight: Weight for semantic scores (linear fusion)
        lexical_weight: Weight for lexical/BM25 scores (linear fusion)
        normalize: Whether to normalize final scores to [0, 1]

    Example:
        # RRF fusion (recommended)
        config = FusionConfig(method=FusionMethod.RRF, rrf_k=60)

        # Weighted linear fusion
        config = FusionConfig(
            method=FusionMethod.LINEAR,
            semantic_weight=0.7,
            lexical_weight=0.3,
        )
    """

    method: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60  # Default from original paper
    semantic_weight: float = 0.7
    lexical_weight: float = 0.3
    normalize: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")

        if self.method == FusionMethod.LINEAR:
            total = self.semantic_weight + self.lexical_weight
            if abs(total - 1.0) > 1e-6:
                logger.warning(
                    f"Linear weights sum to {total}, not 1.0. Scores will be renormalized."
                )


@dataclass
class FusionStats:
    """Statistics for fusion operations."""

    total_fusions: int = 0
    total_items_fused: int = 0
    avg_overlap_ratio: float = 0.0

    # Track by method
    _overlap_ratios: list[float] = field(default_factory=list)

    def record(
        self,
        num_items: int,
        num_in_both: int,
        num_in_semantic_only: int,
        num_in_lexical_only: int,
    ) -> None:
        """Record a fusion operation."""
        self.total_fusions += 1
        self.total_items_fused += num_items

        total = num_in_both + num_in_semantic_only + num_in_lexical_only
        if total > 0:
            overlap = num_in_both / total
            self._overlap_ratios.append(overlap)
            self.avg_overlap_ratio = sum(self._overlap_ratios) / len(self._overlap_ratios)

    @property
    def avg_items_per_fusion(self) -> float:
        """Average items per fusion operation."""
        if self.total_fusions == 0:
            return 0.0
        return self.total_items_fused / self.total_fusions


# =============================================================================
# SCORED ITEM
# =============================================================================


@dataclass
class ScoredItem(Generic[T]):
    """
    An item with its score from fusion.

    Attributes:
        id: Item identifier
        score: Fused score
        semantic_score: Original semantic score (if available)
        lexical_score: Original lexical score (if available)
        semantic_rank: Rank in semantic results (1-indexed, None if not present)
        lexical_rank: Rank in lexical results (1-indexed, None if not present)
        metadata: Additional metadata
    """

    id: T
    score: float
    semantic_score: float | None = None
    lexical_score: float | None = None
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def in_both(self) -> bool:
        """Check if item appeared in both result lists."""
        return self.semantic_rank is not None and self.lexical_rank is not None

    @property
    def boost_from_lexical(self) -> bool:
        """Check if item was boosted by lexical match."""
        if self.semantic_rank is None or self.lexical_rank is None:
            return False
        return self.lexical_rank < self.semantic_rank


# =============================================================================
# CORE FUSION ALGORITHMS
# =============================================================================


def rrf_score(rank: int, k: int = 60) -> float:
    """
    Calculate Reciprocal Rank Fusion score for a single rank.

    Formula: 1 / (k + rank)

    Args:
        rank: 1-indexed rank position
        k: Smoothing constant (default 60)

    Returns:
        RRF score

    Note:
        k=60 was found optimal in the original SIGIR 2009 paper.
        Lower k values give more weight to top ranks.
    """
    if rank < 1:
        raise ValueError("rank must be >= 1 (1-indexed)")
    return 1.0 / (k + rank)


def fuse_rrf(
    semantic_results: list[tuple[T, float]],
    lexical_results: list[tuple[T, float]],
    k: int = 60,
    top_k: int | None = None,
) -> list[ScoredItem[T]]:
    """
    Fuse results using Reciprocal Rank Fusion.

    RRF combines ranked lists by summing reciprocal rank scores.
    It's particularly effective because:
    - No score normalization needed
    - Robust to outlier scores
    - Works well with diverse retrieval methods

    Args:
        semantic_results: List of (id, score) from semantic retrieval
        lexical_results: List of (id, score) from lexical retrieval
        k: RRF smoothing constant (default 60)
        top_k: Return top K results (all if None)

    Returns:
        Fused results sorted by score descending

    Example:
        semantic = [("doc1", 0.95), ("doc2", 0.85), ("doc3", 0.75)]
        lexical = [("doc2", 12.5), ("doc4", 11.2), ("doc1", 10.1)]

        fused = fuse_rrf(semantic, lexical, k=60)
        # doc2 will likely rank highest (appears in both with good ranks)
    """
    # Build rank maps (1-indexed)
    semantic_ranks: dict[T, int] = {
        item_id: rank + 1 for rank, (item_id, _) in enumerate(semantic_results)
    }
    lexical_ranks: dict[T, int] = {
        item_id: rank + 1 for rank, (item_id, _) in enumerate(lexical_results)
    }

    # Build score maps
    semantic_scores: dict[T, float] = dict(semantic_results)
    lexical_scores: dict[T, float] = dict(lexical_results)

    # Get all unique items
    all_items = set(semantic_ranks.keys()) | set(lexical_ranks.keys())

    # Calculate RRF scores
    results: list[ScoredItem[T]] = []

    for item_id in all_items:
        rrf_total = 0.0

        sem_rank = semantic_ranks.get(item_id)
        lex_rank = lexical_ranks.get(item_id)

        if sem_rank is not None:
            rrf_total += rrf_score(sem_rank, k)

        if lex_rank is not None:
            rrf_total += rrf_score(lex_rank, k)

        results.append(
            ScoredItem(
                id=item_id,
                score=rrf_total,
                semantic_score=semantic_scores.get(item_id),
                lexical_score=lexical_scores.get(item_id),
                semantic_rank=sem_rank,
                lexical_rank=lex_rank,
            )
        )

    # Sort by fused score descending
    results.sort(key=lambda x: x.score, reverse=True)

    # Apply top_k
    if top_k is not None:
        results = results[:top_k]

    return results


def fuse_linear(
    semantic_results: list[tuple[T, float]],
    lexical_results: list[tuple[T, float]],
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3,
    normalize_scores: bool = True,
    top_k: int | None = None,
) -> list[ScoredItem[T]]:
    """
    Fuse results using weighted linear combination.

    Combines normalized scores: score = w_s * sem_score + w_l * lex_score

    Args:
        semantic_results: List of (id, score) from semantic retrieval
        lexical_results: List of (id, score) from lexical retrieval
        semantic_weight: Weight for semantic scores
        lexical_weight: Weight for lexical scores
        normalize_scores: Normalize input scores to [0, 1]
        top_k: Return top K results

    Returns:
        Fused results sorted by score descending
    """
    # Normalize weights
    total_weight = semantic_weight + lexical_weight
    sem_w = semantic_weight / total_weight
    lex_w = lexical_weight / total_weight

    # Build score maps
    semantic_scores: dict[T, float] = dict(semantic_results)
    lexical_scores: dict[T, float] = dict(lexical_results)

    # Normalize scores to [0, 1] if requested
    if normalize_scores:
        if semantic_scores:
            sem_max = max(semantic_scores.values())
            sem_min = min(semantic_scores.values())
            sem_range = sem_max - sem_min if sem_max != sem_min else 1.0
            semantic_scores = {k: (v - sem_min) / sem_range for k, v in semantic_scores.items()}

        if lexical_scores:
            lex_max = max(lexical_scores.values())
            lex_min = min(lexical_scores.values())
            lex_range = lex_max - lex_min if lex_max != lex_min else 1.0
            lexical_scores = {k: (v - lex_min) / lex_range for k, v in lexical_scores.items()}

    # Get all unique items
    all_items = set(semantic_scores.keys()) | set(lexical_scores.keys())

    # Build rank maps for metadata
    semantic_ranks: dict[T, int] = {
        item_id: rank + 1 for rank, (item_id, _) in enumerate(semantic_results)
    }
    lexical_ranks: dict[T, int] = {
        item_id: rank + 1 for rank, (item_id, _) in enumerate(lexical_results)
    }

    # Calculate linear combination
    results: list[ScoredItem[T]] = []

    for item_id in all_items:
        sem_score = semantic_scores.get(item_id, 0.0)
        lex_score = lexical_scores.get(item_id, 0.0)

        fused_score = sem_w * sem_score + lex_w * lex_score

        results.append(
            ScoredItem(
                id=item_id,
                score=fused_score,
                semantic_score=sem_score,
                lexical_score=lex_score,
                semantic_rank=semantic_ranks.get(item_id),
                lexical_rank=lexical_ranks.get(item_id),
            )
        )

    # Sort by fused score descending
    results.sort(key=lambda x: x.score, reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


def fuse_combsum(
    semantic_results: list[tuple[T, float]],
    lexical_results: list[tuple[T, float]],
    normalize_scores: bool = True,
    top_k: int | None = None,
) -> list[ScoredItem[T]]:
    """
    Fuse results using CombSUM (sum of normalized scores).

    Simple additive fusion: score = norm_sem_score + norm_lex_score

    Args:
        semantic_results: List of (id, score) from semantic retrieval
        lexical_results: List of (id, score) from lexical retrieval
        normalize_scores: Normalize input scores to [0, 1]
        top_k: Return top K results

    Returns:
        Fused results sorted by score descending
    """
    return fuse_linear(
        semantic_results,
        lexical_results,
        semantic_weight=0.5,
        lexical_weight=0.5,
        normalize_scores=normalize_scores,
        top_k=top_k,
    )


def fuse_combmnz(
    semantic_results: list[tuple[T, float]],
    lexical_results: list[tuple[T, float]],
    normalize_scores: bool = True,
    top_k: int | None = None,
) -> list[ScoredItem[T]]:
    """
    Fuse results using CombMNZ (CombSUM × number of lists containing item).

    Boosts items appearing in multiple lists:
    score = (norm_sem_score + norm_lex_score) × num_lists_containing_item

    Args:
        semantic_results: List of (id, score) from semantic retrieval
        lexical_results: List of (id, score) from lexical retrieval
        normalize_scores: Normalize input scores to [0, 1]
        top_k: Return top K results

    Returns:
        Fused results sorted by score descending
    """
    # Get CombSUM results first
    combsum_results = fuse_combsum(
        semantic_results,
        lexical_results,
        normalize_scores=normalize_scores,
        top_k=None,  # Apply top_k after MNZ
    )

    # Apply MNZ multiplier
    results: list[ScoredItem[T]] = []

    for item in combsum_results:
        # Count how many lists contain this item
        num_lists = 0
        if item.semantic_rank is not None:
            num_lists += 1
        if item.lexical_rank is not None:
            num_lists += 1

        mnz_score = item.score * num_lists

        results.append(
            ScoredItem(
                id=item.id,
                score=mnz_score,
                semantic_score=item.semantic_score,
                lexical_score=item.lexical_score,
                semantic_rank=item.semantic_rank,
                lexical_rank=item.lexical_rank,
            )
        )

    # Re-sort by MNZ score
    results.sort(key=lambda x: x.score, reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


def fuse_max_score(
    semantic_results: list[tuple[T, float]],
    lexical_results: list[tuple[T, float]],
    normalize_scores: bool = True,
    top_k: int | None = None,
) -> list[ScoredItem[T]]:
    """
    Fuse results using maximum score across lists.

    score = max(norm_sem_score, norm_lex_score)

    Args:
        semantic_results: List of (id, score) from semantic retrieval
        lexical_results: List of (id, score) from lexical retrieval
        normalize_scores: Normalize input scores to [0, 1]
        top_k: Return top K results

    Returns:
        Fused results sorted by score descending
    """
    # Build and normalize score maps
    semantic_scores: dict[T, float] = dict(semantic_results)
    lexical_scores: dict[T, float] = dict(lexical_results)

    if normalize_scores:
        if semantic_scores:
            sem_max = max(semantic_scores.values())
            sem_min = min(semantic_scores.values())
            sem_range = sem_max - sem_min if sem_max != sem_min else 1.0
            semantic_scores = {k: (v - sem_min) / sem_range for k, v in semantic_scores.items()}

        if lexical_scores:
            lex_max = max(lexical_scores.values())
            lex_min = min(lexical_scores.values())
            lex_range = lex_max - lex_min if lex_max != lex_min else 1.0
            lexical_scores = {k: (v - lex_min) / lex_range for k, v in lexical_scores.items()}

    # Build rank maps
    semantic_ranks: dict[T, int] = {
        item_id: rank + 1 for rank, (item_id, _) in enumerate(semantic_results)
    }
    lexical_ranks: dict[T, int] = {
        item_id: rank + 1 for rank, (item_id, _) in enumerate(lexical_results)
    }

    # Get all unique items
    all_items = set(semantic_scores.keys()) | set(lexical_scores.keys())

    # Calculate max scores
    results: list[ScoredItem[T]] = []

    for item_id in all_items:
        sem_score = semantic_scores.get(item_id, 0.0)
        lex_score = lexical_scores.get(item_id, 0.0)

        fused_score = max(sem_score, lex_score)

        results.append(
            ScoredItem(
                id=item_id,
                score=fused_score,
                semantic_score=sem_score,
                lexical_score=lex_score,
                semantic_rank=semantic_ranks.get(item_id),
                lexical_rank=lexical_ranks.get(item_id),
            )
        )

    results.sort(key=lambda x: x.score, reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


# =============================================================================
# HYBRID FUSER CLASS
# =============================================================================


class HybridFuser(Generic[T]):
    """
    Hybrid score fuser combining semantic and lexical results.

    Supports multiple fusion methods:
    - RRF (Reciprocal Rank Fusion) - recommended
    - Linear (weighted combination)
    - CombSUM (sum of normalized scores)
    - CombMNZ (CombSUM × list count)
    - MaxScore (maximum across lists)

    Example:
        fuser = HybridFuser(method=FusionMethod.RRF)

        semantic = [("doc1", 0.95), ("doc2", 0.85)]
        lexical = [("doc2", 12.5), ("doc3", 11.2)]

        results = fuser.fuse(semantic, lexical, top_k=10)

        for item in results:
            print(f"{item.id}: {item.score:.4f} (boosted={item.boost_from_lexical})")

    Performance Note:
        RRF is generally preferred because:
        - No score normalization needed
        - Robust to score distribution differences
        - Proven effectiveness (SIGIR 2009)
    """

    def __init__(
        self,
        config: FusionConfig | None = None,
        method: FusionMethod = FusionMethod.RRF,
    ) -> None:
        """
        Initialize hybrid fuser.

        Args:
            config: Fusion configuration
            method: Fusion method (shortcut if config not provided)
        """
        if config is not None:
            self._config = config
        else:
            self._config = FusionConfig(method=method)

        self._stats = FusionStats()

    @property
    def config(self) -> FusionConfig:
        """Get fusion configuration."""
        return self._config

    @property
    def stats(self) -> FusionStats:
        """Get fusion statistics."""
        return self._stats

    def fuse(
        self,
        semantic_results: list[tuple[T, float]],
        lexical_results: list[tuple[T, float]],
        top_k: int | None = None,
    ) -> list[ScoredItem[T]]:
        """
        Fuse semantic and lexical results.

        Args:
            semantic_results: List of (id, score) from semantic retrieval
            lexical_results: List of (id, score) from lexical retrieval
            top_k: Return top K results

        Returns:
            Fused and sorted results
        """
        method = self._config.method

        if method == FusionMethod.RRF:
            results = fuse_rrf(
                semantic_results,
                lexical_results,
                k=self._config.rrf_k,
                top_k=top_k,
            )
        elif method == FusionMethod.LINEAR:
            results = fuse_linear(
                semantic_results,
                lexical_results,
                semantic_weight=self._config.semantic_weight,
                lexical_weight=self._config.lexical_weight,
                normalize_scores=self._config.normalize,
                top_k=top_k,
            )
        elif method == FusionMethod.COMBSUM:
            results = fuse_combsum(
                semantic_results,
                lexical_results,
                normalize_scores=self._config.normalize,
                top_k=top_k,
            )
        elif method == FusionMethod.COMBMNZ:
            results = fuse_combmnz(
                semantic_results,
                lexical_results,
                normalize_scores=self._config.normalize,
                top_k=top_k,
            )
        elif method == FusionMethod.MAX_SCORE:
            results = fuse_max_score(
                semantic_results,
                lexical_results,
                normalize_scores=self._config.normalize,
                top_k=top_k,
            )
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        # Record statistics
        sem_ids = {item_id for item_id, _ in semantic_results}
        lex_ids = {item_id for item_id, _ in lexical_results}

        in_both = len(sem_ids & lex_ids)
        in_sem_only = len(sem_ids - lex_ids)
        in_lex_only = len(lex_ids - sem_ids)

        self._stats.record(
            num_items=len(results),
            num_in_both=in_both,
            num_in_semantic_only=in_sem_only,
            num_in_lexical_only=in_lex_only,
        )

        return results

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information."""
        return {
            "method": self._config.method.value,
            "config": {
                "rrf_k": self._config.rrf_k,
                "semantic_weight": self._config.semantic_weight,
                "lexical_weight": self._config.lexical_weight,
                "normalize": self._config.normalize,
            },
            "stats": {
                "total_fusions": self._stats.total_fusions,
                "total_items_fused": self._stats.total_items_fused,
                "avg_overlap_ratio": self._stats.avg_overlap_ratio,
            },
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_rrf_fuser(k: int = 60) -> HybridFuser:
    """
    Create an RRF fuser with specified k value.

    Args:
        k: RRF smoothing constant (default 60)

    Returns:
        Configured HybridFuser
    """
    config = FusionConfig(method=FusionMethod.RRF, rrf_k=k)
    return HybridFuser(config=config)


def create_linear_fuser(
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> HybridFuser:
    """
    Create a linear fusion fuser.

    Args:
        semantic_weight: Weight for semantic scores
        lexical_weight: Weight for lexical scores

    Returns:
        Configured HybridFuser
    """
    config = FusionConfig(
        method=FusionMethod.LINEAR,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
    )
    return HybridFuser(config=config)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_fusion_info() -> dict[str, Any]:
    """
    Get information about available fusion methods.

    Returns:
        Dictionary with method descriptions and recommendations
    """
    return {
        "methods": [m.value for m in FusionMethod],
        "recommended": FusionMethod.RRF.value,
        "descriptions": {
            FusionMethod.RRF.value: "Reciprocal Rank Fusion - robust, no normalization needed",
            FusionMethod.LINEAR.value: "Weighted linear combination of normalized scores",
            FusionMethod.COMBSUM.value: "Sum of normalized scores (equal weights)",
            FusionMethod.COMBMNZ.value: "CombSUM × number of lists containing item",
            FusionMethod.MAX_SCORE.value: "Maximum score across lists",
        },
        "default_rrf_k": 60,
        "research_reference": "Cormack et al., SIGIR 2009",
    }
