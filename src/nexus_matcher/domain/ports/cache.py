"""
nexus_matcher.domain.ports.cache | Layer: DOMAIN
Port interface for caching search results and embeddings.

## Relationships
# USED_BY    → domain/services/search_service :: result caching
# USED_BY    → infrastructure/adapters/caches/* :: cache implementations
# DEPENDS_ON → shared/types/base :: DocumentId, EmbeddingVector

## Attributes
# Security: Cached data should not include secrets
# Performance: L1 cache should be sub-ms, L2 ~5ms, L3 ~50ms
# Reliability: Cache misses should not cause failures
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

# Type variable for cached values
T = TypeVar("T")


# =============================================================================
# CACHE TYPES
# =============================================================================


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for cache behavior."""

    max_size: int = 10000
    ttl: timedelta | None = timedelta(hours=1)
    eviction_policy: str = "lru"  # lru, lfu, fifo


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1


# =============================================================================
# CACHE PROTOCOL
# =============================================================================


@runtime_checkable
class Cache(Protocol[T]):
    """
    Protocol for cache implementations.

    Generic cache interface that can store any serializable type.

    Example usage:
        cache: Cache[MatchResult] = RedisCache(config)
        cache.set("query_hash", result)
        cached = cache.get("query_hash")
    """

    @property
    def cache_type(self) -> str:
        """Get cache type identifier."""
        ...

    def get(self, key: str) -> T | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        ...

    def set(
        self,
        key: str,
        value: T,
        ttl: timedelta | None = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live (uses default if None)

        Returns:
            True if successfully cached
        """
        ...

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed and was deleted
        """
        ...

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        ...

    def clear(self) -> int:
        """
        Clear all cached values.

        Returns:
            Number of items cleared
        """
        ...

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        ...


# =============================================================================
# SEMANTIC CACHE PROTOCOL
# =============================================================================


@dataclass(frozen=True)
class SemanticCacheConfig(CacheConfig):
    """Configuration for semantic cache."""

    similarity_threshold: float = 0.92
    use_exact_match: bool = True


@runtime_checkable
class SemanticCache(Protocol[T]):
    """
    Protocol for semantic similarity-based caching.

    Extends basic caching with vector similarity lookup
    for semantically similar queries.

    Example usage:
        cache: SemanticCache[list[MatchResult]] = HierarchicalCache(config)
        cache.set_with_embedding("customer email", query_embedding, results)
        cached = cache.get_similar(new_query_embedding, threshold=0.9)
    """

    @property
    def cache_type(self) -> str:
        """Get cache type identifier."""
        ...

    def get_exact(self, key: str) -> T | None:
        """Get by exact key match."""
        ...

    def get_similar(
        self,
        query_embedding: Any,  # EmbeddingVector
        threshold: float | None = None,
    ) -> tuple[T, float] | None:
        """
        Get by semantic similarity.

        Args:
            query_embedding: Query embedding vector
            threshold: Similarity threshold (uses default if None)

        Returns:
            (value, similarity_score) or None if no match above threshold
        """
        ...

    def set_with_embedding(
        self,
        key: str,
        embedding: Any,  # EmbeddingVector
        value: T,
        ttl: timedelta | None = None,
    ) -> bool:
        """
        Set value with associated embedding.

        Args:
            key: Cache key for exact match
            embedding: Embedding vector for similarity search
            value: Value to cache
            ttl: Time-to-live

        Returns:
            True if successfully cached
        """
        ...

    def clear(self) -> int:
        """Clear all cached values."""
        ...

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        ...


# =============================================================================
# HIERARCHICAL CACHE PROTOCOL
# =============================================================================


@dataclass
class HierarchicalCacheStats:
    """Statistics for multi-level cache."""

    l1_stats: CacheStats = field(default_factory=CacheStats)
    l2_stats: CacheStats = field(default_factory=CacheStats)
    l3_stats: CacheStats = field(default_factory=CacheStats)
    total_queries: int = 0

    @property
    def overall_hit_rate(self) -> float:
        """Calculate overall hit rate."""
        if self.total_queries == 0:
            return 0.0
        total_hits = self.l1_stats.hits + self.l2_stats.hits + self.l3_stats.hits
        return total_hits / self.total_queries


@runtime_checkable
class HierarchicalCache(Protocol[T]):
    """
    Protocol for hierarchical multi-level caching.

    Combines multiple cache layers:
    - L1: In-memory (sub-ms latency)
    - L2: Redis/external (2-5ms latency)
    - L3: Semantic similarity (10-50ms latency)

    Example usage:
        cache: HierarchicalCache[list[MatchResult]] = MultiLevelCache(config)
        result = cache.get("query", query_embedding)  # Checks L1 → L2 → L3
    """

    def get(
        self,
        key: str,
        embedding: Any | None = None,
    ) -> T | None:
        """
        Get from hierarchical cache.

        Checks L1 → L2 → L3 in order. Promotes hits to higher levels.

        Args:
            key: Cache key for exact match
            embedding: Optional embedding for L3 semantic search

        Returns:
            Cached value or None
        """
        ...

    def set(
        self,
        key: str,
        value: T,
        embedding: Any | None = None,
        ttl: timedelta | None = None,
    ) -> bool:
        """
        Set in all cache layers.

        Args:
            key: Cache key
            value: Value to cache
            embedding: Optional embedding for L3
            ttl: Time-to-live

        Returns:
            True if successfully cached
        """
        ...

    def invalidate(self, key: str) -> bool:
        """
        Invalidate key in all layers.

        Args:
            key: Cache key

        Returns:
            True if any layer had the key
        """
        ...

    def clear_layer(self, layer: int) -> int:
        """
        Clear specific cache layer.

        Args:
            layer: Layer number (1, 2, or 3)

        Returns:
            Number of items cleared
        """
        ...

    def clear_all(self) -> int:
        """Clear all cache layers."""
        ...

    def get_stats(self) -> HierarchicalCacheStats:
        """Get hierarchical cache statistics."""
        ...


# =============================================================================
# BASE IMPLEMENTATIONS
# =============================================================================


class BaseCache(ABC, Generic[T]):
    """Abstract base class for caches."""

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._stats = CacheStats()

    @property
    @abstractmethod
    def cache_type(self) -> str:
        """Cache type identifier."""
        ...

    @abstractmethod
    def _get_internal(self, key: str) -> T | None:
        """Internal get implementation."""
        ...

    @abstractmethod
    def _set_internal(self, key: str, value: T, ttl: timedelta | None) -> bool:
        """Internal set implementation."""
        ...

    @abstractmethod
    def _delete_internal(self, key: str) -> bool:
        """Internal delete implementation."""
        ...

    @abstractmethod
    def _exists_internal(self, key: str) -> bool:
        """Internal exists implementation."""
        ...

    @abstractmethod
    def _clear_internal(self) -> int:
        """Internal clear implementation."""
        ...

    def get(self, key: str) -> T | None:
        """Get value from cache."""
        value = self._get_internal(key)
        if value is not None:
            self._stats.record_hit()
        else:
            self._stats.record_miss()
        return value

    def set(self, key: str, value: T, ttl: timedelta | None = None) -> bool:
        """Set value in cache."""
        actual_ttl = ttl or self._config.ttl
        return self._set_internal(key, value, actual_ttl)

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return self._delete_internal(key)

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._exists_internal(key)

    def clear(self) -> int:
        """Clear all values."""
        count = self._clear_internal()
        self._stats.evictions += count
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


# =============================================================================
# REGISTRY
# =============================================================================


class CacheRegistry:
    """Registry for cache implementations."""

    def __init__(self) -> None:
        self._factories: dict[str, type] = {}

    def register(self, cache_type: str, factory: type) -> None:
        """Register a cache factory."""
        self._factories[cache_type] = factory

    def create(self, cache_type: str, config: CacheConfig, **kwargs) -> Cache | None:
        """Create a cache instance."""
        factory = self._factories.get(cache_type)
        if factory:
            return factory(config, **kwargs)
        return None

    def list_caches(self) -> list[str]:
        """Get list of registered cache types."""
        return list(self._factories.keys())
