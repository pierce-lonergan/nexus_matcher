"""
nexus_matcher.infrastructure.adapters.caches.memory | Layer: INFRASTRUCTURE
In-memory L1 LRU cache implementation.

## Relationships
# IMPLEMENTS → domain/ports/cache :: BaseCache abstract class
# USED_BY    → application/services :: L1 caching layer (sub-ms)
# USED_BY    → infrastructure/adapters/caches/tiered :: TieredCache L1 layer

## Attributes
# Security: In-memory only, no persistence, clears on restart
# Performance: Sub-millisecond access (<1ms), O(1) get/set
# Reliability: Thread-safe, no external dependencies

## Research Reference
# README_RESEARCH_3.md, Lines 21-37
# Target: 5K entries, sub-ms access, LRU eviction
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, TypeVar

from nexus_matcher.domain.ports.cache import BaseCache, CacheConfig, CacheStats

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    """Internal cache entry with metadata."""

    value: Any
    expires_at: float | None = None  # Unix timestamp or None for no expiry

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class L1LRUCache(BaseCache[Any]):
    """
    In-memory LRU cache for L1 caching layer.

    Provides sub-millisecond access times using Python's OrderedDict
    for efficient LRU eviction. Thread-safe via RLock.

    Research targets (README_RESEARCH_3.md):
    - 5,000 entry capacity
    - Sub-millisecond (<1ms) get/set latency
    - 60-75% latency reduction when combined with L2

    Example:
        cache = L1LRUCache()  # Default 5K capacity
        cache.set("embedding_hash", embedding_vector)

        result = cache.get("embedding_hash")
        if result is not None:
            # Cache hit - sub-ms latency
            return result
        # Cache miss - fall through to L2
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
    ) -> None:
        """
        Initialize L1 LRU cache.

        Args:
            config: Cache configuration. Defaults to 5K entries, 1 hour TTL.
        """
        if config is None:
            config = CacheConfig(max_size=5000, ttl=timedelta(hours=1))

        super().__init__(config)

        self._max_size = config.max_size

        # OrderedDict for LRU - most recently used at end
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        logger.info(f"Initialized L1LRUCache with max_size={self._max_size}")

    @property
    def cache_type(self) -> str:
        """Get cache type identifier."""
        return "memory"

    def _get_internal(self, key: str) -> Any | None:
        """
        Internal get implementation.

        O(1) average case. Moves accessed entry to end (most recently used).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry.value

    def _set_internal(
        self,
        key: str,
        value: Any,
        ttl: timedelta | None = None,
    ) -> bool:
        """
        Internal set implementation.

        O(1) average case. Evicts LRU entry if at capacity.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live

        Returns:
            True if successfully cached
        """
        with self._lock:
            # Calculate expiration time
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl.total_seconds()

            entry = CacheEntry(value=value, expires_at=expires_at)

            # If key exists, update and move to end
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
                return True

            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                # Remove oldest (first) entry
                evicted_key, _ = self._cache.popitem(last=False)
                self._stats.evictions += 1
                logger.debug(f"Evicted key: {evicted_key}")

            # Add new entry at end
            self._cache[key] = entry
            return True

    def _delete_internal(self, key: str) -> bool:
        """
        Internal delete implementation.

        Args:
            key: Cache key

        Returns:
            True if key existed and was deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def _exists_internal(self, key: str) -> bool:
        """
        Internal exists implementation.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return False

            return True

    def _clear_internal(self) -> int:
        """
        Internal clear implementation.

        Returns:
            Number of items cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics with current size."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=len(self._cache),
            )

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key -> value for found keys
        """
        results = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                results[key] = value
        return results

    def set_many(self, items: dict[str, Any], ttl: timedelta | None = None) -> int:
        """
        Set multiple values in cache.

        Args:
            items: Dictionary of key -> value
            ttl: Time-to-live for all items

        Returns:
            Number of items set
        """
        count = 0
        for key, value in items.items():
            if self.set(key, value, ttl):
                count += 1
        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired.append(key)

            for key in expired:
                del self._cache[key]

            return len(expired)


class InMemoryCache(L1LRUCache):
    """
    Alias for L1LRUCache for backward compatibility.

    Use L1LRUCache for new code.
    """

    pass
