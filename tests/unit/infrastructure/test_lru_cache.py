"""
tests.unit.infrastructure.test_lru_cache | Layer: TEST
Tests: L1 LRU Cache | Target: src/infrastructure/adapters/caches/memory.py

TDD Phase: RED → Tests written before implementation
Research Reference: README_RESEARCH_3.md, Lines 21-37
Gap: GAP-003 - L1 LRU Cache Layer
"""

import time
import pytest
from datetime import timedelta


class TestL1CacheProperties:
    """Test basic L1 cache properties."""

    def test_cache_type_is_memory(self):
        """Test cache type is 'memory'."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)
        assert cache.cache_type == "memory"

    def test_accepts_max_size_config(self):
        """Test cache accepts max_size configuration."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=5000)
        cache = L1LRUCache(config)
        assert cache._max_size == 5000

    def test_default_max_size_is_5000(self):
        """Test default max_size per research recommendation."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache

        cache = L1LRUCache()
        assert cache._max_size == 5000


class TestL1CacheBasicOperations:
    """Test basic get/set operations."""

    def test_set_and_get_value(self):
        """Test setting and getting a value."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

    def test_get_nonexistent_returns_none(self):
        """Test getting nonexistent key returns None."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        result = cache.get("nonexistent")
        assert result is None

    def test_set_multiple_values(self):
        """Test setting multiple values."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_overwrite_existing_key(self):
        """Test overwriting an existing key."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"

    def test_delete_key(self):
        """Test deleting a key."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_exists_returns_true_for_existing_key(self):
        """Test exists returns True for existing key."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        assert cache.exists("key1") is True
        assert cache.exists("nonexistent") is False

    def test_clear_removes_all_entries(self):
        """Test clear removes all entries."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cleared = cache.clear()

        assert cleared == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestL1CacheLRUEviction:
    """Test LRU eviction behavior."""

    def test_evicts_lru_when_full(self):
        """Test LRU eviction when cache is full."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=3)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        # Cache is now full

        # Add another entry - should evict key1 (LRU)
        cache.set("key4", "value4")

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_access_updates_lru_order(self):
        """Test accessing a key updates its LRU position."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=3)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 - should move it to most recently used
        _ = cache.get("key1")

        # Add another entry - should evict key2 (now LRU)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Still present (was accessed)
        assert cache.get("key2") is None  # Evicted (was LRU)
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_tracks_eviction_count(self):
        """Test that eviction count is tracked in stats."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=2)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Evicts key1
        cache.set("key4", "value4")  # Evicts key2

        stats = cache.get_stats()
        assert stats.evictions == 2


class TestL1CacheStats:
    """Test cache statistics tracking."""

    def test_tracks_hits(self):
        """Test cache tracks hit count."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("key1")
        cache.get("key1")

        stats = cache.get_stats()
        assert stats.hits == 3

    def test_tracks_misses(self):
        """Test cache tracks miss count."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        cache.get("nonexistent1")
        cache.get("nonexistent2")

        stats = cache.get_stats()
        assert stats.misses == 2

    def test_tracks_size(self):
        """Test cache tracks current size."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.get_stats()
        assert stats.size == 2

    def test_hit_rate_calculation(self):
        """Test hit rate is calculated correctly."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.get_stats()
        assert stats.hit_rate == 0.75  # 3 hits / 4 total


class TestL1CacheTTL:
    """Test TTL (time-to-live) expiration."""

    def test_entry_expires_after_ttl(self):
        """Test entry expires after TTL."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100, ttl=timedelta(milliseconds=50))
        cache = L1LRUCache(config)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for TTL to expire
        time.sleep(0.1)

        assert cache.get("key1") is None

    def test_per_key_ttl_override(self):
        """Test per-key TTL override."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100, ttl=timedelta(hours=1))
        cache = L1LRUCache(config)

        # Set with short TTL override
        cache.set("key1", "value1", ttl=timedelta(milliseconds=50))
        assert cache.get("key1") == "value1"

        # Wait for TTL to expire
        time.sleep(0.1)

        assert cache.get("key1") is None


class TestL1CacheLatency:
    """Test sub-millisecond latency requirement."""

    def test_get_latency_under_1ms(self):
        """Test get operation completes in under 1ms (research target)."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=5000)
        cache = L1LRUCache(config)

        # Populate cache with 5000 entries
        for i in range(5000):
            cache.set(f"key_{i}", f"value_{i}")

        # Measure get latency (warm cache)
        start = time.perf_counter()
        for i in range(100):
            _ = cache.get(f"key_{i % 5000}")
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / 100) * 1000
        assert avg_latency_ms < 1.0, f"Average get latency {avg_latency_ms:.3f}ms exceeds 1ms target"

    def test_set_latency_under_1ms(self):
        """Test set operation completes in under 1ms."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=5000)
        cache = L1LRUCache(config)

        # Measure set latency
        start = time.perf_counter()
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / 100) * 1000
        assert avg_latency_ms < 1.0, f"Average set latency {avg_latency_ms:.3f}ms exceeds 1ms target"


class TestL1CacheComplexValues:
    """Test caching complex value types."""

    def test_cache_dict_values(self):
        """Test caching dictionary values."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        data = {"name": "customer_id", "score": 0.95, "metadata": {"domain": "customer"}}
        cache.set("key1", data)
        result = cache.get("key1")

        assert result == data

    def test_cache_list_values(self):
        """Test caching list values."""
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        data = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]
        cache.set("key1", data)
        result = cache.get("key1")

        assert result == data

    def test_cache_numpy_array(self):
        """Test caching numpy arrays (embeddings)."""
        import numpy as np
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=100)
        cache = L1LRUCache(config)

        embedding = np.random.rand(768).astype(np.float32)
        cache.set("embedding_1", embedding)
        result = cache.get("embedding_1")

        assert result is not None
        assert np.array_equal(result, embedding)


class TestL1CacheThreadSafety:
    """Test thread safety for concurrent access."""

    def test_concurrent_reads_and_writes(self):
        """Test concurrent read/write operations."""
        import threading
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache
        from nexus_matcher.domain.ports.cache import CacheConfig

        config = CacheConfig(max_size=1000)
        cache = L1LRUCache(config)
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f"key_{threading.current_thread().name}_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    _ = cache.get(f"key_{threading.current_thread().name}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, name=f"writer_{i}"))
            threads.append(threading.Thread(target=reader, name=f"reader_{i}"))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
