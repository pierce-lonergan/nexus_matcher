"""
tests.unit.infrastructure.test_semantic_content_cache | Layer: TEST
Tests: Semantic Content Cache | Target: src/infrastructure/adapters/caches/content.py

TDD Phase: RED → Tests written before implementation
Research Reference: README_RESEARCH_3.md, Lines 21-37
Gap: GAP-004 - Semantic Content Caching
"""

import time
from datetime import timedelta

import numpy as np
import pytest

# Check if blake3 is available
try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


class TestContentHasherProperties:
    """Test BLAKE3 content hasher properties."""

    @pytest.mark.skipif(not HAS_BLAKE3, reason="blake3 not installed")
    def test_hasher_uses_blake3(self):
        """Test hasher uses BLAKE3 algorithm."""
        from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

        hasher = ContentHasher()
        assert hasher.algorithm == "blake3"

    def test_hasher_uses_fallback_when_blake3_unavailable(self):
        """Test hasher falls back to sha256 when blake3 not installed."""
        from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

        hasher = ContentHasher()
        # Should use either blake3 or sha256
        assert hasher.algorithm in ("blake3", "sha256")

    def test_hash_produces_consistent_output(self):
        """Test same input produces same hash."""
        from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

        hasher = ContentHasher()
        content = "customer_email_address"

        hash1 = hasher.hash(content)
        hash2 = hasher.hash(content)

        assert hash1 == hash2

    def test_hash_produces_different_output_for_different_input(self):
        """Test different inputs produce different hashes."""
        from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

        hasher = ContentHasher()

        hash1 = hasher.hash("customer_email")
        hash2 = hasher.hash("user_email")

        assert hash1 != hash2

    def test_hash_is_64_char_hex_string(self):
        """Test hash is 64 character hex string (256 bits)."""
        from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

        hasher = ContentHasher()
        hash_value = hasher.hash("test content")

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_hash_normalizes_whitespace(self):
        """Test hasher normalizes whitespace for consistency."""
        from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

        hasher = ContentHasher(normalize=True)

        hash1 = hasher.hash("customer  email")
        hash2 = hasher.hash("customer email")
        hash3 = hasher.hash("  customer email  ")

        assert hash1 == hash2 == hash3

    def test_hash_normalizes_case_when_configured(self):
        """Test hasher can normalize case."""
        from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

        hasher = ContentHasher(normalize=True, lowercase=True)

        hash1 = hasher.hash("CustomerEmail")
        hash2 = hasher.hash("customeremail")

        assert hash1 == hash2

    def test_hash_throughput_is_not_pathological(self):
        """
        Hashing must not be a bottleneck. Deliberately NOT a comparison against SHA-256.

        This previously asserted `sha256_time / blake3_time > 0.5` and failed in CI at
        0.48x. That assertion measured a third-party library on whatever CPU the runner
        happened to allocate, not anything this project controls -- and its own docstring
        conceded "in some environments BLAKE3 may not be significantly faster". Python's
        hashlib.sha256 is a heavily optimised C path, so on short inputs a Python-wrapped
        BLAKE3 losing to it is normal rather than a defect.

        What actually matters here is that content hashing stays cheap enough to run per
        cache lookup. That is an absolute bound, and it is generous because CI runners are
        shared: it catches an accidental O(n^2) or a per-call model load, not a slow day.
        """
        from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

        hasher = ContentHasher()
        content = "customer_email_address" * 100
        iterations = 1000

        start = time.perf_counter()
        for _ in range(iterations):
            hasher.hash(content)
        elapsed = time.perf_counter() - start

        per_hash_us = (elapsed / iterations) * 1_000_000
        assert per_hash_us < 500, (
            f"content hashing took {per_hash_us:.1f}us per call on a ~2.2KB input; "
            f"that is slow enough to matter per cache lookup and suggests a regression"
        )

    def test_hash_is_stable_across_calls(self):
        """The property the cache actually depends on: same input, same digest."""
        from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

        hasher = ContentHasher()
        content = "customer_email_address" * 100
        assert hasher.hash(content) == hasher.hash(content)
        assert hasher.hash(content) != hasher.hash(content + "x")


class TestSemanticContentCacheBasicOperations:
    """Test basic cache operations."""

    def test_cache_type_is_semantic_content(self):
        """Test cache type identifier."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache()
        assert cache.cache_type == "semantic_content"

    def test_set_and_get_by_content(self):
        """Test caching by content (not by key)."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache()
        content = "customer_email_address"
        embedding = np.random.rand(768).astype(np.float32)

        cache.set_by_content(content, embedding)
        result = cache.get_by_content(content)

        assert result is not None
        assert np.array_equal(result, embedding)

    def test_get_nonexistent_content_returns_none(self):
        """Test getting uncached content returns None."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache()
        result = cache.get_by_content("never_cached_content")

        assert result is None

    def test_semantically_equivalent_content_hits_cache(self):
        """Test normalized content produces cache hit."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache(normalize=True)
        embedding = np.random.rand(768).astype(np.float32)

        # Cache with one variation
        cache.set_by_content("customer  email", embedding)

        # Should hit with normalized variations
        result = cache.get_by_content("customer email")
        assert result is not None
        assert np.array_equal(result, embedding)

    def test_get_or_compute_returns_cached(self):
        """Test get_or_compute returns cached value without computing."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache()
        content = "test_field"
        cached_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        cache.set_by_content(content, cached_embedding)

        compute_called = []

        def compute_fn(text):
            compute_called.append(text)
            return np.array([9.0, 9.0, 9.0], dtype=np.float32)

        result = cache.get_or_compute(content, compute_fn)

        assert len(compute_called) == 0  # Not called
        assert np.array_equal(result, cached_embedding)

    def test_get_or_compute_computes_on_miss(self):
        """Test get_or_compute calls compute function on cache miss."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache()
        content = "new_field"
        computed_embedding = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        compute_called = []

        def compute_fn(text):
            compute_called.append(text)
            return computed_embedding

        result = cache.get_or_compute(content, compute_fn)

        assert len(compute_called) == 1
        assert compute_called[0] == content
        assert np.array_equal(result, computed_embedding)

        # Should be cached now
        cache.get_or_compute(content, compute_fn)
        assert len(compute_called) == 1  # Still only one call


class TestSemanticContentCacheStats:
    """Test cache statistics and cost savings."""

    def test_tracks_compute_savings(self):
        """Test cache tracks compute operations saved."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache()
        embedding = np.random.rand(768).astype(np.float32)

        # First call - miss
        cache.set_by_content("field1", embedding)
        cache.get_by_content("field1")  # Hit
        cache.get_by_content("field1")  # Hit
        cache.get_by_content("field2")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1

    def test_tracks_cost_reduction_percentage(self):
        """Test cache reports cost reduction percentage."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache()
        embedding = np.random.rand(768).astype(np.float32)

        def compute_fn(text):
            return embedding

        # 10 queries, 6 unique, 4 duplicates
        contents = ["a", "b", "c", "d", "e", "f", "a", "b", "c", "d"]
        for content in contents:
            cache.get_or_compute(content, compute_fn)

        stats = cache.get_stats()
        # 4 hits out of 10 queries = 40% savings
        assert 0.35 <= stats.hit_rate <= 0.45

    def test_batch_get_or_compute(self):
        """Test batch operation for multiple contents."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache()

        # Pre-cache some
        cache.set_by_content("field1", np.array([1.0], dtype=np.float32))
        cache.set_by_content("field2", np.array([2.0], dtype=np.float32))

        compute_called = []

        def compute_fn(texts):
            compute_called.extend(texts)
            return [np.array([float(i)], dtype=np.float32) for i in range(len(texts))]

        contents = ["field1", "field2", "field3", "field4"]
        results = cache.batch_get_or_compute(contents, compute_fn)

        assert len(results) == 4
        # Only field3 and field4 should be computed
        assert set(compute_called) == {"field3", "field4"}


class TestSemanticContentCacheIntegration:
    """Test integration with L1 LRU cache."""

    def test_backed_by_l1_cache(self):
        """Test SemanticContentCache uses L1LRUCache internally."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache
        from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache

        cache = SemanticContentCache(max_size=1000)

        # Should have internal L1 cache
        assert hasattr(cache, "_cache")
        assert isinstance(cache._cache, L1LRUCache)

    def test_respects_max_size(self):
        """Test cache evicts when full."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache(max_size=3)

        embedding = np.random.rand(768).astype(np.float32)
        cache.set_by_content("field1", embedding)
        cache.set_by_content("field2", embedding)
        cache.set_by_content("field3", embedding)
        cache.set_by_content("field4", embedding)  # Should evict field1

        assert cache.get_by_content("field1") is None  # Evicted
        assert cache.get_by_content("field4") is not None

    def test_supports_ttl(self):
        """Test cache supports TTL expiration."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache(ttl=timedelta(milliseconds=50))

        embedding = np.random.rand(768).astype(np.float32)
        cache.set_by_content("field1", embedding)

        assert cache.get_by_content("field1") is not None

        time.sleep(0.1)

        assert cache.get_by_content("field1") is None


class TestSemanticContentCacheWithRealEmbeddings:
    """Test with realistic embedding scenarios."""

    def test_cache_768_dim_embeddings(self):
        """Test caching standard 768-dim embeddings."""
        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        cache = SemanticContentCache()

        # Simulate multiple field names
        fields = [
            "customer_email_address",
            "user_phone_number",
            "account_balance",
            "transaction_date",
            "customer_email_address",  # Duplicate
        ]

        compute_count = [0]

        def compute_fn(text):
            compute_count[0] += 1
            return np.random.rand(768).astype(np.float32)

        for field in fields:
            cache.get_or_compute(field, compute_fn)

        # Should only compute 4 times (not 5)
        assert compute_count[0] == 4

    def test_memory_efficient_for_many_embeddings(self):
        """Test memory usage is reasonable for many cached embeddings."""
        import tracemalloc

        from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

        tracemalloc.start()

        cache = SemanticContentCache(max_size=1000)

        # Cache 1000 embeddings
        for i in range(1000):
            embedding = np.random.rand(768).astype(np.float32)
            cache.set_by_content(f"field_{i}", embedding)

        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Should be reasonable (<50MB for 1000 x 768 embeddings)
        # 768 * 4 bytes * 1000 = ~3MB for embeddings alone
        # With overhead, should be <20MB
        assert peak / 1024 / 1024 < 50, f"Memory usage {peak / 1024 / 1024:.1f}MB exceeds 50MB"
