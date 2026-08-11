"""
tests.unit.infrastructure.test_semantic_content_cache | Layer: TEST
Tests: Semantic Content Cache | Target: src/infrastructure/adapters/caches/content.py

TDD Phase: RED → Tests written before implementation
Research Reference: README_RESEARCH_3.md, Lines 21-37
Gap: GAP-004 - Semantic Content Caching
"""

import hashlib
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


# =============================================================================
# HASH COST SHAPE -- what replaced this file's wall-clock timing assertion
# =============================================================================
#
# H-007: the noise band on this machine is a function of machine state, not a constant.
# So an absolute microseconds-per-hash assertion is wrong in both directions -- too loose
# on a quiet box to catch a real regression, too tight on a busy one to avoid inventing
# a red build. This file has already had two goes at one:
#
#   1. `sha256_time / blake3_time > 0.5` -- a race between two hash libraries. It failed
#      in CI at 0.48x, on nothing this project controls.
#   2. `per_hash_us < 500` over a single 1000-iteration window. MEASURED HERE 2026-08-10
#      on a 32-core box: 2.04 us idle, 82.03 us with 96 competing processes. A 40x
#      excursion against a 350x margin is one unlucky descheduling away from a red build
#      for reasons that have nothing to do with the hasher, and a test that goes red for
#      reasons unrelated to the code is a test people learn to re-run rather than read.
#
# H-008 established the way out and this is the same instrument at a smaller scale: a
# RATIO BETWEEN TWO INPUT SIZES MEASURED IN ONE RUN. Whatever the machine is doing to the
# 2.2 KB measurement it is also doing to the 1.1 MB one, so machine state largely divides
# out and what is left is the SHAPE of the cost curve -- a property of the algorithm, not
# of the afternoon.
#
# It is also STRICTLY STRONGER against the two defects the old assertion named. Measured
# on this tree, both pathologies below pass `per_hash_us < 500` and fail the shape gate:
#
#     pathology                            per-call us at 2.2 KB     shape ratio
#     a ~155 us per-call fixed cost                        154.9            5.83
#     an O(n^2) rehash                                       0.8        22472.05
#     (today's hasher, for reference)                        1.4          400.15
#
# The O(n^2) row is the sharp one: a timing taken at ONE input size cannot see a
# complexity defect at all, because complexity is not a number, it is a slope. The
# consequence a user feels is a cache lookup that is fine on a field name and pathological
# on a long business definition -- exactly the input this library hashes.

_HASH_UNIT = "customer_email_address"
_SMALL_TEXT = _HASH_UNIT * 100  # 2 200 chars
_HASH_SIZE_RATIO = 512
_LARGE_TEXT = _HASH_UNIT * (100 * _HASH_SIZE_RATIO)  # 1 126 400 chars

# RATCHET. These may tighten and may never loosen; loosening one is how a gate becomes a
# decoration. Both are stated as the exponent they admit, because that is the property
# being gated rather than a number that happened to pass on one afternoon.
#
# A perfectly linear hasher with no per-call overhead reads exactly _HASH_SIZE_RATIO, 512.
# Today's hasher reads a little under that -- the shortfall is its ~0.5 us fixed per-call
# cost, which the 2.2 KB measurement pays proportionally more of.
#
#   upper 2000  ~= 512 ** 1.22, so anything growing worse than O(n^1.22) is red. The
#               O(n^2) rehash below reads 22472, which is 11x past this.
#   lower   25  solves (c + 550) / (c + 1.37) = 25, so a fixed per-call cost above about
#               21 us is red -- flatter than O(n^0.52). A per-call model load, which is
#               the defect named, is milliseconds.
#
# HEADROOM, measured 2026-08-10 over min-of-15 interleaved trials: 398.97 .. 415.30 idle,
# and 324.76 .. 334.25 with 96 competing processes on 32 cores (3x oversubscription).
# The worst observation sits 13.0x above the floor and 6.0x below the ceiling, and load
# moves the ratio DOWN, away from the ceiling.
_SHAPE_LOWER = 25.0
_SHAPE_UPPER = 2000.0


def _per_call_us(hash_fn, text: str, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        hash_fn(text)
    return (time.perf_counter() - start) / iterations * 1_000_000


def _cost_shape(hash_fn, *, trials: int = 15, small_iterations: int = 200) -> float:
    """
    cost(1.1 MB) / cost(2.2 KB), both measured in this run, in this process.

    The scales are INTERLEAVED rather than measured in two blocks. All the small trials
    followed by all the large ones produces a ratio that describes two different minutes
    of machine state; H-008 pins the same property for the perf harness after one
    contended run there read 0.608 against a true 0.86.

    min-of-N at each scale, not the median: interference only ever makes a run slower, so
    the fastest trial is the least contaminated estimate of what this machine can do.
    """
    small: list[float] = []
    large: list[float] = []
    for _ in range(trials):
        small.append(_per_call_us(hash_fn, _SMALL_TEXT, small_iterations))
        large.append(_per_call_us(hash_fn, _LARGE_TEXT, 1))
    return min(large) / min(small)


def _shape_problem(ratio: float) -> str | None:
    """The decision rule, named separately so the controls below exercise the real one."""
    if ratio > _SHAPE_UPPER:
        return (
            f"hash cost grew {ratio:.1f}x for {_HASH_SIZE_RATIO}x the input, ceiling "
            f"{_SHAPE_UPPER:.0f}x. Cost is superlinear in input size, so hashing a long "
            f"business definition is disproportionately expensive per cache lookup."
        )
    if ratio < _SHAPE_LOWER:
        return (
            f"hash cost grew only {ratio:.1f}x for {_HASH_SIZE_RATIO}x the input, floor "
            f"{_SHAPE_LOWER:.0f}x. A fixed per-call cost now dominates the hash itself, "
            f"which is what a model load or a re-import on the hot path looks like."
        )
    return None


_FIXED_COST_BLOB = b"m" * 400_000


def _hash_with_a_per_call_fixed_cost(content: str) -> str:
    """
    A per-call model load, in the cheapest form that is still real CPU work: hash a fixed
    400 KB blob on every call. Cost barely moves with the input, which is the signature.
    """
    return hashlib.sha256(_FIXED_COST_BLOB + content.encode("utf-8")).hexdigest()


_QUADRATIC_CHUNK = 16384


def _hash_quadratically(content: str) -> str:
    """O(n^2): rehash every 16 KB prefix, so the number of passes grows with the input."""
    data = content.encode("utf-8")
    digest = hashlib.sha256()
    for end in range(0, len(data) + 1, _QUADRATIC_CHUNK):
        digest.update(hashlib.sha256(data[:end]).digest())
    return digest.hexdigest()


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

    def test_hash_cost_shape_is_linear(self):
        """
        The gate. Hashing must stay cheap enough to run per cache lookup, and the property
        that says so without measuring a shared machine is the SHAPE of the cost curve.

        Deliberately NOT a race against SHA-256, and no longer an absolute per-call bound:
        see the ratchet block at the top of this file for both, and for the two pathologies
        that passed the absolute bound and fail this.
        """
        from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

        ratio = _cost_shape(ContentHasher().hash)
        problem = _shape_problem(ratio)
        assert problem is None, problem

    def test_a_per_call_fixed_cost_is_caught(self):
        """
        Control, and the reason the assertion above is not a decoration.

        A gate nobody has watched go red is a hypothesis. This drives the real measurement
        with a hasher carrying a ~155 us per-call fixed cost -- a model load, a re-import,
        an env probe on the hot path -- and requires the verdict to be red. The retired
        `per_hash_us < 500` bound passes this pathology, because 155 is less than 500.
        """
        ratio = _cost_shape(_hash_with_a_per_call_fixed_cost, trials=3, small_iterations=20)
        assert ratio < _SHAPE_LOWER, (
            f"a hasher whose cost is dominated by a fixed per-call blob read {ratio:.2f}x, "
            f"which the floor of {_SHAPE_LOWER:.0f}x accepts. The floor no longer separates "
            f"a hash from a model load."
        )
        assert _shape_problem(ratio) is not None

    def test_a_quadratic_hash_is_caught(self):
        """
        Control, the other direction, and the case a single-size timing is STRUCTURALLY
        blind to: an O(n^2) hasher costs 0.8 us on the 2.2 KB input the old assertion
        measured -- faster than today's hasher -- and 18 ms on a 1.1 MB one.

        A user hits this as a cache lookup that is instant on a column name and stalls on a
        long business definition, with no error anywhere.
        """
        ratio = _cost_shape(_hash_quadratically, trials=3)
        assert ratio > _SHAPE_UPPER, (
            f"an O(n^2) hasher read {ratio:.0f}x for {_HASH_SIZE_RATIO}x the input and the "
            f"ceiling of {_SHAPE_UPPER:.0f}x accepted it."
        )
        assert _shape_problem(ratio) is not None

    def test_a_uniform_slowdown_is_invisible_here_and_that_is_stated(self):
        """
        The hole, pinned so it cannot be forgotten.

        Eight times the work at BOTH scales leaves the ratio unchanged and this gate green.
        That is not a bug in the ratio, it is the price of contention immunity: absolute
        cost is judged by benchmarks/optimization_ledger.py against a calibrated noise
        band, shape is judged here. A reader who believes this gate covers absolute cost
        will stop running the other one.
        """
        from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

        hasher = ContentHasher()

        def eight_times_the_work(content: str) -> str:
            digest = ""
            for _ in range(8):
                digest = hasher.hash(content)
            return digest

        ratio = _cost_shape(eight_times_the_work, trials=3)
        assert _shape_problem(ratio) is None, (
            "the shape gate reacted to a uniform slowdown. That is not what it measures, "
            "and a shape gate that drifts into absolute timing inherits H-007's noise "
            "problem -- the one this test exists to have escaped."
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
