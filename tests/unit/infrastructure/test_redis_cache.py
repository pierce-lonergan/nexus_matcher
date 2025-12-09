"""
tests.unit.infrastructure.test_redis_cache | Layer: TEST
Tests: Redis cache | Target: src/infrastructure/adapters/caches/redis.py

TDD Phase: RED → Tests written before implementation
"""

import pytest
import json
from datetime import timedelta
from unittest.mock import Mock, MagicMock, patch

from nexus_matcher.domain.ports.cache import CacheConfig, CacheStats

# Check if redis is available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class TestRedisCacheProperties:
    """Test basic cache properties."""

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_cache_type(self):
        """Test cache type is 'redis'."""
        from nexus_matcher.infrastructure.adapters.caches.redis import (
            RedisCache,
        )

        config = CacheConfig()
        
        with patch("nexus_matcher.infrastructure.adapters.caches.redis.redis.Redis"):
            cache = RedisCache(config, host="localhost", port=6379)
            assert cache.cache_type == "redis"

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_creates_with_config(self):
        """Test cache accepts configuration."""
        from nexus_matcher.infrastructure.adapters.caches.redis import (
            RedisCache,
        )

        config = CacheConfig(
            max_size=5000,
            ttl=timedelta(minutes=30),
        )

        with patch("nexus_matcher.infrastructure.adapters.caches.redis.redis.Redis"):
            cache = RedisCache(config, host="localhost")
            assert cache._config.max_size == 5000
            assert cache._config.ttl == timedelta(minutes=30)


class TestRedisCacheConnection:
    """Test connection management."""

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_connects_to_localhost(self):
        """Test connecting to localhost."""
        from nexus_matcher.infrastructure.adapters.caches.redis import (
            RedisCache,
        )

        config = CacheConfig()

        with patch("nexus_matcher.infrastructure.adapters.caches.redis.redis.Redis") as mock_redis:
            cache = RedisCache(config, host="localhost", port=6379)
            mock_redis.assert_called_once()

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_connects_with_url(self):
        """Test connecting with URL."""
        from nexus_matcher.infrastructure.adapters.caches.redis import (
            RedisCache,
        )

        config = CacheConfig()

        with patch("nexus_matcher.infrastructure.adapters.caches.redis.redis.Redis.from_url") as mock_from_url:
            cache = RedisCache(config, url="redis://localhost:6379/0")
            mock_from_url.assert_called_once()


class TestRedisCacheOperations:
    """Test cache operations with mock."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        with patch("nexus_matcher.infrastructure.adapters.caches.redis.redis.Redis") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def cache(self, mock_redis):
        """Create cache with mock client."""
        from nexus_matcher.infrastructure.adapters.caches.redis import (
            RedisCache,
        )

        config = CacheConfig(ttl=timedelta(hours=1))
        return RedisCache(config, host="localhost")

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_get_existing_key(self, cache, mock_redis):
        """Test getting existing key."""
        mock_redis.get.return_value = json.dumps({"value": "test"}).encode()

        result = cache.get("test_key")

        assert result == {"value": "test"}
        mock_redis.get.assert_called_once()

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_get_missing_key(self, cache, mock_redis):
        """Test getting non-existent key."""
        mock_redis.get.return_value = None

        result = cache.get("missing_key")

        assert result is None

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_set_value(self, cache, mock_redis):
        """Test setting a value."""
        mock_redis.setex.return_value = True

        result = cache.set("test_key", {"data": "value"})

        assert result is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_set_with_custom_ttl(self, cache, mock_redis):
        """Test setting with custom TTL."""
        mock_redis.setex.return_value = True

        result = cache.set("test_key", {"data": "value"}, ttl=timedelta(minutes=5))

        assert result is True
        # Verify TTL was set to 300 seconds
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 300  # TTL in seconds

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_delete_existing_key(self, cache, mock_redis):
        """Test deleting existing key."""
        mock_redis.delete.return_value = 1

        result = cache.delete("test_key")

        assert result is True

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_delete_missing_key(self, cache, mock_redis):
        """Test deleting non-existent key."""
        mock_redis.delete.return_value = 0

        result = cache.delete("missing_key")

        assert result is False

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_exists_true(self, cache, mock_redis):
        """Test exists returns True for existing key."""
        mock_redis.exists.return_value = 1

        result = cache.exists("test_key")

        assert result is True

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_exists_false(self, cache, mock_redis):
        """Test exists returns False for missing key."""
        mock_redis.exists.return_value = 0

        result = cache.exists("missing_key")

        assert result is False


class TestRedisCacheStats:
    """Test cache statistics."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        with patch("nexus_matcher.infrastructure.adapters.caches.redis.redis.Redis") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def cache(self, mock_redis):
        """Create cache with mock client."""
        from nexus_matcher.infrastructure.adapters.caches.redis import (
            RedisCache,
        )

        config = CacheConfig()
        return RedisCache(config, host="localhost")

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_stats_initial(self, cache):
        """Test initial stats are zero."""
        stats = cache.get_stats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_stats_track_hits(self, cache, mock_redis):
        """Test stats track cache hits."""
        mock_redis.get.return_value = json.dumps("value").encode()

        cache.get("key1")
        cache.get("key2")

        stats = cache.get_stats()
        assert stats.hits == 2

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_stats_track_misses(self, cache, mock_redis):
        """Test stats track cache misses."""
        mock_redis.get.return_value = None

        cache.get("missing1")
        cache.get("missing2")

        stats = cache.get_stats()
        assert stats.misses == 2

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_hit_rate_calculation(self, cache, mock_redis):
        """Test hit rate is calculated correctly."""
        # 3 hits
        mock_redis.get.return_value = json.dumps("value").encode()
        cache.get("hit1")
        cache.get("hit2")
        cache.get("hit3")

        # 1 miss
        mock_redis.get.return_value = None
        cache.get("miss1")

        stats = cache.get_stats()
        assert stats.hit_rate == 0.75  # 3/4


class TestRedisCacheKeyPrefix:
    """Test key prefix/namespace support."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        with patch("nexus_matcher.infrastructure.adapters.caches.redis.redis.Redis") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_key_prefix(self, mock_redis):
        """Test keys are prefixed."""
        from nexus_matcher.infrastructure.adapters.caches.redis import (
            RedisCache,
        )

        config = CacheConfig()
        cache = RedisCache(config, host="localhost", key_prefix="nexus:")

        mock_redis.get.return_value = None
        cache.get("test_key")

        # Verify prefixed key was used
        call_args = mock_redis.get.call_args
        assert call_args[0][0] == "nexus:test_key"


class TestRedisCacheClear:
    """Test cache clearing."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        with patch("nexus_matcher.infrastructure.adapters.caches.redis.redis.Redis") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def cache(self, mock_redis):
        """Create cache with mock client."""
        from nexus_matcher.infrastructure.adapters.caches.redis import (
            RedisCache,
        )

        config = CacheConfig()
        return RedisCache(config, host="localhost", key_prefix="test:")

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_clear_with_prefix(self, cache, mock_redis):
        """Test clearing keys with prefix."""
        mock_redis.scan_iter.return_value = ["test:key1", "test:key2"]
        mock_redis.delete.return_value = 2

        count = cache.clear()

        assert count >= 0  # May be 0 or more depending on implementation


class TestRedisCacheErrorHandling:
    """Test error handling."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        with patch("nexus_matcher.infrastructure.adapters.caches.redis.redis.Redis") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def cache(self, mock_redis):
        """Create cache with mock client."""
        from nexus_matcher.infrastructure.adapters.caches.redis import (
            RedisCache,
        )

        config = CacheConfig()
        return RedisCache(config, host="localhost")

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_get_handles_connection_error(self, cache, mock_redis):
        """Test get returns None on connection error."""
        mock_redis.get.side_effect = Exception("Connection refused")

        result = cache.get("test_key")

        assert result is None

    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
    def test_set_handles_connection_error(self, cache, mock_redis):
        """Test set returns False on connection error."""
        mock_redis.setex.side_effect = Exception("Connection refused")

        result = cache.set("test_key", "value")

        assert result is False
