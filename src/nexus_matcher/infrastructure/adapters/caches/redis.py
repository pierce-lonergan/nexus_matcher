"""
nexus_matcher.infrastructure.adapters.caches.redis | Layer: INFRASTRUCTURE
Redis cache implementation for L2 caching.

## Relationships
# IMPLEMENTS → domain/ports/cache :: Cache protocol
# DEPENDS_ON → redis :: Redis Python client
# USED_BY    → application/services :: L2 caching layer

## Attributes
# Security: Supports password authentication, SSL
# Performance: 2-5ms typical latency
# Reliability: Connection pooling, automatic reconnection
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any, TypeVar

from nexus_matcher.domain.ports.cache import BaseCache, CacheConfig

# Lazy import to avoid hard dependency
try:
    import redis
except ImportError:
    redis = None  # type: ignore


logger = logging.getLogger(__name__)

T = TypeVar("T")


class RedisCache(BaseCache[Any]):
    """
    Redis cache implementation.

    Production-grade L2 cache using Redis for:
    - Distributed caching across instances
    - Persistent cache (survives restarts)
    - TTL-based expiration
    - High throughput (100K+ ops/sec)

    Example:
        # Local instance
        config = CacheConfig(ttl=timedelta(hours=1))
        cache = RedisCache(config, host="localhost", port=6379)

        # With URL
        cache = RedisCache(config, url="redis://user:pass@host:6379/0")

        # Use
        cache.set("key", {"data": "value"})
        result = cache.get("key")  # Returns {"data": "value"}

        # Custom TTL
        cache.set("temp", data, ttl=timedelta(minutes=5))
    """

    def __init__(
        self,
        config: CacheConfig,
        host: str | None = None,
        port: int | None = None,
        db: int = 0,
        password: str | None = None,
        url: str | None = None,
        key_prefix: str = "",
        socket_timeout: float = 5.0,
        connection_pool: Any | None = None,
    ) -> None:
        """
        Initialize Redis cache.

        Args:
            config: Cache configuration
            host: Redis host for direct connection
            port: Redis port for direct connection
            db: Redis database number
            password: Redis password
            url: Full Redis URL (overrides host/port/db/password)
            key_prefix: Prefix for all keys (namespace)
            socket_timeout: Socket timeout in seconds
            connection_pool: Optional connection pool
        """
        if redis is None:
            raise ImportError("redis is required for RedisCache. Install with: pip install redis")

        super().__init__(config)

        self._key_prefix = key_prefix

        # Connect via URL or host/port
        if url:
            self._client = redis.Redis.from_url(
                url,
                socket_timeout=socket_timeout,
                decode_responses=False,  # We handle encoding ourselves
            )
        elif connection_pool:
            self._client = redis.Redis(connection_pool=connection_pool)
        else:
            self._client = redis.Redis(
                host=host or "localhost",
                port=port or 6379,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                decode_responses=False,
            )

    @property
    def cache_type(self) -> str:
        """Cache type identifier."""
        return "redis"

    def _make_key(self, key: str) -> str:
        """Create full key with prefix."""
        return f"{self._key_prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        return json.dumps(value).encode("utf-8")

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        return json.loads(data.decode("utf-8"))

    def _get_internal(self, key: str) -> Any | None:
        """Internal get implementation."""
        try:
            full_key = self._make_key(key)
            data = self._client.get(full_key)

            if data is None:
                return None

            return self._deserialize(data)

        except Exception as e:
            logger.warning(f"Redis get failed for key '{key}': {e}")
            return None

    def _set_internal(self, key: str, value: Any, ttl: timedelta | None) -> bool:
        """Internal set implementation."""
        try:
            full_key = self._make_key(key)
            data = self._serialize(value)

            if ttl:
                # Use setex for TTL in seconds
                ttl_seconds = int(ttl.total_seconds())
                self._client.setex(full_key, ttl_seconds, data)
            else:
                # No TTL - use set without expiry
                self._client.set(full_key, data)

            return True

        except Exception as e:
            logger.warning(f"Redis set failed for key '{key}': {e}")
            return False

    def _delete_internal(self, key: str) -> bool:
        """Internal delete implementation."""
        try:
            full_key = self._make_key(key)
            result = self._client.delete(full_key)
            return result > 0

        except Exception as e:
            logger.warning(f"Redis delete failed for key '{key}': {e}")
            return False

    def _exists_internal(self, key: str) -> bool:
        """Internal exists implementation."""
        try:
            full_key = self._make_key(key)
            return self._client.exists(full_key) > 0

        except Exception as e:
            logger.warning(f"Redis exists failed for key '{key}': {e}")
            return False

    def _clear_internal(self) -> int:
        """Internal clear implementation."""
        try:
            if self._key_prefix:
                # Clear only keys with our prefix
                pattern = f"{self._key_prefix}*"
                keys = list(self._client.scan_iter(pattern))
                if keys:
                    return self._client.delete(*keys)
                return 0
            else:
                # Clear entire database - use with caution!
                self._client.flushdb()
                return 0  # Can't know exact count

        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")
            return 0

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values at once.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key -> value for found keys
        """
        if not keys:
            return {}

        try:
            full_keys = [self._make_key(k) for k in keys]
            values = self._client.mget(full_keys)

            result = {}
            for key, value in zip(keys, values, strict=False):
                if value is not None:
                    result[key] = self._deserialize(value)
                    self._stats.record_hit()
                else:
                    self._stats.record_miss()

            return result

        except Exception as e:
            logger.warning(f"Redis mget failed: {e}")
            return {}

    def set_many(
        self,
        items: dict[str, Any],
        ttl: timedelta | None = None,
    ) -> int:
        """
        Set multiple values at once.

        Args:
            items: Dictionary of key -> value
            ttl: Optional TTL for all items

        Returns:
            Number of items successfully set
        """
        if not items:
            return 0

        actual_ttl = ttl or self._config.ttl

        try:
            # Use pipeline for efficiency
            pipe = self._client.pipeline()

            for key, value in items.items():
                full_key = self._make_key(key)
                data = self._serialize(value)

                if actual_ttl:
                    ttl_seconds = int(actual_ttl.total_seconds())
                    pipe.setex(full_key, ttl_seconds, data)
                else:
                    pipe.set(full_key, data)

            pipe.execute()
            return len(items)

        except Exception as e:
            logger.warning(f"Redis mset failed: {e}")
            return 0

    def increment(self, key: str, amount: int = 1) -> int | None:
        """
        Increment a counter.

        Args:
            key: Cache key
            amount: Amount to increment by

        Returns:
            New value or None on error
        """
        try:
            full_key = self._make_key(key)
            return self._client.incrby(full_key, amount)

        except Exception as e:
            logger.warning(f"Redis incr failed for key '{key}': {e}")
            return None

    def ping(self) -> bool:
        """
        Check if Redis is connected.

        Returns:
            True if connected
        """
        try:
            return self._client.ping()
        except Exception:
            return False

    def close(self) -> None:
        """Close the Redis connection."""
        try:
            self._client.close()
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")
