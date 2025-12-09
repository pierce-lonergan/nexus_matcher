# Module: Redis Cache Adapter

## Purpose

Production-grade Redis cache adapter for L2 caching. Provides low-latency distributed caching with TTL support, serialization, and connection pooling.

## Domain Model

### Entities

- **RedisCache**: Production L2 cache implementation
  - Invariants: Keys must be strings, values must be serializable
  - States: Connected, Disconnected
  - Events: N/A (infrastructure adapter)

### Value Objects

- **CacheConfig**: Configuration for cache behavior (TTL, max_size)
- **CacheStats**: Hit/miss statistics

### Domain Services

- Implements `Cache[T]` protocol from domain/ports

## Redis Configuration

### Connection Options

```python
# Local instance
RedisCache(config, host="localhost", port=6379)

# Redis Cloud / Sentinel
RedisCache(config, url="redis://user:pass@host:6379/0")

# With connection pool
RedisCache(config, connection_pool=pool)
```

### Serialization

| Type | Serializer |
|------|------------|
| Default | JSON |
| Binary | Pickle |
| Compressed | Gzip + JSON |

## Planned Implementation

- [x] Domain analysis complete
- [ ] RedisCache class
- [ ] Connection management
- [ ] Key prefix/namespace support
- [ ] TTL management
- [ ] JSON serialization
- [ ] Unit tests (TDD)
- [ ] Integration tests

## Dependencies

```toml
[project.optional-dependencies]
redis = ["redis>=5.0.0"]
```

## Usage Example

```python
from nexus_matcher.infrastructure.adapters.caches.redis import RedisCache
from nexus_matcher.domain.ports.cache import CacheConfig
from datetime import timedelta

# Configure
config = CacheConfig(
    max_size=10000,
    ttl=timedelta(hours=1),
    eviction_policy="lru"
)

# Connect
cache = RedisCache(config, host="localhost", port=6379)

# Or with URL
cache = RedisCache(config, url="redis://localhost:6379/0")

# Use
cache.set("key", {"data": "value"})
result = cache.get("key")

# With custom TTL
cache.set("temp_key", data, ttl=timedelta(minutes=5))

# Check stats
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")
```

## File Structure

```
src/nexus_matcher/infrastructure/adapters/caches/
├── __init__.py
├── memory.py        # In-memory L1 cache
└── redis.py         # Redis L2 cache (new)
```
