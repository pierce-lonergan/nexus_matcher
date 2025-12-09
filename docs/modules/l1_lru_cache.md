# Module: L1 LRU Cache

## Purpose

In-memory LRU (Least Recently Used) cache for sub-millisecond access latency. First tier in the hierarchical caching system.

## Status: RESOLVED (GAP-003)

## Research Reference

README_RESEARCH_3.md, Lines 21-37:
> "Three-tier caching: L1 in-process LRU (5K entries), L2 Redis, L3 semantic"
> "L1 should have sub-millisecond access"

## Domain Model

### Entities

- **L1LRUCache**: In-memory LRU cache implementation
  - Invariants: max_size > 0, thread-safe operations
  - States: Active (always)
  - Events: N/A (infrastructure adapter)

### Value Objects

- **CacheConfig**: Configuration for cache behavior (max_size, ttl, eviction_policy)
- **CacheStats**: Statistics (hits, misses, evictions, size)
- **CacheEntry**: Internal entry with value and expiration time

### Domain Services

- Implements `BaseCache` abstract class from domain/ports/cache

## Implementation Details

### Algorithm

Uses Python's `OrderedDict` for O(1) LRU eviction:
- `get()`: Move accessed key to end (most recent)
- `set()`: Add at end, evict from front if at capacity
- Thread-safe via `threading.RLock`

### Performance Characteristics

| Operation | Time Complexity | Target Latency |
|-----------|-----------------|----------------|
| get() | O(1) | < 1ms |
| set() | O(1) | < 1ms |
| delete() | O(1) | < 1ms |
| exists() | O(1) | < 1ms |
| clear() | O(n) | - |

### Configuration

```python
from nexus_matcher.domain.ports.cache import CacheConfig
from nexus_matcher.infrastructure.adapters.caches import L1LRUCache

# Default: 5K entries, 1 hour TTL
cache = L1LRUCache()

# Custom configuration
config = CacheConfig(
    max_size=10000,
    ttl=timedelta(minutes=30),
    eviction_policy="lru"
)
cache = L1LRUCache(config)
```

## Usage Example

```python
from nexus_matcher.infrastructure.adapters.caches import L1LRUCache
import numpy as np

# Initialize
cache = L1LRUCache()

# Cache embeddings
embedding = np.random.rand(768).astype(np.float32)
cache.set("query_hash_abc123", embedding)

# Retrieve (sub-ms)
result = cache.get("query_hash_abc123")
if result is not None:
    # Cache hit!
    return result
# Cache miss - fall through to L2

# Check statistics
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")
```

## Integration with TieredCache

```python
# Future: TieredCache will combine L1 + L2
class TieredCache:
    def __init__(self, l1: L1LRUCache, l2: RedisCache):
        self._l1 = l1
        self._l2 = l2
    
    def get(self, key: str) -> Any:
        # Check L1 first (sub-ms)
        result = self._l1.get(key)
        if result is not None:
            return result
        
        # Fall through to L2 (2-5ms)
        result = self._l2.get(key)
        if result is not None:
            # Promote to L1
            self._l1.set(key, result)
        return result
```

## Test Coverage

- 25 unit tests covering:
  - Basic operations
  - LRU eviction behavior
  - Statistics tracking
  - TTL expiration
  - Sub-millisecond latency validation
  - Complex value types
  - Thread safety

## File Structure

```
src/nexus_matcher/infrastructure/adapters/caches/
├── __init__.py     # Exports L1LRUCache, InMemoryCache
├── memory.py       # L1LRUCache implementation
└── redis.py        # Redis L2 cache (existing)
```
