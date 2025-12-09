"""
nexus_matcher.infrastructure.adapters.caches | Layer: INFRASTRUCTURE
Cache implementations.

## Relationships
# EXPORTS → L1LRUCache :: In-memory L1 cache (sub-ms)
# EXPORTS → InMemoryCache :: Alias for L1LRUCache
# EXPORTS → SemanticContentCache :: Content-based embedding cache
# EXPORTS → ContentHasher :: BLAKE3 content hasher
# EXPORTS → RedisCache :: Redis L2 cache (optional)
"""

from nexus_matcher.infrastructure.adapters.caches.memory import (
    L1LRUCache,
    InMemoryCache,
)
from nexus_matcher.infrastructure.adapters.caches.content import (
    SemanticContentCache,
    ContentHasher,
)

__all__: list[str] = [
    "L1LRUCache",
    "InMemoryCache",
    "SemanticContentCache",
    "ContentHasher",
]

# Optional Redis support
try:
    from nexus_matcher.infrastructure.adapters.caches.redis import RedisCache
    __all__.append("RedisCache")
except ImportError:
    pass  # redis not installed
