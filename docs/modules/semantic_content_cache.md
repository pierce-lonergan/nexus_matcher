# Module: Semantic Content Cache

## Purpose

Content-based embedding cache using BLAKE3 hashing for deduplication. Eliminates redundant embedding computations by caching results keyed by normalized content hash.

## Status: VALIDATED (GAP-004) ✓

## Research Reference

README_RESEARCH_3.md, Lines 21-37:
> "Cache embeddings by content hash using BLAKE3 to avoid recomputing identical queries, typically achieving 40-60% cache hit rates that translate to 50-70% cost reduction for embedding API calls."

## Domain Model

### Entities

- **SemanticContentCache**: Content-hash-keyed embedding cache
  - Invariants: max_size > 0, valid hashing
  - States: Active (always)

- **ContentHasher**: BLAKE3-based content fingerprinting
  - Invariants: Consistent hashing for identical content
  - Options: normalize, lowercase

### Configuration

```python
from nexus_matcher.infrastructure.adapters.caches import SemanticContentCache

# Default configuration
cache = SemanticContentCache(
    max_size=10000,      # Maximum cached embeddings
    ttl=timedelta(hours=1),
    normalize=True,      # Collapse whitespace
    lowercase=False,     # Case-sensitive
)

# With case normalization
cache = SemanticContentCache(
    normalize=True,
    lowercase=True,      # "CustomerEmail" == "customeremail"
)
```

## Key Operations

### get_or_compute (Primary Interface)

```python
def compute_embedding(text: str) -> np.ndarray:
    return model.encode(text)

# First call - computes and caches
embedding1 = cache.get_or_compute("customer email", compute_embedding)

# Second call - returns cached (no computation)
embedding2 = cache.get_or_compute("customer email", compute_embedding)

# Normalized - also hits cache (if normalize=True)
embedding3 = cache.get_or_compute("customer  email", compute_embedding)
```

### batch_get_or_compute (Batch Efficiency)

```python
def batch_compute(texts: list[str]) -> list[np.ndarray]:
    return model.encode(texts)

contents = ["field_a", "field_b", "field_c", "field_a"]  # field_a is duplicate

# Only computes 3 embeddings, returns 4
results = cache.batch_get_or_compute(contents, batch_compute)
```

## Performance Characteristics

| Metric | Achieved | Target |
|--------|----------|--------|
| Cost Reduction | 99.3% | ≥50% |
| Hit Rate (50% rep) | 50.0% | ≥40% |
| Hashing Throughput | 781K ops/s | - |
| Batch Efficiency | 50.0% | ≥40% |

## Integration Example

```python
from nexus_matcher.infrastructure.adapters.caches import (
    L1LRUCache,
    SemanticContentCache,
)

# L1 for result caching (by hash key)
result_cache = L1LRUCache()

# Semantic cache for embedding deduplication (by content)
embedding_cache = SemanticContentCache(max_size=10000)

def get_embedding(field_name: str) -> np.ndarray:
    return embedding_cache.get_or_compute(
        field_name,
        lambda t: model.encode(t)
    )
```

## Test Coverage

- 21 unit tests covering:
  - ContentHasher consistency and normalization
  - BLAKE3 functionality
  - Basic cache operations
  - get_or_compute behavior
  - Cost savings tracking
  - Batch operations
  - TTL and eviction
  - Memory efficiency

## File Structure

```
src/nexus_matcher/infrastructure/adapters/caches/
├── __init__.py     # Exports SemanticContentCache, ContentHasher
├── memory.py       # L1LRUCache (backing store)
├── content.py      # SemanticContentCache, ContentHasher
└── redis.py        # Redis L2 cache
```
