# Incremental Update Manager — Module Documentation

> **Layer:** Infrastructure / Adapters
> **Gap:** GAP-005 (BLAKE3 Incremental Updates)
> **Status:** VALIDATED ✓
> **Research:** README_RESEARCH_3.md, Lines 33-35

## Overview

The Incremental Update Manager provides efficient change detection for dictionary entries using BLAKE3 content hashing. Instead of recomputing embeddings for all entries during updates, it identifies only the added, modified, and deleted entries, achieving 90-99% computation savings.

## Research Alignment

**Target:** 90-99% computation savings for typical updates
**Achieved:**
- 0.1% changes → 99.9% savings ✓
- 5% changes → 95.0% savings ✓
- 10% changes → 90.0% savings ✓

**Algorithm:** BLAKE3 (8-10× faster than SHA-256)
**Throughput:** 698K hashes/sec, 450K entries/sec change detection

## Components

### ContentHashTracker

Tracks BLAKE3 hashes of content for change detection.

```python
from nexus_matcher.infrastructure.adapters.incremental import ContentHashTracker

tracker = ContentHashTracker()
tracker.track("entry_1", "customer email address")

# Check if content changed
if tracker.has_changed("entry_1", "customer email"):
    print("Content modified!")

# Persistence
state = tracker.to_dict()
tracker2 = ContentHashTracker.from_dict(state)
```

### ChangeDetector

Categorizes entries into added/modified/deleted/unchanged.

```python
from nexus_matcher.infrastructure.adapters.incremental import (
    ChangeDetector,
    ContentHashTracker,
)

tracker = ContentHashTracker()
tracker.track("entry_1", "content 1")
tracker.track("entry_2", "content 2")

detector = ChangeDetector(tracker)
changes = detector.detect_changes({
    "entry_1": "content 1",      # Unchanged
    "entry_2": "modified!",      # Modified
    "entry_3": "new content",    # Added
})
# entry_1 in changes.unchanged
# entry_2 in changes.modified
# entry_3 in changes.added
# (entry that was in tracker but not in new entries → deleted)
```

### ChangeResult

Immutable result with computed properties.

```python
from nexus_matcher.infrastructure.adapters.incremental import ChangeResult

result = ChangeResult(
    added={"a", "b"},
    modified={"c"},
    deleted={"d"},
    unchanged={"e", "f", "g"},
)

result.total_changed  # 4 (added + modified + deleted)
result.total_unchanged  # 3
result.change_ratio  # 0.571
result.estimated_savings_pct  # 42.9%
```

### IncrementalUpdateManager

Orchestrates the complete incremental update workflow.

```python
from nexus_matcher.infrastructure.adapters.incremental import IncrementalUpdateManager

manager = IncrementalUpdateManager()

# Initial load (all entries are "added")
entries = load_dictionary()  # {"DD_001": "desc 1", ...}
result = manager.compute_changes(entries)

print(f"Initial load: {result.added_count} entries to process")
for entry_id, content in result.entries_to_process.items():
    embedding = compute_embedding(content)
    store_embedding(entry_id, embedding)

manager.apply_changes(result)

# Later, incremental update
updated_entries = load_dictionary()  # May have changes
result = manager.compute_changes(updated_entries)

print(f"Savings: {result.savings_pct:.1f}%")
print(f"Only {result.process_count} entries to process")

for entry_id, content in result.entries_to_process.items():
    embedding = compute_embedding(content)
    store_embedding(entry_id, embedding)

manager.apply_changes(result)

# Persistence
manager.save_state("/path/to/state.json")
# Later...
manager = IncrementalUpdateManager.load_state("/path/to/state.json")
```

## Integration Example

Integrating with the dictionary indexing workflow:

```python
class IncrementalDictionaryIndexer:
    def __init__(self, embedding_provider, vector_store):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.update_manager = IncrementalUpdateManager()
    
    def index(self, entries: dict[str, str]) -> dict:
        """Index dictionary entries with incremental optimization."""
        result = self.update_manager.compute_changes(entries)
        
        if result.process_count == 0:
            return {"status": "no_changes", "savings_pct": 100.0}
        
        # Only compute embeddings for changed entries
        texts = list(result.entries_to_process.values())
        ids = list(result.entries_to_process.keys())
        embeddings = self.embedding_provider.embed(texts)
        
        # Update vector store
        for entry_id, embedding in zip(ids, embeddings):
            self.vector_store.upsert(entry_id, embedding)
        
        # Handle deletions
        for entry_id in result.changes.deleted:
            self.vector_store.delete(entry_id)
        
        # Commit changes
        self.update_manager.apply_changes(result)
        
        return {
            "status": "completed",
            "added": result.added_count,
            "modified": result.modified_count,
            "deleted": result.deleted_count,
            "unchanged": result.unchanged_count,
            "savings_pct": result.savings_pct,
        }
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| BLAKE3 Hash Time | 1.43 µs/hash |
| BLAKE3 Throughput | 698K hashes/sec |
| Change Detection | 450K entries/sec |
| Memory per Entry | ~100 bytes (hash + overhead) |
| State Serialization | JSON (human-readable) |

## Design Notes

### Falsy Empty Tracker Bug

The `ContentHashTracker` implements `__len__` which returns the number of tracked entries. An empty tracker returns 0, which is falsy in Python. This caused a subtle bug where:

```python
# WRONG: Creates new tracker if passed tracker is empty
self._tracker = tracker or ContentHashTracker()

# CORRECT: Explicit None check
self._tracker = tracker if tracker is not None else ContentHashTracker()
```

### BLAKE3 Fallback

BLAKE3 is 8-10× faster than SHA-256 but requires the `blake3` package. The implementation falls back to SHA-256 if BLAKE3 is not installed:

```python
pip install blake3  # Recommended for best performance
```

## API Reference

### compute_hash(content: str) -> str
Compute BLAKE3 hash of content (64 hex chars).

### ContentHashTracker
- `track(entry_id, content)` → str (hash)
- `has_hash(entry_id)` → bool
- `get_hash(entry_id)` → str | None
- `has_changed(entry_id, content)` → bool
- `remove(entry_id)`
- `clear()`
- `tracked_ids()` → set[str]
- `to_dict()` → dict
- `from_dict(data)` → ContentHashTracker

### ChangeDetector
- `detect_changes(entries)` → ChangeResult

### ChangeResult (frozen dataclass)
- `added`, `modified`, `deleted`, `unchanged` → frozenset[str]
- `total_changed`, `total_unchanged`, `total_entries` → int
- `change_ratio`, `estimated_savings_pct` → float

### IncrementalUpdateManager
- `compute_changes(entries)` → UpdateResult
- `apply_changes(result)`
- `save_state(path)`
- `load_state(path)` → IncrementalUpdateManager
- `tracker` → ContentHashTracker
- `tracked_count` → int

### UpdateResult (dataclass)
- `changes` → ChangeResult
- `entries_to_process` → dict[str, str]
- `added_count`, `modified_count`, `deleted_count`, `unchanged_count`, `process_count` → int
- `savings_pct` → float
