"""
infrastructure.adapters.incremental | Layer: INFRASTRUCTURE
Implements: GAP-005 BLAKE3 Incremental Updates

Research Reference: README_RESEARCH_3.md, Lines 33-35
Research Quote: "BLAKE3 content-based hashing for change detection (8-10x faster
than SHA-256)... eliminating 90-99% of wasted computation for typical update patterns."

Components:
- ContentHashTracker: BLAKE3-based content hash tracking
- ChangeDetector: Detects added/modified/deleted/unchanged entries
- ChangeResult: Immutable result of change detection
- IncrementalUpdateManager: Orchestrates incremental update workflow
- UpdateResult: Result with entries to process

Performance Targets:
- 100-1000x speedup for <1% corpus changes
- 10-50x speedup for 1-10% corpus changes
- 90-99% computation savings
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# BLAKE3 HASHING
# =============================================================================


def _get_blake3():
    """Lazy import of blake3 with fallback."""
    try:
        import blake3
        return blake3
    except ImportError:
        logger.warning("blake3 not installed, falling back to hashlib (slower)")
        return None


def compute_hash(content: str) -> str:
    """
    Compute BLAKE3 hash of content.
    
    Falls back to SHA-256 if blake3 not installed.
    BLAKE3 is 8-10x faster than SHA-256.
    
    Args:
        content: String content to hash
        
    Returns:
        64-character hex string hash
    """
    encoded = content.encode('utf-8')
    blake3_lib = _get_blake3()
    
    if blake3_lib is not None:
        return blake3_lib.blake3(encoded).hexdigest()
    else:
        import hashlib
        return hashlib.sha256(encoded).hexdigest()


# =============================================================================
# CHANGE RESULT
# =============================================================================


@dataclass(frozen=True)
class ChangeResult:
    """
    Immutable result of change detection.
    
    Attributes:
        added: Entry IDs that are new
        modified: Entry IDs with changed content
        deleted: Entry IDs that were removed
        unchanged: Entry IDs with same content
    """
    added: frozenset[str] = field(default_factory=frozenset)
    modified: frozenset[str] = field(default_factory=frozenset)
    deleted: frozenset[str] = field(default_factory=frozenset)
    unchanged: frozenset[str] = field(default_factory=frozenset)
    
    def __init__(
        self,
        added: set[str] | frozenset[str] | None = None,
        modified: set[str] | frozenset[str] | None = None,
        deleted: set[str] | frozenset[str] | None = None,
        unchanged: set[str] | frozenset[str] | None = None,
    ):
        # Convert to frozenset for immutability
        object.__setattr__(self, 'added', frozenset(added or set()))
        object.__setattr__(self, 'modified', frozenset(modified or set()))
        object.__setattr__(self, 'deleted', frozenset(deleted or set()))
        object.__setattr__(self, 'unchanged', frozenset(unchanged or set()))
    
    @property
    def total_changed(self) -> int:
        """Total entries that changed (added + modified + deleted)."""
        return len(self.added) + len(self.modified) + len(self.deleted)
    
    @property
    def total_unchanged(self) -> int:
        """Total entries that are unchanged."""
        return len(self.unchanged)
    
    @property
    def total_entries(self) -> int:
        """Total entries considered."""
        return self.total_changed + self.total_unchanged
    
    @property
    def change_ratio(self) -> float:
        """Ratio of changed entries to total."""
        if self.total_entries == 0:
            return 0.0
        return self.total_changed / self.total_entries
    
    @property
    def estimated_savings_pct(self) -> float:
        """
        Estimated computation savings percentage.
        
        Research target: 90-99% for typical updates.
        """
        if self.total_entries == 0:
            return 0.0
        return (self.total_unchanged / self.total_entries) * 100.0


# =============================================================================
# CONTENT HASH TRACKER
# =============================================================================


class ContentHashTracker:
    """
    Tracks content hashes for change detection.
    
    Uses BLAKE3 for fast cryptographic hashing (8-10x faster than SHA-256).
    
    Example:
        tracker = ContentHashTracker()
        tracker.track("entry_1", "customer email address")
        
        # Later, check if content changed
        if tracker.has_changed("entry_1", "customer email"):
            print("Content modified!")
    """
    
    def __init__(self):
        """Initialize empty tracker."""
        self._hashes: dict[str, str] = {}
    
    def track(self, entry_id: str, content: str) -> str:
        """
        Track content hash for an entry.
        
        Args:
            entry_id: Unique identifier for the entry
            content: Content to hash
            
        Returns:
            The computed hash
        """
        hash_value = compute_hash(content)
        self._hashes[entry_id] = hash_value
        return hash_value
    
    def has_hash(self, entry_id: str) -> bool:
        """Check if entry is being tracked."""
        return entry_id in self._hashes
    
    def get_hash(self, entry_id: str) -> str | None:
        """Get stored hash for entry, or None if not tracked."""
        return self._hashes.get(entry_id)
    
    def has_changed(self, entry_id: str, content: str) -> bool:
        """
        Check if content has changed from tracked version.
        
        New (untracked) entries are considered "changed".
        
        Args:
            entry_id: Entry identifier
            content: Current content to compare
            
        Returns:
            True if content differs or entry is new
        """
        stored_hash = self._hashes.get(entry_id)
        if stored_hash is None:
            return True  # New entry
        
        current_hash = compute_hash(content)
        return current_hash != stored_hash
    
    def remove(self, entry_id: str) -> None:
        """Stop tracking an entry."""
        self._hashes.pop(entry_id, None)
    
    def clear(self) -> None:
        """Clear all tracked hashes."""
        self._hashes.clear()
    
    def tracked_ids(self) -> set[str]:
        """Get all tracked entry IDs."""
        return set(self._hashes.keys())
    
    def to_dict(self) -> dict[str, str]:
        """Serialize tracker state to dictionary."""
        return dict(self._hashes)
    
    @classmethod
    def from_dict(cls, data: dict[str, str]) -> ContentHashTracker:
        """Deserialize tracker from dictionary."""
        tracker = cls()
        tracker._hashes = dict(data)
        return tracker
    
    def __len__(self) -> int:
        """Number of tracked entries."""
        return len(self._hashes)


# =============================================================================
# CHANGE DETECTOR
# =============================================================================


class ChangeDetector:
    """
    Detects changes between tracked state and new entries.
    
    Categorizes entries into: added, modified, deleted, unchanged.
    
    Example:
        tracker = ContentHashTracker()
        tracker.track("entry_1", "content 1")
        
        detector = ChangeDetector(tracker)
        changes = detector.detect_changes({
            "entry_1": "modified content",  # Modified
            "entry_2": "new content",       # Added
        })
        # entry_1 in changes.modified
        # entry_2 in changes.added
    """
    
    def __init__(self, tracker: ContentHashTracker | None = None):
        """
        Initialize detector with a tracker.
        
        Args:
            tracker: Existing tracker with baseline state.
                     If None, all entries will be considered "added".
        """
        # Note: Can't use `tracker or ContentHashTracker()` because
        # empty tracker is falsy (len == 0)
        self._tracker = tracker if tracker is not None else ContentHashTracker()
    
    def detect_changes(self, entries: dict[str, str]) -> ChangeResult:
        """
        Detect changes between tracked state and new entries.
        
        Args:
            entries: Dictionary mapping entry_id -> content
            
        Returns:
            ChangeResult with categorized entries
        """
        added: set[str] = set()
        modified: set[str] = set()
        unchanged: set[str] = set()
        
        # Check each new entry against tracked state
        for entry_id, content in entries.items():
            if not self._tracker.has_hash(entry_id):
                added.add(entry_id)
            elif self._tracker.has_changed(entry_id, content):
                modified.add(entry_id)
            else:
                unchanged.add(entry_id)
        
        # Find deleted entries (in tracker but not in new entries)
        new_ids = set(entries.keys())
        tracked_ids = self._tracker.tracked_ids()
        deleted = tracked_ids - new_ids
        
        return ChangeResult(
            added=added,
            modified=modified,
            deleted=deleted,
            unchanged=unchanged,
        )


# =============================================================================
# UPDATE RESULT
# =============================================================================


@dataclass
class UpdateResult:
    """
    Result of incremental update computation.
    
    Contains the change analysis and the entries that need processing.
    
    Attributes:
        changes: The ChangeResult with categorization
        entries_to_process: Dict of entry_id -> content for entries needing work
    """
    changes: ChangeResult
    entries_to_process: dict[str, str]
    
    @property
    def added_count(self) -> int:
        """Number of added entries."""
        return len(self.changes.added)
    
    @property
    def modified_count(self) -> int:
        """Number of modified entries."""
        return len(self.changes.modified)
    
    @property
    def deleted_count(self) -> int:
        """Number of deleted entries."""
        return len(self.changes.deleted)
    
    @property
    def unchanged_count(self) -> int:
        """Number of unchanged entries."""
        return len(self.changes.unchanged)
    
    @property
    def process_count(self) -> int:
        """Number of entries that need processing."""
        return len(self.entries_to_process)
    
    @property
    def savings_pct(self) -> float:
        """Estimated computation savings percentage."""
        return self.changes.estimated_savings_pct


# =============================================================================
# INCREMENTAL UPDATE MANAGER
# =============================================================================


class IncrementalUpdateManager:
    """
    Orchestrates incremental update workflow using BLAKE3 change detection.
    
    Research Target: 90-99% computation savings for typical updates.
    
    Workflow:
        1. compute_changes(entries) - Analyze what changed
        2. Process only entries_to_process (added + modified)
        3. apply_changes(result) - Update tracker state
    
    Example:
        manager = IncrementalUpdateManager()
        
        # Initial load (all entries are "added")
        result = manager.compute_changes(dictionary_entries)
        for entry_id, content in result.entries_to_process.items():
            process_entry(entry_id, content)
        manager.apply_changes(result)
        
        # Later, incremental update
        result = manager.compute_changes(updated_entries)
        print(f"Savings: {result.savings_pct:.1f}%")
        for entry_id, content in result.entries_to_process.items():
            process_entry(entry_id, content)
        manager.apply_changes(result)
    """
    
    def __init__(self, tracker: ContentHashTracker | None = None):
        """
        Initialize manager.
        
        Args:
            tracker: Existing tracker with baseline state.
                     If None, creates new empty tracker.
        """
        # Note: Can't use `tracker or ContentHashTracker()` because
        # empty tracker is falsy (len == 0)
        self._tracker = tracker if tracker is not None else ContentHashTracker()
        self._detector = ChangeDetector(self._tracker)
    
    def compute_changes(self, entries: dict[str, str]) -> UpdateResult:
        """
        Compute what has changed and what needs processing.
        
        Args:
            entries: Dictionary mapping entry_id -> content
            
        Returns:
            UpdateResult with changes and entries_to_process
        """
        changes = self._detector.detect_changes(entries)
        
        # Entries to process = added + modified
        entries_to_process = {
            entry_id: entries[entry_id]
            for entry_id in (changes.added | changes.modified)
        }
        
        return UpdateResult(
            changes=changes,
            entries_to_process=entries_to_process,
        )
    
    def apply_changes(self, result: UpdateResult) -> None:
        """
        Apply changes to tracker state.
        
        Call this after processing the entries_to_process.
        
        Args:
            result: The UpdateResult from compute_changes
        """
        # Track new/modified entries
        for entry_id, content in result.entries_to_process.items():
            self._tracker.track(entry_id, content)
        
        # Remove deleted entries
        for entry_id in result.changes.deleted:
            self._tracker.remove(entry_id)
    
    @property
    def tracker(self) -> ContentHashTracker:
        """Get the underlying tracker."""
        return self._tracker
    
    @property
    def tracked_count(self) -> int:
        """Number of tracked entries."""
        return len(self._tracker)
    
    def save_state(self, path: str | Path) -> None:
        """
        Save tracker state to JSON file.
        
        Enables persistence across sessions.
        
        Args:
            path: Path to save state file
        """
        state = {
            "version": 1,
            "hashes": self._tracker.to_dict(),
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved incremental state: {len(self._tracker)} entries to {path}")
    
    @classmethod
    def load_state(cls, path: str | Path) -> IncrementalUpdateManager:
        """
        Load manager from saved state file.
        
        Args:
            path: Path to state file
            
        Returns:
            Manager with restored state
        """
        path = Path(path)
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        tracker = ContentHashTracker.from_dict(state.get("hashes", {}))
        manager = cls(tracker)
        
        logger.info(f"Loaded incremental state: {len(tracker)} entries from {path}")
        return manager


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Core classes
    "ContentHashTracker",
    "ChangeDetector",
    "ChangeResult",
    "IncrementalUpdateManager",
    "UpdateResult",
    # Utilities
    "compute_hash",
]
