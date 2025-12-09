"""
nexus_matcher.application.use_cases | Layer: APPLICATION
Use cases (application services) for NexusMatcher.

## Relationships
# EXPORTS → NexusMatcher :: main matcher class
# EXPORTS → BatchProcessor :: batch processing
# EXPORTS → MatchingConfig, BatchConfig :: configuration
"""

from nexus_matcher.application.use_cases.match_schema import (
    MatchingConfig,
    NexusMatcher,
)
from nexus_matcher.application.use_cases.batch_match import (
    AsyncBatchProcessor,
    BatchConfig,
    BatchProcessor,
    BatchProgress,
    BatchResult,
)

__all__ = [
    # Core matching
    "NexusMatcher",
    "MatchingConfig",
    # Batch processing
    "BatchProcessor",
    "AsyncBatchProcessor",
    "BatchConfig",
    "BatchProgress",
    "BatchResult",
]
