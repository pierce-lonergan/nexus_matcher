"""
nexus_matcher.application.use_cases | Layer: APPLICATION
Use cases (application services) for NexusMatcher.

## Relationships
# EXPORTS → NexusMatcher :: main matcher class
# EXPORTS → BatchProcessor :: batch processing
# EXPORTS → MatchingConfig, BatchConfig :: configuration
"""

from nexus_matcher.application.use_cases.batch_match import (
    AsyncBatchProcessor,
    BatchConfig,
    BatchProcessor,
    BatchProgress,
    BatchResult,
)
from nexus_matcher.application.use_cases.match_schema import (
    MatchingConfig,
    NexusMatcher,
)

__all__ = [
    "AsyncBatchProcessor",
    "BatchConfig",
    # Batch processing
    "BatchProcessor",
    "BatchProgress",
    "BatchResult",
    "MatchingConfig",
    # Core matching
    "NexusMatcher",
]
