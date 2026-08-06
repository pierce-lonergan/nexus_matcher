"""
nexus_matcher.domain.models | Layer: DOMAIN
Core domain models representing business entities.

These are the heart of the domain layer - pure business objects
with no external dependencies.
"""

from nexus_matcher.domain.models.entities import (
    DictionaryEntry,
    MatchingSession,
    MatchResult,
    Schema,
    SchemaField,
)

__all__ = [
    "DictionaryEntry",
    "MatchResult",
    "MatchingSession",
    "Schema",
    "SchemaField",
]
