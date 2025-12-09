"""
nexus_matcher.domain.services | Layer: DOMAIN
Domain services for NexusMatcher.

## Relationships
# EXPORTS → AbbreviationExpander :: abbreviation expansion
# EXPORTS → AbbreviationDictionary :: abbreviation dictionary
# EXPORTS → DomainMatcher :: domain hierarchy matching
# EXPORTS → TypeCompatibilityScorer :: type compatibility scoring
# EXPORTS → ContextEnricher :: context enrichment for nested schemas
"""

from nexus_matcher.domain.services.abbreviation import (
    AbbreviationDictionary,
    AbbreviationExpander,
    AbbreviationMapping,
    ExpandedText,
    DEFAULT_ABBREVIATIONS,
)
from nexus_matcher.domain.services.context_enricher import (
    ContextEnricher,
    EnrichmentConfig,
    enrich_field,
    enrich_fields,
)
from nexus_matcher.domain.services.domain_hierarchy import (
    Domain,
    DomainHierarchy,
    DomainMatch,
    DomainMatcher,
    DomainPath,
    DomainRelationship,
    DEFAULT_HIERARCHY_DATA,
)
from nexus_matcher.domain.services.type_compatibility import (
    CompatibilityLevel,
    TypeCategory,
    TypeCompatibilityResult,
    TypeCompatibilityScorer,
)

__all__ = [
    # Abbreviation expansion
    "AbbreviationExpander",
    "AbbreviationDictionary",
    "AbbreviationMapping",
    "ExpandedText",
    "DEFAULT_ABBREVIATIONS",
    # Context enrichment (GAP-006)
    "ContextEnricher",
    "EnrichmentConfig",
    "enrich_field",
    "enrich_fields",
    # Domain hierarchy
    "Domain",
    "DomainHierarchy",
    "DomainMatch",
    "DomainMatcher",
    "DomainPath",
    "DomainRelationship",
    "DEFAULT_HIERARCHY_DATA",
    # Type compatibility
    "CompatibilityLevel",
    "TypeCategory",
    "TypeCompatibilityResult",
    "TypeCompatibilityScorer",
]
