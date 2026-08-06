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
    DEFAULT_ABBREVIATIONS,
    AbbreviationDictionary,
    AbbreviationExpander,
    AbbreviationMapping,
    ExpandedText,
)
from nexus_matcher.domain.services.context_enricher import (
    ContextEnricher,
    EnrichmentConfig,
    enrich_field,
    enrich_fields,
)
from nexus_matcher.domain.services.domain_hierarchy import (
    DEFAULT_HIERARCHY_DATA,
    Domain,
    DomainHierarchy,
    DomainMatch,
    DomainMatcher,
    DomainPath,
    DomainRelationship,
)
from nexus_matcher.domain.services.type_compatibility import (
    CompatibilityLevel,
    TypeCategory,
    TypeCompatibilityResult,
    TypeCompatibilityScorer,
)

__all__ = [
    "DEFAULT_ABBREVIATIONS",
    "DEFAULT_HIERARCHY_DATA",
    "AbbreviationDictionary",
    # Abbreviation expansion
    "AbbreviationExpander",
    "AbbreviationMapping",
    # Type compatibility
    "CompatibilityLevel",
    # Context enrichment (GAP-006)
    "ContextEnricher",
    # Domain hierarchy
    "Domain",
    "DomainHierarchy",
    "DomainMatch",
    "DomainMatcher",
    "DomainPath",
    "DomainRelationship",
    "EnrichmentConfig",
    "ExpandedText",
    "TypeCategory",
    "TypeCompatibilityResult",
    "TypeCompatibilityScorer",
    "enrich_field",
    "enrich_fields",
]
