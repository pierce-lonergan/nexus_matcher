"""
nexus_matcher.domain.services.domain_hierarchy | Layer: DOMAIN
Domain hierarchy service for computing domain-based matching scores.

## Relationships
# DEPENDS_ON → None (pure domain service)
# USED_BY    → application/use_cases/match_schema :: domain scoring
# USED_BY    → domain/models/entities :: DictionaryEntry.domain

## Invariants
# 1. Domain names are always uppercase (normalized)
# 2. No circular parent references
# 3. Each domain has at most one parent

## Attributes
# Security: No external I/O, pure computation
# Performance: O(d) where d = depth of hierarchy, lookup is O(1)
# Reliability: Unknown domains handled gracefully with neutral scores
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

# =============================================================================
# ENUMS
# =============================================================================


class DomainRelationship(Enum):
    """Types of relationships between domains."""

    EXACT = "exact"  # Same domain
    PARENT = "parent"  # Domain1 is parent of Domain2
    CHILD = "child"  # Domain1 is child of Domain2
    SIBLING = "sibling"  # Share immediate parent
    COUSIN = "cousin"  # Share ancestor within 2 levels
    UNRELATED = "unrelated"  # No common ancestor or distant


# =============================================================================
# VALUE OBJECTS
# =============================================================================


@dataclass(frozen=True)
class Domain:
    """
    A business domain category.

    Domains form a hierarchy representing logical groupings of data
    (e.g., FINANCE → BANKING → ACCOUNTS).

    Example:
        root = Domain("FINANCE")
        child = Domain("BANKING", parent=root)
    """

    name: str
    parent: Domain | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the domain."""
        # Normalize name to uppercase
        object.__setattr__(self, "name", self.name.upper().strip())

        # Validate
        if not self.name:
            raise ValueError("Domain name must be non-empty")

    @property
    def is_root(self) -> bool:
        """Check if this is a root domain (no parent)."""
        return self.parent is None

    @property
    def depth(self) -> int:
        """Get depth in hierarchy (0 for root)."""
        if self.parent is None:
            return 0
        return 1 + self.parent.depth

    def ancestors(self) -> Iterator[Domain]:
        """Iterate over ancestors from parent to root."""
        current = self.parent
        while current is not None:
            yield current
            current = current.parent

    def __hash__(self) -> int:
        """Hash based on name only (for set/dict usage)."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Equality based on name."""
        if isinstance(other, Domain):
            return self.name == other.name
        return False


@dataclass(frozen=True)
class DomainPath:
    """
    Full path from root to a domain.

    Example:
        path = DomainPath(("FINANCE", "BANKING", "ACCOUNTS"))
        str(path) == "FINANCE.BANKING.ACCOUNTS"
    """

    parts: tuple[str, ...]

    @classmethod
    def from_domain(cls, domain: Domain) -> DomainPath:
        """Create path from a domain by traversing to root."""
        parts = [domain.name]
        current = domain.parent
        while current is not None:
            parts.append(current.name)
            current = current.parent
        return cls(tuple(reversed(parts)))

    def __str__(self) -> str:
        """Get dot-separated path string."""
        return ".".join(self.parts)

    def __len__(self) -> int:
        """Get path length."""
        return len(self.parts)

    def __contains__(self, item: str) -> bool:
        """Check if domain name is in path."""
        return item.upper() in self.parts


@dataclass(frozen=True)
class DomainMatch:
    """
    Result of comparing two domains.

    Contains the relationship type and computed score.
    """

    domain1: str
    domain2: str
    relationship: DomainRelationship
    score: float
    common_ancestor: str | None = None

    @property
    def is_related(self) -> bool:
        """Check if domains are related (not UNRELATED)."""
        return self.relationship != DomainRelationship.UNRELATED


# =============================================================================
# DOMAIN HIERARCHY
# =============================================================================


class DomainHierarchy:
    """
    Tree structure of domains with efficient lookup.

    Supports:
    - Adding domains with parent relationships
    - Case-insensitive lookup
    - Path computation
    - Loading from dict

    Example:
        hierarchy = DomainHierarchy()
        hierarchy.add_domain("FINANCE")
        hierarchy.add_domain("BANKING", parent="FINANCE")
    """

    def __init__(self) -> None:
        """Initialize empty hierarchy."""
        self._domains: dict[str, Domain] = {}
        self._children: dict[str, list[Domain]] = {}

    @property
    def size(self) -> int:
        """Get number of domains."""
        return len(self._domains)

    @property
    def is_empty(self) -> bool:
        """Check if hierarchy is empty."""
        return len(self._domains) == 0

    def add_domain(self, name: str, parent: str | None = None) -> Domain:
        """
        Add a domain to the hierarchy.

        Args:
            name: Domain name
            parent: Parent domain name (None for root)

        Returns:
            The created Domain

        Raises:
            ValueError: If parent doesn't exist
        """
        name = name.upper().strip()

        # Get parent domain
        parent_domain = None
        if parent:
            parent_domain = self.get_domain(parent)
            if parent_domain is None:
                raise ValueError(f"Parent domain '{parent}' not found")

        # Create domain
        domain = Domain(name=name, parent=parent_domain)
        self._domains[name] = domain

        # Track children
        parent_key = parent_domain.name if parent_domain else ""
        if parent_key not in self._children:
            self._children[parent_key] = []
        self._children[parent_key].append(domain)

        return domain

    def get_domain(self, name: str) -> Domain | None:
        """
        Get domain by name (case-insensitive).

        Args:
            name: Domain name to look up

        Returns:
            Domain or None if not found
        """
        if not name:
            return None
        return self._domains.get(name.upper().strip())

    def get_children(self, name: str) -> list[Domain]:
        """
        Get child domains.

        Args:
            name: Parent domain name

        Returns:
            List of child domains
        """
        return self._children.get(name.upper().strip(), [])

    def get_path(self, name: str) -> DomainPath | None:
        """
        Get full path for a domain.

        Args:
            name: Domain name

        Returns:
            DomainPath or None if not found
        """
        domain = self.get_domain(name)
        if domain is None:
            return None
        return DomainPath.from_domain(domain)

    def get_roots(self) -> list[Domain]:
        """Get all root domains."""
        return self._children.get("", [])

    def __contains__(self, name: str) -> bool:
        """Check if domain exists."""
        return self.get_domain(name) is not None

    @classmethod
    def from_dict(cls, data: dict) -> DomainHierarchy:
        """
        Load hierarchy from nested dictionary.

        Args:
            data: Nested dict like {"FINANCE": {"BANKING": {"ACCOUNTS": {}}}}

        Returns:
            Populated DomainHierarchy
        """
        hierarchy = cls()

        def add_recursive(items: dict, parent: str | None = None) -> None:
            for name, children in items.items():
                hierarchy.add_domain(name, parent)
                if children:
                    add_recursive(children, name)

        add_recursive(data)
        return hierarchy


# =============================================================================
# DEFAULT HIERARCHY
# =============================================================================

# A generic retail-banking domain hierarchy, supplied as a starting point.
#
# This is an illustrative default, not a recommendation: the terms below are the
# ordinary vocabulary of the industry, and they will not line up with any given
# organisation's data model. Domain scoring is only as good as the hierarchy it
# runs against, so callers with a real taxonomy should load theirs instead --
# see DomainHierarchy.from_dict(). Nothing else in the library assumes these
# particular names.
DEFAULT_HIERARCHY_DATA: dict = {
    "FINANCE": {
        "BANKING": {
            "ACCOUNTS": {},
            "TRANSACTIONS": {},
            "BALANCES": {},
            "STATEMENTS": {},
        },
        "PAYMENTS": {
            "TRANSFERS": {},
            "WIRES": {},
            "ACH": {},
            "CHECKS": {},
        },
        "LENDING": {
            "LOANS": {},
            "MORTGAGES": {},
            "CREDIT_LINES": {},
        },
        "CARDS": {
            "CREDIT": {},
            "DEBIT": {},
            "PREPAID": {},
        },
    },
    "CUSTOMER": {
        "IDENTITY": {
            "KYC": {},
            "VERIFICATION": {},
        },
        "CONTACT": {
            "ADDRESS": {},
            "PHONE": {},
            "EMAIL": {},
        },
        "PREFERENCES": {},
        "RELATIONSHIPS": {},
    },
    "PRODUCT": {
        "DEPOSIT": {
            "CHECKING": {},
            "SAVINGS": {},
            "CD": {},
            "MONEY_MARKET": {},
        },
        "CREDIT": {
            "CREDIT_CARD": {},
            "LINE_OF_CREDIT": {},
        },
        "INVESTMENT": {
            "BROKERAGE": {},
            "RETIREMENT": {},
        },
    },
    "COMPLIANCE": {
        "KYC": {},
        "AML": {},
        "REGULATORY": {},
        "FRAUD": {},
    },
    "OPERATIONS": {
        "AUDIT": {},
        "SYSTEM": {},
        "LOGGING": {},
    },
    "REFERENCE": {
        "GEOGRAPHY": {
            "COUNTRY": {},
            "STATE": {},
            "CITY": {},
        },
        "CURRENCY": {},
        "CODES": {},
    },
}


# =============================================================================
# DOMAIN MATCHER SERVICE
# =============================================================================


# Scoring weights for relationships
RELATIONSHIP_SCORES: dict[DomainRelationship, float] = {
    DomainRelationship.EXACT: 1.00,
    DomainRelationship.PARENT: 0.85,
    DomainRelationship.CHILD: 0.80,
    DomainRelationship.SIBLING: 0.70,
    DomainRelationship.COUSIN: 0.50,
    DomainRelationship.UNRELATED: 0.25,
}


class DomainMatcher:
    """
    Domain service for computing domain compatibility scores.

    Determines the relationship between two domains and computes
    a score based on their position in the hierarchy.

    Example:
        matcher = DomainMatcher.default()
        result = matcher.match("ACCOUNTS", "TRANSACTIONS")
        # result.relationship == DomainRelationship.SIBLING
        # result.score == 0.70
    """

    # Singleton instance for default matcher
    _default_instance: ClassVar[DomainMatcher | None] = None

    def __init__(
        self,
        hierarchy: DomainHierarchy,
        unknown_score: float = 0.5,
    ) -> None:
        """
        Initialize matcher with hierarchy.

        Args:
            hierarchy: The domain hierarchy
            unknown_score: Score for unknown domains
        """
        self._hierarchy = hierarchy
        self._unknown_score = unknown_score

    @classmethod
    def default(cls) -> DomainMatcher:
        """
        Get the default matcher with built-in hierarchy.

        Returns:
            Singleton instance of default matcher
        """
        if cls._default_instance is None:
            hierarchy = DomainHierarchy.from_dict(DEFAULT_HIERARCHY_DATA)
            cls._default_instance = cls(hierarchy)
        return cls._default_instance

    @classmethod
    def reset_default(cls) -> None:
        """Reset the default singleton (for testing)."""
        cls._default_instance = None

    def match(self, domain1: str | None, domain2: str | None) -> DomainMatch:
        """
        Match two domains and compute their relationship score.

        Args:
            domain1: First domain name
            domain2: Second domain name

        Returns:
            DomainMatch with relationship and score
        """
        # Handle None/empty inputs
        if not domain1 or not domain2:
            return DomainMatch(
                domain1=domain1 or "",
                domain2=domain2 or "",
                relationship=DomainRelationship.UNRELATED,
                score=self._unknown_score,
            )

        # Normalize names
        name1 = domain1.upper().strip()
        name2 = domain2.upper().strip()

        # Get domains
        d1 = self._hierarchy.get_domain(name1)
        d2 = self._hierarchy.get_domain(name2)

        # Handle unknown domains
        if d1 is None or d2 is None:
            return DomainMatch(
                domain1=name1,
                domain2=name2,
                relationship=DomainRelationship.UNRELATED,
                score=self._unknown_score,
            )

        # Determine relationship
        relationship, ancestor = self._determine_relationship(d1, d2)
        score = RELATIONSHIP_SCORES[relationship]

        return DomainMatch(
            domain1=name1,
            domain2=name2,
            relationship=relationship,
            score=score,
            common_ancestor=ancestor.name if ancestor else None,
        )

    def score(self, domain1: str, domain2: str) -> float:
        """
        Get just the score for two domains.

        Convenience method for scoring integration.
        """
        return self.match(domain1, domain2).score

    def find_common_ancestor(
        self,
        domain1: str,
        domain2: str,
    ) -> Domain | None:
        """
        Find the common ancestor of two domains.

        Args:
            domain1: First domain name
            domain2: Second domain name

        Returns:
            Common ancestor Domain or None
        """
        d1 = self._hierarchy.get_domain(domain1)
        d2 = self._hierarchy.get_domain(domain2)

        if d1 is None or d2 is None:
            return None

        # Get all ancestors of d1 (including d1)
        d1_ancestors = {d1.name}
        for ancestor in d1.ancestors():
            d1_ancestors.add(ancestor.name)

        # Check d2 and its ancestors
        if d2.name in d1_ancestors:
            return d2

        for ancestor in d2.ancestors():
            if ancestor.name in d1_ancestors:
                return ancestor

        return None

    def _determine_relationship(
        self,
        d1: Domain,
        d2: Domain,
    ) -> tuple[DomainRelationship, Domain | None]:
        """
        Determine the relationship between two domains.

        Returns:
            Tuple of (relationship, common_ancestor)
        """
        # Exact match
        if d1.name == d2.name:
            return DomainRelationship.EXACT, d1

        # Check if d1 is ancestor of d2 (PARENT)
        for ancestor in d2.ancestors():
            if ancestor.name == d1.name:
                return DomainRelationship.PARENT, d1

        # Check if d2 is ancestor of d1 (CHILD)
        for ancestor in d1.ancestors():
            if ancestor.name == d2.name:
                return DomainRelationship.CHILD, d2

        # Find common ancestor
        common = self.find_common_ancestor(d1.name, d2.name)

        if common is None:
            return DomainRelationship.UNRELATED, None

        # Check if siblings (share immediate parent)
        if d1.parent and d2.parent and d1.parent.name == d2.parent.name:
            return DomainRelationship.SIBLING, d1.parent

        # Check distance to common ancestor
        d1_dist = self._distance_to_ancestor(d1, common)
        d2_dist = self._distance_to_ancestor(d2, common)

        # Cousins if within 2 levels
        if d1_dist <= 2 and d2_dist <= 2:
            return DomainRelationship.COUSIN, common

        return DomainRelationship.UNRELATED, common

    def _distance_to_ancestor(self, domain: Domain, ancestor: Domain) -> int:
        """Calculate distance from domain to ancestor."""
        distance = 0
        current = domain
        while current is not None:
            if current.name == ancestor.name:
                return distance
            current = current.parent
            distance += 1
        return float("inf")
