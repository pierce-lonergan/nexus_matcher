"""
tests.unit.domain.test_domain_hierarchy | Layer: TEST
Tests: Domain hierarchy service | Target: src/domain/services/domain_hierarchy.py

TDD Phase: RED → Tests written before implementation
"""

import pytest


class TestDomain:
    """Test Domain value object."""

    def test_creates_valid_domain(self):
        """Test creating a valid domain."""
        from nexus_matcher.domain.services.domain_hierarchy import Domain

        domain = Domain(name="ACCOUNTS")
        assert domain.name == "ACCOUNTS"
        assert domain.parent is None

    def test_normalizes_name_to_uppercase(self):
        """Test domain name is normalized to uppercase."""
        from nexus_matcher.domain.services.domain_hierarchy import Domain

        domain = Domain(name="accounts")
        assert domain.name == "ACCOUNTS"

    def test_rejects_empty_name(self):
        """Test empty name raises error."""
        from nexus_matcher.domain.services.domain_hierarchy import Domain

        with pytest.raises(ValueError, match="name"):
            Domain(name="")

    def test_creates_with_parent(self):
        """Test creating domain with parent."""
        from nexus_matcher.domain.services.domain_hierarchy import Domain

        parent = Domain(name="BANKING")
        child = Domain(name="ACCOUNTS", parent=parent)

        assert child.parent is parent
        assert child.parent.name == "BANKING"

    def test_is_root(self):
        """Test is_root property."""
        from nexus_matcher.domain.services.domain_hierarchy import Domain

        root = Domain(name="FINANCE")
        child = Domain(name="BANKING", parent=root)

        assert root.is_root
        assert not child.is_root

    def test_depth(self):
        """Test depth calculation."""
        from nexus_matcher.domain.services.domain_hierarchy import Domain

        root = Domain(name="FINANCE")
        child = Domain(name="BANKING", parent=root)
        grandchild = Domain(name="ACCOUNTS", parent=child)

        assert root.depth == 0
        assert child.depth == 1
        assert grandchild.depth == 2

    def test_is_immutable(self):
        """Test domain is immutable."""
        from nexus_matcher.domain.services.domain_hierarchy import Domain

        domain = Domain(name="ACCOUNTS")
        with pytest.raises(AttributeError):
            domain.name = "DIFFERENT"


class TestDomainPath:
    """Test DomainPath value object."""

    def test_creates_from_domain(self):
        """Test creating path from domain."""
        from nexus_matcher.domain.services.domain_hierarchy import Domain, DomainPath

        root = Domain(name="FINANCE")
        child = Domain(name="BANKING", parent=root)
        grandchild = Domain(name="ACCOUNTS", parent=child)

        path = DomainPath.from_domain(grandchild)
        assert path.parts == ("FINANCE", "BANKING", "ACCOUNTS")

    def test_path_string(self):
        """Test path string representation."""
        from nexus_matcher.domain.services.domain_hierarchy import Domain, DomainPath

        root = Domain(name="FINANCE")
        child = Domain(name="BANKING", parent=root)

        path = DomainPath.from_domain(child)
        assert str(path) == "FINANCE.BANKING"

    def test_path_length(self):
        """Test path length."""
        from nexus_matcher.domain.services.domain_hierarchy import Domain, DomainPath

        root = Domain(name="FINANCE")
        child = Domain(name="BANKING", parent=root)
        grandchild = Domain(name="ACCOUNTS", parent=child)

        assert len(DomainPath.from_domain(root)) == 1
        assert len(DomainPath.from_domain(grandchild)) == 3

    def test_path_contains(self):
        """Test path containment check."""
        from nexus_matcher.domain.services.domain_hierarchy import Domain, DomainPath

        root = Domain(name="FINANCE")
        child = Domain(name="BANKING", parent=root)

        path = DomainPath.from_domain(child)
        assert "FINANCE" in path
        assert "BANKING" in path
        assert "CUSTOMER" not in path


class TestDomainRelationship:
    """Test DomainRelationship enum."""

    def test_relationship_values(self):
        """Test relationship enum values."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainRelationship

        assert DomainRelationship.EXACT.value == "exact"
        assert DomainRelationship.PARENT.value == "parent"
        assert DomainRelationship.CHILD.value == "child"
        assert DomainRelationship.SIBLING.value == "sibling"
        assert DomainRelationship.COUSIN.value == "cousin"
        assert DomainRelationship.UNRELATED.value == "unrelated"


class TestDomainMatch:
    """Test DomainMatch result class."""

    def test_creates_match_result(self):
        """Test creating match result."""
        from nexus_matcher.domain.services.domain_hierarchy import (
            DomainMatch,
            DomainRelationship,
        )

        match = DomainMatch(
            domain1="ACCOUNTS",
            domain2="ACCOUNTS",
            relationship=DomainRelationship.EXACT,
            score=1.0,
        )

        assert match.domain1 == "ACCOUNTS"
        assert match.relationship == DomainRelationship.EXACT
        assert match.score == 1.0

    def test_is_related(self):
        """Test is_related property."""
        from nexus_matcher.domain.services.domain_hierarchy import (
            DomainMatch,
            DomainRelationship,
        )

        exact = DomainMatch("A", "A", DomainRelationship.EXACT, 1.0)
        unrelated = DomainMatch("A", "B", DomainRelationship.UNRELATED, 0.25)

        assert exact.is_related
        assert not unrelated.is_related


class TestDomainHierarchy:
    """Test DomainHierarchy entity."""

    def test_creates_empty_hierarchy(self):
        """Test creating empty hierarchy."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainHierarchy

        hierarchy = DomainHierarchy()
        assert hierarchy.size == 0
        assert hierarchy.is_empty

    def test_add_root_domain(self):
        """Test adding root domain."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainHierarchy

        hierarchy = DomainHierarchy()
        hierarchy.add_domain("FINANCE")

        assert hierarchy.size == 1
        assert hierarchy.get_domain("FINANCE") is not None
        assert hierarchy.get_domain("FINANCE").is_root

    def test_add_child_domain(self):
        """Test adding child domain."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainHierarchy

        hierarchy = DomainHierarchy()
        hierarchy.add_domain("FINANCE")
        hierarchy.add_domain("BANKING", parent="FINANCE")

        banking = hierarchy.get_domain("BANKING")
        assert banking is not None
        assert banking.parent.name == "FINANCE"

    def test_get_children(self):
        """Test getting child domains."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainHierarchy

        hierarchy = DomainHierarchy()
        hierarchy.add_domain("FINANCE")
        hierarchy.add_domain("BANKING", parent="FINANCE")
        hierarchy.add_domain("LENDING", parent="FINANCE")

        children = hierarchy.get_children("FINANCE")
        names = {c.name for c in children}

        assert "BANKING" in names
        assert "LENDING" in names

    def test_get_path(self):
        """Test getting domain path."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainHierarchy

        hierarchy = DomainHierarchy()
        hierarchy.add_domain("FINANCE")
        hierarchy.add_domain("BANKING", parent="FINANCE")
        hierarchy.add_domain("ACCOUNTS", parent="BANKING")

        path = hierarchy.get_path("ACCOUNTS")
        assert str(path) == "FINANCE.BANKING.ACCOUNTS"

    def test_case_insensitive_lookup(self):
        """Test lookup is case insensitive."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainHierarchy

        hierarchy = DomainHierarchy()
        hierarchy.add_domain("FINANCE")

        assert hierarchy.get_domain("finance") is not None
        assert hierarchy.get_domain("Finance") is not None

    def test_load_from_dict(self):
        """Test loading from dictionary structure."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainHierarchy

        data = {
            "FINANCE": {
                "BANKING": {
                    "ACCOUNTS": {},
                    "TRANSACTIONS": {},
                },
                "LENDING": {},
            },
            "CUSTOMER": {
                "IDENTITY": {},
            },
        }

        hierarchy = DomainHierarchy.from_dict(data)

        assert hierarchy.size == 7
        assert hierarchy.get_domain("ACCOUNTS").parent.name == "BANKING"


class TestDomainMatcher:
    """Test DomainMatcher service."""

    @pytest.fixture
    def matcher(self):
        """Create matcher with test hierarchy."""
        from nexus_matcher.domain.services.domain_hierarchy import (
            DomainHierarchy,
            DomainMatcher,
        )

        hierarchy = DomainHierarchy()
        # Build test hierarchy
        hierarchy.add_domain("FINANCE")
        hierarchy.add_domain("BANKING", parent="FINANCE")
        hierarchy.add_domain("ACCOUNTS", parent="BANKING")
        hierarchy.add_domain("TRANSACTIONS", parent="BANKING")
        hierarchy.add_domain("BALANCES", parent="BANKING")
        hierarchy.add_domain("LENDING", parent="FINANCE")
        hierarchy.add_domain("LOANS", parent="LENDING")

        hierarchy.add_domain("CUSTOMER")
        hierarchy.add_domain("IDENTITY", parent="CUSTOMER")

        return DomainMatcher(hierarchy)

    def test_exact_match(self, matcher):
        """Test exact domain match."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainRelationship

        result = matcher.match("ACCOUNTS", "ACCOUNTS")
        assert result.relationship == DomainRelationship.EXACT
        assert result.score == 1.0

    def test_parent_relationship(self, matcher):
        """Test parent relationship (domain1 is parent of domain2)."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainRelationship

        result = matcher.match("BANKING", "ACCOUNTS")
        assert result.relationship == DomainRelationship.PARENT
        assert result.score == pytest.approx(0.85)

    def test_child_relationship(self, matcher):
        """Test child relationship (domain1 is child of domain2)."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainRelationship

        result = matcher.match("ACCOUNTS", "BANKING")
        assert result.relationship == DomainRelationship.CHILD
        assert result.score == pytest.approx(0.80)

    def test_sibling_relationship(self, matcher):
        """Test sibling relationship (share immediate parent)."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainRelationship

        result = matcher.match("ACCOUNTS", "TRANSACTIONS")
        assert result.relationship == DomainRelationship.SIBLING
        assert result.score == pytest.approx(0.70)

    def test_cousin_relationship(self, matcher):
        """Test cousin relationship (share ancestor within 2 levels)."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainRelationship

        result = matcher.match("ACCOUNTS", "LOANS")
        assert result.relationship == DomainRelationship.COUSIN
        assert result.score == pytest.approx(0.50)

    def test_unrelated_domains(self, matcher):
        """Test unrelated domains."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainRelationship

        result = matcher.match("ACCOUNTS", "IDENTITY")
        assert result.relationship == DomainRelationship.UNRELATED
        assert result.score == pytest.approx(0.25)

    def test_score_method(self, matcher):
        """Test score convenience method."""
        assert matcher.score("ACCOUNTS", "ACCOUNTS") == 1.0
        assert matcher.score("ACCOUNTS", "BANKING") == pytest.approx(0.80)
        assert matcher.score("ACCOUNTS", "TRANSACTIONS") == pytest.approx(0.70)

    def test_unknown_domain(self, matcher):
        """Test with unknown domain."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainRelationship

        result = matcher.match("ACCOUNTS", "UNKNOWN")
        assert result.relationship == DomainRelationship.UNRELATED
        # Unknown domains get a neutral score
        assert result.score <= 0.5

    def test_case_insensitive_matching(self, matcher):
        """Test case insensitive domain matching."""
        result1 = matcher.match("accounts", "ACCOUNTS")
        result2 = matcher.match("ACCOUNTS", "accounts")

        assert result1.score == 1.0
        assert result2.score == 1.0

    def test_find_common_ancestor(self, matcher):
        """Test finding common ancestor."""
        ancestor = matcher.find_common_ancestor("ACCOUNTS", "LOANS")
        assert ancestor is not None
        assert ancestor.name == "FINANCE"

    def test_find_common_ancestor_siblings(self, matcher):
        """Test common ancestor for siblings."""
        ancestor = matcher.find_common_ancestor("ACCOUNTS", "TRANSACTIONS")
        assert ancestor is not None
        assert ancestor.name == "BANKING"


class TestDefaultMatcher:
    """Test default matcher with built-in hierarchy."""

    def test_default_has_common_domains(self):
        """Test default matcher includes common domains."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainMatcher

        matcher = DomainMatcher.default()

        # Should have finance domain
        assert matcher.score("ACCOUNTS", "ACCOUNTS") == 1.0

        # Should recognize common business domains
        assert matcher._hierarchy.get_domain("FINANCE") is not None
        assert matcher._hierarchy.get_domain("CUSTOMER") is not None

    def test_default_is_singleton(self):
        """Test default matcher returns same instance."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainMatcher

        m1 = DomainMatcher.default()
        m2 = DomainMatcher.default()
        assert m1 is m2


class TestDomainMatcherEdgeCases:
    """Test edge cases for domain matcher."""

    @pytest.fixture
    def matcher(self):
        """Create matcher with simple hierarchy."""
        from nexus_matcher.domain.services.domain_hierarchy import (
            DomainHierarchy,
            DomainMatcher,
        )

        hierarchy = DomainHierarchy()
        hierarchy.add_domain("ROOT")
        hierarchy.add_domain("CHILD", parent="ROOT")

        return DomainMatcher(hierarchy)

    def test_empty_domain_name(self, matcher):
        """Test with empty domain name."""
        from nexus_matcher.domain.services.domain_hierarchy import DomainRelationship

        result = matcher.match("", "ROOT")
        assert result.relationship == DomainRelationship.UNRELATED

    def test_none_domain_handling(self, matcher):
        """Test with None domain."""
        # Should handle gracefully
        result = matcher.match(None, "ROOT")
        assert result.score <= 0.5

    def test_special_characters_in_domain(self, matcher):
        """Test handling of special characters."""
        result = matcher.match("ROOT-TEST", "ROOT")
        # Should normalize and handle gracefully
        assert result is not None
