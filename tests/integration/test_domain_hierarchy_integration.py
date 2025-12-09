"""
tests.integration.test_domain_hierarchy_integration | Layer: TEST
Integration tests for domain hierarchy in matching pipeline.

Tests real-world domain matching scenarios.
"""

import pytest

from nexus_matcher.domain.services.domain_hierarchy import (
    DomainHierarchy,
    DomainMatcher,
    DomainRelationship,
)


class TestDomainHierarchyIntegration:
    """Integration tests for domain hierarchy in matching context."""

    @pytest.fixture
    def matcher(self):
        """Get default matcher."""
        return DomainMatcher.default()

    def test_banking_domain_relationships(self, matcher):
        """Test relationships within banking domain."""
        # ACCOUNTS and TRANSACTIONS are siblings under BANKING
        result = matcher.match("ACCOUNTS", "TRANSACTIONS")
        assert result.relationship == DomainRelationship.SIBLING
        assert result.score >= 0.7

        # BANKING is parent of ACCOUNTS
        result = matcher.match("BANKING", "ACCOUNTS")
        assert result.relationship == DomainRelationship.PARENT
        assert result.score >= 0.8

    def test_customer_domain_relationships(self, matcher):
        """Test relationships within customer domain."""
        # IDENTITY and CONTACT are siblings under CUSTOMER
        result = matcher.match("IDENTITY", "CONTACT")
        assert result.relationship == DomainRelationship.SIBLING

        # ADDRESS is child of CONTACT
        result = matcher.match("CONTACT", "ADDRESS")
        assert result.relationship == DomainRelationship.PARENT

    def test_cross_domain_relationships(self, matcher):
        """Test relationships across major domains."""
        # ACCOUNTS (Finance) vs IDENTITY (Customer) - unrelated
        result = matcher.match("ACCOUNTS", "IDENTITY")
        assert result.relationship == DomainRelationship.UNRELATED
        assert result.score <= 0.3

    def test_product_domain_hierarchy(self, matcher):
        """Test product domain relationships."""
        # DEPOSIT products
        result = matcher.match("CHECKING", "SAVINGS")
        assert result.relationship == DomainRelationship.SIBLING

        # DEPOSIT is parent of CHECKING
        result = matcher.match("DEPOSIT", "CHECKING")
        assert result.relationship == DomainRelationship.PARENT

    def test_compliance_domains(self, matcher):
        """Test compliance domain relationships."""
        # KYC and AML are related through COMPLIANCE
        result = matcher.match("KYC", "AML")
        assert result.relationship in [
            DomainRelationship.SIBLING,
            DomainRelationship.COUSIN,
        ]
        assert result.score >= 0.5

    def test_common_ancestor_finding(self, matcher):
        """Test finding common ancestors."""
        # ACCOUNTS and LOANS share FINANCE ancestor
        ancestor = matcher.find_common_ancestor("ACCOUNTS", "LOANS")
        assert ancestor is not None
        assert ancestor.name == "FINANCE"

        # CHECKING and SAVINGS share DEPOSIT ancestor
        ancestor = matcher.find_common_ancestor("CHECKING", "SAVINGS")
        assert ancestor is not None
        assert ancestor.name == "DEPOSIT"

    def test_real_world_field_domain_matching(self, matcher):
        """Test realistic field-to-entry domain matching."""
        # A field from accounts context matching account entry
        assert matcher.score("ACCOUNTS", "ACCOUNTS") == 1.0
        assert matcher.score("ACCOUNTS", "BALANCES") >= 0.7  # Siblings

        # A transaction field vs account entry (siblings)
        assert matcher.score("TRANSACTIONS", "ACCOUNTS") >= 0.7

    def test_hierarchical_scoring_gradient(self, matcher):
        """Test that scores decrease with hierarchy distance."""
        # Exact > Parent > Child > Sibling > Cousin > Unrelated
        exact = matcher.score("ACCOUNTS", "ACCOUNTS")
        parent = matcher.score("BANKING", "ACCOUNTS")
        child = matcher.score("ACCOUNTS", "BANKING")
        sibling = matcher.score("ACCOUNTS", "TRANSACTIONS")

        assert exact > parent
        assert parent >= child
        assert child >= sibling

    def test_case_insensitive_domain_matching(self, matcher):
        """Test case insensitivity in domain matching."""
        assert matcher.score("accounts", "ACCOUNTS") == 1.0
        assert matcher.score("ACCOUNTS", "accounts") == 1.0
        assert matcher.score("Accounts", "ACCOUNTS") == 1.0

    def test_unknown_domain_handling(self, matcher):
        """Test handling of unknown domains."""
        result = matcher.match("UNKNOWN_DOMAIN", "ACCOUNTS")
        assert result.relationship == DomainRelationship.UNRELATED
        assert result.score == 0.5  # Neutral for unknown

    def test_empty_domain_handling(self, matcher):
        """Test handling of empty domain strings."""
        result = matcher.match("", "ACCOUNTS")
        assert result.relationship == DomainRelationship.UNRELATED
        assert result.score <= 0.5


class TestCustomHierarchy:
    """Test custom hierarchy creation."""

    def test_create_jpmc_specific_hierarchy(self):
        """Test creating JPMC-specific domain hierarchy."""
        hierarchy = DomainHierarchy()

        # Build CCB-specific hierarchy
        hierarchy.add_domain("CCB")
        hierarchy.add_domain("CONSUMER_BANKING", parent="CCB")
        hierarchy.add_domain("CHASE_CARD", parent="CCB")
        hierarchy.add_domain("HOME_LENDING", parent="CCB")

        hierarchy.add_domain("CHECKING", parent="CONSUMER_BANKING")
        hierarchy.add_domain("SAVINGS", parent="CONSUMER_BANKING")

        matcher = DomainMatcher(hierarchy)

        # Test relationships
        assert matcher.score("CHECKING", "SAVINGS") >= 0.7  # Siblings
        assert matcher.score("CONSUMER_BANKING", "CHECKING") >= 0.8  # Parent
        assert matcher.score("CHECKING", "CHASE_CARD") >= 0.5  # Cousins

    def test_deep_hierarchy(self):
        """Test deeply nested hierarchy."""
        hierarchy = DomainHierarchy()

        # Create deep hierarchy
        hierarchy.add_domain("L1")
        hierarchy.add_domain("L2", parent="L1")
        hierarchy.add_domain("L3", parent="L2")
        hierarchy.add_domain("L4", parent="L3")
        hierarchy.add_domain("L5", parent="L4")

        matcher = DomainMatcher(hierarchy)

        # Deep child should still work
        path = hierarchy.get_path("L5")
        assert len(path) == 5
        assert str(path) == "L1.L2.L3.L4.L5"

        # Ancestor finding across deep hierarchy
        ancestor = matcher.find_common_ancestor("L5", "L2")
        assert ancestor.name == "L2"
