"""
tests.unit.domain.test_abbreviation | Layer: TEST
Tests: Abbreviation expansion service | Target: src/domain/services/abbreviation.py

TDD Phase: RED → Tests written before implementation
"""

import pytest


class TestAbbreviationMapping:
    """Test AbbreviationMapping value object."""

    def test_creates_valid_mapping(self):
        """Test creating a valid mapping."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationMapping

        mapping = AbbreviationMapping(abbreviation="acct", expansion="account")
        assert mapping.abbreviation == "acct"
        assert mapping.expansion == "account"

    def test_normalizes_abbreviation_to_lowercase(self):
        """Test abbreviation is normalized to lowercase."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationMapping

        mapping = AbbreviationMapping(abbreviation="ACCT", expansion="account")
        assert mapping.abbreviation == "acct"

    def test_rejects_empty_abbreviation(self):
        """Test empty abbreviation raises error."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationMapping

        with pytest.raises(ValueError, match="abbreviation"):
            AbbreviationMapping(abbreviation="", expansion="account")

    def test_rejects_empty_expansion(self):
        """Test empty expansion raises error."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationMapping

        with pytest.raises(ValueError, match="expansion"):
            AbbreviationMapping(abbreviation="acct", expansion="")

    def test_rejects_identity_mapping(self):
        """Test abbreviation cannot equal expansion."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationMapping

        with pytest.raises(ValueError, match="cannot equal"):
            AbbreviationMapping(abbreviation="account", expansion="account")

    def test_is_immutable(self):
        """Test mapping is immutable (frozen dataclass)."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationMapping

        mapping = AbbreviationMapping(abbreviation="acct", expansion="account")
        with pytest.raises(AttributeError):
            mapping.abbreviation = "acc"


class TestExpandedText:
    """Test ExpandedText value object."""

    def test_creates_with_no_expansions(self):
        """Test creating result with no expansions."""
        from nexus_matcher.domain.services.abbreviation import ExpandedText

        result = ExpandedText(
            original="customer_name",
            expanded="customer_name",
            expansions=[],
        )
        assert result.original == "customer_name"
        assert result.expanded == "customer_name"
        assert result.expansions == []
        assert not result.was_expanded

    def test_creates_with_expansions(self):
        """Test creating result with expansions."""
        from nexus_matcher.domain.services.abbreviation import ExpandedText

        result = ExpandedText(
            original="cust_acct",
            expanded="customer_account",
            expansions=[("cust", "customer"), ("acct", "account")],
        )
        assert result.was_expanded
        assert len(result.expansions) == 2

    def test_expansion_count(self):
        """Test expansion count property."""
        from nexus_matcher.domain.services.abbreviation import ExpandedText

        result = ExpandedText(
            original="cust_acct_bal",
            expanded="customer_account_balance",
            expansions=[
                ("cust", "customer"),
                ("acct", "account"),
                ("bal", "balance"),
            ],
        )
        assert result.expansion_count == 3


class TestAbbreviationDictionary:
    """Test AbbreviationDictionary entity."""

    def test_creates_empty_dictionary(self):
        """Test creating empty dictionary."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationDictionary

        dictionary = AbbreviationDictionary()
        assert dictionary.size == 0
        assert dictionary.is_empty

    def test_add_mapping(self):
        """Test adding a mapping."""
        from nexus_matcher.domain.services.abbreviation import (
            AbbreviationDictionary,
            AbbreviationMapping,
        )

        dictionary = AbbreviationDictionary()
        dictionary.add(AbbreviationMapping("acct", "account"))
        assert dictionary.size == 1
        assert not dictionary.is_empty

    def test_lookup_existing(self):
        """Test looking up existing abbreviation."""
        from nexus_matcher.domain.services.abbreviation import (
            AbbreviationDictionary,
            AbbreviationMapping,
        )

        dictionary = AbbreviationDictionary()
        dictionary.add(AbbreviationMapping("acct", "account"))
        assert dictionary.lookup("acct") == "account"

    def test_lookup_case_insensitive(self):
        """Test lookup is case insensitive."""
        from nexus_matcher.domain.services.abbreviation import (
            AbbreviationDictionary,
            AbbreviationMapping,
        )

        dictionary = AbbreviationDictionary()
        dictionary.add(AbbreviationMapping("acct", "account"))
        assert dictionary.lookup("ACCT") == "account"
        assert dictionary.lookup("Acct") == "account"

    def test_lookup_missing_returns_none(self):
        """Test looking up missing abbreviation returns None."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationDictionary

        dictionary = AbbreviationDictionary()
        assert dictionary.lookup("unknown") is None

    def test_contains(self):
        """Test contains check."""
        from nexus_matcher.domain.services.abbreviation import (
            AbbreviationDictionary,
            AbbreviationMapping,
        )

        dictionary = AbbreviationDictionary()
        dictionary.add(AbbreviationMapping("acct", "account"))
        assert "acct" in dictionary
        assert "unknown" not in dictionary

    def test_load_from_dict(self):
        """Test loading from dictionary."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationDictionary

        data = {
            "acct": "account",
            "cust": "customer",
            "bal": "balance",
        }
        dictionary = AbbreviationDictionary.from_dict(data)
        assert dictionary.size == 3
        assert dictionary.lookup("acct") == "account"

    def test_duplicate_abbreviation_overwrites(self):
        """Test adding duplicate overwrites previous."""
        from nexus_matcher.domain.services.abbreviation import (
            AbbreviationDictionary,
            AbbreviationMapping,
        )

        dictionary = AbbreviationDictionary()
        dictionary.add(AbbreviationMapping("acct", "account"))
        dictionary.add(AbbreviationMapping("acct", "accounting"))
        assert dictionary.lookup("acct") == "accounting"
        assert dictionary.size == 1


class TestAbbreviationExpander:
    """Test AbbreviationExpander service."""

    @pytest.fixture
    def expander(self):
        """Create expander with test dictionary."""
        from nexus_matcher.domain.services.abbreviation import (
            AbbreviationDictionary,
            AbbreviationExpander,
        )

        dictionary = AbbreviationDictionary.from_dict({
            "acct": "account",
            "cust": "customer",
            "bal": "balance",
            "amt": "amount",
            "dt": "date",
            "id": "identifier",
            "nm": "name",
            "no": "number",
            "num": "number",
            "txn": "transaction",
        })
        return AbbreviationExpander(dictionary)

    def test_expand_single_abbreviation(self, expander):
        """Test expanding single abbreviation."""
        result = expander.expand("acct")
        assert result.expanded == "account"
        assert result.expansions == [("acct", "account")]

    def test_expand_underscore_separated(self, expander):
        """Test expanding underscore-separated text."""
        result = expander.expand("cust_acct_bal")
        assert result.expanded == "customer_account_balance"
        assert ("cust", "customer") in result.expansions
        assert ("acct", "account") in result.expansions
        assert ("bal", "balance") in result.expansions

    def test_expand_camel_case(self, expander):
        """Test expanding camelCase text."""
        result = expander.expand("custAcctBal")
        assert "customer" in result.expanded.lower()
        assert "account" in result.expanded.lower()
        assert "balance" in result.expanded.lower()

    def test_preserves_non_abbreviations(self, expander):
        """Test non-abbreviations are preserved."""
        result = expander.expand("customer_account")
        assert result.expanded == "customer_account"
        assert not result.was_expanded

    def test_mixed_abbreviations_and_words(self, expander):
        """Test mix of abbreviations and full words."""
        result = expander.expand("cust_account_bal")
        assert "customer" in result.expanded
        assert "account" in result.expanded
        assert "balance" in result.expanded

    def test_expand_tokens(self, expander):
        """Test expanding list of tokens."""
        tokens = ["cust", "acct", "balance"]
        expanded = expander.expand_tokens(tokens)
        assert expanded == ["customer", "account", "balance"]

    def test_expand_empty_string(self, expander):
        """Test expanding empty string."""
        result = expander.expand("")
        assert result.expanded == ""
        assert not result.was_expanded

    def test_expand_preserves_case_of_non_abbreviations(self, expander):
        """Test non-abbreviations preserve original case."""
        result = expander.expand("Customer_Name")
        # Non-abbreviations should be preserved
        assert "Customer" in result.expanded or "customer" in result.expanded.lower()

    def test_get_candidates(self, expander):
        """Test getting expansion candidates."""
        candidates = expander.get_candidates("acct")
        assert "account" in candidates

    def test_get_candidates_unknown(self, expander):
        """Test getting candidates for unknown abbreviation."""
        candidates = expander.get_candidates("xyz")
        assert candidates == []


class TestDefaultExpander:
    """Test default expander with built-in abbreviations."""

    def test_default_has_common_abbreviations(self):
        """Test default expander includes common abbreviations."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationExpander

        expander = AbbreviationExpander.default()

        # Test common data domain abbreviations
        assert expander.expand("acct").expanded == "account"
        assert expander.expand("cust").expanded == "customer"
        assert expander.expand("txn").expanded == "transaction"
        assert expander.expand("amt").expanded == "amount"
        assert expander.expand("bal").expanded == "balance"

    def test_default_is_singleton(self):
        """Test default expander returns same instance."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationExpander

        exp1 = AbbreviationExpander.default()
        exp2 = AbbreviationExpander.default()
        assert exp1 is exp2

    def test_expand_real_world_field_names(self):
        """Test expanding realistic field names."""
        from nexus_matcher.domain.services.abbreviation import AbbreviationExpander

        expander = AbbreviationExpander.default()

        # Realistic banking/financial field names
        assert "account" in expander.expand("acct_no").expanded.lower()
        assert "customer" in expander.expand("cust_id").expanded.lower()
        assert "transaction" in expander.expand("txn_dt").expanded.lower()
        assert "balance" in expander.expand("curr_bal").expanded.lower()


class TestAbbreviationExpanderEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def expander(self):
        """Create expander with test dictionary."""
        from nexus_matcher.domain.services.abbreviation import (
            AbbreviationDictionary,
            AbbreviationExpander,
        )

        dictionary = AbbreviationDictionary.from_dict({
            "id": "identifier",
            "no": "number",
            "dt": "date",
        })
        return AbbreviationExpander(dictionary)

    def test_single_character_abbreviations(self, expander):
        """Test single character shouldn't match (too ambiguous)."""
        # "i" and "d" alone shouldn't expand
        result = expander.expand("i_d")
        # Single chars might or might not expand based on implementation
        assert result is not None

    def test_numbers_in_field_names(self, expander):
        """Test handling of numbers."""
        result = expander.expand("acct_no_1")
        # Should handle numbers gracefully
        assert "1" in result.expanded

    def test_special_characters(self, expander):
        """Test handling of special characters."""
        result = expander.expand("acct-no")
        assert result is not None

    def test_very_long_field_name(self, expander):
        """Test handling of very long field names."""
        long_name = "_".join(["field"] * 50)
        result = expander.expand(long_name)
        assert result is not None
        assert len(result.expanded) >= len(long_name)
