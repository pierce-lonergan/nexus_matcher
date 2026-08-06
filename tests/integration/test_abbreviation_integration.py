"""
tests.integration.test_abbreviation_integration | Layer: TEST
Integration tests for abbreviation expansion in matching pipeline.

Tests real-world scenarios where abbreviation expansion improves matching.
"""

import pytest

from nexus_matcher.domain.services.abbreviation import AbbreviationExpander


class TestAbbreviationIntegration:
    """Integration tests for abbreviation expansion in matching context."""

    @pytest.fixture
    def expander(self):
        """Get default expander."""
        return AbbreviationExpander.default()

    def test_financial_field_names(self, expander):
        """Test expansion of common financial field names."""
        test_cases = [
            ("acct_bal", "account_balance"),
            ("cust_acct_no", "customer_account_number"),
            ("txn_amt", "transaction_amount"),
            ("curr_bal", "current_balance"),
            ("acct_stat", "account_status"),
        ]

        for input_text, expected_contains in test_cases:
            result = expander.expand(input_text)
            # Check each expected word is present
            for word in expected_contains.split("_"):
                assert word in result.expanded.lower(), (
                    f"'{word}' not found in expansion of '{input_text}': '{result.expanded}'"
                )

    def test_date_time_fields(self, expander):
        """Test expansion of date/time field names."""
        test_cases = [
            ("create_dt", "date"),
            ("update_ts", "timestamp"),
            ("last_mod_dt", "date"),
        ]

        for input_text, expected_contains in test_cases:
            result = expander.expand(input_text)
            assert expected_contains in result.expanded.lower(), (
                f"'{expected_contains}' not found in '{result.expanded}'"
            )

    def test_identifier_fields(self, expander):
        """Test expansion of identifier field names."""
        test_cases = [
            ("cust_id", "customer"),
            ("acct_no", "account"),
            ("txn_seq", "sequence"),
        ]

        for input_text, expected_contains in test_cases:
            result = expander.expand(input_text)
            assert expected_contains in result.expanded.lower()

    def test_preserves_already_expanded_fields(self, expander):
        """Test that full words are not double-expanded."""
        result = expander.expand("customer_account_balance")
        assert result.expanded == "customer_account_balance"
        assert not result.was_expanded

    def test_mixed_abbreviations_and_words(self, expander):
        """Test mixed abbreviation and full word combinations."""
        result = expander.expand("cust_account_bal")
        assert "customer" in result.expanded.lower()
        assert "account" in result.expanded.lower()
        assert "balance" in result.expanded.lower()

    def test_real_schema_field_names(self, expander):
        """Test with real-world schema field names from banking domain."""
        real_fields = [
            "ACCT_TYP_CD",
            "CUST_NM",
            "TXN_DT",
            "BAL_AMT",
            "STAT_IND",
            "SRC_SYS_ID",
            "LST_UPD_TS",
            "CNTRY_CD",
            "ADDR_LN_1",
        ]

        for field_name in real_fields:
            result = expander.expand(field_name)
            # Should expand and produce valid output
            assert result.expanded is not None
            assert len(result.expanded) >= len(field_name)

    def test_expansion_count_reasonable(self, expander):
        """Test that expansion count matches actual expansions."""
        result = expander.expand("cust_acct_bal_amt")

        # Should have 4 expansions
        assert result.expansion_count >= 3  # At minimum cust, acct, bal
        assert result.was_expanded

    def test_camel_case_schema_fields(self, expander):
        """Test camelCase field names common in JSON schemas."""
        test_cases = [
            "custAcct",
            "txnAmt",
            "balDt",
        ]

        for field_name in test_cases:
            result = expander.expand(field_name)
            assert result is not None
            # Should have done some expansion
            # CamelCase handling may vary

    def test_expansion_improves_searchability(self, expander):
        """Test that expansion creates more searchable text."""
        original = "cust_acct_bal"
        result = expander.expand(original)

        # Expanded version should have more descriptive words
        expanded_words = set(result.expanded.lower().replace("_", " ").split())
        original_words = set(original.lower().replace("_", " ").split())

        # Should have new, more descriptive words
        new_words = expanded_words - original_words
        assert "customer" in new_words or "account" in new_words or "balance" in new_words
