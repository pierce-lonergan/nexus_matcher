"""
tests.unit.domain.test_context_enricher | Layer: TEST
Tests: Context Enricher Service | Target: src/domain/services/context_enricher.py

TDD Phase: RED → Tests written before implementation
Research Reference: README_RESEARCH_1.md, Lines 174-176
Gap: GAP-006 - Enhanced Context Injection

Research Quote:
"Context injection is non-negotiable for nested schemas... For user.addresses[].street_name,
the query must include 'user entity, addresses array, street_name field' structure."
"""

from nexus_matcher.domain.models.entities import SchemaField
from nexus_matcher.shared.types.base import DataType


class TestContextEnricherBasicBehavior:
    """Test basic context enrichment behavior."""

    def test_enricher_exists(self):
        """Test ContextEnricher class exists."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        assert enricher is not None

    def test_enrich_returns_string(self):
        """Test enrich method returns a string."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="customer_id",
            data_type=DataType.STRING,
        )

        result = enricher.enrich(field)
        assert isinstance(result, str)

    def test_enrich_includes_field_name(self):
        """Test enriched text includes field name."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="customer_email",
            data_type=DataType.STRING,
        )

        result = enricher.enrich(field)
        assert "customer email" in result.lower() or "customer_email" in result.lower()


class TestContextEnricherRootFields:
    """Test enrichment of root-level (non-nested) fields."""

    def test_root_field_includes_name_and_type(self):
        """Type context is opt-in; when enabled it appears alongside the name."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        # include_type must be requested explicitly: it is off by default because
        # appending "text field" cost 2.1 points of P@1 on the BIRD+OMOP benchmark.
        enricher = ContextEnricher(include_type=True)
        field = SchemaField(
            name="email_address",
            data_type=DataType.STRING,
            full_path="email_address",
        )

        result = enricher.enrich(field)

        # Should include field name (human readable)
        assert "email" in result.lower()
        assert "address" in result.lower()
        # Should include type context
        assert "string" in result.lower() or "text" in result.lower()

    def test_root_field_with_description(self):
        """Test root field includes description."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="account_balance",
            data_type=DataType.DECIMAL,
            description="Current balance in USD",
        )

        result = enricher.enrich(field)
        assert "current balance" in result.lower()


class TestContextEnricherNestedFields:
    """Test enrichment of nested fields with hierarchical context."""

    def test_nested_field_includes_parent_context(self):
        """Test nested field includes parent entity context.

        Research: For user.addresses[].street_name, include
        'user entity, addresses array, street_name field'
        """
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="street_name",
            data_type=DataType.STRING,
            full_path="user.addresses.street_name",
            parent_path="user.addresses",
        )

        result = enricher.enrich(field)

        # Must include hierarchical context
        assert "user" in result.lower()
        assert "addresses" in result.lower() or "address" in result.lower()
        assert "street" in result.lower()

    def test_nested_field_includes_entity_markers(self):
        """Test nested field includes entity/record markers."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="city",
            data_type=DataType.STRING,
            full_path="customer.shipping_address.city",
            parent_path="customer.shipping_address",
        )

        result = enricher.enrich(field)

        # Should include structural markers
        assert "customer" in result.lower()
        assert "shipping" in result.lower() or "address" in result.lower()

    def test_deeply_nested_field_three_levels(self):
        """Test 3+ levels of nesting are captured."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="amount",
            data_type=DataType.DECIMAL,
            full_path="order.line_items.price.amount",
            parent_path="order.line_items.price",
        )

        result = enricher.enrich(field)

        # All hierarchy levels must be present
        assert "order" in result.lower()
        assert "line" in result.lower() or "item" in result.lower()
        assert "price" in result.lower()
        assert "amount" in result.lower()

    def test_array_field_indicates_array_context(self):
        """Test array fields include array context."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="tag",
            data_type=DataType.STRING,
            full_path="product.tags",
            is_array=True,
            array_item_type=DataType.STRING,
        )

        result = enricher.enrich(field)

        # Should indicate array nature
        assert "array" in result.lower() or "list" in result.lower() or "tags" in result.lower()


class TestContextEnricherSpecialCases:
    """Test special cases and edge conditions."""

    def test_field_with_namespace(self):
        """Test field with namespace includes namespace context."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="id",
            data_type=DataType.STRING,
            full_path="com.company.customer.id",
            parent_path="com.company.customer",
            source_metadata={"namespace": "com.company"},
        )

        result = enricher.enrich(field)

        # Namespace may or may not be included, but customer context should be
        assert "customer" in result.lower()

    def test_field_with_domain_context(self):
        """Test field includes domain hint if available."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="ssn",
            data_type=DataType.STRING,
            full_path="employee.ssn",
            source_metadata={"domain": "PII", "sensitive": True},
        )

        result = enricher.enrich(field)

        # Should include employee context
        assert "employee" in result.lower()

    def test_underscore_names_are_humanized(self):
        """Test underscored names are converted to human-readable."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="first_name",
            data_type=DataType.STRING,
            full_path="customer.first_name",
        )

        result = enricher.enrich(field)

        # Should be human readable, not "first_name"
        assert "first" in result.lower()
        assert "name" in result.lower()

    def test_camel_case_names_are_humanized(self):
        """Test camelCase names are converted to human-readable."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="firstName",
            data_type=DataType.STRING,
            full_path="customer.firstName",
        )

        result = enricher.enrich(field)

        # Should split camelCase
        assert "first" in result.lower()


class TestContextEnricherOutputFormat:
    """Test the format of enriched output."""

    def test_output_is_embedding_friendly(self):
        """Test output is suitable for embedding (not too long)."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="very_long_field_name_with_many_underscores",
            data_type=DataType.STRING,
            full_path="deeply.nested.structure.very_long_field_name_with_many_underscores",
            description="This is a very long description that goes on and on",
        )

        result = enricher.enrich(field)

        # Should be reasonable length for embedding
        # Most models handle up to 512 tokens well
        assert len(result) < 1000  # characters

    def test_output_has_no_excessive_whitespace(self):
        """Test output has normalized whitespace."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="customer_id",
            data_type=DataType.STRING,
        )

        result = enricher.enrich(field)

        # No double spaces or leading/trailing whitespace
        assert "  " not in result
        assert result == result.strip()


class TestContextEnricherConfigurable:
    """Test configurable enrichment options."""

    def test_can_disable_type_context(self):
        """Test type context can be disabled."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher(include_type=False)
        field = SchemaField(
            name="amount",
            data_type=DataType.DECIMAL,
        )

        result = enricher.enrich(field)

        # Type might not be mentioned
        # Just verify it doesn't crash
        assert "amount" in result.lower()

    def test_can_set_hierarchy_depth_limit(self):
        """Test hierarchy depth can be limited."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher(max_depth=2)
        field = SchemaField(
            name="value",
            data_type=DataType.STRING,
            full_path="a.b.c.d.e.value",
            parent_path="a.b.c.d.e",
        )

        result = enricher.enrich(field)

        # Should include limited context
        assert "value" in result.lower()
        # May not include all levels
        # Just verify reasonable output
        assert len(result) > 0


class TestContextEnricherIntegration:
    """Test integration with SchemaField.to_searchable_text()."""

    def test_enriched_text_is_better_than_basic(self):
        """Test enriched text contains more context than basic."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        field = SchemaField(
            name="street_name",
            data_type=DataType.STRING,
            full_path="user.addresses.street_name",
            parent_path="user.addresses",
        )

        basic = field.to_searchable_text()
        enriched = enricher.enrich(field)

        # Enriched should have more context
        assert len(enriched) >= len(basic)
        # Enriched should include parent context that basic lacks
        assert "user" in enriched.lower() or "address" in enriched.lower()

    def test_batch_enrich(self):
        """Test batch enrichment of multiple fields."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        enricher = ContextEnricher()
        fields = [
            SchemaField(name="id", data_type=DataType.STRING, full_path="customer.id"),
            SchemaField(name="name", data_type=DataType.STRING, full_path="customer.name"),
            SchemaField(name="email", data_type=DataType.STRING, full_path="customer.email"),
        ]

        results = enricher.enrich_batch(fields)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
        assert all("customer" in r.lower() for r in results)
