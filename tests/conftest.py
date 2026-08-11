"""
tests.conftest | Layer: TEST
Shared test fixtures and configuration.

## Relationships
# USED_BY → tests/* :: all test modules
"""

from pathlib import Path

import pytest

from nexus_matcher.domain.models import DictionaryEntry, Schema, SchemaField
from nexus_matcher.shared.types import DataType, ProtectionLevel
from nexus_matcher.shared.types.base import PerformanceMetrics, ScoreBreakdown

# =============================================================================
# PATH FIXTURES
# =============================================================================


@pytest.fixture
def test_data_dir() -> Path:
    """Get path to test data directory."""
    return Path(__file__).parent / "fixtures"


# =============================================================================
# MODEL FIXTURES
# =============================================================================


@pytest.fixture
def sample_schema_field() -> SchemaField:
    """Create a sample schema field."""
    return SchemaField(
        name="customer_email",
        data_type=DataType.STRING,
        full_path="customer.contact.email",
        parent_path="customer.contact",
        description="Customer email address for communications",
    )


@pytest.fixture
def sample_dictionary_entry() -> DictionaryEntry:
    """Create a sample dictionary entry."""
    return DictionaryEntry(
        id="field_001",
        business_name="Customer Email Address",
        logical_name="cust_email_addr",
        definition="Primary email address for customer contact and notifications",
        data_type=DataType.STRING,
        protection_level=ProtectionLevel.PII,
        domain="Sales",
        parent_table="Customer",
    )


@pytest.fixture
def sample_schema(sample_schema_field: SchemaField) -> Schema:
    """Create a sample schema."""
    fields = (
        sample_schema_field,
        SchemaField(
            name="customer_id",
            data_type=DataType.STRING,
            full_path="customer.id",
            parent_path="customer",
        ),
        SchemaField(
            name="amount",
            data_type=DataType.DOUBLE,
            full_path="transaction.amount",
            parent_path="transaction",
        ),
    )
    return Schema(
        name="CustomerTransaction",
        fields=fields,
        namespace="com.example.banking",
        source_format="avro",
    )


@pytest.fixture
def sample_dictionary_entries() -> list[DictionaryEntry]:
    """Create sample dictionary entries."""
    return [
        DictionaryEntry(
            id="field_001",
            business_name="Customer Email Address",
            logical_name="cust_email",
            definition="Email address for customer",
            data_type=DataType.STRING,
            protection_level=ProtectionLevel.PII,
        ),
        DictionaryEntry(
            id="field_002",
            business_name="Transaction Amount",
            logical_name="txn_amt",
            definition="Monetary value of the transaction",
            data_type=DataType.DOUBLE,
            protection_level=ProtectionLevel.INTERNAL,
        ),
        DictionaryEntry(
            id="field_003",
            business_name="Customer Identifier",
            logical_name="cust_id",
            definition="Unique identifier for customer",
            data_type=DataType.STRING,
            protection_level=ProtectionLevel.INTERNAL,
        ),
    ]


# =============================================================================
# SCORE FIXTURES
# =============================================================================


@pytest.fixture
def sample_score_breakdown() -> ScoreBreakdown:
    """Create a sample score breakdown."""
    return ScoreBreakdown(
        semantic_score=0.92,
        lexical_score=0.85,
        edit_distance_score=0.78,
        type_compatibility_score=1.0,
        domain_score=0.5,
    )


@pytest.fixture
def sample_performance_metrics() -> PerformanceMetrics:
    """Create sample performance metrics."""
    return PerformanceMetrics(
        latency_ms=45.2,
        cache_hit=False,
        retrieval_stage="cross_encoder",
        candidates_evaluated=50,
        reranking_applied=True,
    )
