# Context Enricher Module

> Gap: GAP-006 - Enhanced Context Injection  
> Status: VALIDATED ✓  
> Research Reference: README_RESEARCH_1.md, Lines 174-176

## Overview

The Context Enricher provides hierarchical context injection for nested schema fields. Research from ReMatch (2024) and Starmie (2023) consistently shows 10-20% accuracy degradation when schemas lose hierarchical context.

**Research Quote:**
> "Context injection is non-negotiable for nested schemas... For user.addresses[].street_name, the query must include 'user entity, addresses array, street_name field' structure."

## Components

### ContextEnricher

Main service class for enriching SchemaField with hierarchical context.

```python
from nexus_matcher.domain.services.context_enricher import ContextEnricher

enricher = ContextEnricher()

# For user.addresses[].street_name:
field = SchemaField(
    name="street_name",
    data_type=DataType.STRING,
    full_path="user.addresses.street_name",
    parent_path="user.addresses",
)

# Returns: "user, addresses street name text field"
enriched = enricher.enrich(field)
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| include_type | True | Include data type context |
| include_hierarchy | True | Include parent hierarchy |
| include_description | True | Include field description |
| max_depth | 5 | Maximum hierarchy depth |
| humanize_names | True | Convert snake_case/camelCase |

### EnrichmentConfig

Configuration dataclass for customizing enrichment behavior.

## Enrichment Examples

| Field Path | Basic Text | Enriched Text |
|------------|------------|---------------|
| email_address | email address | email address text field |
| user.addresses.street_name | street name | user, addresses street name text field |
| order.line_items.price.amount | amount | order, line items, price amount decimal number field |
| product.tags (array) | tags | product tags array of text |

## Integration

The ContextEnricher is integrated into `NexusMatcher._match_field()` and is used automatically for all field matching operations.

```python
# In NexusMatcher._match_field():
enriched_query = self._context_enricher.enrich(field)
expanded = self._abbreviation_expander.expand(enriched_query)
query_text = expanded.expanded
```

## Benchmark Results (SUITE-004c)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Depth 3+ Coverage | 100% | ≥80% | ✓ |
| Hierarchy Tokens | 1.78 avg | ≥1.5 | ✓ |
| Humanization Rate | 100% | ≥95% | ✓ |
| Throughput | 103,721 fields/s | ≥50K | ✓ |

**Latency (per field):**
- P50: 8.47 µs
- P95: 16.90 µs
- P99: 17.39 µs

## Files

| File | Purpose |
|------|---------|
| `src/nexus_matcher/domain/services/context_enricher.py` | Implementation |
| `tests/unit/domain/test_context_enricher.py` | Unit tests (19) |
| `benchmarks/suite_004c_context_enrichment.py` | Benchmark suite |

## Dependencies

- None (uses only stdlib and existing domain models)

## Design Decisions

1. **OrderedDict-based path parsing** - Simple, efficient hierarchy extraction
2. **Humanization by default** - Converts identifiers to embedding-friendly text
3. **Configurable depth limit** - Prevents excessive context for deeply nested schemas
4. **Type context inclusion** - Adds semantic type information ("text field", "decimal number")
5. **Namespace filtering** - Ignores common namespace prefixes (com, org, etc.)
