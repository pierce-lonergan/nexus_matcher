# Context Enricher Module

> Measured impact: **+20.1 points of P@1** (0.491 -> 0.691) on the 793-pair combined
> benchmark. This is the largest single accuracy factor in the pipeline.
> Artifact: `benchmarks/results/exp_query_repr_combined.json`.

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

# Returns: "user, addresses street name"
enriched = enricher.enrich(field)
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_type` | **False** | Append a scalar type descriptor ("text field"). **Off because it measurably hurts:** enabling it cost 2.1 points of P@1 (0.493 -> 0.469). Nearly every entry is compatible with a generic type phrase, so the tokens dilute the part of the embedding that discriminates. Type compatibility is still used, as a separate numeric scoring signal. |
| `include_hierarchy` | True | Include parent hierarchy — this is where the +20.1 points come from |
| `include_description` | True | Include field description |
| `include_array_indicator` | True | Append "array" for collection fields. Deliberately separate from `include_type`: whether a field is a collection is structural information. The benchmark that showed type text to be harmful is entirely flat relational schemas with no arrays, so it says nothing either way about array markers. |
| `max_depth` | 5 | Maximum hierarchy depth |
| `humanize_names` | True | Convert snake_case/camelCase |

### EnrichmentConfig

Configuration dataclass for customizing enrichment behavior.

## Enrichment Examples

Actual outputs at default settings:

| Field path | Enriched text |
|------------|---------------|
| `email_address` | `email address` |
| `user.addresses.street_name` | `user, addresses street name` |
| `order.line_items.price.amount` | `order, line items, price amount` |
| `product.tags` (array) | `product tags array` |
| `satscores.sname` | `satscores sname` |

The last row is the whole point: `sname` alone is ambiguous against a 793-entry glossary;
`satscores sname` is not.

With `include_type=True`, `user.addresses.street_name` becomes
`user, addresses street name text field` — and P@1 drops.

## Integration

The ContextEnricher is integrated into `NexusMatcher._match_field()` and is used automatically for all field matching operations.

```python
# In NexusMatcher._build_query_text():
enriched_query = self._context_enricher.enrich(field)
return self._abbreviation_expander.expand(enriched_query).expanded
```

Ordering caveat: the abbreviation expander runs on the *already enriched* string. It
previously collapsed that whole string into a single camelCase token, which drove BM25
P@1 to 0.005 with 787 of 793 queries returning zero hits. See
[ENHANCEMENT_JOURNEY.md](../ENHANCEMENT_JOURNEY.md#3-two-defects-the-old-benchmark-could-not-see).

## Benchmark Results

**Accuracy** (`benchmarks/results/exp_query_repr_combined.json`, 793 labelled pairs) —
this is the number that matters:

| Query representation | P@1 |
|---|---|
| bare field name | 0.488 |
| + underscore / camelCase splitting | 0.493 |
| **+ parent-path context** | **0.691** |
| + scalar type words | 0.469 |

**Throughput** (`benchmarks/results/suite_004c_context_enrichment_20251209_131426.json`,
40 synthetic fields) — measures speed and structural coverage only, not accuracy:

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
4. **Type context is OFF by default** - appending "text field" / "decimal number" was
   measured to cost 2.1 points of P@1. Type information is used as a numeric scoring
   signal instead of as text in the query.
5. **Namespace filtering** - Ignores common namespace prefixes (com, org, etc.)
