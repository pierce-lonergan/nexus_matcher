# Module: Abbreviation Expansion

## Purpose

Expand common abbreviations in field names and dictionary entries to improve semantic matching quality. Maps abbreviated tokens like "acct" → "account", "cust" → "customer", enabling better alignment between terse schema field names and verbose business dictionary terms.

> **Measured contribution is small — and this component once did real damage.**
>
> On the 793-pair combined benchmark, abbreviation expansion is worth **+0.8 points of
> P@1** on its own (0.488 → 0.496, `benchmarks/results/exp_query_repr_combined.json`).
> Parent-path context, by comparison, is worth +20.1 points. Expansion is worth keeping,
> but it is not where the accuracy comes from.
>
> More importantly: this expander runs on the **already-enriched, multi-word** query
> string, and it used to collapse that whole string into a single camelCase mega-token.
> While that was true the production pipeline measured **dense P@1 0.309 and BM25 P@1
> 0.005, with 787 of 793 queries returning zero BM25 hits.** Any change to tokenisation
> here must be validated against `benchmarks/eval_pipeline.py`, not just unit tests —
> the entire unit suite passed throughout.

## Domain Model

### Entities

- **AbbreviationMapping**: A single abbreviation → expansion mapping
  - Invariants:
    - Abbreviation must be non-empty lowercase string
    - Expansion must be non-empty string
    - Abbreviation ≠ expansion (no identity mappings)
  - States: N/A (immutable value object)
  - Events: N/A

- **AbbreviationDictionary**: Collection of mappings with lookup capability
  - Invariants:
    - No duplicate abbreviations (last wins or error)
    - All abbreviations normalized to lowercase
  - States: EMPTY → LOADED
  - Events: DictionaryLoaded, MappingAdded

### Value Objects

- **ExpandedText**: Text with expansion metadata
  - original: str — Original input text
  - expanded: str — Text with abbreviations expanded
  - expansions: list[tuple[str, str]] — Applied (abbrev, expansion) pairs

### Domain Services

- **AbbreviationExpander**: Expands abbreviations in text
  - expand(text: str) → ExpandedText
  - expand_tokens(tokens: list[str]) → list[str]
  - get_candidates(abbrev: str) → list[str]

## Common Abbreviations (Data Domain)

| Abbreviation | Expansion      | Domain       |
|--------------|----------------|--------------|
| acct         | account        | Finance      |
| addr         | address        | General      |
| amt          | amount         | Finance      |
| bal          | balance        | Finance      |
| cd           | code           | General      |
| cust         | customer       | Business     |
| dt           | date           | General      |
| desc         | description    | General      |
| id           | identifier     | General      |
| ind          | indicator      | General      |
| msg          | message        | General      |
| nm           | name           | General      |
| no           | number         | General      |
| num          | number         | General      |
| pct          | percent        | Finance      |
| qty          | quantity       | General      |
| seq          | sequence       | General      |
| stat         | status         | General      |
| ts           | timestamp      | General      |
| txn          | transaction    | Finance      |
| typ          | type           | General      |
| val          | value          | General      |

## Integration Points

- **SchemaField.to_searchable_text()**: Use expander before generating search text
- **DictionaryEntry.to_searchable_text()**: Expand abbreviations in business names
- **NexusMatcher._match_field()**: Expand field name before embedding

## Planned Implementation

- [x] Domain analysis complete
- [x] AbbreviationMapping value object
- [x] AbbreviationDictionary entity
- [x] AbbreviationExpander service
- [x] Default abbreviation mappings (60+ mappings)
- [x] Integration with NexusMatcher
- [x] Unit tests (34 tests)
- [x] Integration tests (9 tests)

## Configuration

```python
# Environment variables
NEXUS_ABBREVIATION_FILE=path/to/custom_abbreviations.json
NEXUS_ABBREVIATION_EXPAND_BOTH_DIRECTIONS=false  # Also map "account"→"acct"?
```

## Usage Example

```python
from nexus_matcher.domain.services.abbreviation import AbbreviationExpander

expander = AbbreviationExpander.default()

# Single text
result = expander.expand("cust_acct_bal")
# result.expanded == "customer_account_balance"
# result.expansions == [("cust", "customer"), ("acct", "account"), ("bal", "balance")]

# Field matching integration (automatic)
matcher = NexusMatcher.from_config()
# Internally expands "cust_id" to "customer identifier" for better matching
```
