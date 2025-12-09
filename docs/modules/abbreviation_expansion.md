# Module: Abbreviation Expansion

## Purpose

Expand common abbreviations in field names and dictionary entries to improve semantic matching quality. Maps abbreviated tokens like "acct" → "account", "cust" → "customer", enabling better alignment between terse schema field names and verbose business dictionary terms.

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
