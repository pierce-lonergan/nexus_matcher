# Module: Domain Hierarchy Service

## Purpose

Manage business domain hierarchies and compute domain-based matching scores. Domains represent logical groupings of data (e.g., Finance, Customer, Product) that help disambiguate matches when multiple dictionary entries have similar semantic content but belong to different business contexts.

## Domain Model

### Entities

- **Domain**: A business domain category
  - Invariants:
    - Name must be non-empty, normalized to uppercase
    - Parent domain must exist if specified
    - No circular parent references
  - States: N/A (immutable)
  - Events: N/A

- **DomainHierarchy**: Tree structure of domains
  - Invariants:
    - Root domains have no parent
    - Each domain has at most one parent
    - No orphaned domains
  - States: EMPTY → LOADED
  - Events: HierarchyLoaded

### Value Objects

- **DomainPath**: Full path from root to domain (e.g., "FINANCE.BANKING.ACCOUNTS")
- **DomainMatch**: Result of domain comparison with score and relationship type
- **DomainRelationship**: Enum of relationship types (EXACT, PARENT, CHILD, SIBLING, COUSIN, UNRELATED)

### Domain Services

- **DomainMatcher**: Computes domain compatibility scores
  - match(domain1: str, domain2: str) → DomainMatch
  - score(domain1: str, domain2: str) → float (0.0 to 1.0)
  - get_path(domain: str) → DomainPath
  - find_common_ancestor(d1: str, d2: str) → Domain | None

## Domain Hierarchy (JPMorgan CCB Context)

```
ROOT
├── FINANCE
│   ├── BANKING
│   │   ├── ACCOUNTS
│   │   ├── TRANSACTIONS
│   │   └── BALANCES
│   ├── PAYMENTS
│   │   ├── TRANSFERS
│   │   └── WIRES
│   └── LENDING
│       ├── LOANS
│       └── MORTGAGES
├── CUSTOMER
│   ├── IDENTITY
│   ├── CONTACT
│   └── PREFERENCES
├── PRODUCT
│   ├── DEPOSIT
│   ├── CREDIT
│   └── INVESTMENT
├── COMPLIANCE
│   ├── KYC
│   ├── AML
│   └── REGULATORY
└── OPERATIONS
    ├── AUDIT
    └── SYSTEM
```

## Scoring Rules

| Relationship | Score | Description |
|--------------|-------|-------------|
| EXACT        | 1.00  | Same domain |
| PARENT       | 0.85  | Field domain is parent of entry domain |
| CHILD        | 0.80  | Field domain is child of entry domain |
| SIBLING      | 0.70  | Share immediate parent |
| COUSIN       | 0.50  | Share ancestor within 2 levels |
| UNRELATED    | 0.25  | No common ancestor or >3 levels apart |

## Integration Points

- **DictionaryEntry.domain**: Domain field for each entry
- **MatchResult.score_breakdown.domain_score**: Domain contribution to final score
- **NexusMatcher.config.domain_weight**: Weight for domain score (default 0.15)

## Planned Implementation

- [x] Domain analysis complete
- [x] Domain value object
- [x] DomainPath value object
- [x] DomainRelationship enum
- [x] DomainMatch result class
- [x] DomainHierarchy entity
- [x] DomainMatcher service
- [x] Default hierarchy (50+ CCB domains)
- [x] Integration with scoring
- [x] Unit tests (37 tests)
- [x] Integration tests (13 tests)

## Configuration

```python
# Environment variables
NEXUS_DOMAIN_HIERARCHY_FILE=path/to/custom_hierarchy.json
NEXUS_DOMAIN_UNKNOWN_SCORE=0.5  # Score when domain unknown
```

## Usage Example

```python
from nexus_matcher.domain.services.domain_hierarchy import DomainMatcher

matcher = DomainMatcher.default()

# Exact match
result = matcher.match("ACCOUNTS", "ACCOUNTS")
# result.relationship == DomainRelationship.EXACT
# result.score == 1.0

# Parent-child relationship
result = matcher.match("BANKING", "ACCOUNTS")
# result.relationship == DomainRelationship.PARENT
# result.score == 0.85

# Sibling relationship
result = matcher.match("ACCOUNTS", "TRANSACTIONS")
# result.relationship == DomainRelationship.SIBLING
# result.score == 0.70
```
