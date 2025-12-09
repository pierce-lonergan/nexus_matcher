# Module: Type Compatibility Scoring

## Purpose

Compute type compatibility scores between schema field types and dictionary entry types. Goes beyond simple equality checks to handle semantic type compatibility (e.g., INTEGER can match LONG, VARCHAR can match STRING) and provide nuanced scoring for partial matches.

## Domain Model

### Entities

- **TypeCompatibility**: Result of type comparison
  - Invariants: Score between 0.0 and 1.0
  - States: N/A (immutable value object)
  - Events: N/A

### Value Objects

- **TypeCategory**: Grouping of related types (NUMERIC, STRING, TEMPORAL, BOOLEAN, BINARY, COMPLEX)
- **TypeProjection**: Mapping of source type to compatible target types with scores
- **TypeCompatibilityResult**: Full result with score, reason, and compatibility level

### Domain Services

- **TypeCompatibilityScorer**: Computes type compatibility scores
  - score(source: DataType, target: DataType) → float
  - get_compatibility(source: DataType, target: DataType) → TypeCompatibilityResult
  - is_compatible(source: DataType, target: DataType) → bool
  - get_compatible_types(source: DataType) → list[DataType]

## Type Categories

| Category | Types |
|----------|-------|
| NUMERIC | INTEGER, LONG, FLOAT, DOUBLE, DECIMAL |
| STRING | STRING, VARCHAR, CHAR, TEXT |
| TEMPORAL | DATE, TIME, DATETIME, TIMESTAMP |
| BOOLEAN | BOOLEAN |
| BINARY | BYTES, BINARY, BLOB |
| COMPLEX | ARRAY, MAP, RECORD, UNION |

## Compatibility Matrix

### Within Category (High Compatibility)

| Source | Target | Score | Notes |
|--------|--------|-------|-------|
| INTEGER | LONG | 0.95 | Widening conversion |
| LONG | INTEGER | 0.80 | Narrowing (possible overflow) |
| FLOAT | DOUBLE | 0.95 | Widening conversion |
| DOUBLE | FLOAT | 0.80 | Precision loss |
| STRING | VARCHAR | 1.00 | Semantically equivalent |
| VARCHAR | TEXT | 0.95 | Widening |
| DATE | TIMESTAMP | 0.85 | Contains date component |
| TIMESTAMP | DATE | 0.75 | Loses time component |

### Cross Category (Low/No Compatibility)

| Source | Target | Score | Notes |
|--------|--------|-------|-------|
| INTEGER | STRING | 0.30 | Possible but lossy |
| STRING | INTEGER | 0.20 | Parsing required |
| BOOLEAN | INTEGER | 0.40 | 0/1 mapping |
| DATE | STRING | 0.50 | Format dependent |

## Scoring Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| EXACT | 1.00 | Same type |
| EQUIVALENT | 0.95-0.99 | Semantically identical (STRING≈VARCHAR) |
| COMPATIBLE | 0.80-0.94 | Safe conversion (INT→LONG) |
| CONVERTIBLE | 0.50-0.79 | Possible conversion with care |
| INCOMPATIBLE | 0.00-0.49 | Likely incompatible |

## Integration Points

- **DictionaryEntry.matches_type()**: Enhanced type matching
- **MatchResult.score_breakdown.type_compatibility_score**: Type contribution
- **NexusMatcher.config.type_weight**: Weight for type score (default 0.10)

## Planned Implementation

- [x] Domain analysis complete
- [x] TypeCategory enum
- [x] CompatibilityLevel enum
- [x] TypeCompatibilityResult value object
- [x] TypeCompatibilityScorer service
- [x] Compatibility matrix (60+ mappings)
- [x] Category cross-scores
- [x] Unit tests (39 tests)
- [x] Integration tests (15 tests)

## Configuration

```python
# Environment variables
NEXUS_TYPE_STRICT_MODE=false  # If true, only exact matches score 1.0
NEXUS_TYPE_UNKNOWN_SCORE=0.5  # Score for unknown types
```

## Usage Example

```python
from nexus_matcher.domain.services.type_compatibility import TypeCompatibilityScorer
from nexus_matcher.shared.types.base import DataType

scorer = TypeCompatibilityScorer()

# Exact match
result = scorer.get_compatibility(DataType.STRING, DataType.STRING)
# result.score == 1.0, result.level == CompatibilityLevel.EXACT

# Widening conversion
result = scorer.get_compatibility(DataType.INTEGER, DataType.LONG)
# result.score == 0.95, result.level == CompatibilityLevel.COMPATIBLE

# Cross-category
result = scorer.get_compatibility(DataType.INTEGER, DataType.STRING)
# result.score == 0.30, result.level == CompatibilityLevel.INCOMPATIBLE
```
