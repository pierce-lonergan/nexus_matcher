# Module: CrossEncoder Reranker Adapter

## Purpose

Production-grade CrossEncoder reranker adapter for semantic reranking. Uses transformer-based cross-attention models to score query-document pairs with high precision.

## Domain Model

### Entities

- **CrossEncoderReranker**: Production reranker implementation
  - Invariants: Model must be loaded before scoring
  - States: Loaded, Unloaded
  - Events: N/A (infrastructure adapter)

### Value Objects

- **RerankCandidate**: Candidate document for reranking
- **RerankResult**: Result with score, rank, original_rank

### Domain Services

- Implements `Reranker` protocol from domain/ports/retrieval

## CrossEncoder Models

### Recommended Models

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| BAAI/bge-reranker-v2-m3 | 568M | Excellent | Slow |
| BAAI/bge-reranker-base | 278M | Good | Medium |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M | Good | Fast |
| cross-encoder/ms-marco-TinyBERT-L-2-v2 | 4.4M | Fair | Very Fast |

### How CrossEncoders Work

Unlike bi-encoders that independently encode query and document, CrossEncoders:
1. Concatenate query and document: "[CLS] query [SEP] document [SEP]"
2. Pass through transformer to get cross-attention between query and document tokens
3. Output a single relevance score

This gives much higher quality scores but is O(n) per candidate vs O(1) for bi-encoders.

## Planned Implementation

- [x] Domain analysis complete
- [ ] CrossEncoderReranker class
- [ ] Model loading with caching
- [ ] Batch scoring optimization
- [ ] GPU/CPU device handling
- [ ] Unit tests (TDD)
- [ ] Integration tests

## Dependencies

```toml
[project.optional-dependencies]
reranker = ["sentence-transformers>=2.2.0"]
```

## Usage Example

```python
from nexus_matcher.infrastructure.adapters.rerankers.cross_encoder import (
    CrossEncoderReranker,
)
from nexus_matcher.domain.ports.retrieval import RerankCandidate

# Create reranker (downloads model on first use)
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"
)

# Rerank candidates
candidates = [
    RerankCandidate(id="1", text="customer email address"),
    RerankCandidate(id="2", text="account phone number"),
    RerankCandidate(id="3", text="email contact information"),
]

result = reranker.rerank(
    query="customer email",
    candidates=candidates,
    top_k=2
)

for r in result.unwrap():
    print(f"{r.id}: {r.score:.3f} (rank {r.rank} from {r.original_rank})")
```

## File Structure

```
src/nexus_matcher/infrastructure/adapters/rerankers/
├── __init__.py
└── cross_encoder.py  # CrossEncoder implementation (new)
```
