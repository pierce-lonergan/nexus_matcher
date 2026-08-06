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

### Measured on this corpus

`benchmarks/results/exp_rerank_combined.json`, reranking the dense shortlist on the
793-pair combined benchmark. First stage alone: P@1 0.691, MRR@10 0.771.

| Model | Size | P@1 | MRR@10 | Rerank throughput |
|-------|------|-----|--------|-------------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | **0.747** (+0.055) | 0.809 | 18.1 queries/sec |

**Bigger is not better here.** Also measured, though not preserved in the committed
artifact: `BAAI/bge-reranker-base` scored **4.7 points below** the 22M MiniLM-L-6, and
`ms-marco-MiniLM-L-12-v2` also underperformed the L-6. Re-run
`python benchmarks/exp_rerank.py --benchmark combined` to regenerate the comparison.

Reranking is **off by default**: +5.5 points of P@1 costs roughly an order of magnitude
of throughput (652 fields/sec un-reranked).

### Size and speed, for reference

| Model | Size | Speed |
|-------|------|-------|
| BAAI/bge-reranker-v2-m3 | 568M | Slow |
| BAAI/bge-reranker-base | 278M | Medium — **measured worse than MiniLM-L-6 here** |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M | Fast — best measured |
| cross-encoder/ms-marco-TinyBERT-L-2-v2 | 4.4M | Very fast — unmeasured on this corpus |

Quality columns have been removed: general-purpose quality rankings did not predict
performance on this corpus.

### How CrossEncoders Work

Unlike bi-encoders that independently encode query and document, CrossEncoders:
1. Concatenate query and document: "[CLS] query [SEP] document [SEP]"
2. Pass through transformer to get cross-attention between query and document tokens
3. Output a single relevance score

This gives much higher quality scores but is O(n) per candidate vs O(1) for bi-encoders.

## Implementation

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
