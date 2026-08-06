# ColBERT MaxSim Reranker — Module Documentation

> **Layer:** Infrastructure / Adapters / Rerankers
> **Gap:** GAP-001 (ColBERT MaxSim Implementation)
> **Status:** IMPLEMENTED, optional, off by default
> **Research:** README_RESEARCH_3.md, Lines 5-8

## Overview

The ColBERT MaxSim reranker implements proper late interaction scoring instead of the incorrect bi-encoder approach. The **+10-20% accuracy improvement** figure is a projection from the literature, not a
measurement of this implementation. What *was* measured
(`benchmarks/results/suite_003_real_20251209_162900.json`) is a **latency** result:
pre-computing document token embeddings takes 100-candidate reranking from 274.0 ms
average to 2.93 ms (93.6x). The same artifact records **100% top-5 ranking agreement
with the plain bi-encoder** on its sample — i.e. MaxSim did not change the ranking at
all there. Treat the accuracy benefit as unproven on this corpus. For measured reranking
accuracy, see the cross-encoder result in
[BENCHMARK_REGISTRY.md](../BENCHMARK_REGISTRY.md#exp-rerank--cross-encoder-reranking)
(+5.5 points P@1).

## Research Context

**The Problem (from research):**
> "Your current ColBERT implementation fundamentally misuses the architecture by treating it as a bi-encoder (pooling tokens into single vectors) instead of implementing proper MaxSim late interaction."

**The Solution:**
> "The correct approach uses token-level embeddings where each query and document becomes multiple 128-dimensional vectors, then computes the sum of maximum similarities across query tokens."

## Key Differences: Bi-Encoder vs MaxSim

| Aspect | Bi-Encoder (WRONG) | MaxSim (CORRECT) |
|--------|-------------------|------------------|
| Embedding | Single pooled vector | Multiple token vectors |
| Dimensions | 768+ (model hidden size) | 128 per token |
| Scoring | Cosine similarity | Sum of max similarities |
| Accuracy | -10-20% | Reference baseline |

## Components

### MaxSimConfig

Configuration for ColBERT MaxSim reranking.

```python
from nexus_matcher.infrastructure.adapters.rerankers.colbert import MaxSimConfig

config = MaxSimConfig(
    model_name="answerai-colbert-small-v1",  # 33M params, CPU-optimized
    embedding_dim=128,  # ColBERT standard
    device="cpu",
    batch_size=32,
    max_query_length=32,
    max_doc_length=180,
)
```

### MockColBERTReranker

Testing mock that simulates MaxSim behavior without model dependencies.

```python
from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
    MockColBERTReranker,
)
from nexus_matcher.domain.ports.retrieval import RerankCandidate

reranker = MockColBERTReranker()

candidates = [
    RerankCandidate(id="1", text="customer email address"),
    RerankCandidate(id="2", text="account balance"),
    RerankCandidate(id="3", text="transaction date"),
]

result = reranker.rerank("find email field", candidates)
for r in result.unwrap():
    print(f"{r.rank}. {r.id}: {r.score:.3f}")
```

### ColBERTMaxSimReranker

Production reranker using RAGatouille library.

```python
from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
    ColBERTMaxSimReranker,
)

# Requires: pip install ragatouille
reranker = ColBERTMaxSimReranker()

result = reranker.rerank("customer email", candidates, top_k=10)
```

### ColBERTStatistics

Performance tracking for reranking operations.

```python
from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
    ColBERTStatistics,
)

stats = reranker.get_statistics()
print(f"Total reranks: {stats.total_reranks}")
print(f"Avg latency: {stats.avg_latency_ms:.2f}ms")
print(f"P95 latency: {stats.p95_latency_ms:.2f}ms")

# Reset statistics
reranker.reset_statistics()
```

## MaxSim Scoring Algorithm

The MaxSim score is computed as follows:

```
MaxSim(Q, D) = Σ max(Qi · Dj) for each query token Qi
              j∈D

Where:
- Q = {Q1, Q2, ..., Qn} are query token embeddings (128-dim each)
- D = {D1, D2, ..., Dm} are document token embeddings (128-dim each)
- · represents dot product (cosine similarity with normalized vectors)
```

**Key insight:** For each query token, we find the document token that best matches it, then sum all these maximum similarities.

## Performance Targets

| Metric | Target | Achieved (Mock) |
|--------|--------|-----------------|
| 100 candidates P95 | ≤60ms | 9.81ms ✓ |
| Throughput | ≥1000/s | 11,413/s ✓ |
| Accuracy gain | +10-20% | Pending real model |

## Integration Example

Integrating with the matching pipeline:

```python
class SchemaMatcherPipeline:
    def __init__(self):
        self.retriever = VectorRetriever()  # First stage
        self.reranker = ColBERTMaxSimReranker()  # Second stage
    
    def match(self, query: str, top_k: int = 5) -> list[Match]:
        # Stage 1: Fast retrieval (BM25 + vector)
        candidates = self.retriever.retrieve(query, k=100)
        
        # Stage 2: Precise reranking with ColBERT MaxSim
        rerank_candidates = [
            RerankCandidate(id=c.id, text=c.text, initial_score=c.score)
            for c in candidates
        ]
        
        result = self.reranker.rerank(query, rerank_candidates, top_k=top_k)
        
        return result.unwrap()
```

## Installation

For production use with real ColBERT model:

```bash
pip install ragatouille
```

The first call to `rerank()` will download the model automatically.

## API Reference

### MockColBERTReranker / ColBERTMaxSimReranker

**Properties:**
- `reranker_type` → str: "colbert_maxsim"
- `model_name` → str: Model identifier
- `uses_token_embeddings` → bool: True (NOT pooled)
- `is_bi_encoder` → bool: False (MaxSim, not bi-encoder)

**Methods:**
- `rerank(query, candidates, top_k=None)` → Result[list[RerankResult]]
- `score_pair(query, document)` → Result[float]
- `rerank_batch(queries, candidates_per_query, top_k=None)` → Result[list[list[RerankResult]]]
- `get_statistics()` → ColBERTStatistics
- `reset_statistics()`

### RerankResult (from domain.ports.retrieval)

- `id`: Document ID
- `score`: MaxSim score
- `rank`: New rank after reranking
- `original_rank`: Rank before reranking
- `metadata`: Preserved from candidate

## Next Steps

1. Install RAGatouille: `pip install ragatouille`
2. Run benchmark with real model: `USE_REAL_MODEL=1 python benchmarks/suite_003_colbert_reranking.py`
3. Validate +10-20% accuracy improvement on schema matching benchmark
4. Mark GAP-001 as VALIDATED
