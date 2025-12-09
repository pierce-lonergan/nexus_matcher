# Benchmark Registry — NexusMatcher
> All performance measurements with reproducible test conditions
> Last Updated: 2025-12-09

## Benchmark Suites

### SUITE-001: Core Retrieval Performance
**Test Dataset:** 30,000 Excel dictionary entries (simulated)
**Query Set:** 500 Avro schema fields (stratified by complexity)
**Environment:** TBD (capture on first run)

| Run ID | Date | Model | Quantization | Precision@5 | Recall@10 | MRR | P95 Latency | Notes |
|--------|------|-------|--------------|-------------|-----------|-----|-------------|-------|
| - | - | - | - | - | - | - | - | Baseline pending |

### SUITE-002: Embedding Model Performance (GAP-002)
**Test Dataset:** 1,000+ field descriptions (varied length)
**Batch Sizes:** 1, 8, 16, 32, 64
**Environment:** Python 3.12.3, Linux-4.4.0-x86_64-with-glibc2.39
**CPU Features:** VNNI ✓, AVX-512 ✓, AVX2 ✓

| Run ID | Date | Model | Quantization | Batch Size | Throughput (texts/s) | P50 (ms) | P95 (ms) | Memory (MB) | Notes |
|--------|------|-------|--------------|------------|---------------------|----------|----------|-------------|-------|
| run_20251209_132441 | 2025-12-09 | mock-int8 | INT8 (mock) | 32 | 5,657 | 5.52 | 6.32 | - | Infrastructure validated |

**Detailed Results — run_20251209_132441 (Mock Provider):**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| P95 Latency | 6.32ms | ≤15ms | ✓ PASS |
| P99 Latency | 6.68ms | - | Info |
| Throughput | 5,657 texts/s | ≥5000 | ✓ PASS |
| Speedup Ratio | 2.3x (mock) | ≥3x | Pending real model |

**CPU Feature Detection:**
- VNNI: ✓ Available (optimal for INT8)
- AVX-512: ✓ Available
- AVX2: ✓ Available
- ONNX Runtime: ✓ Available

**Note:** Results from MockQuantizedProvider. Full validation pending integration with actual quantized ONNX/OpenVINO models. Infrastructure supports 3-10x speedup target per research.

### SUITE-003: Reranking Stage Performance (GAP-001)
**Test Dataset:** 10-200 candidates per query, 50 iterations
**Models:** ColBERT MaxSim (MockColBERTReranker)
**Environment:** Python 3.12.3, Linux-4.4.0-x86_64-with-glibc2.39

| Run ID | Date | Reranker | Candidates | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Notes |
|--------|------|----------|------------|----------|----------|----------|------------|-------|
| run_20251209_134730 | 2025-12-09 | mock_colbert | 10 | 1.09 | 1.39 | 1.89 | 8,902/s | Mock |
| run_20251209_134730 | 2025-12-09 | mock_colbert | 50 | 4.36 | 4.99 | 5.40 | 11,268/s | Mock |
| run_20251209_134730 | 2025-12-09 | mock_colbert | 100 | 8.61 | 9.81 | 10.66 | 11,413/s | ✓ Target |
| run_20251209_134730 | 2025-12-09 | mock_colbert | 200 | 16.45 | 18.21 | 19.62 | 12,029/s | Mock |

**Detailed Results — run_20251209_134730:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| 100 Candidates P95 | 9.81ms | ≤60ms | ✓ PASS |
| Throughput | 11,413/s | ≥1000 | ✓ PASS |

**Key Implementation Details:**
- Uses token-level embeddings (NOT bi-encoder pooling)
- MaxSim late interaction: sum of max similarities per query token
- 128-dimensional vectors (ColBERT standard)
- Model: answerai-colbert-small-v1 (when RAGatouille installed)

**GAP-001 Status:** IN_PROGRESS — Infrastructure validated, awaiting RAGatouille integration

### SUITE-004: Cache Performance
**Test Dataset:** 5,000 embeddings (768 dimensions), 1,000 queries
**Query Pattern:** 60% repetition rate (simulates realistic workload)
**Environment:** Python 3.12.3, Linux-4.4.0-x86_64-with-glibc2.39

| Run ID | Date | L1 Size | GET P95 (ms) | Hit Rate | Throughput (ops/s) | Memory (MB) | Notes |
|--------|------|---------|--------------|----------|-------------------|-------------|-------|
| run_20251209_062416 | 2025-12-09 | 5000 | 0.0008 | 56.99% | 1,332,126 | 16.58 | GAP-003 VALIDATED |

**Detailed Results — run_20251209_062416:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| GET P50 | 0.0007ms | <1ms | ✓ PASS |
| GET P95 | 0.0008ms | <1ms | ✓ PASS |
| GET P99 | 0.0026ms | <1ms | ✓ PASS |
| SET P50 | 0.0011ms | <1ms | ✓ PASS |
| Hit Rate | 56.99% | ≥15% | ✓ PASS |
| Throughput | 1,332,126 ops/s | >100K | ✓ PASS |
| Memory Peak | 16.58 MB | <100MB | ✓ PASS |

### SUITE-004b: Semantic Content Cache (GAP-004)
**Test Dataset:** 1,300 queries (realistic field names with duplicates)
**Configuration:** max_size=10000, normalize=True
**Environment:** Python 3.12.3, Linux-4.4.0-x86_64-with-glibc2.39

| Run ID | Date | Cost Reduction | Hit Rate | Throughput | Batch Eff | Notes |
|--------|------|---------------|----------|------------|-----------|-------|
| run_20251209_062xxx | 2025-12-09 | 99.3% | 50% | 781K ops/s | 50% | GAP-004 VALIDATED |

**Detailed Results:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Cost Reduction | 99.3% | ≥50% | ✓ PASS |
| Hit Rate (50% rep) | 50.0% | ≥40% | ✓ PASS |
| Hit Rate (30% rep) | 30.0% | - | Info |
| Hit Rate (70% rep) | 70.0% | - | Info |
| BLAKE3 Throughput | 781,172 ops/s | Functional | ✓ PASS |
| Batch Efficiency | 50.0% | ≥40% | ✓ PASS |

### SUITE-005: Incremental Update Performance (GAP-005)
**Test Dataset:** 50,000 dictionary entries (synthetic)
**Change Scenarios:** 0.1%, 1%, 5%, 10% modification rates
**Environment:** Python 3.12.3, Linux-4.4.0-x86_64-with-glibc2.39, BLAKE3 ✓

| Run ID | Date | Change Rate | Entries Changed | Savings % | Detection (ms) | Throughput | Notes |
|--------|------|-------------|-----------------|-----------|----------------|------------|-------|
| run_20251209_133428 | 2025-12-09 | 0.1% | 50 | 99.9% | 111.72 | 448K/s | ✓ PASS |
| run_20251209_133428 | 2025-12-09 | 1% | 500 | 99.0% | 105.13 | 476K/s | ✓ PASS |
| run_20251209_133428 | 2025-12-09 | 5% | 2,500 | 95.0% | 106.93 | 468K/s | ✓ PASS |
| run_20251209_133428 | 2025-12-09 | 10% | 5,000 | 90.0% | 107.95 | 463K/s | ✓ PASS |

**Detailed Results — run_20251209_133428:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| 0.1% Change Savings | 99.9% | ≥99% | ✓ PASS |
| 1% Change Savings | 99.0% | ≥99% | ✓ PASS |
| 5% Change Savings | 95.0% | ≥95% | ✓ PASS |
| 10% Change Savings | 90.0% | ≥90% | ✓ PASS |
| BLAKE3 Hash Time | 1.43 µs | <10 µs | ✓ PASS |
| BLAKE3 Throughput | 698,646 hashes/s | Functional | ✓ PASS |
| Change Detection Throughput | 450K+ entries/s | >100K | ✓ PASS |

**GAP-005 Status:** VALIDATED ✓ — All research targets met

### SUITE-004c: Context Enrichment (GAP-006)
**Test Dataset:** 40 schema fields (mixed depths: root, 1-parent, 2-parent, 3+-parent)
**Configuration:** ContextEnricher with defaults (include_type=True, humanize=True)
**Environment:** Python 3.12.3, Linux-4.4.0-x86_64-with-glibc2.39

| Run ID | Date | Depth 3+ Coverage | Hierarchy Tokens | Humanization | Throughput | Notes |
|--------|------|-------------------|------------------|--------------|------------|-------|
| run_20251209_131426 | 2025-12-09 | 100% | 1.78 avg | 100% | 103K fields/s | GAP-006 VALIDATED |

**Detailed Results — run_20251209_131426:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Depth 3+ Coverage | 100% | ≥80% | ✓ PASS |
| Hierarchy Tokens | 1.78 avg | ≥1.5 | ✓ PASS |
| Humanization Rate | 100% | ≥95% | ✓ PASS |
| Throughput | 103,721 fields/s | ≥50K | ✓ PASS |
| Latency P50 | 8.47 µs | - | Info |
| Latency P95 | 16.90 µs | - | Info |
| Latency P99 | 17.39 µs | - | Info |

**Enrichment Examples:**

| Field Path | Basic | Enriched |
|------------|-------|----------|
| email_address | email address | email address text field |
| user.addresses.street_name | street name | user, addresses street name text field |
| order.line_items.price.amount | amount | order, line items, price amount decimal number field |

### SUITE-006: End-to-End Pipeline Performance
**Test Dataset:** 100 complete Avro schemas (5-50 fields each)
**Full Pipeline:** Parse → Embed → Retrieve → Rerank → Score
**Environment:** TBD

| Run ID | Date | Schema Count | Fields | P50 (ms/field) | P95 (ms/field) | Total Time (s) | Precision@5 | Notes |
|--------|------|--------------|--------|----------------|----------------|----------------|-------------|-------|
| - | - | - | - | - | - | - | - | Baseline pending |

## Baseline Capture Checklist

- [ ] SUITE-001: Core retrieval baseline
- [ ] SUITE-002: Embedding model baseline (BGE-base, FP32)
- [ ] SUITE-003: Reranking baseline (CrossEncoder only)
- [x] SUITE-004: L1 Cache baseline — validated (2025-12-09)
- [x] SUITE-004b: Semantic Cache — validated (2025-12-09)
- [x] SUITE-004c: Context Enrichment — validated (2025-12-09)
- [ ] SUITE-005: Update baseline (full reindex only)
- [ ] SUITE-006: End-to-end baseline

## Environment Configuration Template

```yaml
hardware:
  cpu: [model]
  cores: [count]
  ram_gb: [size]
  storage: [type]

software:
  os: [version]
  python: [version]
  pytorch: [version]
  sentence_transformers: [version]
  qdrant_client: [version]

models:
  embedding: [name, size]
  reranker: [name, size]
  
qdrant:
  version: [version]
  mode: [memory/disk]
  hnsw_m: [value]
  hnsw_ef_construct: [value]
```

## Research Targets (from RESEARCH_ALIGNMENT.md)

| Metric | Phase 1 | Phase 2 | Final |
|--------|---------|---------|-------|
| Precision@5 | 95%+ | 97%+ | 97-99% |
| Recall@10 | 93%+ | 95%+ | 96%+ |
| MRR | 0.72+ | 0.80+ | 0.87+ |
| P95 Latency | ≤200ms | ≤150ms | 120-180ms |
| Cache Hit Rate | ≥40% | ≥60% | 60-70% |
| Embedding Latency | ≤30ms | ≤15ms | 10-15ms |

## Benchmark Execution Log

### [Date] — [Suite] — [Purpose]
```
Environment: [captured]
Command: [exact command]
Duration: [time]
Results: [summary]
```
