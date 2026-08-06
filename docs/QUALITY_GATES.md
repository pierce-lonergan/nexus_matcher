# Quality Gates

> **Corrections applied after audit.** Several gates below were marked passing on the
> strength of results that have no artifact, or that measured something other than what
> the gate asks. Those rows are now marked **UNVERIFIED** or **RETRACTED** with the
> reason. The accuracy and latency gates that were pending are still pending, because
> the suites they name (SUITE-001, SUITE-006) have never produced a result.
>
> The end-to-end accuracy question those gates were reaching for is now answered by
> `benchmarks/eval_pipeline.py`: **P@1 0.700, P@5 0.888, MRR@10 0.781, Recall@10 0.919,
> 652 fields/sec** on 793 labelled pairs. See
> [BENCHMARK_REGISTRY.md](BENCHMARK_REGISTRY.md).

## Current measured state

| Gate | Status | Evidence |
|---|---|---|
| Test suite passes | ✓ | 551 passed, 0 failed, 35 skipped (skips are uninstalled optional deps) |
| Line coverage ≥80% | ✗ | **60%** measured against `fail_under = 80` in `pyproject.toml` |
| Entry points all importable | ✓ | Each `pyproject.toml` entry point target imports cleanly |
| P@5 ≥95% | ✗ | Measured **0.888** end-to-end on the combined benchmark |
| MRR ≥0.80 | ✗ | Measured **0.781** end-to-end (0.809 with cross-encoder reranking) |
| Recall@10 ≥93% | ✗ | Measured **0.919** |
| Auto-approve precision ≥95% | ~ | **0.947** at threshold 0.85, over 42.7% coverage |

## Genesis Gates

Infrastructure gates that must pass before advancing to Construction.

### Core Architecture

| Gate | Status | Validation |
|------|--------|------------|
| Repository follows hexagonal architecture | ✓ | Manual review |
| pyproject.toml supports multi-target builds | ✓ | `pip install -e ".[full]"` |
| Package imports cleanly | ✓ | `python -c "from nexus_matcher import *"` |
| Type hints on all public interfaces | ✓ | Manual review |

### Domain Layer

| Gate | Status | Validation |
|------|--------|------------|
| Domain models are immutable (frozen dataclasses) | ✓ | Unit tests |
| All ports defined as Protocol classes | ✓ | Manual review |
| Domain has no infrastructure dependencies | ✓ | Import analysis |
| Validation in model constructors | ✓ | Unit tests |

### Infrastructure Layer

| Gate | Status | Validation |
|------|--------|------------|
| Configuration loads from environment | ✓ | `NEXUS_*` vars |
| Secrets never hardcoded | ✓ | gitleaks scan |
| Adapters implement port protocols | ✓ | Type checking |
| Entry points registered in pyproject.toml | ✓ | Plugin system |

### Presentation Layer

| Gate | Status | Validation |
|------|--------|------------|
| API health endpoints respond | ✓ | `/health` returns 200 |
| API has OpenAPI documentation | ✓ | `/docs` available |
| Request ID middleware active | ✓ | X-Request-ID header |
| Error responses follow standard format | ✓ | Unit tests |

### Quality Infrastructure

| Gate | Status | Validation |
|------|--------|------------|
| At least one unit test per layer | ✓ | pytest |
| Tests use fixtures (no hardcoded data) | ✓ | conftest.py |
| Structured logging configured | ✓ | JSON output |
| CI pipeline defined | ✓ | .github/workflows/ci.yml |

### Packaging & Deployment

| Gate | Status | Validation |
|------|--------|------------|
| README.md exists with quick start | ✓ | Manual review |
| Dockerfile builds | ⚠ | Not validated in env |
| docker-compose.yml works | ⚠ | Not validated in env |
| Package builds (`python -m build`) | ⚠ | Not validated |

### Outstanding Gates

| Gate | Status | Blocker |
|------|--------|---------|
| All gates pass | ✓ | None |

## Construction Gates

**Status:** COMPLETE (8/8 features)

| Gate | Status | Validation |
|------|--------|------------|
| Abbreviation expansion works | ✓ | 34 unit tests |
| Domain hierarchy matching works | ✓ | 37 unit tests |
| Type compatibility scoring works | ✓ | 39 unit tests |
| JSON Schema parser works | ✓ | 38 unit tests |
| SQL DDL parser works | ✓ | 38 unit tests |
| Qdrant adapter works | ✓ | 14 unit tests |
| Redis cache adapter works | ✓ | 20 unit tests |
| CrossEncoder reranker defined | ✓ | 13 tests (skipped w/o deps) |

---

## Enhancement Protocol Gates

### Phase 1: Foundation

| Gate | Status | Target | Validation |
|------|--------|--------|------------|
| L1 cache sub-ms access | **UNVERIFIED** | <1ms | SUITE-004 writes no artifact; nothing to check |
| Cache hit rate ≥40% | **RETRACTED** | ≥40% | The 56.99% figure has no artifact, and a cache's hit rate is a property of the workload (a 60% repetition rate was configured), not of the system |
| Nested schema context (3+ depth) | ✓ | Full hierarchy | `suite_004c_context_enrichment_20251209_131426.json`: 100% coverage at all depths. Measures coverage, not accuracy — for accuracy see `exp_query_repr_combined.json` (+20.1 points P@1) |
| No Precision@5 regression | ⬜ | ≥baseline | SUITE-001 has never run. Current end-to-end P@5 is 0.888 (`eval_pipeline_combined.json`) |
| P95 latency ≤200ms | ⬜ | ≤200ms | SUITE-006 has never run. End-to-end throughput is 652 fields/sec; per-field P95 is unmeasured |
| All tests passing | ✓ | — | 551 passed, 0 failed, 35 skipped |
| SUITE-004 executed (3+ runs) | **UNVERIFIED** | 3+ runs | Cited run id `run_20251209_062416` corresponds to no file in `benchmarks/results/` |

### Phase 2: Acceleration

| Gate | Status | Target | Validation |
|------|--------|--------|------------|
| INT8 quantization deployed | ⬜ | OpenVINO/ONNX | Model loading test |
| Embedding latency ≤15ms (batch-32) | ⬜ | ≤15ms | SUITE-002 benchmark |
| ColBERT MaxSim (not bi-encoder) | ⬜ | RAGatouille | Architecture review |
| ColBERT reranking ≤60ms (top-100) | ⬜ | ≤60ms | SUITE-003 benchmark |
| Incremental update ≥90% faster | ⬜ | ≥90% savings | SUITE-005 benchmark |
| Precision@5 ≥95% | ⬜ | ≥95% | SUITE-001 benchmark |
| P95 latency ≤150ms | ⬜ | ≤150ms | SUITE-006 benchmark |

### Phase 3: Precision

| Gate | Status | Target | Validation |
|------|--------|--------|------------|
| ModernBERT or equivalent deployed | ⬜ | 4x faster | SUITE-002 benchmark |
| Learned type projections trained | ⬜ | LoRA fine-tuning | Training logs |
| MRR ≥0.80 | ⬜ | ≥0.80 | SUITE-001 benchmark |
| P95 latency 120-180ms | ⬜ | 120-180ms | SUITE-006 benchmark |
| Precision@5 ≥97% | ⬜ | 97-99% | SUITE-001 benchmark |
| Research alignment ≥97% | ⬜ | ≥97% | Gap resolution count |

---

## Integration Gates

_Not yet applicable_

## Summary

Structural gates (architecture, ports, packaging, entry points, logging, CI) pass.

Quality gates do **not** all pass:

- Coverage is 60% against a configured 80% gate.
- The accuracy targets these gates were written against (P@5 ≥95%, MRR ≥0.80) are not
  met: measured 0.888 and 0.781. Those targets were set from a literature survey before
  any benchmark existed on this corpus; whether they are the right targets for
  abbreviation-heavy schema matching is itself an open question — see the bird/omop
  split (0.490 vs 0.819) in [BENCHMARK_REGISTRY.md](BENCHMARK_REGISTRY.md).
- Latency gates (P95 per field) remain unmeasured; only aggregate throughput is known.
- Cache gates are unverifiable: the benchmarks that were said to validate them write no
  artifacts.

**Recommendation:** do not describe the system as gate-complete. The genuine next steps
are listed in [PROJECT_STATE.md](PROJECT_STATE.md#outstanding-work-roughly-in-priority-order).
