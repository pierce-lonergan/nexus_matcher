# Project State

Current, verified state of the repository. Anything not listed as measured here should
be treated as unverified.

> This file previously tracked an "Enhancement Protocol" through phases marked COMPLETE
> and VALIDATED on the strength of numbers that had no artifacts behind them, and
> concluded "Production deployment ready!". That assessment was not supported. It has
> been replaced with what can actually be checked.

---

## Headline

| | |
|---|---|
| End-to-end P@1 (793-pair labelled benchmark) | **0.700** |
| P@1 on the abbreviation-heavy split (bird) | **0.490** |
| P@1 on the descriptive split (omop) | 0.819 |
| Throughput | 652 fields/sec, CPU |
| Auto-approve precision at default threshold | 0.947 over 42.7% of fields |
| Tests | 551 passed, 0 failed, 35 skipped |
| Line coverage | 60% against a configured gate of 80% |

Artifact: `benchmarks/results/eval_pipeline_combined.json`. Environment and caveats:
[BENCHMARK_REGISTRY.md](BENCHMARK_REGISTRY.md).

---

## What works and is measured

- Matching from the Python API and the CLI, against Avro / JSON Schema / SQL DDL
  schemas and Excel / CSV dictionaries.
- Dense retrieval (sentence-transformers) + BM25 sparse retrieval, combined with linear
  min-max fusion at `fusion_alpha = 0.90`.
- Hierarchical context enrichment of the query — the single largest accuracy factor.
- Multi-signal confidence scoring and a calibrated three-way decision policy.
- Optional cross-encoder reranking: +5.5 points P@1 at roughly an order of magnitude
  less throughput.

## What works but is not measured end-to-end

Implemented, unit-tested, and not exercised by the accuracy benchmark. Do not assume a
production benefit without measuring it on your data.

- L1 LRU / Redis / semantic content caches. **Not wired into the matching pipeline** —
  `NexusMatcher` performs no cache lookups and hardcodes `cache_hit=False`.
- ColBERT MaxSim reranker.
- INT8 quantized embedding provider.
- Incremental update manager (BLAKE3 change detection).
- Qdrant and HNSW vector stores (the benchmark uses the in-memory store).
- Learned type projections and graph matching — experimental; their reported numbers
  come from a 17-field hand-written toy set.

## What does not exist

- **HTTP matching.** The REST app serves health and introspection routes only.
- **Dictionary CRUD, cache, or metrics endpoints.**
- **API authentication or rate limiting.** The OpenAPI description mentions API keys;
  no route enforces one. CORS is currently `allow_origins=["*"]`.
- **Configuration of matching via YAML or environment variables.**
  `NexusMatcher.from_config()` ignores its `config_path`; `NEXUS_*` settings are read
  only by the logging setup.
- **Dependency health probing.** `/health/ready` reports hardcoded component status.
- **GPU support.** Every measurement in this repo is CPU-only, one machine.

---

## Known risks

| Risk | Detail |
|---|---|
| Abbreviation-heavy schemas | bird P@1 0.490 — roughly half of such fields miss at rank 1. Recall@10 is much higher, which is why the intended surface is a ranked review list. |
| Threshold portability | `auto_approve_threshold = 0.85` is calibrated on this corpus only. Re-calibrate on a labelled sample of your own data before trusting auto-approval. |
| Threshold drift | Improving retrieval shifts scores upward and can *lower* auto-approve precision at a fixed threshold. Re-run `exp_calibration.py` after any retrieval change. |
| Scale | Accuracy is measured against a ~1,200-entry dictionary. Behaviour at catalogue scale is unmeasured. |
| Coverage gate | Line coverage is 60% against a configured 80% gate, so a bare `pytest` run exits non-zero. |
| Single environment | One Windows workstation, CPU, no AVX-512/VNNI. Throughput will not transfer. |
| Language | English only. |

---

## Outstanding work, roughly in priority order

1. Re-calibrate and re-run `exp_calibration.py` whenever retrieval changes.
2. Raise coverage to the configured 80% gate, or lower the gate deliberately.
3. Wire the cache layer into the matching pipeline, or remove it from the documented
   architecture.
4. Implement an HTTP matching endpoint, or stop describing the service as a matching API.
5. Make `from_config()` honour a config file, or remove the parameter.
6. Make `/health/ready` probe real dependencies.
7. Expose a `/metrics` route for the existing Prometheus backend.
8. Regenerate the multi-reranker comparison artifact.
9. Measure at catalogue scale and on GPU.

---

## Layout

`src/nexus_matcher/` is canonical. The duplicate `nexus_matcher_src/` tree has been
deleted. Entry points in `pyproject.toml` have been reduced to those whose target
modules are importable.
