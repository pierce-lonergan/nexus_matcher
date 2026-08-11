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

- Matching from the Python API, the CLI, and over HTTP (`POST /api/v1/match`,
  `/api/v1/match/batch`, `/api/v1/feedback`), against Avro / JSON Schema / SQL DDL
  schemas and Excel / CSV dictionaries. The endpoint calls the same matcher, so it makes
  no accuracy claim of its own; it is the numbers above, over a wire.
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

- **Dictionary CRUD, cache, or metrics endpoints.**
- **API authentication or rate limiting.** No route enforces a key and `/openapi.json`
  declares no security scheme; the description says the service ships unauthenticated
  instead of offering a header nothing reads. CORS is refused unless
  `NEXUS_API_CORS_ORIGINS` names the origins.
- **Configuration of matching via YAML, or via an environment variable per setting.**
  `from_config()` does honour a config file now — JSON or TOML, with an unknown key
  raising rather than being discarded — and the service reads one from
  `NEXUS_API_MATCHING_CONFIG`. A `.yaml` file is parsed as JSON and fails, and no
  `NEXUS_*` variable sets an individual matching parameter; the `NEXUS_API_*` set
  configures the HTTP service, not the matcher's numbers.
- **Dependency health probing.** `/health/ready` checks `matcher` for real — 503 when no
  dictionary loaded — but `api` and `config` are still hardcoded `True`, and nothing
  probes a vector store or a cache.
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
4. Probe real dependencies from `/health/ready`. `matcher` is a real check; `api` and
   `config` are still set unconditionally.
5. Expose a `/metrics` route for the existing Prometheus backend.
6. Regenerate the multi-reranker comparison artifact.
7. Measure at catalogue scale and on GPU.

An earlier revision of this list carried a demand to build the HTTP matching endpoint. It
had already been built. This list and eight other places went on describing it as absent for
as long as nothing checked — `tests/packaging/test_documented_routes.py` is the gate that now
fails the build when a document contradicts the live route table, in either direction.

Three statements in the section above went stale the same way in the session that landed
those routes: CORS, the readiness components and the OpenAPI description all changed
underneath sentences nobody re-read. Those are claims about behaviour rather than about
routes, so `tests/packaging/test_documented_behaviour.py` settles them against a live
`create_app()` instead. Its own docstring is explicit that it checks four behaviours and
cannot read prose.

---

## Layout

`src/nexus_matcher/` is canonical. The duplicate `nexus_matcher_src/` tree has been
deleted. Entry points in `pyproject.toml` have been reduced to those whose target
modules are importable.
