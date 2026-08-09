# Changelog

All notable changes to NexusMatcher are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every performance or accuracy number below names the artifact in `benchmarks/results/`
that it came from. Numbers without an artifact are not stated.

---

## [Unreleased]

### Added — measurement

- **A real, labelled benchmark.** `benchmarks/datasets/build_benchmarks.py` builds 793
  query→entry pairs from BIRD-SQL dev `database_descriptions` (361) and the OHDSI OMOP
  CDM v5.4 field-level spec (432). Dictionary entries are indexed on **business name and
  definition only** — the source system's technical column name is deliberately excluded,
  so nothing can be solved by string identity.
- **`benchmarks/eval_pipeline.py`** — end-to-end evaluation that drives the real
  `NexusMatcher` orchestrator, not a hand-rolled cosine loop. Artifact:
  `eval_pipeline_combined.json`.

  | Metric | combined | bird | omop |
  |---|---|---|---|
  | P@1 | 0.700 | 0.490 | 0.819 |

  Also: P@5 0.888, MRR@10 0.781, Recall@10 0.919, 652 fields/sec, 1.76 s index build
  for 793 entries, on CPU.
- **Ablation and calibration experiments**, each writing its own artifact:
  `exp_query_repr.py`, `exp_fusion.py`, `exp_calibration.py`, `exp_rerank.py`.
- **`tests/unit/test_regression_guards.py`** — tests that fail when accuracy-destroying
  changes are made, rather than only when APIs break.

### Fixed — accuracy defects found by the new benchmark

- **`AbbreviationExpander` destroyed enriched queries.** It collapsed multi-word
  natural-language queries into a single camelCase mega-token. The production path was
  measuring dense P@1 0.309 and BM25 P@1 0.005, with **787 of 793 queries returning zero
  BM25 hits** — the sparse arm was contributing essentially nothing. After the fix:
  dense P@1 0.636, BM25 P@1 0.531, zero zero-hit queries.
- **Missing BGE query-instruction prefix.** BGE retrieval models are trained with an
  instruction on the query side only. Adding it asymmetrically (queries prefixed,
  documents not) was worth +5.3 points of P@1.
- **Loading a second dictionary left the first one's vectors searchable.**
  `_dictionary_entries` was replaced but the vector store was only ever upserted into,
  producing silent misses and matches against unresolvable entries. The store is now
  cleared of previously indexed ids first.
- **Failed sparse index builds were silent.** `SparseRetriever.index()` returned a
  `Result` that was discarded, so a failure left the matcher running dense-only with no
  indication. It now raises.
- **`match_schema_session()` parsed the source twice**, doubling parse cost and risking
  a mismatch between the returned schema and the results computed from it.

### Changed — defaults, all measurement-driven

- **Query text now includes the parent path.** `satscores sname` instead of `sname`.
  Worth **+20.1 points of P@1** (0.491 → 0.691) — the largest single accuracy factor in
  the pipeline. Artifact: `exp_query_repr_combined.json`.
- **Scalar type words are no longer appended to queries.** Adding "text field" to the
  query *cost* 2.1 points of P@1. Now off by default.
- **Fusion is linear min-max with `fusion_alpha = 0.90`,** not RRF. Measured on the
  combined benchmark: linear dense=0.9 → 0.7024, dense-only → 0.6910, RRF k=60 → 0.6103.
  **RRF was the worst method measured and worse than not fusing at all.** Artifact:
  `exp_fusion_combined.json`.
- **`auto_approve_threshold` raised from 0.75 to 0.85.** At 0.75 the auto-approved slice
  was only 86.3% precise. At 0.85 it is 94.7% precise over 42.7% coverage. Auto-approving
  a wrong mapping costs more than sending a field to review. Artifact:
  `exp_calibration_combined.json`.

### Changed — performance

- `InMemoryVectorStore` no longer re-normalises the entire corpus matrix on every query
  (this also removed a large per-query allocation).
- Edit distance now uses `rapidfuzz` instead of a pure-Python DP loop; results are
  bit-identical.
- `_match_fields` embeds all query strings in one batched call rather than one per field.

  These three are micro-benchmarks without committed artifacts — see the "unarchived"
  section of [docs/BENCHMARK_REGISTRY.md](docs/BENCHMARK_REGISTRY.md).

### Removed

- **The duplicate `nexus_matcher_src/` tree** (127 files). `src/nexus_matcher/` is
  canonical.
- **The stray second readme** (`README (1).md`). There is one `README.md`.
- **Broken plugin entry points.** `pyproject.toml` declared entry points for
  `csv_headers`, `database`, `faiss` and `openai` modules that do not exist; a broken
  entry point makes plugin discovery raise at import time for every consumer of the
  package. Remaining entry points were each verified importable.

### Documentation — retractions

The following claims appeared in the README, this changelog, the package docstring and
`docs/ENHANCEMENT_JOURNEY.md`. They were false and have been removed.

| Retracted claim | What is actually true |
|---|---|
| **"100% Precision@1"** | Came from `benchmarks/suite_008_combined.py`, which **never calls `NexusMatcher`** — it computes raw cosine similarity over 17 hand-written source fields against a 20-entry hand-written target set, and got 17/17. Measured end-to-end P@1 is **0.700**. |
| **"1.68× INT8 speedup"** | Not in any artifact. `suite_002_real_20251209_162836.json` measures 1.27× at batch 32 (the batch size the claim cited), ranging 1.26×–2.93× across batch sizes, on a machine without VNNI. |
| **"3.07% accuracy loss" from INT8** | No accuracy figure was ever recorded. The artifact carries `accuracy_pass: false` and `overall_pass: false`. |
| **"56.99% cache hit rate", "99.3% cost reduction"** cited as VALIDATED | `benchmarks/suite_004_cache_performance.py` and `suite_004b_semantic_cache.py` write **no artifact at all**. One cited run ID, `run_20251209_062xxx`, is a literal placeholder. Also: a cache's hit rate is a property of the workload, which in this case was a synthetic 60%-repetition query pattern. |
| **"86x faster reranking" / "93.7x"** | The same measurement compared two ways — cold 274.0 ms avg vs warm 2.93 ms avg (93.6×) or vs warm 3.17 ms p95 (86×), at 100 candidates. It is a **latency** result for pre-computing document token embeddings, and the same artifact shows MaxSim did not change the top-5 ranking at all on its sample. It is not evidence of an accuracy gain. |
| **10 documented REST endpoints** | The FastAPI app implements health and introspection endpoints only. There is no matching endpoint, no dictionary CRUD, no cache endpoints, no `/metrics`, no API-key auth and no rate limiting. See [docs/API_REFERENCE.md](docs/API_REFERENCE.md). |
| **"Prometheus metrics endpoint"** listed under Added in 2.0.0 | A `PrometheusMetrics` backend class exists; no route exposes it. |
| **Default model `all-MiniLM-L6-v2`** | The shipped default in `SentenceTransformersProvider` is `BAAI/bge-base-en-v1.5`. The published benchmark uses `BAAI/bge-small-en-v1.5`. |
| **YAML / environment configuration of matching** | `NexusMatcher.from_config()` accepts a `MatchingConfig` or a JSON/TOML file, but the `NEXUS_*` settings classes are still consumed only by the logging setup. There is no YAML path and no environment-variable control of matching behaviour. |
| **Test count "433 tests"** | Current measured state: 551 passed, 0 failed, 35 skipped (skips are uninstalled optional dependencies). Line coverage 60% against a configured gate of 80%. |

### Planned

- GPU measurement. Every number in this repository is CPU-only, single machine.
- Accuracy measurement at catalogue scale; the benchmark corpus is ~1,200 entries.
- Non-English schema matching. All measurement to date is English.
- An HTTP matching endpoint.

---

## [2.0.1] - 2026-08-09

Fixes from a verification sweep of the published 2.0.0 wheel — installed into a clean
environment and driven through the documented quickstart. Packaging, CLI and
documentation only. The matching pipeline is untouched and no measured number in this
file moves.

### Fixed

- **`match` and `sync` crashed with `UnicodeEncodeError` on non-UTF-8 Windows consoles.**
  Rich's default spinner animates with Braille code points that cp437, cp850 and cp1252
  all refuse to encode, and its `no_wrap` truncation adds U+2026, so the only two commands
  that do real work died with a bare codec error and exit 1 — after the matching had been
  paid for, and with nothing to suggest that a terminal or a field name was the problem.
  The CLI now picks decorations the console can encode and escapes what is left, keeping
  the user's code page rather than forcing UTF-8 onto it.
- **The `nexus-matcher` console script was installed without typer or rich.** The entry
  point is declared unconditionally but its dependencies sat in the `cli` extra, so a
  plain `pip install nexus-matcher` put a command on `PATH` that could not start. Both
  moved into the core dependencies; `cli` is kept as a name so existing pins still
  resolve. Its `typer[all]` marker is also gone — typer has published no extras since
  0.12, so that asked for something which does not exist and pip warned about it on every
  install.
- **`--output/-o` was silently ignored when `--format` was left at its default.** The
  default was `table` and the table branch had no write path, so
  `nexus-matcher match schema.avsc -d dictionary.csv -o results.json` printed to stdout,
  wrote no file and exited 0 — a scripted run could not tell that it had produced nothing.
  The format is now inferred from the `--output` extension when it is not given.
- **`match_schema` silently dropped fields whose dotted paths collided.** Results are
  keyed by field path, and where two fields produced the same path the later one replaced
  the earlier. The displaced field then never appeared in the results and so never
  received the governance classification of the entry it would have matched; the returned
  mapping was simply shorter than the schema, with nothing to say which fields were gone.
- **Flattened Avro results were keyed by names the caller had not supplied.** The keys in
  the returned mapping did not match the flattened field names passed in, so looking a
  result up by the name you provided missed.
- **`create_app` was listed in `__all__` but needs the `api` extra.** `from nexus_matcher
  import *` therefore raised `ImportError` on any install without `[api]`, including the
  bare install the README documents as the complete pipeline.
- **The `Documentation` project URL 404'd, and the README's relative links were dead on
  PyPI.** README.md is the PyPI `long_description`, where a relative markdown link
  resolves against pypi.org rather than against the repository: twelve of them — including
  every entry in the README's own Documentation section — went nowhere for anyone arriving
  from the package page. They are absolute GitHub URLs now. One dead in-page anchor went
  with them; `#known-limits` named no heading in the file and scrolled nowhere.

---

## [2.0.0] - 2025-12-09

Complete rewrite from a procedural single-file implementation to a hexagonal
(ports and adapters) architecture.

> **The performance table originally published with this release was not valid.** See
> the retractions above. The entries below have had unsupported numbers removed.

### Added

- Hexagonal architecture: domain / application / infrastructure / presentation layers,
  dependency-injection container, plugin system via entry points.
- Three-stage matching pipeline: retrieval → optional reranking → multi-signal scoring
  with a decision policy (`AUTO_APPROVE` / `REVIEW` / `REJECT`).
- Schema parsers: Avro, JSON Schema, SQL DDL.
- Dictionary loaders: Excel, CSV.
- Vector stores: in-memory, Qdrant, HNSW.
- Sparse retrieval: BM25.
- Rerankers: cross-encoder and ColBERT MaxSim. Both optional, off by default.
- Caches: L1 LRU in-memory, Redis, content-addressed semantic cache. Implemented and
  unit-tested; not exercised by the accuracy benchmark.
- Incremental update manager with BLAKE3 content hashing.
- Learned type projections and a graph matcher. Experimental.
- INT8 quantized embedding provider via ONNX Runtime.
- REST API with health and readiness probes; CLI with Typer; Python library API.

### Changed

- Configuration moved to YAML with environment-variable overrides — **note that this
  affects logging only; the matching pipeline does not read it.**
- Test suite expanded substantially.

### Removed

- Legacy single-file implementation.
- OpenAI embedding provider.

### Security

- Non-root Docker container.
- CORS is currently configured with `allow_origins=["*"]`; narrow it before exposing the
  service.

---

## [1.0.0] - 2025-10-15

- Initial release: basic semantic schema matching, Excel dictionary loader, Avro parser,
  sentence-transformers embeddings, cosine similarity scoring.
- Single-threaded, no caching. No labelled benchmark existed at this point, so the
  accuracy of this release is unknown; earlier claims of "~85%" are unsupported.

---

## Upgrade guide: 1.x → 2.x

**Import paths changed.**

```python
# Old
from schema_matcher import match_schema

# New
from nexus_matcher import NexusMatcher
matcher = NexusMatcher.from_config()      # NOT NexusMatcher()
matcher.load_dictionary("dictionary.csv")
results = matcher.match_schema("schema.avsc")
```

`NexusMatcher()` with no arguments raises `TypeError` — `embedding_provider` and
`vector_store` are required. `from_config()` supplies the defaults.

**Result structure changed.**

```python
# Old
result = {"field": "matched_entry", "score": 0.95}

# New
results: dict[str, tuple[MatchResult, ...]]     # keyed by SchemaField.full_path
top = results["Customer.email"][0]
top.dictionary_entry.business_name
top.final_confidence          # float in [0, 1]
top.decision                  # MatchDecision enum (str-backed)
top.score_breakdown           # per-signal components
```

---

## Links

- [Repository](https://github.com/pierce-lonergan/nexus_matcher)
- [Issues](https://github.com/pierce-lonergan/nexus_matcher/issues)
- [Benchmark registry](docs/BENCHMARK_REGISTRY.md)
