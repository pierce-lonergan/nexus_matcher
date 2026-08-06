# NexusMatcher API Reference

This document describes the interfaces that **exist in the code**. Every endpoint,
method and CLI flag below was verified against `src/nexus_matcher/` by enumerating the
live FastAPI route table and the Typer command table.

Contents:

1. [Python API](#python-api) — the primary and only benchmarked interface
2. [CLI](#cli)
3. [REST API](#rest-api) — health and introspection only
4. [Domain types](#domain-types)
5. [Not implemented](#not-implemented)

---

## Python API

### `NexusMatcher`

`nexus_matcher.application.use_cases.match_schema.NexusMatcher`, re-exported lazily as
`nexus_matcher.NexusMatcher`.

#### Construction

```python
NexusMatcher(
    embedding_provider,                    # required
    vector_store,                          # required
    sparse_retriever=None,
    reranker=None,
    schema_parser_registry=None,           # dict[str, SchemaParser]
    dictionary_loader_registry=None,       # dict[str, DictionaryLoader]
    abbreviation_expander=None,
    context_enricher=None,
    domain_matcher=None,
    config=None,                           # MatchingConfig
)
```

`embedding_provider` and `vector_store` are positional-or-keyword and **required**.
`NexusMatcher()` with no arguments raises `TypeError`.

```python
NexusMatcher.from_config(config: MatchingConfig | str | Path | None = None) -> NexusMatcher
```

Returns a matcher wired with the **bundled int8 ONNX encoder** (falling back to
sentence-transformers when the `embeddings` extra is installed, then to static
embeddings), `InMemoryVectorStore`, `BM25Retriever`, the Avro and flattened-Avro parsers,
and the Excel + CSV dictionary loaders.

`config` accepts a `MatchingConfig`, a path to a JSON or TOML file holding its fields, or
`None` for the calibrated defaults. A file may wrap the fields in a `[matching]` table.

> **An unknown key raises `ValueError`.** Every field in `MatchingConfig` is a measured
> number; silently discarding a mistyped `auto_approve_treshold` would leave the caller
> believing they had raised the auto-approve bar while the matcher kept approving at
> 0.87. A missing file raises `FileNotFoundError`.
>
> The parameter was previously named `config_path` and was accepted and then never read.
> The `NEXUS_*` settings classes in `nexus_matcher.infrastructure.config.settings` remain
> consumed only by `nexus_matcher.shared.logging`.

#### Methods

| Method | Returns | Notes |
|---|---|---|
| `load_dictionary(source, column_mapping=None, source_type=None)` | `LoadStatistics` | Loads **and indexes**. Loader auto-detected from the file extension against the registry. Raises `ValueError` if no loader matches. Re-loading replaces the previous index. |
| `match_schema(schema_source, schema_format=None)` | `dict[str, tuple[MatchResult, ...]]` | Keyed by `SchemaField.full_path`. Raises `RuntimeError` if no dictionary is loaded. |
| `match_schema_session(schema_source, schema_format=None)` | `MatchingSession` | Same matching, plus the parsed `Schema` and timing metadata. |

| Property | Returns |
|---|---|
| `dictionary_size` | `int` |
| `is_ready` | `bool` |

There is no public method to register a parser or loader on an existing matcher; pass
`schema_parser_registry` / `dictionary_loader_registry` at construction time.

#### `MatchingConfig`

Frozen dataclass, `nexus_matcher.application.use_cases.match_schema.MatchingConfig`.

| Field | Default | Meaning |
|---|---|---|
| `dense_top_k` | 100 | Dense candidates retrieved |
| `sparse_top_k` | 100 | BM25 candidates retrieved |
| `fusion_alpha` | 0.90 | Dense weight in linear min-max fusion. Measured optimum — see `benchmarks/results/exp_fusion_combined.json` |
| `colbert_top_k` | 50 | Candidates passed to a ColBERT reranker, if one is supplied |
| `cross_encoder_top_k` | 20 | Candidates passed to a cross-encoder reranker, if one is supplied |
| `semantic_weight` | 0.70 | Confidence weights; sum to 1.0 |
| `lexical_weight` | 0.05 | |
| `edit_distance_weight` | 0.05 | |
| `type_weight` | 0.05 | |
| `domain_weight` | 0.15 | |
| `auto_approve_threshold` | 0.85 | Calibrated — `benchmarks/results/exp_calibration_combined.json` |
| `review_threshold` | 0.50 | Below this, `REJECT` |
| `min_confidence_gap` | 0.10 | Minimum margin over the runner-up required to auto-approve |
| `results_per_field` | 5 | Matches returned per field |

#### `BatchProcessor`

`nexus_matcher.application.use_cases.batch_match.BatchProcessor`

```python
BatchProcessor(matcher, config=None)   # config: BatchConfig

processor.process_schemas(schema_paths, **options) -> BatchResult
processor.process_directory(...)       -> BatchResult
processor.process_manifest(...)        -> BatchResult
```

`BatchConfig` fields: `max_workers` (4), `chunk_size` (10), `fail_fast` (False),
`max_errors` (100), `progress_callback`, plus checkpoint options.
`BatchResult` exposes `sessions`, `errors`, `total_schemas`, `successful`, `failed`,
`total_duration_ms`, `avg_duration_ms`, `success_rate`.

Parallelism uses a `ThreadPoolExecutor`, so it parallelises I/O and parsing; the
embedding model itself is shared.

---

## CLI

Entry point `nexus-matcher` → `nexus_matcher.presentation.cli.main:app` (Typer).

Global options: `--version` / `-v`, `--install-completion`, `--show-completion`, `--help`.

### `match`

```
nexus-matcher match SCHEMA -d DICTIONARY [options]
```

| Option | Default | Meaning |
|---|---|---|
| `SCHEMA` (arg, required) | — | Schema file path |
| `--dictionary` / `-d` (required) | — | Dictionary file (Excel, CSV) |
| `--output` / `-o` | — | Write results to a file |
| `--format` / `-f` | `table` | `json`, `csv`, or `table` |
| `--top-k` / `-k` | 5 | Matches per field, 1–20 |
| `--threshold` / `-t` | 0.0 | Minimum confidence to display, 0.0–1.0 |
| `--verbose` / `-V` | off | Detailed output and full tracebacks |

### `sync`

Sync a dictionary to a vector store.

### `api`

Start the REST API server (see below for what it serves).

### `info`

Print system information and configuration.

> **Windows note.** Table output and the progress spinner use Unicode box-drawing and
> Braille characters. In a console using the legacy code page this raises
> `'charmap' codec can't encode character`. Set `PYTHONIOENCODING=utf-8`.

---

## REST API

Factory: `nexus_matcher.presentation.api.app.create_app()`. A module-level `app` instance
is also created for `uvicorn nexus_matcher.presentation.api.app:app`.

**The complete route table**, enumerated from a live `create_app()`:

| Method | Path | Response |
|---|---|---|
| GET | `/` | `{"service": "nexus-matcher", "version": "2.0.0", "docs": "/docs"}` |
| GET | `/health` | `HealthResponse` — `status`, `timestamp`, `version`, `checks.uptime_seconds`. `status` is `healthy` unless a registered component is unhealthy, then `degraded`. |
| GET | `/health/live` | `{"status": "alive"}` — Kubernetes liveness probe |
| GET | `/health/ready` | `ReadinessResponse` — `ready`, `timestamp`, `components`. Returns **503** if any registered component is not ready. |
| GET | `/health/startup` | `{"status": "started", "startup_time": ...}`. Returns **503** while starting. |
| GET | `/docs`, `/redoc`, `/openapi.json` | Generated OpenAPI documentation |

Middleware and behaviour that does exist:

- Request-ID middleware. Reads `X-Request-ID` or generates one; echoes `X-Request-ID`
  and `X-Response-Time-Ms` on every response.
- CORS, currently `allow_origins=["*"]` — narrow this before exposing the service.
- Exception handlers mapping `NexusMatcherError` to its status code and everything else
  to a 500 with error code `NEXUS-1000`.

The `components` map reported by `/health/ready` is populated in the lifespan handler
with hardcoded `True` values for `api`, `config`, `vector_store` and `cache`. It does
**not** currently probe a real vector store or cache connection — the code paths that
would do so are empty `try` blocks. Treat readiness as "the process started", not "the
dependencies are reachable".

---

## Domain types

### `SchemaField`

`name`, `data_type`, `full_path`, `parent_path`, `description`, `is_nullable`,
`is_array`, `array_item_type`, `default_value`, `constraints`, `metadata`.

`parent_path` matters: it is the hierarchical context injected into the retrieval query,
and it is the single largest accuracy factor measured on the benchmark
(+20 points of P@1 — `benchmarks/results/exp_query_repr_combined.json`).

### `DictionaryEntry`

`id`, `business_name`, `logical_name`, `definition`, `data_type`, `protection_level`,
`domain`, `parent_table`, `sample_values`, `synonyms`, `is_enum`, `enum_values`,
`source_metadata`.

`id` and `business_name` must be non-empty. `to_searchable_text()` concatenates
business name, logical name (underscores replaced), definition, and synonyms — that
string is what gets embedded and BM25-indexed.

There is no `technical_name` attribute; the field is called `logical_name`.

### `MatchResult`

| Attribute | Type |
|---|---|
| `schema_field` | `SchemaField` |
| `dictionary_entry` | `DictionaryEntry` |
| `rank` | `int`, 1-based |
| `final_confidence` | `float` in [0, 1] |
| `score_breakdown` | `ScoreBreakdown` |
| `decision` | `MatchDecision` enum |
| `performance` | `PerformanceMetrics` |

Convenience properties: `is_auto_approved`, `needs_review`, `is_rejected`.

`MatchDecision` is a `str`-backed enum, so `result.decision == "AUTO_APPROVE"`,
`result.decision.value` and `result.decision.name` all work. Prefer comparing against
`MatchDecision.AUTO_APPROVE`.

### `ScoreBreakdown`

`semantic_score`, `lexical_score`, `edit_distance_score`, `type_compatibility_score`,
`domain_score`, `graph_boost`, plus optional `colbert_score` and `cross_encoder_score`
(both `None` when no reranker is configured). Property `has_reranking`.

### `PerformanceMetrics`

`latency_ms`, `cache_hit`, `retrieval_stage`, `candidates_evaluated`,
`reranking_applied`.

---

## Plugin entry points

Declared in `pyproject.toml`. Each target module was verified importable:

| Group | Names |
|---|---|
| `nexus_matcher.schema_parsers` | `avro`, `json_schema`, `sql_ddl` |
| `nexus_matcher.dictionary_loaders` | `excel`, `csv` (both from `…dictionary_loaders.excel`) |
| `nexus_matcher.vector_stores` | `qdrant`, `memory` |
| `nexus_matcher.embedding_providers` | `sentence_transformers` |

Earlier revisions of `pyproject.toml` also declared `csv_headers`, `database`, `faiss`
and `openai` entry points pointing at modules that do not exist. Those have been removed.
If you installed the package before that cleanup, the stale entry points survive in the
installed `.dist-info` metadata and will raise `ModuleNotFoundError` during plugin
discovery — reinstall (`pip install -e .`) to regenerate it.

---

## Not implemented

Previous revisions of this document described the following. **None of it exists.**
It is listed here so that nobody re-derives it from an old copy.

| Documented previously | Reality |
|---|---|
| `POST /match` | No matching endpoint exists. Matching over HTTP is not implemented. |
| `POST /batch` | Not implemented. Use `BatchProcessor` in-process. |
| `GET/POST/PUT/DELETE /dictionary`, `GET /dictionary/{id}` | No dictionary CRUD endpoints. |
| `POST /cache/clear`, `GET /cache/stats` | No cache endpoints. |
| `GET /metrics` (Prometheus) | Not routed. A `PrometheusMetrics` backend class exists in `nexus_matcher.shared.metrics`, but no endpoint exposes it. |
| `GET /ready` | The path is `/health/ready`. |
| API key auth via `NEXUS_API_KEY` / `X-API-Key` | No authentication dependency is attached to any route. The OpenAPI description text mentions it; the code does not implement it. |
| Rate limiting | No rate-limiting middleware is installed. |
| `nexus-matcher batch-match` | Not a CLI command. The commands are `match`, `sync`, `api`, `info`. |
| `NexusMatcher(config_path="config.yaml")` | Not a valid constructor call. |
| `matcher.match_field(...)`, `matcher.precompute_embeddings()`, `matcher.set_type_projection_manager(...)` | Not methods on `NexusMatcher`. |
| Loading a dictionary from a `postgresql://` URL or a JSON file | Only Excel and CSV loaders are registered. |
