# NexusMatcher API Reference

This document describes the interfaces that **exist in the code**. Every endpoint,
method and CLI flag below was verified against `src/nexus_matcher/` by enumerating the
live FastAPI route table and the Typer command table.

Contents:

1. [Python API](#python-api) — the primary and only benchmarked interface
2. [CLI](#cli)
3. [REST API](#rest-api) — matching, feedback, health and introspection
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
| `match_schema(schema_source, schema_format=None)` | `dict[str, tuple[MatchResult, ...]]` | Keyed by the caller's own field identity: `source_metadata['flattened_name']` when the parser set one (flattened Avro), otherwise `full_path`. Keys are unique, and every input field appears exactly once. Raises `RuntimeError` if no dictionary is loaded. |
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
| POST | `/api/v1/match` | `MatchResponseView` — `results`, one entry per input field, keyed by the caller's own `path`, in the order sent. Field cap `NEXUS_API_MAX_FIELDS` (default 100). |
| POST | `/api/v1/match/batch` | Identical contract and one shared implementation; the only difference is the cap, `NEXUS_API_MAX_BATCH_FIELDS` (default 250). |
| POST | `/api/v1/feedback` | **201** and `FeedbackResponseView` — the stored record echoed back, server `receivedAt` included. Appended to `NEXUS_API_FEEDBACK_PATH`; **503** when that is unset, **422** on a malformed record, **500** when the append itself fails. |
| GET | `/` | `{"service": "nexus-matcher", "version": …, "docs": "/docs"}`. `version` is the installed package's `__version__`, resolved at startup — this document used to print a literal here, and a literal is a second copy that drifts. |
| GET | `/health` | `HealthResponse` — `status`, `timestamp`, `version`, `checks.uptime_seconds`. `status` is `healthy` unless a registered component is unhealthy, then `degraded`. |
| GET | `/health/live` | `{"status": "alive"}` — Kubernetes liveness probe |
| GET | `/health/ready` | `ReadinessResponse` — `ready`, `timestamp`, `components`. Returns **503** if any registered component is not ready. |
| GET | `/health/startup` | `{"status": "started", "startup_time": ...}`. Returns **503** while starting. |
| GET | `/docs`, `/redoc`, `/openapi.json` | Generated OpenAPI documentation |

### The matching request body

`FieldSpec` sets `extra="forbid"`, so an unrecognised key is a **422** rather than a
silently dropped input — a misspelled `documentation` would drop the column comment, and
the column comment is real retrieval signal.

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | The column's own name. |
| `path` | no | The caller's identifier, and the key the response is keyed by. Defaults to `name`. A **dotted** path is strongly preferred: the segment before the last dot becomes the query's parent context, the single largest accuracy factor measured on this task. |
| `doc` | no | Column comment or description. |
| `type` | no | Source type name, normalised server-side. Unknown types are accepted. |

Plus `top_k` (default 5; a value above the server's `results_per_field` is a 422 naming
the cap) and `explain` (default false).

Worked request, response and failure modes: [GOVERNANCE.md](GOVERNANCE.md#matching-over-http).

Both match routes answer **503** until a dictionary is loaded — see
[DEPLOYMENT.md](DEPLOYMENT.md#2-environment-configuration) for the variables that load one.

### Published failure responses

Enumerated from a live `/openapi.json`, not from this document's memory: **15 non-2xx
responses across 5 routes**, every one of them `ErrorResponse`. So a generated client gets
the same typed DTO — `{"error": {"code", "message", "details"}}` — on every documented
failure, and never a bare map on some of them.

The two match routes publish an identical set; they are one implementation behind two field
caps.

| Route | Status | Code | When |
|---|---|---|---|
| `POST /api/v1/match`, `POST /api/v1/match/batch` | **413** | `NEXUS-8004` | Two different refusals under one status. More fields than the route's cap: `details.fields` and `details.limit`. Or a raw body over the server's byte cap, answered by middleware *before* the body is parsed: `details.limit_bytes`, `details.observed_bytes` and `details.source` (`content-length` or `stream`). |
| `POST /api/v1/match`, `POST /api/v1/match/batch` | **422** | `NEXUS-8004` | Malformed request; `details.violations` names the offending field. |
| `POST /api/v1/match`, `POST /api/v1/match/batch` | **500** | `NEXUS-6000`, `NEXUS-1003` | The matcher failed (`NEXUS-6000`, `details.cause`), this layer refused a response whose field count or keys did not conserve (`NEXUS-6000`, `details.fields_in` and `details.results_out`), or the matcher object has drifted from what this surface calls (`NEXUS-1003`). No field was classified — treat the request as unanswered, not as a partial answer. |
| `POST /api/v1/match`, `POST /api/v1/match/batch` | **503** | `NEXUS-1002`, `NEXUS-8000` | No dictionary loaded (`NEXUS-1002`), or the request was shed by admission control with the pool already full (`NEXUS-8000`, `details.capacity` and `details.in_flight`). |
| `POST /api/v1/match`, `POST /api/v1/match/batch` | **504** | `NEXUS-6002` | `NEXUS_API_DEADLINE_SECONDS` (default 25.0) fired before matching finished; `details.deadline_seconds`. The work may still be running — the deadline promises a response, not a stop. |
| `POST /api/v1/feedback` | **422** | `NEXUS-8004` | Malformed record. |
| `POST /api/v1/feedback` | **500** | `NEXUS-6000` | The append failed; `details.cause`. **Nothing was recorded** — do not treat the verdict as filed. |
| `POST /api/v1/feedback` | **503** | `NEXUS-1002` | `NEXUS_API_FEEDBACK_PATH` is unset. |
| `GET /health/ready` | **503** | `NEXUS-8000` | Not ready; `details.components` names the component that is red. |
| `GET /health/startup` | **503** | `NEXUS-8000` | Startup has not completed; no component has reported in yet. |

**413 and 504 are the two a client author cannot guess**, which is why they were the
expensive omission: a Java client generated from a spec that published neither has no branch
for the status the chunking path depends on, and none for the one that ends a long request.

`GET /health` and `GET /health/live` publish no failure at all, because neither can fail:
`/health` answers 200 with `status: "degraded"` when a component is red, and `/health/live`
returns a constant. Do not point a rollout gate at either — see the readiness paragraph
below.

**404 and 405 are answered but not published**, and they cannot be: they are raised by the
router, so no path object in the spec owns them. They carry the same `{"error": {…}}`
envelope as everything above, and the 405 keeps its `Allow` header.

Middleware and behaviour that does exist:

- Request-ID middleware. Reads `X-Request-ID` or generates one; echoes `X-Request-ID`
  and `X-Response-Time-Ms` on every response — verified on 200, 201, 404, 405, 413, 422,
  500, 503 and 504. The 413 is answered by the body-size middleware, which sits *outside*
  this one, so it stamps the two headers itself and hands its id down in the ASGI scope;
  that is what makes the client's 413 and the server's log line for the same request
  joinable on one id.
- CORS, only when `NEXUS_API_CORS_ORIGINS` names the origins that may use a browser.
  Empty is the default and mounts no `CORSMiddleware` at all; methods are narrowed to
  `GET`, `POST`, `OPTIONS`; `NEXUS_API_CORS_ALLOW_CREDENTIALS=true` alongside `*` is
  refused at startup rather than quietly reflecting the caller's own origin.
- Exception handlers mapping `NexusMatcherError` to its status code and any unhandled
  exception to a 500 with error code `NEXUS-1000`. Two more join them so that one service
  answers in one shape: a `StarletteHTTPException` handler renders 404, 405 and the health
  503s (the 405 keeps its `Allow: POST`), and a `RequestValidationError` handler replaces
  FastAPI's `{"detail": [...]}`. Every failure body is `{"error": {code, message,
  details}}` — verified: 404, 405 and 422 all have exactly the top-level key `error`.

The `components` map reported by `/health/ready` carries `api`, `config` and `matcher`.
`api` and `config` are still hardcoded `True`. `matcher` is a real check: it is `False`
when no dictionary loaded, which is the misconfiguration a rollout is most likely to
produce, and the map is repeated in the 503 body so an operator can see which component is
red rather than only that something is. Under `NEXUS_API_MATCHING_OPTIONAL` it disappears
from the map when no dictionary was configured, and survives as a non-gating `False` when
one *was* named and failed to load — the opt-out excuses a deployment that wants no
matcher, not one whose matcher is broken.

`vector_store` and `cache` were removed from the map. They were set `True` inside `try`
blocks whose bodies were comments, so they could never be anything else, and a component
that cannot fail is a claim rather than a check. Nothing here probes a vector store or a
cache connection today.

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
| `POST /match` | The path is `/api/v1/match` (see the route table above). This row previously denied the endpoint; that denial is retracted. |
| `POST /batch` | The path is `/api/v1/match/batch`. `BatchProcessor` remains the in-process route. |
| `GET/POST/PUT/DELETE /dictionary`, `GET /dictionary/{id}` | No dictionary CRUD endpoints. |
| `POST /cache/clear`, `GET /cache/stats` | No cache endpoints. |
| `GET /metrics` (Prometheus) | Not routed. A `PrometheusMetrics` backend class exists in `nexus_matcher.shared.metrics`, but no endpoint exposes it. |
| `GET /ready` | The path is `/health/ready`. |
| API key auth via `NEXUS_API_KEY` / `X-API-Key` | No authentication dependency is attached to any route, and `/openapi.json` carries no `securitySchemes`. The description used to offer the header and name the variable; it now says the service ships unauthenticated and repeats neither, so nobody sends a credential believing it is checked. |
| Rate limiting | No rate-limiting middleware is installed. |
| `nexus-matcher batch-match` | Not a CLI command. The commands are `match`, `sync`, `api`, `info`. |
| `NexusMatcher(config_path="config.yaml")` | Not a valid constructor call. |
| `matcher.match_field(...)`, `matcher.precompute_embeddings()`, `matcher.set_type_projection_manager(...)` | Not methods on `NexusMatcher`. |
| Loading a dictionary from a `postgresql://` URL or a JSON file | Only Excel and CSV loaders are registered. |
