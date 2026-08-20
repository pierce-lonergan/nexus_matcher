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
    abbreviation_expander=None,            # AbbreviationExpander
    context_enricher=None,
    domain_matcher=None,
    config=None,                           # MatchingConfig
    governance=None,                       # GovernanceVocabulary | str | Path
    feedback_consumer=None,                # FeedbackConsumer
)
```

`embedding_provider` and `vector_store` are positional-or-keyword and **required**.
`NexusMatcher()` with no arguments raises `TypeError`.

> **`feedback_consumer` is opt-in and the shipped default consumes nothing.** It takes a
> `nexus_matcher.domain.ports.review_feedback.FeedbackConsumer` — an object that is
> allowed to answer for a field *before* retrieval runs, from a reviewer's recorded
> verdicts. Nothing in this package constructs one: `from_config()` does not take it and
> `create_app()` does not build it, so the only way to attach one is to pass it here. With
> `None` the matcher is bound to no consumer and every field is matched by retrieval
> exactly as it was before the parameter existed.
>
> It is *read*, not merely accepted — `load_dictionary()` binds the consumer to the
> freshly indexed entries on every index, and `match_schema()` consults it per field. That
> distinction is the whole reason this block is gated:
> `tests/packaging/test_documented_construction.py` compares this signature against the
> real `__init__`, because `config_path` was once accepted here and never read.

```python
NexusMatcher.from_config(
    config: MatchingConfig | str | Path | None = None,
    governance: GovernanceVocabulary | str | Path | None = None,
) -> NexusMatcher
```

Returns a matcher wired with the **bundled int8 ONNX encoder** (falling back to
sentence-transformers when the `embeddings` extra is installed, then to static
embeddings), `InMemoryVectorStore`, `BM25Retriever`, the Avro and flattened-Avro parsers,
and the Excel + CSV dictionary loaders.

> **`from_config` does not take `abbreviation_expander`.** It builds one from the bundled
> generic dictionary and there is no setter afterwards, so supplying your own
> approved-abbreviation catalog requires the full constructor above. See
> [governed abbreviations](guides/governed_abbreviations.md).

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
| `load_dictionary(source, column_mapping=None, source_type=None, governance_strict=True)` | `LoadStatistics` | Loads **and indexes**, reading the source *through* the vocabulary given to `from_config(governance=…)`. Loader auto-detected from the file extension. Re-loading replaces the previous index. Raises `ValueError` if no loader matches, if the source's governance is defective, or if the source has a protection-code column and no vocabulary was configured to read it — pass `governance_strict=False` to load anyway and treat that column as plain metadata. |
| `match_schema(schema_source, schema_format=None)` | `dict[str, tuple[MatchResult, ...]]` | Keyed by the caller's own field identity: `source_metadata['flattened_name']` when the parser set one (flattened Avro), otherwise `full_path`. Keys are unique, and every input field appears exactly once. Raises `RuntimeError` if no dictionary is loaded. |
| `match_schema_session(schema_source, schema_format=None)` | `MatchingSession` | Same matching, plus the parsed `Schema` and timing metadata. **Also the only way to get the per-field verdicts from Python**: `session.field_decisions()` returns `dict[str, FieldDecision]`, one entry per field in schema order, and it is a *method*, not a property. `match_schema` returns candidates only. |

| Property | Returns |
|---|---|
| `dictionary_size` | `int` |
| `is_ready` | `bool` |
| `absolute_score_floor` | `float` or `None` — the configured floor, `None` when off. The same number the service publishes as `scoring.absoluteScoreFloor`. |
| `absolute_score_metric` | `str` — what produced `ScoreBreakdown.absolute_cosine`; `"cosine"` under the shipped wiring, `"unknown"` when the store does not declare one. |
| `minimum_achievable_confidence` | `float` or `None` — the structural floor of `final_confidence`, `None` when a reranker makes the derivation unsound. |

There is no public method to register a parser or loader on an existing matcher; pass
`schema_parser_registry` / `dictionary_loader_registry` at construction time.

#### `MatchingConfig`

Frozen dataclass, `nexus_matcher.application.use_cases.match_schema.MatchingConfig`.

| Field | Default | Meaning |
|---|---|---|
| `dense_top_k` | 100 | Dense candidates retrieved |
| `sparse_top_k` | 100 | BM25 candidates retrieved |
| `fusion_alpha` | 0.90 | Dense weight in linear min-max fusion. Measured optimum — but the artifact behind it (`exp_fusion_combined.json`) is the only A-grade result **not re-run since the benchmark leakage fix**, so it is a 793-pair pre-fix measurement. The ordering it establishes (linear beats RRF, decisively) is robust; the exact optimum between 0.8 and 0.9 is a 1.6-point margin on a corpus that no longer exists. See [BENCHMARK_REGISTRY.md](BENCHMARK_REGISTRY.md#exp-fusion--combining-dense-and-sparse). |
| `colbert_top_k` | 50 | Candidates passed to a ColBERT reranker, if one is supplied |
| `cross_encoder_top_k` | 20 | Candidates passed to a cross-encoder reranker, if one is supplied |
| `semantic_weight` | 0.70 | Confidence weights; sum to 1.0 |
| `lexical_weight` | 0.05 | |
| `edit_distance_weight` | 0.05 | |
| `type_weight` | 0.05 | |
| `domain_weight` | 0.15 | |
| `auto_approve_threshold` | 0.87 | Calibrated — `benchmarks/results/exp_calibration_combined.json`, where 0.87 is the point that holds auto-approve precision at 0.95 (coverage 0.12). An earlier revision of this table printed 0.85; that is the 0.90-precision point on the same curve, and it was never the shipped default. |
| `review_threshold` | 0.50 | Below this, `REJECT` |
| `min_confidence_gap` | 0.10 | Minimum margin over the runner-up required to auto-approve |
| `absolute_score_floor` | `None` | The rank-1 `absolute_cosine` beneath which a field is reported `NO_MATCH`. `None` means **off**, and it is the only default a library with no view of the caller's corpus may ship — `0.0` means on with a floor at zero, which is a different thing. It is compared against the raw retrieval score, not against `final_confidence`, because `final_confidence` has a structural floor of `semantic_weight × fusion_alpha` = 0.63 that sits above `review_threshold` and makes "nothing matched" inexpressible. Measuring one for your own corpus: [Calibrating the absolute score floor](guides/absolute_score_floor.md). |
| `results_per_field` | 5 | Matches returned per field |
| `expand_query_abbreviations` | `False` | Query-side abbreviation expansion. Off because the evidence does not support turning it on — **not** because it is proven harmful. The "-2.0 points" an earlier revision of this table printed does not survive a paired test; re-measured on the full 688-pair corpus it is **-1.60 points, exact McNemar p = 0.099**: inconclusive, point estimate negative. The mechanism is the bundled *dictionary*, not the idea — it asserts wrong long forms on short tokens (`st` → state, `no` → number). Supply your own approved catalog and this flag becomes the largest lever in the pipeline: see [governed abbreviations](guides/governed_abbreviations.md). |
| `dictionary_alias_count` | 0 | Dictionary-side alias generation. Off because the gain inverts with corpus size — +1.9 at 688 entries, **-13.7 at 10k and -18.8 at 30k**. Never enable without re-measuring at your own corpus size. |

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
| POST | `/api/v1/match` | `MatchResponseView` — four top-level keys in this order: `results` (one entry per input field, keyed by the caller's own `path`, in the order sent), `vocabulary`, `fieldDecisions` (one verdict per column, the only place `NO_MATCH` is expressible) and `scoring`, plus `contrast` and `consistency` **appended when the request asks for them**. Each candidate carries `absoluteScore` beside `confidence`. Every key is spelled out under [The matching response](#the-matching-response) rather than restated here. Field cap `NEXUS_API_MAX_FIELDS` (default 100). |
| POST | `/api/v1/match/batch` | Identical contract and one shared implementation; the only difference is the cap, `NEXUS_API_MAX_BATCH_FIELDS` (default 250). |
| POST | `/api/v1/feedback` | **201** and `FeedbackResponseView` — the stored record echoed back, server `receivedAt` included. Appended to `NEXUS_API_FEEDBACK_PATH`; **503** when that is unset, **422** on a malformed record, **500** when the append itself fails. |
| POST | `/api/v1/lookup` | `LookupResponseView` — `results` maps every id you sent, once and in the order sent, to an entry or an explicit `null`; `missing` names exactly the nulls; `vocabulary` is the same block `/match` carries. No score, no rank, no decision. Id cap `NEXUS_API_MAX_BATCH_FIELDS`; **413** over it, **422** on a duplicate, blank or oversized id, **503** with no dictionary. |
| GET | `/api/v1/lookup/{governance_id:path}` | The single-id form, answering the identical `LookupResponseView` under one key. A miss is **200** with `null` — 404 on this service means no such route. The `:path` converter is what keeps an id containing `/` addressable; OpenAPI publishes the path as `/api/v1/lookup/{governance_id}`. |
| GET | `/api/v1/status` | `StatusResponseView` — `ready`, `degraded`, `warnings[]`, `dictionary`, `encoder`, `thresholds`, `limits`, `calibration`. Always **200**, including with no dictionary loaded, because a pre-run degradation check that refused when the condition holds is unusable exactly when it is needed. Every threshold is nullable: `null` means this matcher does not expose it, never `0.0` — except `absoluteScoreFloor`, where `null` means no floor is configured. `calibration` says which profile is in force and what the shipped one was fitted on. |
| POST | `/api/v1/diag/retrieval` | `RetrievalDiagnosticView` — `queryText`, the `dense`/`sparse`/`fused` channels with their own raw scores and full result counts, `rerankerWired`, and, when `expected_governance_id` is given, that entry's rank per channel plus `inDictionary`. Retrieval only: it does not reproduce scoring, the decision, or reranking. Scores are each channel's own number and are **not** comparable across channels. |
| GET | `/` | `{"service": "nexus-matcher", "version": …, "docs": "/docs"}`. `version` is the installed package's `__version__`, resolved at startup — this document used to print a literal here, and a literal is a second copy that drifts. |
| GET | `/health` | `HealthResponse` — `status`, `timestamp`, `version`, `checks.uptime_seconds`. `status` is `healthy` unless a registered component is unhealthy, then `degraded`. |
| GET | `/health/live` | `{"status": "alive"}` — Kubernetes liveness probe |
| GET | `/health/ready` | `ReadinessResponse` — `ready`, `timestamp`, `components`. Returns **503** if any registered component is not ready. |
| GET | `/health/startup` | `{"status": "started", "startup_time": ...}`. Returns **503** while starting. |
| GET | `/docs`, `/redoc`, `/openapi.json` | Generated OpenAPI documentation |

### The matching request body

The two levels have deliberately different strictness, and the difference is the contract:

- **Inside a field, unknown keys are refused.** `FieldSpec` sets `extra="forbid"`, so an
  unrecognised key is a **422** rather than a silently dropped input — a misspelled
  `documentation` would drop the column comment, and the column comment is real retrieval
  signal. A field that matched worse because a key was ignored is a defect you would never
  trace back to a typo.
- **At the top level, unknown keys are accepted.** `MatchRequest` sets `extra="ignore"`,
  so a caller sending an envelope key a newer server understands is not rejected by an
  older one. Forbidding them made the contract non-extensible in both directions at once,
  which is a poor trade for a strictness that catches nothing: an envelope typo does not
  silently degrade a match the way a field typo does.

A misspelled `fields` is still an error — it is `missing`, not `extra`.

#### Every `path` in one request must be distinct

**This is a client obligation and the failure is a 422 for the whole batch.** The response
is a map keyed by your `path`, and a map cannot hold two entries for one key — collapsing
them silently would hand one of your columns another column's governance and hand you a
shorter map with nothing saying which answer was dropped.

So a repeated `path` is refused, and the refusal **names the offenders**: they are in
`error.details.duplicate_paths` as a sorted list for a program, and in `error.message` for
whoever is reading the log. That matters most in exactly the case that produces it — a
200-field chunk out of a flattening step, where leaf names repeat across records, and where
a 422 that said only "duplicate path" would read as an inexplicable rejection of the whole
batch.

If your source is flattened nested data, uniqueness is yours to guarantee before you send:
the leaf name alone is not unique, and the flattened path is.

The same rule and the same shape apply to `ids` on `POST /api/v1/lookup`, under
`error.details.duplicate_ids`.

#### Which route takes a 200-field chunk

`/api/v1/match/batch`. `/api/v1/match` caps at `NEXUS_API_MAX_FIELDS` (default **100**),
`/api/v1/match/batch` at `NEXUS_API_MAX_BATCH_FIELDS` (default **250**) — so a 200-field
chunk goes to the batch route and only to the batch route. Both are the same contract and
the same implementation; the cap is the only difference.

Over the cap is a **413**, and it carries the number to retry with: `error.details.limit` is
the server's cap and `error.details.fields` is what you sent, with both repeated in
`message`. The Java client surfaces the first as
`PayloadTooLargeException.suggestedChunkSize()`. The raw-body 413 is a different path and
has no such number — nothing counted your fields, because the body was refused before it was
parsed — so it carries `limit_bytes` instead and the honest response is to halve and retry.

**The per-field ceiling has been measured against real data, not assumed.** Over 4,598 HL7
FHIR R5 element definitions and 1,556 real flattened element paths, the largest single field
is **1,267 characters** (a 1,099-character definition, a 65-character path, a 71-character
leaf name) against a ceiling of 10,880 — 8.6x headroom. A 200-field chunk with *every* field
at those maxima serialises to **263,241 bytes** against the derived body cap of 10,897,024 —
41.4x. A realistic glossary-grade `doc` is nowhere near either bound;
`tests/unit/presentation/api/test_payload_headroom.py` holds that measurement against the
shipped caps, so lowering a cap under real data fails there.

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | The column's own name. |
| `path` | no | The caller's identifier, and the key the response is keyed by. Defaults to `name`. A **dotted** path is strongly preferred: the segment before the last dot becomes the query's parent context, the single largest accuracy factor measured on this task. |
| `doc` | no | Column comment or description. |
| `type` | no | Source type name, normalised server-side. Unknown types are accepted. |
| `signals` | no | Query signals for this one field — a map, not free text. A field-level key beats the request-level key of the same name, and the two merge **key by key**, so a request-level overlay and a field-level entity coexist. |

Plus the request-level knobs, all optional and all defaulted server-side:

| Knob | Default | Meaning |
|---|---|---|
| `top_k` | 5 | Candidates per field. Above the server's `results_per_field` is a **422** naming the cap. |
| `explain` | false | The score components and weights behind each confidence. |
| `signals` | `{}` | Request-level query signals, applied to every field unless that field overrides the same key. The abbreviation overlay belongs here rather than on a field: it is a catalog, and it is scoped to this one request. |
| `contrast` | false | Append the [`contrast` block](#contrast--why-not-the-other-one). |
| `consistency` | false | Append the [`consistency` block](#consistency-and-why-it-is-off). |
| `consistency_qualifier_segments` | 1 | The grouping key `consistency` uses. Bounded `0..MAX_QUALIFIER_SEGMENTS`, which the server **derives** from its own `path` length limit rather than declaring as a literal. A negative value is a 422; a value deeper than your deepest path is inert. |

Both request keys and both blocks are **strictly additive**. With `contrast` and
`consistency` unset the response is byte-identical to the one this service sent before they
existed.

Worked request, response and failure modes: [GOVERNANCE.md](GOVERNANCE.md#matching-over-http).

Both match routes answer **503** until a dictionary is loaded — see
[DEPLOYMENT.md](DEPLOYMENT.md#2-environment-configuration) for the variables that load one.

### The matching response

**Four top-level keys, in this order: `results`, `vocabulary`, `fieldDecisions`,
`scoring`.** The order is the wire contract. `results` was once the whole body, so
everything since has been **appended** rather than placed in front of it — a client
generated against any earlier shape keeps reading every key it already knew at the key it
already knew.

| Key | What it is | Why it is not derivable from the ones before it |
|---|---|---|
| `results` | One entry per input field, keyed by the caller's own `path`, in the order sent. A field nothing matched gets `[]`, never a missing key. | — |
| `vocabulary` | The caller's own `openClassification` and tier ladder, echoed back. | Without it a `"governance": null` cannot be read. |
| `fieldDecisions` | **One verdict per column**, under the same keys in the same order. Vocabulary: `AUTO_APPROVE`, `REVIEW`, `REJECT`, **`NO_MATCH`**. | The per-candidate `decision` cannot express `NO_MATCH`. See below. |
| `scoring` | What every number in this body *means*: which are comparable across fields, which floors are in force, and what metric produced `absoluteScore`. | A governance artifact whose numbers can only be interpreted by reading this library's source is not one. |

Two further keys are **appended, and only when the request asks for them**:
`contrast` and `consistency`, in that order. A request that asks for neither gets the four
keys above and nothing else, which is why they are appended rather than inserted.

`fieldDecisions` and `results` are checked against each other before the response is sent;
a body where one covers a field the other does not is refused rather than returned.

Everything below was captured from a live server started with
[`examples/governance/serve.sh`](../examples/governance/serve.sh), which loads the
fictional Gravel Bay Ferry Authority pack. The request:

```json
{
  "fields": [
    {"name": "pax_legal_nm",   "path": "booking.pax_legal_nm",  "doc": "Passenger legal name as printed on the manifest.", "type": "string"},
    {"name": "crew_member_id", "path": "roster.crew_member_id", "doc": "Permanent staff identifier for a vessel crew member.", "type": "string"},
    {"name": "route_cd",       "path": "timetable.route_cd",    "doc": "Short code identifying a scheduled route.", "type": "string"}
  ],
  "top_k": 1
}
```

and the response, pretty-printed — the server sends it compact, and nothing else here is
edited:

```json
{
  "results": {
    "booking.pax_legal_nm": [
      {
        "rank": 1,
        "governanceId": "GBF-0001",
        "businessName": "Passenger Legal Name",
        "definition": "The full legal name of a ticketed passenger as printed on the Gravel Bay sailing manifest.",
        "domain": "Passenger",
        "governance": {
          "code": "MANIFEST_NAME",
          "name": "Passenger manifest identity",
          "classification": "SEALED_RESTRICTED",
          "personalInformation": true,
          "directIdentifier": true,
          "enhancement": "MASK_IN_LOGS"
        },
        "confidence": 0.925,
        "decision": "AUTO_APPROVE",
        "absoluteScore": 0.881982,
        "sourceMetadata": {
          "values": {
            "personal_information": "yes",
            "direct_identifier": "yes"
          },
          "droppedKeyCount": 0,
          "renderedKeys": []
        }
      }
    ],
    "roster.crew_member_id": [
      {
        "rank": 1,
        "governanceId": "GBF-0009",
        "businessName": "Crew Member Identifier",
        "definition": "The authority's permanent staff identifier for a member of a vessel crew.",
        "domain": "Crew",
        "governance": {
          "code": "CREW_ROSTER",
          "name": "Crew employment record",
          "classification": "BRIDGE_SENSITIVE",
          "personalInformation": true,
          "directIdentifier": true,
          "enhancement": null
        },
        "confidence": 0.925,
        "decision": "AUTO_APPROVE",
        "absoluteScore": 0.935079,
        "sourceMetadata": {
          "values": {
            "personal_information": "yes",
            "direct_identifier": "yes"
          },
          "droppedKeyCount": 0,
          "renderedKeys": []
        }
      }
    ],
    "timetable.route_cd": [
      {
        "rank": 1,
        "governanceId": "GBF-0028",
        "businessName": "Sailing Route Code",
        "definition": "The short code that identifies a scheduled route between two terminals.",
        "domain": "Published",
        "governance": null,
        "confidence": 0.925,
        "decision": "AUTO_APPROVE",
        "absoluteScore": 0.879752,
        "sourceMetadata": {
          "values": {
            "personal_information": "",
            "direct_identifier": ""
          },
          "droppedKeyCount": 0,
          "renderedKeys": []
        }
      }
    ]
  },
  "vocabulary": {
    "openClassification": "OPEN_DECK",
    "tiersMostOpenFirst": [
      "OPEN_DECK",
      "CREW_ONLY",
      "BRIDGE_SENSITIVE",
      "SEALED_RESTRICTED"
    ]
  },
  "fieldDecisions": {
    "booking.pax_legal_nm": "AUTO_APPROVE",
    "roster.crew_member_id": "AUTO_APPROVE",
    "timetable.route_cd": "AUTO_APPROVE"
  },
  "scoring": {
    "confidenceFloor": 0.63,
    "absoluteScoreFloor": null,
    "absoluteScoreMetric": "cosine",
    "absoluteScorePooledOverAliases": false,
    "thresholdableAcrossFields": [
      "absoluteScore",
      "explain.absoluteCosine",
      "explain.scores.lexical",
      "explain.scores.editDistance",
      "explain.scores.type",
      "explain.scores.domain"
    ],
    "comparabilityScopesNarrowestFirst": [
      "WITHIN_FIELD",
      "ACROSS_FIELDS",
      "ACROSS_RUNS"
    ],
    "comparability": {
      "confidence": "WITHIN_FIELD",
      "absoluteScore": "ACROSS_FIELDS",
      "explain.absoluteCosine": "ACROSS_FIELDS",
      "explain.scores.fusedRetrieval": "WITHIN_FIELD",
      "explain.scores.lexical": "ACROSS_FIELDS",
      "explain.scores.editDistance": "ACROSS_FIELDS",
      "explain.scores.type": "ACROSS_FIELDS",
      "explain.scores.domain": "ACROSS_FIELDS"
    }
  }
}
```

`POST /api/v1/match/batch` returns the same body for the same request — one implementation
behind two field caps.

#### The candidate

Eleven keys, in this order: `rank`, `governanceId`, `businessName`, `definition`,
`domain`, `governance`, `confidence`, `decision`, `absoluteScore`, `sourceMetadata`,
`provenance`. A twelfth, `explain`, is appended when the request asks for it.

`provenance` is `RETRIEVAL` or `APPROVED_PAIR`, and it is the ONLY way to tell a
scored candidate from one a reviewer decided. Do not use `confidence` for that: the
weights sum to exactly 1.0 and every signal caps at 1.0, so ordinary retrieval can
reach 1.0 — the same value the approved-pair path writes. An `APPROVED_PAIR`
candidate carries no `absoluteScore` and no `explain`, because nothing measured it.

| Key | Type | Meaning |
|---|---|---|
| `rank` | `int` | 1-based position within this field's candidate list. |
| `governanceId` | `string` | The dictionary entry's id — the thing you join on, and **the caller's own string**. Never parsed, never normalised, never compared numerically: a zero-padded numeric id comes back with its padding. [Why that matters, with the capture](#governanceid-is-an-opaque-string-and-lookup-is-where-that-becomes-visible). |
| `businessName` | `string` | The entry's business name. |
| `definition` | `string` | The entry's definition. |
| `domain` | `string` | The entry's subject area. |
| `governance` | object or `null` | The protection class the entry confers. Six keys; see below. `null` is a documented value and **it is not "no restriction"** — see [the fail-open hazard](GOVERNANCE.md#the-fail-open-hazard-a-null-class-is-not-no-restriction) before mapping this onto read permissions. |
| `confidence` | `float` | The scored confidence. **Comparable within this field only** — see `scoring.comparability`. |
| `decision` | `string` | `AUTO_APPROVE`, `REVIEW` or `REJECT`. **Per candidate.** It cannot say `NO_MATCH`; the per-field verdict is in `fieldDecisions`. |
| `absoluteScore` | `float` or `null` | The raw dense-retrieval score for this candidate. `null` means the dense arm never proposed it — which is not zero: zero is a similarity that was measured, `null` is one that was never taken. |
| `sourceMetadata` | object | The matched entry's **pass-through plane**: `values` (your glossary's own enrichment columns, in loader order, never interpreted), `droppedKeyCount` (how many keys the loader's per-entry size cap discarded; `0` means this is the whole plane) and `renderedKeys` (keys whose value JSON could not hold natively and which were rendered as text). Always present, `values` empty rather than the key missing. Nothing in it is read by any score, ranking, threshold or governance decision. |

**`absoluteScore` is the only per-candidate number comparable across fields**, and it is
the number `absolute_score_floor` is compared against. It is present on every candidate
whether or not `explain` was requested; when `explain` is requested it is read once and
emitted twice, so `absoluteScore` and `explain.absoluteCosine` cannot disagree. Read
`scoring.absoluteScoreMetric` before treating it as a cosine, and
`scoring.absoluteScorePooledOverAliases` before comparing it against a floor measured on a
deployment that pooled differently. Choosing a floor is
[a measurement, not a constant](guides/absolute_score_floor.md).

A candidate carries an eleventh key, `explain`, only when the request set `explain: true`.
It is **absent**, not present-and-null, otherwise. The identical request above with
`"explain": true` returns, on `booking.pax_legal_nm`:

```json
"explain": {
  "scores": {"fusedRetrieval": 1.0, "lexical": 1.0, "editDistance": 1.0, "type": 1.0, "domain": 0.5},
  "weights": {"fusedRetrieval": 0.7, "lexical": 0.05, "editDistance": 0.05, "type": 0.05, "domain": 0.15},
  "absoluteCosine": 0.881982
}
```

`absoluteCosine` here is byte-for-byte the `absoluteScore` in the response above, because
both are the same read.

`scores` and `weights` are open maps carrying the same keys, so
`sum(scores[k] * weights[k])` clamped to [0, 1] reproduces `confidence` from the response
alone; the weights are the live matcher's, not the shipped defaults, so a tuned deployment
gets numbers that reproduce *its* confidences. `fusedRetrieval` is min-max normalised
within this field's shortlist — 0.9 means "ranked first here", not "90% similar" — and
`absoluteCosine` is the only figure comparable across fields. It is `null` when the dense
retriever did not return that candidate.

#### `governance`, and its sixth key

Six keys, in this order: `code`, `name`, `classification`, `personalInformation`,
`directIdentifier`, `enhancement`.

**This block is security-relevant, not descriptive.** `governance.code` is [an
access-control class](GOVERNANCE.md#governancecode--an-access-control-class-not-a-label): a
deployment's own description of how protected a data element is, of the kind an
organisation writes in order to decide who or what may read a column. The library defines
none of them, resolves them only against the caller's own vocabulary, and enforces nothing
— but a consumer wiring `code` into read permissions must read [the fail-open
hazard](GOVERNANCE.md#the-fail-open-hazard-a-null-class-is-not-no-restriction) first,
because the whole block is `null` in five different situations and none of them is a grant.
The procedure that gets all five right is
[Governance as access control](guides/governance_as_access_control.md).

`enhancement` is the newest and was **appended**, so the order of the five before it is
unchanged and a client reading them by name is unaffected.

| Key | Type | Meaning |
|---|---|---|
| `code` | `string` | **The access-control class**: how protected this element is, in the caller's own vocabulary, resolved with aliases already followed. This is the value a consumer maps onto read permissions. The example pack maps `GBF-LEGACY-NAME` to `MANIFEST_NAME`, which is why `GBF-0001` comes back as the latter. |
| `name` | `string` | The class's human-readable name. |
| `classification` | `string` | The tier. Rankable only against `vocabulary.tiersMostOpenFirst`. |
| `personalInformation` | `bool` | |
| `directIdentifier` | `bool` | |
| `enhancement` | `string` or `null` | The caller's own instruction for **how to protect the field** — masking, tokenisation, a retention rule. Passed through untouched and never interpreted by this library. |

`enhancement` is the only member that says what to *do* rather than what the field *is*,
which is why it is carried: it was resolved on every `MatchResult` and then dropped at the
wire, so the answer to "mask it, tokenise it, or retain it seven years" lived only in a
file the HTTP caller does not have.

**`null` is a documented value, not a defect.** It means the class declares no instruction
— the tier is the whole instruction. Five of the nine classes in
`examples/governance/protection_classes.json` declare none, so a client that treats it as
required refuses this repository's own example pack. The other four declare one, across
three distinct values — `MASK_IN_LOGS` (twice), `TOKENISE_AT_REST`, `RETAIN_SEVEN_YEARS` —
and those strings are entirely that fictional pack's own invention. It is free text,
deliberately not a closed set.

`code`, `name`, `classification` and `enhancement` are all typed `string` in the published
schema, never an enum. They carry the caller's controlled vocabulary, and this library
ships no taxonomy at all — closing them would hard-code one organisation's into the schema
a Java client is generated from.

#### `vocabulary`, and why a response needs one

```json
"vocabulary": {
  "openClassification": "OPEN_DECK",
  "tiersMostOpenFirst": ["OPEN_DECK", "CREW_ONLY", "BRIDGE_SENSITIVE", "SEALED_RESTRICTED"]
}
```

Both values are the caller's own, echoed back from the vocabulary file
`NEXUS_API_GOVERNANCE` names. This library supplies neither and ranks nothing.

It exists because **without it a `"governance": null` cannot be read.** `null` is not
"unclassified" — on any candidate that is not a rejected rank 1 it means the entry carries
no protection code, which is a real tier with a name only the caller's vocabulary knows.
In the response above, `timetable.route_cd` matched `GBF-0028` and came back `null`;
`openClassification` is what says that field sits at `OPEN_DECK`. A consumer holding only
the response cannot derive that, and a separate service — the case this endpoint exists
for — has no copy of the vocabulary file to look it up in. The body is a governance
artifact that gets pasted into a ticket and diffed, and an artifact whose `null` means
"ask a second system" is not one.

`tiersMostOpenFirst` is the only thing that can rank two classifications against each
other. Empty means the vocabulary declares no ordering: treat tiers as incomparable there,
and never as alphabetical, which sorts `CONFIDENTIAL` above `PUBLIC`.

The block is constant per deployment and roughly 120 bytes, so it is a rounding error
against a response carrying candidates. It rides on the response rather than sitting on an
endpoint somebody has to know to call.

**When no vocabulary is configured** the block reads:

```json
"vocabulary": {"openClassification": "UNCLASSIFIED", "tiersMostOpenFirst": []}
```

`UNCLASSIFIED` is this library's own sentinel, not a tier: it is deliberately not a word a
real taxonomy uses, so it cannot be mistaken for one. Reaching it requires a glossary with
no protection-code column at all — pointing `NEXUS_API_DICTIONARY` at a glossary that *has*
one while leaving `NEXUS_API_GOVERNANCE` unset is refused at startup, and both match routes
then answer **503** (`NEXUS-1002`) rather than serving every field as `null`.

#### `fieldDecisions`, the one verdict per column

```json
"fieldDecisions": {
  "booking.pax_legal_nm": "AUTO_APPROVE",
  "roster.crew_member_id": "AUTO_APPROVE",
  "timetable.route_cd": "AUTO_APPROVE"
}
```

Same keys as `results`, same order, one string each. **This is the value a consumer writes
down.** `results[path][0].decision` is a statement about a candidate; this is a statement
about the column.

Four values, and the fourth is the reason this block exists:

| Value | Meaning |
|---|---|
| `AUTO_APPROVE` | Rank 1 cleared the auto-approve bar and its margin over the runner-up. Inherit. |
| `REVIEW` | A human decides. Never read it as "probably fine". |
| `REJECT` | Rank 1 is below `review_threshold`. Unreachable at the shipped numbers — see below. |
| **`NO_MATCH`** | **This response carries nothing this field may inherit.** |

`NO_MATCH` is the state the per-candidate vocabulary cannot express, and it is not a
convenience. `confidence` is min-max normalised within a field, so rank 1 has a structural
floor of `semantic_weight × fusion_alpha` = **0.63** (published as
`scoring.confidenceFloor`), which sits above `review_threshold` = 0.50. No rank-1 candidate
can therefore be `REJECT` on score alone: **every field comes back at least `REVIEW`,
however irrelevant its best candidate is.** No setting of `review_threshold` recovers the
missing state, because the floor moves with the weights and the thresholds do not move
with it.

A field is `NO_MATCH` when either:

1. **it came back with no candidates at all** — which happens with **no floor configured**,
   so `NO_MATCH` is reachable on the shipped defaults; or
2. **`scoring.absoluteScoreFloor` is configured and rank 1's `absoluteScore` does not clear
   it** — including when that `absoluteScore` is `null`, meaning the dense retriever never
   proposed the candidate and so offers no evidence it clears anything.

The verdict alone does not distinguish the two. Read `results[path]` beside it: empty is
case 1, non-empty is case 2.

**The candidates on a `NO_MATCH` field are still returned and still carry `governance`.**
They are evidence for a reviewer, not a classification. Captured from a live server with
`absolute_score_floor` set to 0.70, sending one field the pack answers and one it cannot:

```json
{
  "results": {
    "telemetry.quasar_flux_index": [
      {
        "rank": 1,
        "governanceId": "GBF-0022",
        "businessName": "Vessel Heading Degrees",
        "definition": "The compass heading a vessel reported at the last telemetry ping.",
        "domain": "Voyage",
        "governance": {
          "code": "VESSEL_TELEMETRY",
          "name": "Vessel operational telemetry",
          "classification": "CREW_ONLY",
          "personalInformation": false,
          "directIdentifier": false,
          "enhancement": null
        },
        "confidence": 0.823333,
        "decision": "REVIEW",
        "absoluteScore": 0.586716,
        "sourceMetadata": {
          "values": {
            "personal_information": "no",
            "direct_identifier": "no"
          },
          "droppedKeyCount": 0,
          "renderedKeys": []
        }
      }
    ]
  },
  "fieldDecisions": {
    "booking.passenger.legal_name": "AUTO_APPROVE",
    "telemetry.quasar_flux_index": "NO_MATCH"
  },
  "scoring": {
    "absoluteScoreFloor": 0.7,
    "absoluteScoreMetric": "cosine"
  }
}
```

That candidate has a `confidence` of 0.82 — well above `review_threshold` — a `decision` of
`REVIEW`, and a fully populated `CREW_ONLY` protection class. Nothing in it says the field
is unanswerable. Its `absoluteScore` of 0.5867 does, and `fieldDecisions` is where that
reading is published. **Read `fieldDecisions[path]` first.**

That instruction is load-bearing rather than tidy when `governance.code` drives read
permissions. `NO_MATCH` beside a populated `CREW_ONLY` block is one half of the trap; the
other half is a `governance: null` read as "no restriction", which grants the column to
everyone. Both halves, the five situations that produce a `null`, and the seven-step recipe
that gets them all right are in [the fail-open
hazard](GOVERNANCE.md#the-fail-open-hazard-a-null-class-is-not-no-restriction).

`absolute_score_floor` is off by default and this library ships no value for it: a floor is
a statement about a score distribution, and the distribution belongs to your dictionary.
[Calibrating the absolute score floor](guides/absolute_score_floor.md) is the procedure for
measuring one, and includes the measurement showing that a plausible-sounding floor can
produce zero `NO_MATCH` verdicts on any real corpus.

#### `scoring`, so the numbers can be read without this library's source

```json
"scoring": {
  "confidenceFloor": 0.63,
  "absoluteScoreFloor": null,
  "absoluteScoreMetric": "cosine",
  "absoluteScorePooledOverAliases": false,
  "thresholdableAcrossFields": ["absoluteScore", "explain.absoluteCosine", "explain.scores.lexical", "explain.scores.editDistance", "explain.scores.type", "explain.scores.domain"],
  "comparabilityScopesNarrowestFirst": ["WITHIN_FIELD", "ACROSS_FIELDS", "ACROSS_RUNS"],
  "comparability": {"confidence": "WITHIN_FIELD", "absoluteScore": "ACROSS_FIELDS", "explain.absoluteCosine": "ACROSS_FIELDS", "explain.scores.fusedRetrieval": "WITHIN_FIELD", "explain.scores.lexical": "ACROSS_FIELDS", "explain.scores.editDistance": "ACROSS_FIELDS", "explain.scores.type": "ACROSS_FIELDS", "explain.scores.domain": "ACROSS_FIELDS"}
}
```

Same argument as `vocabulary` one section up: the body is a governance artifact that gets
pasted into a ticket and diffed, so an artifact whose numbers can only be interpreted by
reading this library's source is not one.

| Key | Type | Meaning |
|---|---|---|
| `confidenceFloor` | `float` or `null` | The lowest `confidence` a rank-1 candidate can structurally take, `semantic_weight × fusion_alpha`. **Self-verifying**: it is checked against the rank-1 confidences in *this* response and published as `null` if any of them is below it, because the bound has preconditions (no reranker; at least two distinct dense scores) that an ordinary one-entry dictionary violates. A threshold set above a floor your own fields sit under is this repository's NM-0027 defect on the wire. |
| `absoluteScoreFloor` | `float` or `null` | The `absoluteScore` beneath which this server reports a field as `NO_MATCH`. `null` — the shipped default — means no floor is configured, and `fieldDecisions` can then only report `NO_MATCH` for a field with no candidates at all. |
| `absoluteScoreMetric` | `string` | What actually produced `absoluteScore`. `cosine` under the shipped wiring; a store that cannot say reports `unknown` rather than guessing. Open string, never an enum. |
| `absoluteScorePooledOverAliases` | `bool` | Whether the score is the best over an entry's generated aliases rather than over the entry alone. `true` shifts the whole distribution upward, so a floor measured on a deployment with a different setting means something else here. |
| `thresholdableAcrossFields` | `string[]` | The response paths whose numbers a caller may compare against a fixed constant. |
| `comparabilityScopesNarrowestFirst` | `string[]` | The scope ladder, narrowest first, so a client can order two scopes without hard-coding them. |
| `comparability` | `object` | Response path → scope. `WITHIN_FIELD` means the number is normalised inside one field's shortlist and comparing it across fields is meaningless. |

The single most important row: **`confidence` is `WITHIN_FIELD` and `absoluteScore` is
`ACROSS_FIELDS`.** A rank-1 `confidence` lands near `fusion_alpha` whether the match is
excellent or absurd, so thresholding it across a schema compares nothing. That is why a
floor exists on the absolute number and not on the confidence.

`absoluteScoreFloor` is **also** published on `GET /api/v1/status`, in that route's
`thresholds` block, alongside `absoluteScoreMetric`. It is the same number read off the
same property, so the two surfaces cannot disagree about which floor governs a verdict.
Read it there when you want a deployment's floor without matching anything; read it here
when you want it travelling with the verdicts it produced.

> This paragraph previously stated that the status route carried no floor, and that sending
> a one-field match was the only way to read one. That was true and is no longer: an
> operator who could not see the active floor could not tell an emitted `NO_MATCH` from a
> field the matcher simply had nothing for.

### Contrast — why not the other one

`"contrast": true` on the request appends one block. It answers the question `explain`
cannot: `explain` describes why the winner scored what it did, using weights that are the
same for every candidate and are already published, while a reviewer looking at a
surprising match wants to know **why not the other one** — which is a subtraction between
two candidates rather than a description of one. The two are independent; you can have the
comparison without a weight breakdown on every candidate.

Captured from the live pack, one field of three, two signals of five:

```json
{
  "resolution": 1e-06,
  "comparability": {
    "confidenceGap": "WITHIN_FIELD",
    "signals": {"fusedRetrieval": "WITHIN_FIELD", "lexical": "ACROSS_FIELDS", "editDistance": "ACROSS_FIELDS", "type": "ACROSS_FIELDS", "domain": "ACROSS_FIELDS"}
  },
  "fields": {
    "published.terminal.name": {
      "topGovernanceId": "GBF-0027",
      "runnerUpGovernanceId": "GBF-0001",
      "topConfidence": 0.884091,
      "runnerUpConfidence": 0.598656,
      "confidenceGap": 0.285435,
      "signalGap": 0.285436,
      "separation": "SEPARATED",
      "largestDifference": "fusedRetrieval",
      "decidingSignals": [],
      "governanceDiffers": true,
      "domainDiffers": true,
      "signals": [
        {"signal": "fusedRetrieval", "topScore": 1.0, "runnerUpScore": 0.593317, "delta": 0.406683, "weight": 0.7, "weightedDelta": 0.284678, "separating": true, "deciding": false},
        {"signal": "editDistance", "topScore": 0.181818, "runnerUpScore": 0.166667, "delta": 0.015151, "weight": 0.05, "weightedDelta": 0.000758, "separating": true, "deciding": false}
      ]
    }
  }
}
```

| Key | Meaning |
|---|---|
| `resolution` | The smallest difference the numbers in this response can express — the precision every published float is rounded to. **Derived** from the serialiser's own precision, not typed as a constant. Nothing below it is reported as separating and nothing below it is named as a cause. |
| `comparability` | The scale contract for the contrast's own numbers, in the vocabulary `scoring.comparabilityScopesNarrowestFirst` publishes. `confidenceGap` names the scope of the gap; `signals` names the scope of each signal's `delta` and `weightedDelta`. Derived from `scoring.comparability` rather than restated, so a number whose scope changes cannot leave a stale entry here. |
| `fields` | **Every input path, keyed and ordered exactly like `results`**, with an explicit `null` where the field has fewer than two candidates. "This field had one candidate" and "this pass skipped it" must not look alike. |

Inside one contrast:

| Key | Meaning |
|---|---|
| `topGovernanceId`, `runnerUpGovernanceId` | The two entries being compared. The runner-up is read from the **full** match list, not the `top_k` slice, so a caller who asks for one candidate is still told what the one they cannot see was. |
| `topConfidence`, `runnerUpConfidence` | Their confidences, at `resolution`. |
| `confidenceGap` | `topConfidence - runnerUpConfidence`. Comparable **within this field only**, because `confidence` is — a difference is no more comparable than its operands. |
| `signalGap` | The same margin reached the other way: the sum of every `weightedDelta`. Published so the arithmetic can be checked from the response alone. The service verifies the two against each other and **refuses to answer** rather than send a contrast that does not close. The two can sit a unit of the last place apart — 0.285435 against 0.285436 above — because both operands of every delta are rounded before being subtracted; the server's own tolerance is one order above `resolution`. |
| `separation` | `SEPARATED` when the margin exceeds `resolution`; `TIED` when it does not, meaning the two are level in every number this response publishes and the ordering came from the matcher's sort. An open string, not a closed enum. |
| `largestDifference` | The separating signal with the largest weighted difference — the headline answer. `null` on a `TIED` contrast and when no signal differs by more than `resolution`. |
| `decidingSignals` | Every signal whose removal would leave rank 2 level with or ahead of rank 1. **Empty is a real answer and the common one on a wide margin** — it means no single signal carried it, as above, where `fusedRetrieval` accounts for 0.2847 of a 0.2854 margin and removing it still leaves rank 1 ahead. Always empty on a `TIED` contrast. |
| `governanceDiffers`, `domainDiffers` | Read from the two glossary **entries**, not from any signal, and usually what settles a review. Taken from the entries' own codes, so a rank-1 `REJECT` — which confers no class by design — does not read as a difference that is not there. |
| `signals` | One entry per weighted signal, largest weighted difference first, ties broken by declaration order so two identical requests order the list identically. |

Inside one signal difference: `signal` (the same key it carries in `explain.scores` and
`explain.weights`), `topScore`, `runnerUpScore`, `delta` (`topScore - runnerUpScore`,
negative where rank 2 won it), `weight`, `weightedDelta` (`delta * weight`), `separating`
(false when the two differ by no more than `resolution`; a signal that is not separating is
never named as a cause) and `deciding`.

### Consistency, and why it is off

`"consistency": true` appends a second block: which columns this request believes are the
same business concept, and whether their rank-1 answers agree. It is **reporting only** —
nothing in `results` or `fieldDecisions` changes, whatever it finds, and `promotionApplied`
states that machine-readably rather than leaving a consumer to infer it.

**It is off by default because its grouping was measured and the measurement came back
negative.** The idea is sound: fields are matched one at a time and independently, and
nothing notices when two columns that are the same concept get different answers. Detecting
that needs no labelled data. But everything depends on the grouping rule, and the rule is a
heuristic over column names:

- at the default `consistency_qualifier_segments` of **1** — a leaf groups only with a leaf
  under the same declared parent — it emits **no group at all** on every profile in this
  repository's generated corpus. It reports nothing, and therefore claims nothing false;
- at **0**, the leaf alone, it scores 0.86–1.00 pair-precision on a parent-diverse mixture
  and **0.0233 at recall 1.00** on a repeated-leaf schema — one leaf governed separately in
  ~30 domains, which is the shape the feature was proposed for. There it emitted four groups
  containing **zero concepts and four collisions**: 87 columns spanning 29 genuinely distinct
  answers merged into one "concept" and reported as a contradiction;
- and there is **no operating point**. The whole published policy space was searched — 684
  policies over two profiles, two scales and two repetition depths — and the best precision
  reached by any policy that reports anything at all on that shape is **0.0235**.

So a `DISAGREE` is a prompt to look, never a defect report about the matcher. The check to
run first is `distinctAnswers` against the number of members that answered: when the two are
close, the group is a collision of distinct concepts that happen to share a column name.

Captured from the live pack at `consistency_qualifier_segments: 0`, which is what it takes
to make these three fields group at all:

```json
{
  "grouping": {"qualifierSegments": 0, "includeDataType": true, "orderSensitive": false, "minGroupSize": 2},
  "groupsFound": 1,
  "fieldsGrouped": 2,
  "groupsDisagreeing": 1,
  "promotionApplied": false,
  "groups": [
    {
      "concept": "|name|name|string",
      "fields": ["published.terminal.name", "booking.passenger.name"],
      "answers": {"published.terminal.name": "GBF-0027", "booking.passenger.name": "GBF-0001"},
      "distinctAnswers": 2,
      "agreement": "DISAGREE",
      "majorityGovernanceId": null,
      "majorityCount": 0
    }
  ]
}
```

That finding is a **false positive, and it is shown here rather than a tidy `AGREE` on
purpose**: a ferry terminal's name and a passenger's name are not one business concept, they
share four letters. Two members answered and gave two different answers — `distinctAnswers`
equals the number that answered, which is the collision signature. At the shipped default of
1 the same three fields produce `"groupsFound": 0`.

| Key | Meaning |
|---|---|
| `grouping` | The policy these groups were built under — `qualifierSegments`, `includeDataType`, `orderSensitive`, `minGroupSize` — published because a finding cannot be judged without the rule that produced it. |
| `groupsFound` | How many groups of two or more columns were found. Zero is the expected answer at the default. |
| `fieldsGrouped` | How many of this request's fields fell into a group of two or more. A column that shares its concept with nothing is not reported: it cannot disagree with anyone. |
| `groupsDisagreeing` | How many groups have an `agreement` of `DISAGREE`. A count of groups, not a count of problems. |
| `promotionApplied` | Always `false`. Promoting a group's majority can move a correct answer to an incorrect one, which surfacing a disagreement cannot, and the measurement that would justify it does not exist. |
| `groups` | Ordered by where each group's first member appeared in the request, so two identical requests produce the same list. |

Inside one group:

| Key | Meaning |
|---|---|
| `concept` | The concept key as a printable label: the qualifier segments, the leaf's normalised tokens, its class word and the data type, separated by `\|`. A grouping **artifact**, stable for a given request and policy — quotable in a ticket, and not a name anyone chose. Do not key a downstream system on it. |
| `fields` | The group's members, in the order they were sent. |
| `answers` | Each member's rank-1 governance id, or `null` where the field had no answer to give — no candidates, or a `fieldDecisions` verdict of `NO_MATCH`, which inherits nothing. **A null is silence, not a dissenting answer.** |
| `distinctAnswers` | How many different non-null answers the group got. |
| `agreement` | `AGREE` when two or more answered and all agree, `DISAGREE` when two or more answered and they do not, `UNDECIDED` when fewer than two answered at all. `UNDECIDED` is deliberately not `AGREE`: one answer and five blanks is not five columns confirming each other. An open string, not a closed enum. |
| `majorityGovernanceId` | The modal answer, or `null` when no single answer holds a plurality. **Evidence, never an instruction.** |
| `majorityCount` | How many members gave it; `0` when there is none. |

The concept key is built from the **response key** — your own `path` — and not from `name`.
Segments are boundaries you declared: dots, or the `__` array boundary. Single underscores
are tokens inside a segment, so `a_b__c_d_e` has two segments and `a.b.c` has three.

### Resolving an id: POST /api/v1/lookup

The match routes answer "which entry governs this column". This route answers "what is
entry X", for an id you already hold — the `governanceId` off a stored match, a row in your
own catalog, an id typed into a ticket. No scoring, no ranking, no decision: an entry is
either there or it is not.

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/lookup \
  -H 'Content-Type: application/json' \
  -d '{"ids":["0000123","123"]}'
```

| Key | Meaning |
|---|---|
| `results` | Every id you sent, **once and in the order sent**, mapped to an entry or an explicit `null`. An id cannot vanish from the map. |
| `missing` | Exactly the ids whose value is `null`, so a program does not have to scan for them. |
| `vocabulary` | The same block the match routes carry: `openClassification` and `tiersMostOpenFirst`. A `LookupEntryView` can carry `"governance": null` for the same reason a candidate can, and the block is what makes that null readable. |

Caps and refusals: id cap `NEXUS_API_MAX_BATCH_FIELDS`, **413** over it, **422** on a
duplicate, blank or oversized id (duplicates come back in `error.details.duplicate_ids`,
the same shape `duplicate_paths` uses on `/match`), **503** with no dictionary loaded.

`GET /api/v1/lookup/{governance_id}` is the single-id form of the same body. A miss is a
**200** carrying `null`, never a 404 — a 404 on this service means the route does not exist.
The route is declared with the `:path` converter so an id containing `/` stays addressable.

#### governanceId is an opaque string, and lookup is where that becomes visible

The id is the caller's own, [carried through
unchanged](GOVERNANCE.md#governanceid--your-own-identifier-for-the-entry-carried-through-unchanged):
never parsed, never normalised beyond the loader stripping whitespace around the cell,
never compared numerically. `governanceId` is `"type": "string"` on both
`MatchCandidateView` and `LookupEntryView`, `ids` is `"items": {"type": "string"}`, the
path parameter is a string, and the generated Java client binds it as `String`.

Lookup is where a consumer who ignored that finds out. Two captures, both from live servers
today, both against glossaries whose ids are zero-padded seven-digit numbers.

**A padded id resolves; the unpadded form does not.** The glossary holds `0000123` and no
`123`:

```json
{
  "results": {
    "0000123": {
      "governanceId": "0000123",
      "businessName": "Passenger Legal Name",
      "definition": "The full legal name of a ticketed passenger as printed on the Gravel Bay sailing manifest.",
      "domain": "Passenger",
      "governance": {
        "code": "MANIFEST_NAME",
        "name": "Passenger manifest identity",
        "classification": "SEALED_RESTRICTED",
        "personalInformation": true,
        "directIdentifier": true,
        "enhancement": "MASK_IN_LOGS"
      },
      "sourceMetadata": {
        "values": {"personal_information": "yes", "direct_identifier": "yes"},
        "droppedKeyCount": 0,
        "renderedKeys": []
      }
    },
    "123": null
  },
  "missing": ["123"],
  "vocabulary": {
    "openClassification": "OPEN_DECK",
    "tiersMostOpenFirst": ["OPEN_DECK", "CREW_ONLY", "BRIDGE_SENSITIVE", "SEALED_RESTRICTED"]
  }
}
```

`GET /api/v1/lookup/123` against the same server is a **200** with
`{"results":{"123":null},"missing":["123"], ...}`. So an `int()` on the way out and a
`str()` on the way back does not fail loudly; it returns a clean, well-formed answer that
the entry does not exist.

**And when both forms exist, they are two entries.** The same glossary plus one row whose
id is the literal `123`, same request, `governance` and `businessName` shown:

```json
{
  "results": {
    "0000123": {
      "governanceId": "0000123",
      "businessName": "Passenger Legal Name",
      "governance": {
        "code": "MANIFEST_NAME",
        "name": "Passenger manifest identity",
        "classification": "SEALED_RESTRICTED",
        "personalInformation": true,
        "directIdentifier": true,
        "enhancement": "MASK_IN_LOGS"
      }
    },
    "123": {
      "governanceId": "123",
      "businessName": "Crew Watch Rota Identifier",
      "governance": {
        "code": "CREW_ROSTER",
        "name": "Crew employment record",
        "classification": "BRIDGE_SENSITIVE",
        "personalInformation": true,
        "directIdentifier": true,
        "enhancement": null
      }
    }
  },
  "missing": []
}
```

*(`definition`, `domain` and `sourceMetadata` elided from both entries; nothing else is
edited.)*

Two ids that are the same number, two different terms, two different protection classes. A
consumer that had parsed either one as an integer would hold a single key for both and
would inherit whichever class it happened to write last. Since `governance.code` is [an
access-control class](GOVERNANCE.md#governancecode--an-access-control-class-not-a-label),
that is not a display defect.

The ingest side has the same property, measured on one CSV through `load_entries` today:

```
  id column        entry id
  ---------        --------
  "  0000123  " -> '0000123'            surrounding whitespace stripped, padding kept
  "123"         -> '123'
  "0123"        -> '0123'
  ""            -> 'ad77dbe304e1d05a'   no id in the row, so a content digest
```

Store it as text, join on it as text, log it as text.

### The feedback request body

`POST /api/v1/feedback` records one reviewer's verdict. **Recorded only** — it is not read
back into ranking.

| Key | Required | Meaning |
|---|---|---|
| `field` | yes | The path the match response was keyed by. |
| `doc` | no | The column comment the reviewer had in front of them. |
| `chosenGovernanceId` | yes | The glossary id the reviewer chose. |
| `suggestedGovernanceId` | no | The id the matcher had suggested, when it differed. |
| `wasCorrect` | yes | Whether the matcher's suggestion was right. |
| `reviewer` | yes | Who decided. |
| `ts` | yes | The client's timestamp. Stored verbatim and **not** trusted for ordering — the server stamps its own `receivedAt` beside it, and that is the field to sort by. |
| `verdict` | no | What the reviewer *did*: `APPROVED`, `REJECTED` or `MANUAL_OVERRIDE`. |

`verdict` exists because `wasCorrect` has two states and the vocabulary has three. The one a
boolean cannot express is the reviewer who chose a term **the matcher never proposed** — not
rank 2, not rank 20: absent from the candidate list entirely. Collapsed into `false`, that
record is byte-identical to "the top match was wrong and I took the third one", and those are
opposite diagnoses. The second says the answer was retrieved and mis-ranked, which weights or
a reranker can fix; the first says it was never retrieved, which no amount of re-ranking a
list that never contained it will fix. `MANUAL_OVERRIDE` is that state.

The two must agree, and the server **refuses the disagreement rather than picking a winner**:
`APPROVED` requires `wasCorrect: true`, `REJECTED` and `MANUAL_OVERRIDE` both require
`false`, and any other pairing is a **422**. A trail is evidence, and evidence that
contradicts itself is worse than a refusal.

The three values are published **inline on the property** and deliberately *not* as a named
schema component, so a generated client gets them as documentation rather than as a closed
type that stops decoding the day a fourth value is added.

> **`verdict` is additive, and something did change for an unchanged request.** A body that
> predates the member is still **201** and still stores the same eight values — but the
> echoed record and the appended trail line now carry a ninth key, `"verdict": null`. A
> tolerant reader is unaffected; a trail-consuming script asserting an exact key set is not.
> See [CHANGELOG.md](../CHANGELOG.md).

### The status response

`GET /api/v1/status`, captured from the same live server:

```json
{
  "ready": true,
  "degraded": false,
  "warnings": [],
  "dictionary": {
    "entryCount": 30,
    "source": "examples/governance/glossary.csv",
    "indexedAt": "2026-08-20T01:33:09.809092+00:00"
  },
  "encoder": {
    "provider": "BundledOnnxProvider",
    "modelName": "bge-small-en-v1.5-onnx-int8 (bundled)",
    "dimension": 384,
    "tier": "bundled",
    "bundledEncoderAvailable": true,
    "fallbackInForce": false
  },
  "thresholds": {
    "autoApprove": 0.87,
    "review": 0.5,
    "minConfidenceGap": 0.1,
    "resultsPerField": 5,
    "fusionAlpha": 0.9,
    "minimumAchievableConfidence": 0.63,
    "reviewThresholdBelowFloor": true,
    "absoluteScoreFloor": null,
    "absoluteScoreMetric": "cosine"
  },
  "limits": {
    "maxFields": 100,
    "maxBatchFields": 250,
    "bodyByteCap": 10897024,
    "deadlineSeconds": 25.0,
    "capacity": 36
  },
  "calibration": {
    "defaultsInForce": true,
    "overrides": {},
    "dictionarySizeRatio": 0.043605,
    "warnAboveSizeRatio": 10.0,
    "corpus": {
      "name": "bird+omop combined",
      "fields": 688,
      "dictionaryEntries": 688,
      "splits": {"bird": 361, "omop": 327},
      "domains": [
        "public relational database schemas (BIRD-SQL dev set)",
        "clinical common data model (OMOP CDM v5.4 field specification)"
      ],
      "fieldNaming": "ordinary SQL and CDM column identifiers. The bird split is heavily abbreviated and the omop split is not; neither is a flattened nested path, and neither was contracted by a governed abbreviation standard.",
      "ambiguity": "one gold entry per field, drawn from two unrelated domains, so a query competes against 687 distractors that are mostly from a different subject area.",
      "measuredBy": "benchmarks/exp_calibration.py",
      "artifact": "benchmarks/results/exp_calibration_combined.json",
      "autoApproveThreshold": 0.87,
      "autoApprovePrecision": 0.952941,
      "autoApproveCoverage": 0.123547,
      "precisionAtRank1": 0.581395
    }
  }
}
```

Always **200**, including with no dictionary loaded — a pre-run degradation check that
refused whenever the condition it reports on held would be unusable exactly when it is
needed.

| Key | Type | Meaning |
|---|---|---|
| `ready` | `bool` | A dictionary is loaded and the match routes will answer. |
| `degraded` | `bool` | Something is answerable but not as configured — read `warnings` for what. |
| `warnings` | `object[]` | Zero or more `{code, message}` records. Empty list, never `null`. **Branch on `code`, never on `message`**: the codes are `NO_DICTIONARY`, `EMPTY_DICTIONARY`, `FALLBACK_ENCODER` and `UNCALIBRATED_SIZE`; the message is human-readable and not part of the contract. |
| `dictionary.entryCount` | `int` or `null` | Entries indexed. `null` — not `0` — when no dictionary is loaded, so "loaded and empty" stays distinguishable from "not loaded". |
| `dictionary.source` | `string` or `null` | What `NEXUS_API_DICTIONARY` named, stamped at startup — the provenance of the numbers this server is producing. `null` when the matcher was handed over already indexed, in which case this server did not load it and cannot name a source. |
| `dictionary.indexedAt` | `string` or `null` | UTC ISO-8601 instant the index was built. |
| `encoder.provider` | `string` | The provider class actually in use. |
| `encoder.modelName` | `string` | The model actually loaded. |
| `encoder.dimension` | `int` | Embedding width. |
| `encoder.tier` | `string` | Which tier of the preference order resolved — `bundled`, and the other tiers a deployment can land on. An open string, not an enum. |
| `encoder.bundledEncoderAvailable` | `bool` | Whether the bundled int8 ONNX encoder could be loaded at all. |
| `encoder.fallbackInForce` | `bool` | **`true` means this server is not scoring with the encoder its benchmarks were measured on.** Read it before comparing any number against a published one. |
| `thresholds.*` | `float`/`int`/`bool` or `null` | The live matcher's numbers, not the shipped defaults. **Every threshold is nullable, and `null` means "this matcher does not expose it" — never `0.0`.** |
| `thresholds.minimumAchievableConfidence` | `float` or `null` | The structural floor of rank-1 `confidence`, `semantic_weight × fusion_alpha`. |
| `thresholds.reviewThresholdBelowFloor` | `bool` or `null` | `true` says `review_threshold` sits *under* that floor, so no rank-1 match can be `REJECT` on score alone. It is `true` at the shipped numbers, and it is the condition `absolute_score_floor` exists to answer. |
| `thresholds.absoluteScoreFloor` | `float` or `null` | The active floor beneath which a field is reported `NO_MATCH`. **`null` here means no floor is configured** — the shipped default — not "unreadable", which is what `null` means for every other member of this block. With no floor this deployment **cannot emit `NO_MATCH` at all**, whatever the scores. Same number as `scoring.absoluteScoreFloor` on a match response. |
| `thresholds.absoluteScoreMetric` | `string` or `null` | What that floor is compared against: the distance metric the wired vector store declares. `cosine` under the shipped wiring; `unknown` means the store declares none, and a floor chosen against an unknown metric is a guess. |
| `calibration.defaultsInForce` | `bool` or `null` | `true` when all three decision thresholds (`auto_approve_threshold`, `review_threshold`, `min_confidence_gap`) are still the shipped numbers. |
| `calibration.overrides` | `object` or `null` | Every matching setting whose live value differs from the shipped default, mapped to the **live** value. Keys are **snake_case** — the names you put in a `NEXUS_API_MATCHING_CONFIG` file, so the key you read is the key you set. `{}` means a stock profile. Derived from the config dataclass's own fields, so a setting this table has never heard of still appears the day a deployment changes it. |
| `calibration.dictionarySizeRatio` | `float` or `null` | `dictionary.entryCount / calibration.corpus.dictionaryEntries`. |
| `calibration.warnAboveSizeRatio` | `float` | The ratio above which running on shipped defaults raises `UNCALIBRATED_SIZE`. Published so the rule is arithmetic you can check rather than a judgement inside the server. |
| `calibration.corpus.*` | mixed | **The corpus the shipped defaults were fitted on**, and the answer to "were these numbers measured on anything like my data?". Size (`fields`, `dictionaryEntries`, `splits`), shape (`domains`, `fieldNaming`, `ambiguity`), provenance (`measuredBy`, `artifact`) and the operating point (`autoApproveThreshold`, `autoApprovePrecision`, `autoApproveCoverage`, `precisionAtRank1`). Present even with no dictionary loaded — it describes the build, not the deployment. |
| `limits.maxFields` | `int` | Cap on `/api/v1/match`. |
| `limits.maxBatchFields` | `int` | Cap on `/api/v1/match/batch`. |
| `limits.bodyByteCap` | `int` | Request bytes accepted before a **413**, derived from the field caps rather than typed as a literal. |
| `limits.deadlineSeconds` | `float` | Wall-clock budget for one match before it is shed. |
| `limits.capacity` | `int` | Concurrent + queued requests admitted before a **503**. |

**`absoluteScoreFloor` is deliberately not here.** The `thresholds` block is the seven keys
above and no floor; the active floor is published per response at
`scoring.absoluteScoreFloor`. To read a deployment's floor without matching anything, send a
one-field match and read `scoring`.

#### Reading `calibration`, and the one warning it raises

A threshold is a statement about a **score distribution**. The distribution is a property of
your dictionary and your field names, so the same number means something different on a
different corpus. `calibration` exists so you can tell, from the wire, whether the numbers
this server is auto-approving with were fitted on anything resembling your data.

Two questions, two members:

* **"Which calibration is in force?"** — `overrides`. Empty means this server is running the
  numbers this library shipped. Non-empty names every setting that differs and its live
  value, in the spelling you would use in a config file.
* **"What were the shipped numbers fitted on?"** — `corpus`. 688 labelled fields against a
  688-entry pooled dictionary, half public SQL schemas and half a clinical data model, in
  ordinary column identifiers. At the shipped `auto_approve_threshold` of 0.87 that measured
  0.952941 auto-approve precision at 0.123547 coverage, over a corpus where rank-1 accuracy
  is 0.581395. Every one of those numbers is checked against
  `benchmarks/results/exp_calibration_combined.json` by a packaging gate, so this block
  cannot drift away from the measurement it summarises.

**`UNCALIBRATED_SIZE`** is raised when both hold: the three decision thresholds are still the
shipped ones, **and** your dictionary is more than `warnAboveSizeRatio` times the calibration
corpus. It fires in one direction only, on one dimension only, and both limits are
deliberate:

* **Size, and not domain or naming style.** Comparing those would need a similarity metric
  this library has never validated, and a warning computed from an invented metric is wrong
  in a direction nobody can audit. They are *described* on `corpus.domains`,
  `corpus.fieldNaming` and `corpus.ambiguity` instead, for you to compare by eye.
* **Larger only, and only by an order of magnitude.** Ten is the smallest ratio this
  repository can point at a measurement across: in `exp_alias_scale.json` a retrieval
  setting is worth **+1.9** points of P@1 on a corpus the size of the calibration one, and
  **-13.7** on a corpus ten times it. The sign inverts. Below that ratio there is no
  evidence, and a `degraded: true` on every small deployment would teach operators to
  ignore the field.

The warning is not a defect report. It says the shipped auto-approve precision is a fact
about somebody else's corpus and not about yours. The fix is to fit your own —
[Calibration profiles](guides/calibration_profiles.md).

### The retrieval diagnostic response

`POST /api/v1/diag/retrieval` answers **why a field retrieved what it did**. It is retrieval
only: it does not reproduce scoring, the decision, or reranking. Same server, one field,
`top_k: 3`, `expected_governance_id: "GBF-0001"` — abbreviated here to the `dense` channel,
with `sparse` and `fused` carrying the identical shape:

```json
{
  "field": {"name": "legal_name", "path": "booking.passenger.legal_name", "doc": "Full legal name of the passenger as printed on the sailing manifest.", "type": "string"},
  "queryText": "booking, passenger legal name Full legal name of the passenger as printed on the sailing manifest.",
  "encoderModel": "bge-small-en-v1.5-onnx-int8 (bundled)",
  "rerankerWired": false,
  "channels": {
    "dense": {
      "available": true,
      "detail": null,
      "requestedTopK": 100,
      "returned": 30,
      "candidates": [
        {"rank": 1, "governanceId": "GBF-0001", "businessName": "Passenger Legal Name", "score": 0.784332},
        {"rank": 2, "governanceId": "GBF-0002", "businessName": "Passenger Date Of Birth", "score": 0.668792},
        {"rank": 3, "governanceId": "GBF-0012", "businessName": "Boarding Pass Serial", "score": 0.650669}
      ]
    }
  },
  "expected": {
    "governanceId": "GBF-0001",
    "inDictionary": true,
    "rankByChannel": {"dense": 1, "sparse": 1, "fused": 1}
  }
}
```

| Key | Type | Meaning |
|---|---|---|
| `field` | object | The field as parsed, echoed back. |
| `queryText` | `string` | **The text that was actually embedded**, after the parent context is prepended and any expansion applied. The single most useful line here: most retrieval surprises are a query that does not say what the caller thought it said. |
| `encoderModel` | `string` | The model that produced the dense scores in this body. |
| `rerankerWired` | `bool` | Whether a reranker is configured. `false` means these channel orders *are* the retrieval order. |
| `channels.{dense,sparse,fused}.available` | `bool` | Whether the channel ran. `false` with `detail` saying why. |
| `channels.*.detail` | `string` or `null` | Why an unavailable channel did not run. `null` when it did. |
| `channels.*.requestedTopK` | `int` or `null` | What this channel was asked for — `dense_top_k`, `sparse_top_k`. `null` on `fused`, which asks for nothing: it re-ranks what the two arms already returned. |
| `channels.*.returned` | `int` | How many the channel actually produced — the **full** count, not the truncated candidate list. `returned` far below `requestedTopK` is the diagnosis. |
| `channels.*.candidates` | object[] | `rank`, `governanceId`, `businessName`, `score`, truncated to `top_k`. |
| `expected.governanceId` | `string` | The id you asked about. Block absent when you asked about none. |
| `expected.inDictionary` | `bool` | Whether that id is indexed at all. `false` means the retrieval question is moot. |
| `expected.rankByChannel` | object | Where that id landed per channel, or `null` for a channel that did not return it. |

**Scores are each channel's own number and are not comparable across channels.** `dense`
is the raw retrieval score — the same quantity as a candidate's `absoluteScore`, which is
why 0.784332 here is 0.784332 there. `sparse` is a BM25 score on its own unbounded scale
(33.7 above). `fused` is min-max normalised, so its rank 1 is 1.0 by construction and says
nothing about quality. Compare ranks across channels; compare scores only within one.

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
`is_array`, `array_item_type`, `default_value`, `constraints`, `source_metadata`.

The last one is `source_metadata`. An earlier revision of this list called it `metadata`;
there is no such attribute, and `getattr(field, "metadata")` raises `AttributeError`.

`parent_path` matters: it is the hierarchical context injected into the retrieval query,
and it is the single largest accuracy factor measured on the benchmark
(+20 points of P@1 — `benchmarks/results/exp_query_repr_combined.json`).

### `DictionaryEntry`

`id`, `business_name`, `logical_name`, `definition`, `data_type`, `protection_level`,
`governance_code`, `domain`, `parent_table`, `sample_values`, `synonyms`, `is_enum`,
`enum_values`, `source_metadata`.

`governance_code` is the controlled protection code the entry carries, in the caller's own
vocabulary. A code the vocabulary does not define does not land here — it survives only as
`source_metadata['governance_code_raw']`, because a stored code nobody defined reads as
governance and is not.

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
| `governance` | `ProtectionClass` or `None` |
| `governance_id` | `str`, always populated |

Convenience properties: `is_auto_approved`, `needs_review`, `is_rejected`.

A caller matches a field in order to inherit the entry's governance, so both are
first-class attributes rather than something to fish out of `source_metadata`, and both are
populated on **every** candidate rather than on rank 1 alone — deciding between rank 1 and
rank 2 usually turns on which of them is a direct-identifier class, and that comparison is
impossible if only the top match carries one.

`governance` is `None` when the entry carries no code (the open tier), and also on a rank-1
`REJECT`, which confers nothing: a rejected top match means no entry in the glossary
describes this field, so a novel field would otherwise arrive carrying the class of a
candidate the matcher itself rejected. A rejected runner-up keeps the class its entry
confers — nothing inherits from rank 2, so nothing needs to be withheld there. The rank
qualifier is the rule, not a detail of it. `decision` alone does not separate the two nulls;
over HTTP, `vocabulary.openClassification` is what resolves the first.

`governance_id` is the matched entry's id, which *is* the governance id. It is derived from
`dictionary_entry` when not supplied and refused when supplied and different, because two
answers to "whose class is this?" is worse than none. It is a `str` and it is the caller's
own: `str(dictionary_entry.id)` and nothing else, [never parsed, normalised or compared
numerically](GOVERNANCE.md#governanceid--your-own-identifier-for-the-entry-carried-through-unchanged).

`governance` is an [access-control
class](GOVERNANCE.md#governancecode--an-access-control-class-not-a-label), so `None` here
is a value a caller must handle deliberately rather than fall through: see [the fail-open
hazard](GOVERNANCE.md#the-fail-open-hazard-a-null-class-is-not-no-restriction).

`MatchDecision` is a `str`-backed enum, so `result.decision == "AUTO_APPROVE"`,
`result.decision.value` and `result.decision.name` all work. Prefer comparing against
`MatchDecision.AUTO_APPROVE`.

### `FieldDecision`

`nexus_matcher.domain.models.entities.FieldDecision`. **Not re-exported from the package
root** — import it from that path.

A `str`-backed enum with four members: `AUTO_APPROVE`, `REVIEW`, `REJECT`, **`NO_MATCH`**.
It is the verdict for a **field**, where `MatchDecision` is the verdict for a **candidate**.
The first three spellings are shared with `MatchDecision` by contract; `NO_MATCH` is the
member that closes the hole `MatchDecision` cannot express, and the reason the two enums are
separate rather than one enum widened — widening `MatchDecision` would have sent a fourth
value down a wire whose existing clients deserialise three.

```python
session = matcher.match_schema_session("customer.avsc")
for path, verdict in session.field_decisions().items():
    if verdict is FieldDecision.NO_MATCH:
        ...   # inherit nothing, whatever session.results[path] contains
```

`field_decisions()` returns one entry per field in schema order, including fields with no
candidates, which come back `NO_MATCH` rather than being dropped. It is derived through
`derive_field_decision(matches, absolute_score_floor)` in the same module, so a library
caller and an HTTP caller are reading the same rule and not two implementations of it.
`MatchingSession.absolute_score_floor` carries the floor the session was matched under; over
HTTP the same number is `scoring.absoluteScoreFloor`. Measuring one:
[Calibrating the absolute score floor](guides/absolute_score_floor.md).

### `ScoreBreakdown`

`fused_retrieval_score`, `lexical_score`, `edit_distance_score`,
`type_compatibility_score`, `domain_score`, `graph_boost`, plus optional `colbert_score`
and `cross_encoder_score` (both `None` when no reranker is configured) and
`absolute_cosine`. Property `has_reranking`.

**The first field is `fused_retrieval_score`, and the rename matters.** It used to be
called `semantic_score`, and that name misled every reader of it — including an earlier
revision of this document. It is not a cosine similarity: it is the min-max-normalised
fused retrieval score, with each arm rescaled by its own min and max **over the candidates
retrieved for this field**, then combined with `fusion_alpha`. It is therefore
rank-relative. The rank-1 candidate lands at or just above `fusion_alpha` for essentially
every field, whether the match is excellent or barely plausible — which is why the value is
so often 0.9: 0.9 is `fusion_alpha`. Telling an auditor "semantic score 0.9" claims 90%
semantic similarity and delivers "ranked first among N candidates".

`absolute_cosine` is the number that answers "how similar were they really?" — the raw
dense score before normalisation, and the only one here comparable **across** fields.
`None` when the dense arm did not return that candidate at all, i.e. it reached the result
list through the lexical arm.

`semantic_score` still works, as a property and as a constructor keyword, and emits a
`DeprecationWarning`. Passing both it and `fused_retrieval_score` is a `TypeError` rather
than a coin toss. It is scheduled for removal in 3.0.

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
| `nexus_matcher.vector_stores` | `qdrant`, `memory`, `hnsw` |
| `nexus_matcher.embedding_providers` | `sentence_transformers`, `bundled_onnx` |

`bundled_onnx` is the default `from_config` wires, registered here so plugin discovery can
find the offline encoder too. An earlier revision of this table omitted both it and `hnsw`.

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
