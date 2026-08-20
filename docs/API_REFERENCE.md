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
)
```

`embedding_provider` and `vector_store` are positional-or-keyword and **required**.
`NexusMatcher()` with no arguments raises `TypeError`.

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
| POST | `/api/v1/match` | `MatchResponseView` — four top-level keys in this order: `results` (one entry per input field, keyed by the caller's own `path`, in the order sent), `vocabulary`, `fieldDecisions` (one verdict per column, the only place `NO_MATCH` is expressible) and `scoring`. Each candidate carries `absoluteScore` beside `confidence`. Every key is spelled out under [The matching response](#the-matching-response) rather than restated here. Field cap `NEXUS_API_MAX_FIELDS` (default 100). |
| POST | `/api/v1/match/batch` | Identical contract and one shared implementation; the only difference is the cap, `NEXUS_API_MAX_BATCH_FIELDS` (default 250). |
| POST | `/api/v1/feedback` | **201** and `FeedbackResponseView` — the stored record echoed back, server `receivedAt` included. Appended to `NEXUS_API_FEEDBACK_PATH`; **503** when that is unset, **422** on a malformed record, **500** when the append itself fails. |
| POST | `/api/v1/lookup` | `LookupResponseView` — `results` maps every id you sent, once and in the order sent, to an entry or an explicit `null`; `missing` names exactly the nulls; `vocabulary` is the same block `/match` carries. No score, no rank, no decision. Id cap `NEXUS_API_MAX_BATCH_FIELDS`; **413** over it, **422** on a duplicate, blank or oversized id, **503** with no dictionary. |
| GET | `/api/v1/lookup/{governance_id:path}` | The single-id form, answering the identical `LookupResponseView` under one key. A miss is **200** with `null` — 404 on this service means no such route. The `:path` converter is what keeps an id containing `/` addressable; OpenAPI publishes the path as `/api/v1/lookup/{governance_id}`. |
| GET | `/api/v1/status` | `StatusResponseView` — `ready`, `degraded`, `warnings[]`, `dictionary`, `encoder`, `thresholds`, `limits`. Always **200**, including with no dictionary loaded, because a pre-run degradation check that refused when the condition holds is unusable exactly when it is needed. Every threshold is nullable: `null` means this matcher does not expose it, never `0.0`. |
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

Ten keys, in this order: `rank`, `governanceId`, `businessName`, `definition`, `domain`,
`governance`, `confidence`, `decision`, `absoluteScore`, `sourceMetadata`.

| Key | Type | Meaning |
|---|---|---|
| `rank` | `int` | 1-based position within this field's candidate list. |
| `governanceId` | `string` | The dictionary entry's id — the thing you join on. |
| `businessName` | `string` | The entry's business name. |
| `definition` | `string` | The entry's definition. |
| `domain` | `string` | The entry's subject area. |
| `governance` | object or `null` | The protection class the entry confers. Six keys; see below. `null` is a documented value. |
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

`enhancement` is the newest and was **appended**, so the order of the five before it is
unchanged and a client reading them by name is unaffected.

| Key | Type | Meaning |
|---|---|---|
| `code` | `string` | The protection code, resolved through the caller's vocabulary — aliases already followed. The example pack maps `GBF-LEGACY-NAME` to `MANIFEST_NAME`, which is why `GBF-0001` comes back as the latter. |
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

`absoluteScoreFloor` is **not** published on `GET /api/v1/status` — that route's
`thresholds` block carries `autoApprove`, `review`, `minConfidenceGap`, `resultsPerField`,
`fusionAlpha`, `minimumAchievableConfidence` and `reviewThresholdBelowFloor`, and no floor.
To read a deployment's floor without matching anything, send a one-field match and read
this block.

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
    "reviewThresholdBelowFloor": true
  },
  "limits": {
    "maxFields": 100,
    "maxBatchFields": 250,
    "bodyByteCap": 9873024,
    "deadlineSeconds": 25.0,
    "capacity": 36
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
| `warnings` | `object[]` | Zero or more `{code, message}` records. Empty list, never `null`. **Branch on `code`, never on `message`**: the codes are `NO_DICTIONARY`, `EMPTY_DICTIONARY` and `FALLBACK_ENCODER`; the message is human-readable and not part of the contract. |
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
| `limits.maxFields` | `int` | Cap on `/api/v1/match`. |
| `limits.maxBatchFields` | `int` | Cap on `/api/v1/match/batch`. |
| `limits.bodyByteCap` | `int` | Request bytes accepted before a **413**, derived from the field caps rather than typed as a literal. |
| `limits.deadlineSeconds` | `float` | Wall-clock budget for one match before it is shed. |
| `limits.capacity` | `int` | Concurrent + queued requests admitted before a **503**. |

**`absoluteScoreFloor` is deliberately not here.** The `thresholds` block is the seven keys
above and no floor; the active floor is published per response at
`scoring.absoluteScoreFloor`. To read a deployment's floor without matching anything, send a
one-field match and read `scoring`.

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
answers to "whose class is this?" is worse than none.

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
