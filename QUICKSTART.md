# NexusMatcher Quick Start

Every command and code block on this page was executed against this repository and the
outputs shown are the real ones.

---

## 1. Install

```bash
git clone https://github.com/pierce-lonergan/nexus_matcher.git
cd nexus_matcher
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[cli]"
```

No model download and no HuggingFace account: the package carries an int8 ONNX build
of `bge-small-en-v1.5` and uses it by default. Everything runs on CPU. No extra is
required for anything on this page: `[cli]` still resolves, so the line above keeps
working, but typer and rich are core dependencies now and `pip install -e .` gets you
the `nexus-matcher` command too.

---

## 2. Two input files

**`dictionary.csv`** — headers are auto-detected (`Term`, `Business Definition`,
`Classification` and similar are all recognised). If yours
differ, pass a `ColumnMapping` to `load_dictionary()`.

```csv
ID,Business Name,Logical Name,Definition,Data Type,Domain
DICT-001,Customer Identifier,cust_id,Unique identifier assigned to a customer account,integer,customer
DICT-002,Customer Full Name,cust_nm,Full legal name of the customer,string,customer
DICT-003,Email Address,email_addr,Primary contact email address for the customer,string,customer
DICT-004,Account Balance,acct_bal,Current monetary balance of the account,float,finance
DICT-005,Date of Birth,dob,Calendar date on which the customer was born,date,customer
```

**`customer.avsc`**:

```json
{
  "type": "record",
  "name": "Customer",
  "fields": [
    {"name": "cid", "type": "int"},
    {"name": "full_nm", "type": "string"},
    {"name": "email", "type": "string"},
    {"name": "bal", "type": "double"},
    {"name": "birth_dt", "type": "string"}
  ]
}
```

---

## 3. Match, from Python

```python
from nexus_matcher import NexusMatcher

matcher = NexusMatcher.from_config()
matcher.load_dictionary("dictionary.csv")

results = matcher.match_schema("customer.avsc")
for field_path, matches in results.items():
    top = matches[0]
    print(f"{field_path:10s} -> {top.dictionary_entry.business_name:22s} "
          f"{top.final_confidence:.2f}  {top.decision.name}")
```

Real output:

```
cid        -> Account Balance        0.75  REVIEW
full_nm    -> Customer Full Name     0.83  REVIEW
email      -> Email Address          0.83  REVIEW
bal        -> Account Balance        0.86  REVIEW
birth_dt   -> Date of Birth          0.76  REVIEW
```

Four of five top-1 matches are right and nothing is auto-approved -- every score is under
the 0.87 bar, so all five go to a human. That is the design: `auto_approve_threshold` is
calibrated for ~95% precision on the fields it approves, not for coverage.

`cid` is the miss, and it is the expected one: a bare three-letter abbreviation with no
parent path and no description. The fp32 torch encoder gets it wrong too. See the
calibration table in the
[README](README.md#decision-policy).

### Two things that are easy to get wrong

**`NexusMatcher()` does not work.** The constructor requires at minimum an
`embedding_provider` and a `vector_store`. Use `NexusMatcher.from_config()` for the
wired-up defaults, or pass components explicitly (below).

**`from_config()` honours what you pass it.** It takes a `MatchingConfig`, or a path to
a JSON/TOML file holding its fields, or nothing for the calibrated defaults:

```python
from nexus_matcher import MatchingConfig, NexusMatcher

matcher = NexusMatcher.from_config(MatchingConfig(auto_approve_threshold=0.85))
matcher = NexusMatcher.from_config("matching.toml")
```

An unknown key in that file raises rather than being ignored -- every field is a measured
number, and a silently dropped `auto_approve_treshold` would leave you believing you had
raised the bar. (This argument was previously named `config_path` and was accepted and
then never read.) The `NEXUS_*` settings in
`nexus_matcher.infrastructure.config.settings` remain logging-only.

---

## 4. Match, from the CLI

```bash
nexus-matcher match customer.avsc -d dictionary.csv
nexus-matcher match customer.avsc -d dictionary.csv -f json -o results.json
nexus-matcher match customer.avsc -d dictionary.csv -k 3 -t 0.5
```

Available commands: `match`, `sync`, `api`, `info`. Run `nexus-matcher --help`.

On a Windows console using a legacy code page the CLI drops to ASCII decorations and an
ASCII spinner, and escapes anything else it cannot encode rather than aborting on it.
Nothing needs setting. Use Windows Terminal, or set `PYTHONIOENCODING=utf-8`, if you want
box drawing and non-ASCII field names rendered exactly.

---

## 5. Explicit component wiring

Use this when you want a specific model, extra schema parsers, or non-default thresholds.

```python
from nexus_matcher import NexusMatcher
from nexus_matcher.application.use_cases.match_schema import MatchingConfig
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.embedding_providers.sentence_transformers import (
    SentenceTransformersProvider,
)
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
from nexus_matcher.infrastructure.adapters.schema_parsers.avro import AvroSchemaParser
from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import JsonSchemaParser
from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import SqlDdlParser
from nexus_matcher.infrastructure.adapters.dictionary_loaders.excel import (
    CsvDictionaryLoader,
    ExcelDictionaryLoader,
)

provider = SentenceTransformersProvider(model_name="BAAI/bge-small-en-v1.5")

matcher = NexusMatcher(
    embedding_provider=provider,
    vector_store=InMemoryVectorStore(
        VectorStoreConfig(collection_name="dictionary", dimension=provider.dimension)
    ),
    sparse_retriever=BM25Retriever(),
    schema_parser_registry={
        "avro": AvroSchemaParser(),
        "json_schema": JsonSchemaParser(),
        "sql_ddl": SqlDdlParser(),
    },
    dictionary_loader_registry={
        "csv": CsvDictionaryLoader(),
        "excel": ExcelDictionaryLoader(),
    },
    config=MatchingConfig(auto_approve_threshold=0.85, results_per_field=5),
)

matcher.load_dictionary("dictionary.csv")
results = matcher.match_schema("customer.schema.json", schema_format="json_schema")
```

`from_config()` registers the Avro parser only, so registering JSON Schema and SQL DDL
is the reason to wire things by hand.

Real output of the above against the same dictionary and an equivalent JSON Schema:

```
cid          -> Customer Full Name     0.74
full_nm      -> Customer Full Name     0.83
email        -> Email Address          0.83
bal          -> Account Balance        0.86
birth_dt     -> Date of Birth          0.78
```

Note `cid` is now **wrong** — `bge-small-en-v1.5` (the model used for the published
benchmark) misses it where the larger default `bge-base-en-v1.5` gets it. Small models
are cheaper and measurably weaker on short abbreviations. Pick deliberately.

---

## 6. Batch processing

```python
from nexus_matcher import NexusMatcher
from nexus_matcher.application.use_cases.batch_match import BatchProcessor, BatchConfig

matcher = NexusMatcher.from_config()
matcher.load_dictionary("dictionary.csv")

processor = BatchProcessor(matcher, BatchConfig(max_workers=4))
result = processor.process_schemas(["customer.avsc", "order.avsc"])

print(f"{result.successful}/{result.total_schemas} schemas processed, {result.failed} failed")
for session in result.sessions:
    print(" ", session.schema.name, len(session.results), "fields")
```

Real output:

```
2/2 schemas processed, 0 failed
  Customer 5 fields
  Customer 5 fields
```

`BatchProcessor` also exposes `process_directory()` and `process_manifest()`.

---

## 7. The API server

Started bare, the server has no dictionary and every match answers 503 naming the setting
to change. Both shipped entry points call `create_app()` with no arguments, so the
dictionary and the vocabulary are passed through the environment:

```bash
export NEXUS_API_DICTIONARY=examples/governance/glossary.csv
export NEXUS_API_GOVERNANCE=examples/governance/protection_classes.json
nexus-matcher api --host 127.0.0.1 --port 8000
```

`examples/governance/serve.sh` (and `serve.ps1`) is exactly that, plus
`NEXUS_API_FEEDBACK_PATH` so `/api/v1/feedback` works too.

The complete route table:

| Method | Path | What it serves |
|---|---|---|
| POST | `/api/v1/match` | Up to 100 schema fields, keyed by the caller's own `path` |
| POST | `/api/v1/match/batch` | The same contract with a 250-field cap |
| POST | `/api/v1/feedback` | Appends a reviewer's verdict to an audit log |
| POST | `/api/v1/lookup` | Resolve dictionary ids you already hold — entry text and governance, no scoring |
| GET | `/api/v1/lookup/{governance_id:path}` | The single-id form; a miss is 200 with `null`, not a 404 |
| GET | `/api/v1/status` | Entry count, encoder, thresholds and caps. Always 200; `degraded` is the pre-run check |
| POST | `/api/v1/diag/retrieval` | Retrieval trace for one field: query text, per-channel candidates, expected-entry rank |
| GET | `/` | Service identity |
| GET | `/health` | Health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe (503 if a registered component is not ready) |
| GET | `/health/startup` | Startup probe (503 while starting) |
| GET | `/docs`, `/redoc`, `/openapi.json` | Generated OpenAPI documentation |

One request, run against the server started above:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/match \
  -H 'Content-Type: application/json' \
  -d '{"fields":[{"name":"legal_name","path":"booking.passenger.legal_name","doc":"Full legal name of the passenger as printed on the sailing manifest.","type":"string"}],"top_k":1}'
```

Real output, captured from a live app on 2026-08-19 and pasted here unedited except for
pretty-printing. The response itself is one line, in exactly this key order, ASCII-only,
and byte-identical between two identical requests — all three checked on the capture:

```json
{
  "results": {
    "booking.passenger.legal_name": [
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
        "confidence": 0.904167,
        "decision": "AUTO_APPROVE",
        "absoluteScore": 0.784332,
        "sourceMetadata": {
          "values": {
            "personal_information": "yes",
            "direct_identifier": "yes"
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
    "booking.passenger.legal_name": "AUTO_APPROVE"
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

**Four top-level keys, in this order: `results`, `vocabulary`, `fieldDecisions`,
`scoring`.** `results` was once the whole body, so everything since has been appended
rather than placed in front of it.

`vocabulary` echoes back the two things a caller cannot otherwise infer from a response:
which tier an uncoded field sits at, and the order the caller's own file declares its
tiers in. Without it a `"governance": null` is uninterpretable without a second copy of
the vocabulary file, which a service consuming this API may not have.

`enhancement` is the caller's own instruction for how to protect the field — it is
whatever their vocabulary declares, and it is `null` for a class that declares none.

`sourceMetadata` is your glossary's own enrichment columns for the matched entry, carried
through untouched. Nothing in the library reads it.

**Read `fieldDecisions[path]`, not `decision` and not `confidence`.** The three are
different claims:

| Where | Scope | Vocabulary |
|---|---|---|
| `fieldDecisions[path]` | the **column** — the answer you write down | `AUTO_APPROVE`, `REVIEW`, `REJECT`, **`NO_MATCH`** |
| `results[path][n].decision` | one **candidate** | `AUTO_APPROVE`, `REVIEW`, `REJECT` |
| `results[path][n].confidence` | a rank-relative score, normalised inside this field | a number, comparable within the field only |

`NO_MATCH` — *this response carries nothing this field may inherit* — exists only on the
first row, and it is the state the other two cannot express. `confidence` has a structural
floor of 0.63 (published as `scoring.confidenceFloor`) that sits above `review_threshold` =
0.50, so a rank-1 candidate can never be `REJECT` on score alone: **every field comes back
at least `REVIEW`, however irrelevant its best candidate is.**

To make a field report `NO_MATCH` on a low score you have to configure
`absolute_score_floor`, which is off by default because a floor is a statement about a
score distribution and the distribution belongs to your glossary.
[docs/guides/absolute_score_floor.md](docs/guides/absolute_score_floor.md) is the procedure
for measuring one — including the measurement showing that a plausible-looking value can
produce zero `NO_MATCH` verdicts and never fire at all.

`scoring` says what the numbers beside it *mean*: `scoring.comparability` marks
`confidence` `WITHIN_FIELD` and `absoluteScore` `ACROSS_FIELDS`, so `absoluteScore` is the
only per-candidate number you may compare against a fixed constant or between two columns.

Do not diff against `confidence`. It is rank-relative and will move with any retrieval
change.

The field keys are `name`, `path`, `doc` and `type` — **not** the `flattenedName` /
`dataType` spellings used by `examples/governance/fields.json`, which is the pack's own
input format and not the wire contract. Sending those gives a 422, on purpose. The whole
contract, and why the 422 is deliberate, is in
[docs/GOVERNANCE.md](docs/GOVERNANCE.md#matching-over-http).

---

## Troubleshooting

**`No module named 'sentence_transformers'`** — only needed for the optional torch
encoder: `pip install -e ".[embeddings]"`. The default bundled encoder needs nothing.

**`RuntimeError: Dictionary not loaded. Call load_dictionary() first.`** — `match_schema()`
requires an indexed dictionary; `load_dictionary()` both loads and indexes.

**`ValueError: No loader found for extension .xyz`** — the loader registry is keyed by
file extension. `from_config()` registers `.csv`/`.tsv`/`.txt` and Excel formats only.

**`'charmap' codec can't encode character`** on Windows — fixed in 2.0.1, where it killed
`match` and `sync` outright. On 2.0.0, set `PYTHONIOENCODING=utf-8`.

**Low confidence across the board** — confidence is dominated by the semantic score
(weight 0.70) plus a domain score (0.15). Dictionary entries with an empty `Definition`
give the embedding model very little to work with; filling in definitions is the single
highest-leverage fix. Field names alone score materially worse — see the query
representation ablation in the [README](README.md#where-the-accuracy-comes-from).

---

## Next

- [README.md](README.md) — measured accuracy, limitations, how to reproduce the benchmark
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) — the real Python, CLI and REST surface, key by key
- [docs/GOVERNANCE.md](docs/GOVERNANCE.md) — what a match confers, and what a `NO_MATCH` withholds
- [docs/guides/absolute_score_floor.md](docs/guides/absolute_score_floor.md) — how to measure a floor for your own corpus
- [docs/BENCHMARK_REGISTRY.md](docs/BENCHMARK_REGISTRY.md) — every benchmark run and its artifact
