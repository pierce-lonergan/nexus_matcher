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
of `bge-small-en-v1.5` and uses it by default. Everything runs on CPU. `[cli]` is only
for the `nexus-matcher` command -- the Python API needs no extras at all.

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

On Windows, the CLI writes box-drawing characters and a spinner. In a console using the
legacy code page this raises `'charmap' codec can't encode character`. Set
`PYTHONIOENCODING=utf-8` (or use Windows Terminal) before running.

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

```bash
nexus-matcher api
```

The server exposes **health and introspection endpoints only**: `/`, `/health`,
`/health/live`, `/health/ready`, `/health/startup`, plus `/docs`, `/redoc` and
`/openapi.json`.

**There is no HTTP matching endpoint.** Matching over HTTP is not implemented. Use the
Python API or the CLI. See [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

---

## Troubleshooting

**`No module named 'sentence_transformers'`** — only needed for the optional torch
encoder: `pip install -e ".[embeddings]"`. The default bundled encoder needs nothing.

**`RuntimeError: Dictionary not loaded. Call load_dictionary() first.`** — `match_schema()`
requires an indexed dictionary; `load_dictionary()` both loads and indexes.

**`ValueError: No loader found for extension .xyz`** — the loader registry is keyed by
file extension. `from_config()` registers `.csv`/`.tsv`/`.txt` and Excel formats only.

**`'charmap' codec can't encode character`** on Windows — set `PYTHONIOENCODING=utf-8`.

**Low confidence across the board** — confidence is dominated by the semantic score
(weight 0.70) plus a domain score (0.15). Dictionary entries with an empty `Definition`
give the embedding model very little to work with; filling in definitions is the single
highest-leverage fix. Field names alone score materially worse — see the query
representation ablation in the [README](README.md#where-the-accuracy-comes-from).

---

## Next

- [README.md](README.md) — measured accuracy, limitations, how to reproduce the benchmark
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) — the real Python, CLI and REST surface
- [docs/BENCHMARK_REGISTRY.md](docs/BENCHMARK_REGISTRY.md) — every benchmark run and its artifact
