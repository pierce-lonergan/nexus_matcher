# NexusMatcher

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/pierce-lonergan/nexus_matcher/blob/main/LICENSE)

Map schema fields onto data dictionary entries by meaning, not by string equality.

You give it a data dictionary (business names + human definitions) and a schema
(Avro, JSON Schema, SQL DDL). For each field it returns ranked candidate entries,
a confidence score, and a decision: `AUTO_APPROVE`, `REVIEW`, or `REJECT`.

It is designed for the case where the incoming column is called `sname` or `AvgScrRead`
and the dictionary says "School Name" or "Average Scholastic Reading Score" — where
there is no common substring to match on.

---

## Honest summary

On a 688-pair labelled benchmark built from BIRD-SQL and OMOP CDM v5.4:

- **P@1 = 0.581**, Recall@10 = 0.878, on CPU, at ~364 fields/sec.
- The half with heavy abbreviations (BIRD) scores **P@1 = 0.601**; the half where the
  signal lives in prose definitions (OMOP) scores **0.575**. Both halves are hard.
- Auto-approve is deliberately conservative: at the default threshold it fires on
  **~12% of fields** and is **95.3% precise** on those. Everything else goes to a human.

Every number in this README comes from a JSON artifact in `benchmarks/results/`,
named next to the number. All of it is single-machine CPU measurement — see
[Limitations](#limitations).

---

## Install

```bash
pip install nexus-matcher
```

That is the whole setup. The wheel **carries its own encoder** -- an int8 ONNX build of
bge-small-en-v1.5, 33.8 MB inside a 22.6 MB wheel -- so there is no model download, no
HuggingFace account, and no torch. It works in an airgapped container on first run.

The base install is the *complete* pipeline, not a stub: the bundled encoder, BM25
lexical retrieval, and CSV/Excel glossary loading all work with nothing else installed.
Anything the default path needs is a real dependency, not an extra -- an extra you have
to install before the quickstart runs is a bug, and this package shipped three of them.

For the last ~2 points of accuracy, the transformer path is still there:

```bash
pip install "nexus-matcher[embeddings]"   # adds torch + sentence-transformers (~800 MB)
```

### If you cannot reach PyPI

Governance tooling often runs where a package index is blocked, so there are three other
routes — all ending with the same wheel and the same bundled encoder. Full detail in
[docs/INSTALL.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/INSTALL.md).

```bash
# GitHub reachable, PyPI not. The encoder is committed, so it comes with the source.
pip install "git+https://github.com/pierce-lonergan/nexus_matcher.git@v2.0.1"

# Airgapped: build the bundle where there IS network, carry it over, install with no index.
python scripts/make_offline_bundle.py
pip install --no-index --find-links wheels nexus-matcher
```

The airgapped path is verified rather than asserted: built here, installed into a fresh
venv with `socket.connect` and `socket.getaddrinfo` patched to raise, then a real
governance match run to completion — **0 network connection attempts**, and no torch, no
pandas.

Or from source:

```bash
git clone https://github.com/pierce-lonergan/nexus_matcher.git
cd nexus_matcher
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
```

Extras defined in `pyproject.toml`: `embeddings`, `vector-stores`, `sparse`, `accel`,
`graph`, `observability`, `quantization`, `static-embeddings`, `colbert`, `cache`,
`parsers`, `loaders`, `api`, `cli`, `async`, `full`, `dev`, and `docs`. **None of them
are needed for the quickstart, and that now includes the CLI** — install extras only for
what they add: `api` for the REST server, `vector-stores` for Qdrant/HNSW, `cache` for
Redis, `embeddings` for the torch encoder, `loaders` to read a dictionary out of a
database or a Parquet file, `accel` for BLAKE3 hashing and CPU feature detection, and
`graph`, `observability`, `quantization`, `static-embeddings` and `colbert` for the
experimental components under
[What is actually implemented](#what-is-actually-implemented). `full` is everything
except `colbert`, which has no resolvable install on Python 3.12+.

`sparse` and `cli` are kept as names so existing pins keep resolving. `typer` and `rich`
moved into the core dependencies, because the console script is declared unconditionally
and cannot start without them.

`rank-bm25` did NOT: BM25 is built in. `BM25Retriever` runs on a numpy inverted index and
imports nothing, so the lexical arm of hybrid retrieval works on a bare install with no
extra at all. The `sparse` extra now installs only the reference implementation the tests
compare against. (An earlier version of this paragraph said rank-bm25 was a core
dependency — it was, until the inverted-index rewrite stopped importing it, and this text
was not updated with the code.)

---

## Quickstart

**`dictionary.csv`** — column names are auto-detected, so `Term` / `Business Definition`
/ `Subject Area` / `Classification` work just as well as the canonical names below. Pass
`column_mapping=ColumnMapping(...)` to override the detection:

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

**Match them**:

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

Actual output of that script, on a bare `pip install nexus-matcher`:

```
cid        -> Account Balance        0.75  REVIEW
full_nm    -> Customer Full Name     0.83  REVIEW
email      -> Email Address          0.83  REVIEW
bal        -> Account Balance        0.86  REVIEW
birth_dt   -> Date of Birth          0.76  REVIEW
```

Four of the five top-1 matches are correct and **nothing is auto-approved** — every score
sits under the 0.87 bar, so all five go to a human. That is the intended behaviour: the
threshold is tuned for precision on auto-approval, not coverage
(see [Decision policy](#decision-policy)).

The miss is worth reading rather than hiding. `cid` is a bare three-letter abbreviation
with no parent path and no `doc`, which is the hardest input this system takes and the
one case it is measurably weakest on (see [Limitations](#limitations)). It is not a
quantization artifact — the fp32 sentence-transformers encoder also gets it wrong, just
differently (`Customer Full Name`, 0.74). Give that field either a parent path or a one
-line description and it resolves; that single signal is worth +19.3 P@1 on the
benchmark, far more than any change of model.

Note `NexusMatcher.from_config()`, not `NexusMatcher()`. The constructor requires an
embedding provider and a vector store to be passed in; `from_config()` wires up the
defaults (**bundled int8 ONNX encoder**, in-memory vector store, BM25 sparse retriever,
Avro + flattened-Avro parsers, Excel/CSV dictionary loaders). It also accepts a
`MatchingConfig`, or a path to a JSON/TOML file holding its fields:

```python
from nexus_matcher import MatchingConfig, NexusMatcher

matcher = NexusMatcher.from_config(MatchingConfig(auto_approve_threshold=0.85))
matcher = NexusMatcher.from_config("matching.toml")
```

### Same thing from the CLI

```bash
nexus-matcher match customer.avsc -d dictionary.csv
nexus-matcher match customer.avsc -d dictionary.csv -f json -o results.json
```

---

## Encoders

Three tiers. The default needs no setup and no network.

| Provider | Size | P@1 | Throughput | Needs |
|---|---|---|---|---|
| **`BundledOnnxProvider`** (default) | 33.8 MB, in the wheel | ~0.536 | ~1240 q/s | onnxruntime |
| `SentenceTransformersProvider` | 130 MB + ~800 MB torch | 0.560 | ~973 q/s | torch, HF download |
| `StaticEmbeddingProvider` | ~30 MB | 0.494 | ~71000 q/s | model2vec |

```python
from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
    default_embedding_provider,
)

provider = default_embedding_provider()           # bundled -> transformer -> static
provider = default_embedding_provider("bundled")  # force fully offline
```

**int8 is not batch-invariant.** ONNX Runtime selects different quantised GEMM kernels per
input shape, so the same corpus scores P@1 0.5276 at batch 8, 0.5378 at batch 64 and
0.5436 at batch 128. That 1.6-point spread is wider than several effects worth acting on,
so hold batch size fixed when comparing anything -- otherwise you are measuring kernel
selection rather than your change. The fp32 torch path does not behave this way.

**Static embeddings are a fallback, not a shortcut.** They cost 6.5 points on this
benchmark and 8.9 on a FHIR-derived corpus built to mirror flattened-Avro matching. The
fact that seven *transformer* encoders from 22M to 335M parameters all landed within 0.03
P@1 does not transfer to static models. If you want their speed without the accuracy cost,
use them as a first stage: static top-25 followed by transformer rescoring measured 0.5581
against 0.5596 for transformer-only.

---

## Flattened Avro schemas

Match a flattened Avro field to a governed glossary entry, so it inherits that entry's
classification.

```python
import json
from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import (
    FlattenedAvroParser, flatten_avro_schema,
)

# From a flattener's output -- dict, list-of-rows, CSV and JSONL all work
schema = FlattenedAvroParser().parse_file("customer_flat.json").unwrap()

# Or straight from the .avsc, which also preserves `doc`
fields = flatten_avro_schema(json.load(open("customer.avsc")))
```

Conventions follow `GAvroSchemaFlattener`: `_` joins path segments, `__` marks an array
boundary, unions unwrap to their non-null branch, arrays of primitives serialise to a
single column. Unrecognised columns (a `governance_status`, say) survive into
`field.source_metadata`.

**The parser rebuilds the hierarchy instead of treating the name as one token**, which is
where the accuracy lives:

```
customer_addresses__street_name
  -> 'customer, addresses, street name array Street line of the postal address'
```

Parent-path context is worth **+19.3 points of P@1** here, and reproduced at **+19.0** on
an independent FHIR corpus. A flattened name already contains the path; embedding the raw
identifier throws away more accuracy than any model choice would recover.

**One gap worth closing upstream:** `GAvroSchemaFlattener` does not propagate the Avro
`doc` attribute, so its output carries names but no definitions -- and definitions are the
strongest signal after the path. `flatten_avro_schema()` flattens the `.avsc` directly and
keeps `doc`, including inheriting a parent record's doc when a leaf has none.

---

## Measured accuracy

**Benchmark.** 688 labelled query → entry pairs:

| Split | Pairs | Source |
|---|---|---|
| `bird` | 361 | BIRD-SQL dev `database_description/*.csv` — technical column name → human-authored business name + description. Heavily abbreviated. |
| `omop` | 327 | OHDSI OMOP CDM v5.4 field-level spec — `cdmFieldName` → `userGuidance` prose. The entry's business name is the TABLE only (see leakage note), so all discriminating signal is in the definition. |
| `combined` | 688 | Both dictionaries pooled, so every query competes against 687 distractors from two unrelated domains. |

### Known defects in the benchmark corpora

These are properties of the source data, not of the matcher. They cap the achievable
score and should temper how any number here is read.

**OMOP business names collide 9-to-1.** After the leakage fix the entry's business name is
the TABLE, so 327 entries share only **36 distinct names** (`measurement` x21,
`observation` x20, `drug exposure` x18). Retrieval finds the right table easily -- OMOP
Recall@10 is 0.936 -- and then has to separate ~20 fields whose only distinguishing text
is a paragraph of `userGuidance`. That is why OMOP P@1 is 0.575 despite high recall.

**87% of BIRD entries have no description.** 314 of 361 carry `description: ""`, so the
entire document is a ~5-word business name. There is very little text to match against.

**BIRD contains label noise.** The description CSVs are misaligned in places. The first
record in `data/benchmarks/bird/dictionary.jsonl` maps the field `County Name` to the gold
label **"County Code"**. Some fraction of measured errors is therefore unwinnable. Spot
checks suggest low single digits; it is not cleanly detectable automatically, because the
obvious heuristic (gold label matches a different field in the same table) flags 4.4% of
entries and most of those are correct labels caught on a substring.

**Most errors are near-misses, not wild misses.** Instrumenting the error set:
**72% of failures pick a wrong entry from the RIGHT source table** -- `lap` -> "lap time"
instead of "lap number", `amount` -> "amount of money" instead of "amount budgeted". The
task is not "find the right neighbourhood", which the system already does; it is
fine-grained discrimination inside it. That is consistent with the measured ~0.002 cosine
margin between the gold entry and the best wrong one, and it is why techniques aimed at
recall or at abbreviation expansion do not move the number.

### Benchmark construction, and a leak we found in it

An earlier revision of this benchmark reported **P@1 0.715**. That number was wrong and is
retracted. The OMOP split synthesised each entry's business name as
`humanise(table) + humanise(field)` — but the *query* representation is also
`humanise(table) + humanise(field)`, so query and gold label were token-identical. Mean
token overlap measured **1.000** for OMOP against **0.240** for BIRD. Half the benchmark
was scoring string identity and reporting it as semantic matching.

OMOP publishes no human-authored business name; the only prose it ships is `userGuidance`.
So the entry's business name is now the **table only** — shared by every field in that
table and therefore unable to identify any one of them — and all discriminating signal
must come from the definition. Verified: indexing on business name alone now scores
**P@1 0.046**, i.e. chance. Rows with no definition are dropped (432 → 327 entries).

This mattered beyond the headline. Two techniques that looked like wins on the leaky
benchmark **inverted** once it was fixed:

| Technique | On leaky benchmark | On corrected benchmark |
|---|---|---|
| cross-encoder reranking | +5.5 P@1 | **−1.3 P@1** |
| dictionary-side alias generation | +12.1 P@1 | **+1.5 P@1** |

Both had been credited for recovering identity matches. If you take one thing from this
README, take that: measure the benchmark before you measure the model.

Leakage control: dictionary entries are indexed on **business name + definition only**.
The source system's technical column name is blanked before indexing
(`eval_pipeline.py`, `logical_name=""`), so nothing can be solved by string identity.
BIRD rows whose business label is just the technical name modulo case and underscores
are dropped at build time for the same reason.

**End-to-end `NexusMatcher` on `combined`** — BAAI/bge-small-en-v1.5, CPU, sparse
retrieval on, BGE query-instruction prefix on
(`benchmarks/results/eval_pipeline_combined.json`):

| Metric | Value |
|---|---|
| P@1 | 0.581 |
| P@5 | 0.826 |
| MRR@10 | 0.685 |
| Recall@10 | 0.878 |
| Throughput | ~364 fields/sec |
| Index build (688 entries) | ~1.8 s |

Accuracy figures vary by about ±0.002 between runs and throughput varies more than that with machine load, which is why they are
quoted to three decimals and with a `~`. Treat a change smaller than ±0.005 P@1 as noise
rather than as a result.

Per split, same configuration: **bird P@1 0.601** (`eval_pipeline_bird.json`), **omop
P@1 0.575** (`eval_pipeline_omop.json`). They are hard for different reasons — BIRD's
names are opaque abbreviations, OMOP's meaning sits in a paragraph of prose — and the
system is roughly equally bad at both.

### Decision policy

Confidence is thresholded into `AUTO_APPROVE` / `REVIEW` / `REJECT`. The threshold is
calibrated against the benchmark rather than picked by feel
(`benchmarks/results/exp_calibration_combined.json`, n=688):

| Threshold | Coverage | Precision on auto-approved |
|---|---|---|
| 0.85 | 20.8% | 0.916 |
| **0.87 (default)** | **12.4%** | **0.953** |

### Dictionary aliasing: a gain that inverts at scale

`MatchingConfig.dictionary_alias_count` indexes fabricated technical spellings of each
glossary term ("Number of Test Takers" -> "num tst takr", "ntt") and max-pools them.
It is **off by default**, and the reason is the most useful negative result here
(`benchmarks/results/exp_alias_scale.json`):

| Dictionary entries | aliases off | aliases on | delta |
|---|---|---|---|
| 688 | 0.5814 | 0.6003 | **+1.9** |
| 10,000 | 0.5044 | 0.3677 | **-13.7** |
| 30,000 | 0.4666 | 0.2791 | **-18.8** |

The gain inverts between 688 and 10k entries. The mechanism is inherent to max-pooling
rather than a tuning problem: every DISTRACTOR also gets N extra chances to beat the
gold entry, so alias noise grows with corpus size while the signal does not. Enable it
only on a genuinely small dictionary, and only after measuring on your own data.

Auto-approving a wrong mapping costs more than sending a field to review, so the default
targets ~95% precision on the auto-approved slice and accepts that the majority of fields
get reviewed by a human.

These numbers move when retrieval changes. Better retrieval shifts the whole score
distribution up, pushes more candidates over a fixed bar, and can *lower* auto-approve
precision. Re-run `benchmarks/exp_calibration.py` after changing the model, the fusion
weights, or the query representation.

### Where the accuracy comes from

Query representation ablation, first-stage dense retrieval on `combined`
(`benchmarks/results/exp_query_repr_combined.json`):

| Query text | P@1 |
|---|---|
| raw field name | 0.361 |
| + underscores split | 0.366 |
| + abbreviation expansion | 0.377 |
| **+ parent-path context** | **0.560** |
| + scalar type words ("text field") | 0.353 |

Parent-path context is worth **+19.3 points of P@1** and is by far the largest single
factor: `sname` is ambiguous, `satscores sname` is not. Appending scalar type words
*hurt* by 2.1 points and is off by default.

Adding the BGE query-instruction prefix to queries (and not to documents) is worth a
further **+5.3 points** (0.438 -> 0.491, measured before parent-path context was added).
The provider owns applying it asymmetrically so a caller cannot forget.

Fusion of dense and sparse arms, same benchmark
(`benchmarks/results/exp_fusion_combined.json`):

| Method | P@1 | Recall@10 |
|---|---|---|
| **linear min-max, dense=0.9 (default)** | **0.702** | 0.910 |
| dense only | 0.691 | 0.908 |
| linear, dense=0.8 | 0.686 | 0.916 |
| combsum / combmnz | 0.657 | 0.923 |
| sparse only | 0.542 | 0.799 |
| RRF (k=60) | 0.610 | 0.903 |

Worth stating plainly: **RRF was the worst fusion method measured here, and worse than
not fusing at all.** It is widely recommended as a default; on this corpus it is not.
RRF discards score magnitude, and the dense arm's magnitudes carry real signal when one
retriever is much stronger than the other. Any docstring in this repo that recommends RRF
as best practice is contradicted by `exp_fusion_combined.json`.

### Optional: cross-encoder reranking

Rerankers are **off by default**. Turning one on buys accuracy and costs throughput
(`benchmarks/results/exp_rerank_combined.json`, reranking the dense shortlist on
`combined`):

| Stage | P@1 | P@5 | MRR@10 | Rerank-stage throughput |
|---|---|---|---|---|
| first stage only | 0.560 | 0.810 | 0.667 | n/a |
| + `cross-encoder/ms-marco-MiniLM-L-6-v2` (depth 10) | **0.547** | 0.824 | 0.663 | 87 queries/sec |

**Cross-encoder reranking makes this task WORSE, and is off by default.** It costs 1.3
points of P@1 on `combined` and 2.8 points on `bird`, while cutting throughput ~7x.

An earlier version of this README reported +5.5 points for the same reranker. That number
was measured against a **defective benchmark**: the OMOP half then derived its business
name from the field name, so query and gold label were token-identical and the task was
string identity rather than semantic matching. The reranker was good at confirming
identity matches. Once the leak was removed (see "Benchmark construction" above), the
gain inverted to a loss. The old number is retracted.

The mechanism is worth internalising: MS-MARCO cross-encoders are trained on
natural-language QUESTION -> passage relevance. Our query is `satscores sname` — not a
question, barely even language. Nothing about that training transfers.

Also measured, and consistent with the above: **larger rerankers did worse still.**
`BAAI/bge-reranker-base` scored 4.7 points below the MiniLM-L-6 and was 7x slower;
`ms-marco-MiniLM-L-12-v2` underperformed the L-6. Do not assume a bigger reranker is a
better one — measure it on your own corpus before shipping it.

A reranker is still supported, because a reranker fine-tuned on YOUR schema pairs is a
different proposition from a stock MS-MARCO one. Measure before enabling:

```python
from nexus_matcher.infrastructure.adapters.rerankers.cross_encoder import CrossEncoderReranker

matcher = NexusMatcher(
    embedding_provider=...,
    vector_store=...,
    reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
)
```

Verify with `python benchmarks/exp_rerank.py --benchmark combined --depth 10`. On the
corpora shipped here that command reports a LOSS.

---

## Reproducing the benchmark

The benchmark is built from third-party corpora that are **not vendored in this repo**.
Place the raw sources under `data/raw/` first:

| Path | Source |
|---|---|
| `data/raw/dev_20240627/dev_databases/*/database_description/*.csv` | BIRD-SQL dev set |
| `data/raw/omop_cdm_field_level.csv` | OHDSI OMOP CDM v5.4 field-level specification |

Then:

```bash
# 1. Build data/benchmarks/{bird,omop,combined}/{dictionary,queries}.jsonl
python benchmarks/datasets/build_benchmarks.py

# 2. End-to-end accuracy + throughput through the real NexusMatcher pipeline
python benchmarks/eval_pipeline.py --benchmark combined --save
python benchmarks/eval_pipeline.py --benchmark bird
python benchmarks/eval_pipeline.py --benchmark omop
```

`eval_pipeline.py` drives the actual `NexusMatcher` orchestrator — enrichment,
abbreviation expansion, dense retrieval, BM25, fusion, multi-signal scoring, decision
policy. It is the number a library user actually gets.

Supporting experiments, each writing its own artifact into `benchmarks/results/`:

```bash
python benchmarks/exp_query_repr.py   --benchmark combined   # query representation ablation
python benchmarks/exp_fusion.py       --benchmark combined   # fusion method comparison
python benchmarks/exp_calibration.py  --benchmark combined   # threshold / coverage curve
python benchmarks/exp_rerank.py       --benchmark combined   # reranker accuracy vs cost
```

`benchmarks/eval_harness.py` measures retrieval strategies in isolation, without the
orchestrator.

---

## Limitations

Read this before trusting the headline number.

- **One machine, CPU only.** Every measurement here was taken on a single Windows 11
  workstation (AMD64, 32 logical cores, no AVX-512/VNNI), Python 3.13, torch 2.13 CPU
  build. Throughput figures will not transfer to your hardware. No GPU numbers exist.
- **Abbreviation-heavy schemas are hard.** bird P@1 is 0.598. Roughly half the fields in
  a BIRD-style schema will *not* have the right answer at rank 1. Recall@10 is much
  better than P@1, which is why the product surface is a ranked review list, not a
  silent auto-mapping.
- **793-entry dictionary for the headline numbers.** The corpus those are measured on is
  enterprise-glossary-sized, not catalogue-sized. Accuracy at catalogue scale IS measured,
  separately: `benchmarks/results/exp_scale_combined.json` records P@1 **0.589** (in-memory)
  and **0.591** (HNSW) at **100,000 entries** — so it degrades with corpus size, as the
  0.0024 gold-vs-nearest-wrong margin predicts it must. Treat the headline figures as an
  upper bound for a glossary of a few hundred entries, not as a promise at catalogue scale.
- **English only.** All measurement is English-language field names and definitions.
- **Two domains.** Healthcare (OMOP) and assorted OLTP (BIRD). Your domain vocabulary is
  not represented; re-run the calibration on your own labelled sample before relying on
  auto-approve.
- **The auto-approve threshold is corpus-specific.** 0.85 is calibrated on this
  benchmark. It is not a universal constant.

---

## What is actually implemented

Being explicit, because earlier versions of this document were not.

**Python library** — the primary interface, and the one that is benchmarked.

**CLI** (`nexus-matcher`, entry point `nexus_matcher.presentation.cli.main:app`) — four
commands: `match`, `sync`, `api`, `info`.

**REST API** (`nexus_matcher.presentation.api.app:create_app`) — the complete route
table, enumerated from a live app:

| Method | Path | What it serves |
|---|---|---|
| POST | `/api/v1/match` | Up to 100 schema fields; returns the ranked dictionary entries and the protection class each field would inherit, one **verdict per column** in `fieldDecisions`, and a `scoring` block saying what the numbers mean |
| POST | `/api/v1/match/batch` | The same contract with a 250-field cap, for chunked clients |
| POST | `/api/v1/feedback` | Appends a reviewer's verdict to an audit log. Recorded, never fed back into ranking |
| POST | `/api/v1/lookup` | Resolve dictionary ids you already hold. No scoring, no ranking, no decision. Every id comes back once, in the order sent, carrying an entry or an explicit `null`. Ids are matched as **exact strings** — `123` does not resolve `0000123` |
| GET | `/api/v1/lookup/{governance_id:path}` | The single-id form of the above, answering the identical body under one key. A miss is **200** with `null`, not a 404 |
| GET | `/api/v1/status` | Entry count, dictionary provenance, the active encoder and whether a fallback one is in force, the live thresholds and caps. Always **200**; read `degraded` before a bulk run |
| POST | `/api/v1/diag/retrieval` | Why a field retrieved what it did: the query text it became, each channel's candidates with that channel's own raw scores, and where an expected entry ranked. Retrieval only |
| GET | `/` | Service identity |
| GET | `/health` | Health check |
| GET | `/health/live` | Kubernetes liveness probe |
| GET | `/health/ready` | Readiness probe (503 if a registered component is not ready) |
| GET | `/health/startup` | Startup probe (503 while starting) |
| GET | `/docs`, `/redoc`, `/openapi.json` | Generated OpenAPI documentation |

**A match response has four top-level keys, in this order: `results`, `vocabulary`,
`fieldDecisions`, `scoring`** — plus `contrast` and `consistency`, appended only when the
request asks for them. `results` was once the whole body and everything since has
been appended to it, never placed in front, so a client generated against an earlier shape
still reads every key it knew at the key it knew. The one a consumer writes down is
`fieldDecisions[path]` — **one verdict per column**, and the only place the value
`NO_MATCH` can appear. Each candidate carries `absoluteScore` beside `confidence`, and
`scoring.comparability` says which of the two may be compared across fields (`absoluteScore`)
and which is meaningful only inside one field (`confidence`). Every key is spelled out, with
a captured response, in
[docs/API_REFERENCE.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/API_REFERENCE.md#the-matching-response).

`NO_MATCH` — *nothing in this response may be inherited by this field* — is the verdict the
per-candidate `decision` cannot express, because `confidence` has a structural floor above
`review_threshold` and no rank-1 match can therefore be rejected on score alone. Reaching it
on a low score means configuring `absolute_score_floor`, which ships **off**: a floor is a
statement about a score distribution and the distribution belongs to your glossary, not to
this library. Measuring one is
[docs/guides/absolute_score_floor.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/guides/absolute_score_floor.md).

### The two members you build against, and the one way to get them wrong

**`governanceId` is your own identifier for the matched glossary entry, carried through
unchanged** — the handle you join a match back to your system with. It is **opaque** and
typed **`string`**: the library never parses it, never normalises it beyond the loader
stripping whitespace around the cell, and never compares it numerically. A deployment whose
ids are zero-padded numbers gets those bytes back, so `0000123` does not resolve from
`123` — `POST /api/v1/lookup` with `{"ids":["0000123","123"]}` puts the unpadded form in
`missing`. Store it as text and join on it as text.

**`governance.code` is an access-control class** — your own description of how protected a
data element is, of the kind an organisation writes in order to decide who or what may read
a column. This library still defines none of them; your vocabulary file is the only source.
But the value is **security-relevant**, not descriptive metadata, and that has one
consequence worth putting in a README:

> **`governance: null` is not "no restriction".** A consumer that maps `code` onto read
> permissions and treats a `null` as "no rule applies" has made that column
> **world-readable**. The safe reading of "I could not classify this" is your **most
> restrictive** class, not your least. `null` is produced by five different situations —
> the entry carries no code and sits at your open tier; the top candidate was rejected and
> confers nothing; a reviewer decided the field; the field has no answer at all; or the
> server was never given a vocabulary — and the response publishes
> `vocabulary.openClassification`, `fieldDecisions` and `provenance` precisely so a
> consumer can tell them apart. Check `vocabulary.openClassification` is not
> `UNCLASSIFIED` once per response, then read `fieldDecisions[path]` per field, before
> anything else.

The hazard is argued, with captures of all five, in
[docs/GOVERNANCE.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/GOVERNANCE.md#the-fail-open-hazard-a-null-class-is-not-no-restriction);
the seven-step recipe and the three checks to run on your own deployment are in
[docs/guides/governance_as_access_control.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/guides/governance_as_access_control.md).

The two match routes answer **503** until a dictionary is loaded, and `/api/v1/feedback`
answers **503** until a feedback file is configured; each 503 names the setting to change.
Both shipped ways to start the server call `create_app()` with no arguments, so the wiring
is environment-driven: `NEXUS_API_DICTIONARY` and `NEXUS_API_GOVERNANCE` for matching,
`NEXUS_API_FEEDBACK_PATH` for recording. The wire contract, an operator's start command and
a verified request and response are in
[docs/GOVERNANCE.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/GOVERNANCE.md#matching-over-http);
[examples/governance/serve.sh](https://github.com/pierce-lonergan/nexus_matcher/blob/main/examples/governance/serve.sh)
starts a server over the example pack in one command.

There are still no dictionary CRUD endpoints, no cache endpoints and no `/metrics`
endpoint. Earlier revisions of this section denied that the three `/api/v1` routes
existed, in nine places across the documentation, while `create_app()` registered them;
that claim is retracted and
[tests/packaging/test_documented_routes.py](https://github.com/pierce-lonergan/nexus_matcher/blob/main/tests/packaging/test_documented_routes.py)
now fails the build in both directions. See
[docs/API_REFERENCE.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/API_REFERENCE.md).

**Schema parsers**: Avro, JSON Schema, SQL DDL. (`from_config()` registers Avro only;
pass the others explicitly via `schema_parser_registry`.)

**Dictionary loaders**: Excel, CSV.

**Vector stores**: in-memory, Qdrant, HNSW.

**Caches**: L1 LRU in-memory, Redis, content-addressed semantic cache. These are
implemented and unit-tested but are **not** exercised by the accuracy benchmark and have
no committed performance artifacts — see [docs/BENCHMARK_REGISTRY.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/BENCHMARK_REGISTRY.md).

**Also present, not part of the benchmarked path**: ColBERT MaxSim reranker, INT8
quantized embedding provider, incremental update manager, learned type projections,
graph matcher. Treat these as experimental; their measurements are component-level
microbenchmarks, recorded with their caveats in the benchmark registry.

---

## Repository layout

```
src/nexus_matcher/          canonical package
  domain/                   entities, ports, services (abbreviation, context, domain, types)
  application/use_cases/    NexusMatcher orchestrator, batch matching
  infrastructure/adapters/  parsers, loaders, embedding providers, vector stores,
                            sparse retrievers, rerankers, caches, incremental
  presentation/             CLI (typer), API (fastapi)
  core/                     fusion, graph matcher, type projections
  shared/                   types, DI container, logging, metrics, plugins
benchmarks/                 eval_pipeline.py, eval_harness.py, exp_*.py, suite_*.py
benchmarks/datasets/        build_benchmarks.py
benchmarks/results/         JSON artifacts — the source of every number above
data/benchmarks/            generated benchmark corpora
tests/                      unit + integration
docs/                       architecture, API reference, benchmark registry
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
mypy src
```

Current state, measured: **551 passed, 0 failed, 35 skipped**. The skips are optional
dependencies that are not installed in this environment (`qdrant-client`, `redis`,
`blake3`) — install the relevant extra to run them. Line coverage is **60%** against a
configured gate of 80% (`pyproject.toml`, `[tool.coverage.report] fail_under = 80`), so
a bare `pytest` run exits non-zero on the coverage gate even when every test passes.

One test, `test_hash_performance_faster_than_sha256`, is timing-sensitive and has been
observed to fail intermittently under load.

---

## Documentation

- [QUICKSTART.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/QUICKSTART.md) — the verified five-minute path
- [docs/API_REFERENCE.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/API_REFERENCE.md) — the Python, CLI and REST surfaces
- [docs/GOVERNANCE.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/GOVERNANCE.md) — governance inheritance, what `governanceId` and `governance.code` are, the fail-open hazard, and the matching endpoint's wire contract
- [docs/guides/governance_as_access_control.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/guides/governance_as_access_control.md) — turning a match into a read permission without failing open: the recipe, the five nulls, and the checks to run
- [docs/guides/absolute_score_floor.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/guides/absolute_score_floor.md) — how to measure a `NO_MATCH` floor for your own corpus, and why no default ships
- [docs/guides/governed_abbreviations.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/guides/governed_abbreviations.md) — using your own approved-abbreviation catalog
- [docs/BENCHMARK_REGISTRY.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/BENCHMARK_REGISTRY.md) — every benchmark run, with its artifact or an explicit note that it has none
- [docs/ARCHITECTURE.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/ARCHITECTURE.md) — hexagonal layering and component wiring
- [docs/ENHANCEMENT_JOURNEY.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/docs/ENHANCEMENT_JOURNEY.md) — what was changed and what it measured
- [CHANGELOG.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/CHANGELOG.md)
- [CONTRIBUTING.md](https://github.com/pierce-lonergan/nexus_matcher/blob/main/CONTRIBUTING.md)

---

## License

Apache 2.0 — see [LICENSE](https://github.com/pierce-lonergan/nexus_matcher/blob/main/LICENSE).

---

## Acknowledgments

- **Pierce Lonergan** - Architecture and implementation
- **Sentence-Transformers** team - Embedding models
- **Qdrant** team - Vector search infrastructure
- **FastAPI** team - Web framework

---

<div align="center">

**Built for enterprise data engineering**

[Documentation](https://github.com/pierce-lonergan/nexus_matcher/tree/main/docs) • [Issues](https://github.com/pierce-lonergan/nexus_matcher/issues) • [Discussions](https://github.com/pierce-lonergan/nexus_matcher/discussions)

</div>
