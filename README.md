# NexusMatcher

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

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
git clone https://github.com/pierce-lonergan/nexus_matcher.git
cd nexus_matcher
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[embeddings,parsers,loaders,sparse,cli]"
```

Extras defined in `pyproject.toml`: `embeddings`, `parsers`, `loaders`, `sparse`,
`vector-stores`, `cache`, `api`, `cli`, `async`, `docs`, `dev`, and `full`. The four
above are what the quickstart needs; the CLI additionally needs `cli`. The first run
downloads a sentence-transformers model from HuggingFace (the shipped default is
`BAAI/bge-base-en-v1.5`; the published benchmark uses the smaller
`BAAI/bge-small-en-v1.5`).

---

## Quickstart

**`dictionary.csv`** — column names are the loader's defaults:

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

Actual output of that script:

```
cid        -> Customer Identifier      0.78  REVIEW
full_nm    -> Customer Full Name       0.83  REVIEW
email      -> Email Address            0.83  REVIEW
bal        -> Account Balance          0.86  AUTO_APPROVE
birth_dt   -> Date of Birth            0.76  REVIEW
```

All five top-1 matches are correct; only one clears the auto-approve bar. That is the
intended behaviour — the threshold is tuned for precision on auto-approval, not for
coverage (see [Decision policy](#decision-policy)).

Note `NexusMatcher.from_config()`, not `NexusMatcher()`. The constructor requires an
embedding provider and a vector store to be passed in; `from_config()` wires up the
defaults (sentence-transformers provider, in-memory vector store, BM25 sparse retriever,
Avro parser, Excel/CSV dictionary loaders).

### Same thing from the CLI

```bash
nexus-matcher match customer.avsc -d dictionary.csv
nexus-matcher match customer.avsc -d dictionary.csv -f json -o results.json
```

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
- **793-entry dictionary.** The benchmark corpus is enterprise-glossary-sized, not
  catalogue-sized. Accuracy at 100k entries is unmeasured.
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

**REST API** (`nexus_matcher.presentation.api.app:create_app`) — health and
introspection endpoints **only**:

| Method | Path |
|---|---|
| GET | `/` |
| GET | `/health` |
| GET | `/health/live` |
| GET | `/health/ready` |
| GET | `/health/startup` |
| GET | `/docs`, `/redoc`, `/openapi.json` |

There is **no** HTTP matching endpoint, no dictionary CRUD endpoint, no cache endpoint,
and no `/metrics` endpoint. Matching over HTTP is not implemented. See
[docs/API_REFERENCE.md](docs/API_REFERENCE.md).

**Schema parsers**: Avro, JSON Schema, SQL DDL. (`from_config()` registers Avro only;
pass the others explicitly via `schema_parser_registry`.)

**Dictionary loaders**: Excel, CSV.

**Vector stores**: in-memory, Qdrant, HNSW.

**Caches**: L1 LRU in-memory, Redis, content-addressed semantic cache. These are
implemented and unit-tested but are **not** exercised by the accuracy benchmark and have
no committed performance artifacts — see [docs/BENCHMARK_REGISTRY.md](docs/BENCHMARK_REGISTRY.md).

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

- [QUICKSTART.md](QUICKSTART.md) — the verified five-minute path
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) — Python, CLI, and the (health-only) REST surface
- [docs/BENCHMARK_REGISTRY.md](docs/BENCHMARK_REGISTRY.md) — every benchmark run, with its artifact or an explicit note that it has none
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — hexagonal layering and component wiring
- [docs/ENHANCEMENT_JOURNEY.md](docs/ENHANCEMENT_JOURNEY.md) — what was changed and what it measured
- [CHANGELOG.md](CHANGELOG.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Acknowledgments

- **Pierce Lonergan** - Architecture and implementation
- **Sentence-Transformers** team - Embedding models
- **Qdrant** team - Vector search infrastructure
- **FastAPI** team - Web framework

---

<div align="center">

**Built for enterprise data engineering**

[Documentation](docs/) • [Issues](https://github.com/pierce-lonergan/nexus_matcher/issues) • [Discussions](https://github.com/pierce-lonergan/nexus_matcher/discussions)

</div>
