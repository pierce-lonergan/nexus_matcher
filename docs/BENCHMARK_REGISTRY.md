# Benchmark Registry

Every performance and accuracy claim made anywhere in this repository should be
traceable to a row in this file, and every row should either name a JSON artifact under
`benchmarks/results/` or say plainly that it has none.

Rows are graded:

| Grade | Meaning |
|---|---|
| **A — end-to-end** | Drives the real `NexusMatcher` pipeline against labelled data. Artifact committed. |
| **B — component, real model** | Measures one component with a real model, not the full pipeline. Artifact committed. |
| **C — component, mock** | Measures infrastructure with a mock provider. Tells you the plumbing runs; tells you nothing about accuracy or real-model speed. Artifact committed. |
| **D — unverifiable** | No artifact was ever written. Numbers cannot be checked. Do not cite. |

---

## A — End-to-end accuracy

### EVAL-PIPELINE — `NexusMatcher` on the labelled benchmark

- **Script:** `benchmarks/eval_pipeline.py`
- **Artifact:** `benchmarks/results/eval_pipeline_combined.json`
- **Data:** `data/benchmarks/combined/` — 793 labelled query→entry pairs, built by
  `benchmarks/datasets/build_benchmarks.py` from BIRD-SQL dev `database_descriptions`
  (361) and OHDSI OMOP CDM v5.4 field-level spec (432).
- **Leakage control:** dictionary entries indexed with `logical_name` blanked, so the
  source system's technical column name is not in the corpus. Retrieval works from
  business name + human definition only.
- **Config:** `BAAI/bge-small-en-v1.5`, CPU, sparse retrieval on, BGE query-instruction
  prefix on.

| Metric | Value |
|---|---|
| P@1 | 0.6999 |
| P@5 | 0.8878 |
| MRR@10 | 0.7814 |
| Recall@1 / @5 / @10 | 0.6999 / 0.8878 / 0.9193 |
| Throughput | 652.3 fields/sec |
| Index build (793 entries) | 1.756 s |
| Decisions | AUTO_APPROVE 421, REVIEW 372, REJECT 0 |
| Auto-approve precision at that operating point | 0.9264 |

Per split, same configuration: **bird P@1 0.490**, **omop P@1 0.819**.

Reproduce:

```bash
python benchmarks/datasets/build_benchmarks.py
python benchmarks/eval_pipeline.py --benchmark combined --save
```

---

## A — Ablations and calibration

All of these run against the same 793-pair `combined` benchmark.

### EXP-QUERY-REPR — what the query text should contain

- **Script:** `benchmarks/exp_query_repr.py`
- **Artifact:** `benchmarks/results/exp_query_repr_combined.json`

| Variant | P@1 | P@5 | MRR@10 | R@50 |
|---|---|---|---|---|
| `raw` — bare field name | 0.4880 | 0.7352 | 0.5896 | 0.9016 |
| `underscores` | 0.4905 | 0.7402 | 0.5963 | 0.9029 |
| `split` | 0.4931 | 0.7427 | 0.5995 | 0.9042 |
| `abbrev` | 0.4956 | 0.7503 | 0.6022 | 0.9029 |
| **`context` — + parent path** | **0.6910** | **0.8726** | **0.7706** | **0.9596** |
| `type` — + scalar type words | 0.4691 | 0.7087 | 0.5713 | 0.8953 |
| `full_no_type` | 0.6709 | 0.8764 | 0.7602 | 0.9609 |
| `full` | 0.6671 | 0.8588 | 0.7530 | 0.9533 |

Findings: parent-path context is worth **+20.1 points of P@1** over the bare field name
and is the largest single accuracy factor in the pipeline. Appending scalar type words
("text field") **costs 2.1 points** and is off by default.

### EXP-FUSION — combining dense and sparse

- **Script:** `benchmarks/exp_fusion.py`
- **Artifact:** `benchmarks/results/exp_fusion_combined.json`

| Method | P@1 | P@5 | MRR@10 | Recall@10 |
|---|---|---|---|---|
| **linear min-max, dense=0.9** | **0.7024** | 0.8789 | 0.7773 | 0.9105 |
| dense only | 0.6910 | 0.8726 | 0.7706 | 0.9079 |
| linear, dense=0.8 | 0.6860 | 0.8802 | 0.7707 | 0.9155 |
| linear, dense=0.7 | 0.6822 | 0.8802 | 0.7702 | 0.9193 |
| combsum / linear dense=0.5 | 0.6570 | 0.8865 | 0.7524 | 0.9231 |
| combmnz | 0.6570 | 0.8827 | 0.7508 | 0.9218 |
| rrf k=10 | 0.6192 | 0.8638 | 0.7200 | 0.9105 |
| rrf k=20 | 0.6129 | 0.8499 | 0.7129 | 0.9067 |
| rrf k=60 | 0.6103 | 0.8373 | 0.7076 | 0.9029 |
| rrf k=200 | 0.6103 | 0.8335 | 0.7061 | 0.8953 |
| max_score | 0.6053 | 0.8752 | 0.7251 | 0.9168 |
| sparse only | 0.5422 | 0.7465 | 0.6303 | 0.7995 |

Finding: **RRF is the worst fusion method measured on this corpus at every `k` tried,
and worse than not fusing at all** (dense-only 0.6910 vs best RRF 0.6192). RRF throws
away score magnitude, which matters when one retriever is much stronger than the other.
Docstrings or references in this repo that present RRF as a best-practice default are
contradicted by this artifact.

Note the tension between P@1 and Recall@10: `combsum` has the best Recall@10 (0.9231)
and a materially worse P@1. `fusion_alpha = 0.90` optimises P@1.

### EXP-CALIBRATION — auto-approve threshold

- **Script:** `benchmarks/exp_calibration.py`
- **Artifact:** `benchmarks/results/exp_calibration_combined.json` (n = 793,
  `min_gap` = 0.10). Contains the full threshold sweep from 0.50 in 0.01 steps.

| Threshold | Coverage | Auto-approve precision | n auto |
|---|---|---|---|
| ≤ 0.75 | 59.5% | 0.8559 | 472 |
| 0.79 | 54.6% | 0.9007 | 433 |
| 0.80 | 53.1% | 0.9264 | 421 |
| **0.85 (default)** | **42.7%** | **0.9469** | **339** |
| 0.86 | 39.6% | 0.9586 | 314 |

Lowest threshold reaching a given precision target, per the artifact's `recommended`
block: 0.95 → threshold 0.86; 0.90 → threshold 0.79; 0.85 and 0.80 → threshold 0.50
(i.e. the floor already clears those targets).

**Warning recorded with the measurement:** these numbers move with the retriever.
Improving retrieval shifts the score distribution upward, pushes more candidates over a
fixed bar, and can *lower* auto-approve precision. Re-run this experiment after any
change to the model, the fusion weights, or the query representation.

### EXP-RERANK — cross-encoder reranking

- **Script:** `benchmarks/exp_rerank.py`
- **Artifact:** `benchmarks/results/exp_rerank_combined.json`
- **Setup:** dense first stage (`context` query representation + BGE prefix), then
  cross-encoder rerank of the shortlist. Shortlist recall ceiling 0.9596.

| Stage | P@1 | P@5 | MRR@10 | Throughput |
|---|---|---|---|---|
| first stage only | 0.6910 | 0.8726 | 0.7706 | — |
| + `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0.7465 | 0.8916 | 0.8089 | 18.1 queries/sec (43.7 s for 793 queries) |

Gain +0.0555 P@1, recovering 20.7% of the headroom between first-stage P@1 and the
shortlist ceiling. Rerankers are **off by default** because of the throughput cost.

Measured but **not preserved in the committed artifact** (the saved run contains only
the L-6 entry): `BAAI/bge-reranker-base` scored 4.7 points *below* the MiniLM-L-6, and
`ms-marco-MiniLM-L-12-v2` also underperformed the L-6. Re-run
`python benchmarks/exp_rerank.py --benchmark combined` with the default `--rerankers`
list to regenerate the comparison. Until then, treat the "bigger rerankers are worse"
finding as reported-but-unarchived.

---

## B — Component benchmarks with real models

### SUITE-002-REAL — INT8 quantization vs FP32

- **Script:** `benchmarks/suite_002_real_quantization.py`
- **Artifact:** `benchmarks/results/suite_002_real_20251209_162836.json`
  (two earlier runs also present; `…_162220.json` is **truncated / invalid JSON**)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Environment:** Windows, AMD64, AVX2 yes, **AVX-512 no, VNNI no**, onnxruntime 1.23.2

| Batch size | FP32 latency | INT8 latency | Speedup |
|---|---|---|---|
| 1 | 3.42 ms | 1.17 ms | 2.93× |
| 8 | 6.11 ms | 4.57 ms | 1.34× |
| 16 | 8.18 ms | 6.47 ms | 1.26× |
| 32 | 12.53 ms | 9.85 ms | **1.27×** |
| 64 | 24.18 ms | 15.03 ms | 1.61× |

The artifact's own verdict: `{"speedup_pass": true, "accuracy_pass": false,
"latency_pass": true, "overall_pass": false}`.

**Corrections to earlier documentation.** Previous revisions claimed a "1.68× INT8
speedup" with "3.07% accuracy loss". Neither number appears in any artifact. The
batch-32 figures those docs cited (12.5 ms → 9.85 ms) are a **1.27×** speedup, not
1.68×. No accuracy field was ever recorded — the artifact carries only
`accuracy_pass: false`, which is the opposite of a validated 3.07% loss. Speedup here is
also strongly batch-size dependent and was measured on a machine without VNNI, the
instruction set INT8 inference benefits most from.

### SUITE-003-REAL — ColBERT MaxSim, cold vs pre-computed

- **Script:** `benchmarks/suite_003_real_colbert.py`
- **Artifact:** `benchmarks/results/suite_003_real_20251209_162900.json`
  (an earlier run, `…_161540.json`, is **truncated / invalid JSON**)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` token embeddings, CPU

| Candidates | Cold (encode at query time) | Warm (doc embeddings pre-computed) |
|---|---|---|
| 10 | 25.78 ms avg | 1.92 ms avg |
| 50 | 131.92 ms avg | 2.51 ms avg |
| 100 | 274.04 ms avg | 2.93 ms avg (3.17 ms p95) |
| 200 | 494.00 ms avg | — |

At 100 candidates that is **93.6× on average latency** (274.04 / 2.93), or 86× if you
divide cold average by warm p95. Earlier docs quoted both "93.7×" and "86×" as if they
were different results; they are the same measurement compared two different ways. State
which one you mean.

Ranking agreement with the bi-encoder at top-5 was 100% on the sample used
(`ranking_comparison.overlap_pct = 100.0`) — i.e. on that sample MaxSim did not change
the ranking at all. This is a latency benchmark, **not** evidence that MaxSim reranking
improves accuracy. For measured reranking accuracy see EXP-RERANK above.

### SUITE-007 — ModernBERT embedding quality probe

- **Artifacts:** `benchmarks/results/suite_007_modernbert_20251209_1644*.json`

| Model | Avg similarity, similar pairs | Avg similarity, dissimilar pairs | Separation |
|---|---|---|---|
| baseline | 0.679 | 0.111 | 0.568 |
| ModernBERT | 0.721 | 0.401 | 0.320 |

ModernBERT scores similar pairs slightly higher but dissimilar pairs *much* higher, so
its separation is materially worse on this probe. Combined with being slower on CPU,
this is why ModernBERT was not adopted.

### SUITE-008 — combined GAP-008/GAP-009 demo

- **Script:** `benchmarks/suite_008_combined.py`
- **Artifacts:** `benchmarks/results/suite_008_combined_20251209_16*.json`

> **This suite does not measure NexusMatcher.** It defines 17 hand-written source
> fields, 20 hand-written target fields and 17 ground-truth pairs inline in the script,
> then computes raw cosine similarity between `all-MiniLM-L6-v2` sentence embeddings.
> The string `NexusMatcher` does not appear in the file.
>
> Its `baseline.precision_at_1 = 1.0` is **17 correct out of 17 hand-written pairs on a
> 20-entry corpus**. This is the origin of the "100% Precision@1" headline that appeared
> in the README, CHANGELOG, package docstring and enhancement narrative. **That claim is
> retracted.** The measured end-to-end P@1 is 0.700 (see EVAL-PIPELINE).
>
> Recorded here so the number is not resurrected. The suite is retained as a smoke test
> for the graph-matching and type-projection code paths, not as an accuracy benchmark.

Also in these artifacts: `gap009.mrr = 0.477` (graph matching alone, well below the
semantic baseline on the same toy set) and `gap008` type-projection numbers
(`schema_mrr` 0.9706, `test_accuracy` 0.89–0.905, varying between the two runs). All are
on the same 17/20-field toy set and should not be quoted as system accuracy.

---

## C — Component benchmarks with mock providers

These validate that infrastructure runs. They say nothing about real-model performance.

### SUITE-002 — quantization infrastructure (mock)

- **Artifact:** `benchmarks/results/suite_002_quantization_20251209_132441.json`
- Model field is literally `"mock-int8"`. Reports 5,657 texts/s, P50 5.52 ms,
  P95 6.32 ms, and a 2.32× "speedup" computed against a **hardcoded**
  `baseline_latency_ms: 12.8`. Not a measured baseline.
- The artifact records `cpu_features` with VNNI and AVX-512 **available** — a different
  machine from the one every A-grade result was measured on, which has neither. It
  records no OS or Python version.

### SUITE-003 — reranking infrastructure (mock)

- **Artifact:** `benchmarks/results/suite_003_run_20251209_134730.json`
- `mock_colbert`: 1.09 / 4.36 / 8.61 / 16.45 ms P50 at 10 / 50 / 100 / 200 candidates.
  Superseded by SUITE-003-REAL above.

---

## D — Claims with no artifact

**Do not cite these numbers.** They were presented as VALIDATED in earlier revisions of
this document with run IDs that do not correspond to any file in `benchmarks/results/`.

| Previous claim | Cited run ID | Status |
|---|---|---|
| L1 LRU cache: 56.99% hit rate, 0.0008 ms GET P95, 1,332,126 ops/s, 16.58 MB | `run_20251209_062416` | **No artifact exists.** `benchmarks/suite_004_cache_performance.py` contains no serialisation code — it prints and exits. Nothing was ever persisted. |
| Semantic content cache: 99.3% cost reduction, 50% hit rate, 781K ops/s | `run_20251209_062xxx` | **No artifact, and the run ID is a literal placeholder** — `062xxx` is not a timestamp. `benchmarks/suite_004b_semantic_cache.py` also writes no file. |
| "1.68× INT8 speedup", "3.07% accuracy loss" | — | Contradicted by SUITE-002-REAL; see the correction above. |
| "100% Precision@1" as a system result | — | Retracted; see SUITE-008 above. |

The 56.99% figure in particular is a property of the synthetic query pattern that the
cache benchmark generated (a 60% repetition rate was configured), not a property of the
cache. A cache's hit rate is determined by the workload; reporting it as a system
capability is meaningless without the workload.

### Also unarchived: in-session engineering micro-benchmarks

Measured while fixing hot paths, reported here for context but **without committed
artifacts**. Reproduce before relying on them.

| Change | Reported effect |
|---|---|
| `InMemoryVectorStore` no longer re-normalises the whole corpus matrix per query | 68.9 ms → 3.2 ms at 50k × 768, and one 153 MB allocation per query removed |
| Edit distance: pure-Python DP → `rapidfuzz` | ~145× faster, bit-identical results |
| BM25 backend comparison: `rank_bm25` vs `bm25s` at 50k docs | 38.7 ms/query vs 0.12 ms/query. **`bm25s` is not adopted** — the shipped `BM25Retriever` still uses `rank_bm25`. |
| Embedding batch size | ~128 texts/sec at batch 1 vs ~1691 at batch 128 |

### Fixed defects that had inflated or destroyed earlier measurements

| Defect | Effect while present |
|---|---|
| `AbbreviationExpander` collapsed enriched natural-language queries into a single camelCase mega-token | Production path measured dense P@1 0.309 and **BM25 P@1 0.005, with 787 of 793 queries returning zero BM25 hits**. After the fix: dense 0.636, BM25 0.531, zero zero-hit queries. |
| Missing BGE query-instruction prefix | −5.3 points P@1 (0.438 vs 0.491 at the time) |

---

## SUITE-004c and SUITE-005 — artifact exists, but read the caveat

### SUITE-004c — context enrichment throughput

- **Artifact:** `benchmarks/results/suite_004c_context_enrichment_20251209_131426.json`
- 40 fields, 103,721 fields/sec, P50 8.47 µs, P95 16.90 µs, 100% depth coverage at all
  depths, 1.78 average hierarchy tokens.
- This measures the **speed and coverage** of string enrichment on 40 synthetic fields.
  The *accuracy* value of context enrichment is measured properly by EXP-QUERY-REPR
  (+20.1 points P@1) — cite that instead.
- The artifact records no environment block, so the machine it ran on is unknown. Its
  timestamp places it alongside the Linux/Python 3.12 runs.

### SUITE-005 — incremental update change detection

- **Artifact:** `benchmarks/results/suite_005_run_20251209_133428.json`
- 50,000 synthetic entries, BLAKE3 hashing, 447K–476K entries/sec change detection,
  105–112 ms detection time.
- The reported "savings" column is **arithmetically trivial**: 99.9% savings at a 0.1%
  change rate, 99.0% at 1%, 95.0% at 5%, 90.0% at 10% — it is just `100 − change_rate`,
  i.e. the definition of only re-embedding what changed, not an empirical result. The
  genuine measurements here are the hashing and detection throughput.

---

## Environment

Every A-grade result in this file was measured on one machine:

```
OS        Windows 11 (10.0.26200)
CPU       AMD64 family 26, 32 logical cores; AVX2 yes, AVX-512 no, VNNI no
Python    3.13.4
torch     2.13.0+cpu           (CPU build; no GPU results exist)
sentence-transformers 5.6.1
numpy     2.3.5
```

The C-grade mock suites and SUITE-004c / SUITE-005 were recorded earlier on
Linux / Python 3.12 with AVX-512 and VNNI available. **Do not compare timings across
those two environments.**

---

## Adding a benchmark

1. Write the script under `benchmarks/`.
2. Make it write a JSON artifact into `benchmarks/results/` with a stable filename — an
   unpersisted benchmark is a D-grade claim the moment the terminal scrolls.
3. Record the environment inside the artifact.
4. Add a row here, with its grade and its artifact filename.
5. If it does not drive `NexusMatcher` against labelled data, do not describe its output
   as system accuracy.
