# Enhancement Journey

What was changed in NexusMatcher, what it was worth, and what turned out to be wrong.

> **This document was rewritten after an audit.** The previous version was a narrative
> built on a benchmark that never ran the system: it reported "100% Precision@1",
> "1.68× INT8 speedup with 3.07% accuracy loss", and cache results with no artifacts.
> Those claims are retracted — see [CHANGELOG.md](../CHANGELOG.md#documentation--retractions)
> and [BENCHMARK_REGISTRY.md](BENCHMARK_REGISTRY.md). Every number below names its
> artifact.

---

## 1. The measurement problem came first

For most of this project's life there was no way to tell whether a change helped.

The suite that produced the headline accuracy number,
`benchmarks/suite_008_combined.py`, defines 17 source fields, 20 target fields and 17
ground-truth pairs inline in the script, embeds them with `all-MiniLM-L6-v2`, and takes
the argmax of a cosine similarity matrix. The string `NexusMatcher` does not appear in
the file. It scored 17/17 and that became "100% Precision@1" in the README, the
CHANGELOG, the package docstring and this document.

A benchmark that cannot fail teaches nothing. Worse, it actively hid two defects that
were destroying accuracy in the real pipeline (§3).

The fix was a labelled benchmark against real corpora:

- **BIRD-SQL dev** `database_description/*.csv` — 361 pairs. Technical column name is
  the query; the human business name and description is the dictionary entry. Heavily
  abbreviated (`sname` → "school name"), which is the hard case.
- **OMOP CDM v5.4** field-level spec — 432 pairs. `cdmFieldName` is the query;
  humanised name plus `userGuidance` is the entry. A real healthcare standard.
- **combined** — both pooled, 793 pairs, so every query competes against 792 distractors
  from two unrelated domains.

Two design decisions make it honest:

1. **No leakage.** Dictionary entries are indexed with `logical_name` blanked. The
   source system's technical column name is not in the corpus. Retrieval has to work
   from the business name and human definition alone.
2. **No trivial pairs.** BIRD rows whose business label equals the technical name modulo
   case, spaces and underscores are dropped at build time. They would be string-identity
   matches and would inflate the score meaninglessly.

Built by `benchmarks/datasets/build_benchmarks.py`; evaluated end-to-end through the
real orchestrator by `benchmarks/eval_pipeline.py`.

---

## 2. Where the system landed

Artifact: `benchmarks/results/eval_pipeline_combined.json`.
`BAAI/bge-small-en-v1.5`, CPU, sparse on, BGE query prefix on.

| Metric | combined | bird | omop |
|---|---|---|---|
| P@1 | **0.700** | 0.490 | 0.819 |
| P@5 | 0.888 | — | — |
| MRR@10 | 0.781 | — | — |
| Recall@10 | 0.919 | — | — |
| Throughput | 652 fields/sec | | |

The bird/omop split is the most informative row in this document. OMOP field names are
descriptive; BIRD's are abbreviations. Same code, same model, 33 points of P@1 between
them. Schema-matching accuracy is mostly a property of how much natural language your
metadata contains, not of the matcher.

---

## 3. Two defects the old benchmark could not see

### The abbreviation expander was destroying its own input

`AbbreviationExpander` was written to expand short forms in a field name. But by the
time it ran, the query had already been enriched into a natural-language string. The
expander collapsed that whole string into a single camelCase mega-token.

Effect on the production path, measured on the combined benchmark:

| | Before | After |
|---|---|---|
| dense P@1 | 0.309 | 0.636 |
| BM25 P@1 | **0.005** | 0.531 |
| queries with zero BM25 hits | **787 / 793** | 0 |

The sparse arm was contributing nothing at all — 99.2% of queries retrieved literally
nothing from BM25 — and no test noticed, because every test asserted on API shape rather
than on retrieval quality.

### The BGE query-instruction prefix was missing

BGE retrieval models are trained with an instruction on the **query side only**. It was
being applied to neither side. Adding it asymmetrically — queries prefixed, documents
not — was worth **+5.3 points of P@1**. The provider now owns applying it, so a caller
cannot forget.

---

## 4. What actually moves accuracy

Artifact: `benchmarks/results/exp_query_repr_combined.json`. First-stage dense
retrieval, combined benchmark, varying only the query text.

| Query text | P@1 | Δ |
|---|---|---|
| bare field name | 0.488 | — |
| + split underscores | 0.491 | +0.003 |
| + camelCase split | 0.493 | +0.005 |
| + abbreviation expansion | 0.496 | +0.008 |
| **+ parent path** | **0.691** | **+0.203** |
| + scalar type words | 0.469 | **−0.019** |

**Parent-path context is the whole game.** `sname` is ambiguous against a 793-entry
glossary; `satscores sname` is not. +20.1 points from one string concatenation — an
order of magnitude more than every lexical normalisation combined.

**Appending type words hurt.** "text field" appended to the query cost 2.1 points. The
intuition was that type information disambiguates; the measurement was that it dilutes
the semantic signal, because nearly every entry is a text field and the phrase carries
no discriminative information. Now off by default.

---

## 5. Fusion: the recommended default was the worst option

Artifact: `benchmarks/results/exp_fusion_combined.json`.

| Method | P@1 |
|---|---|
| linear min-max, dense=0.9 | **0.7024** |
| dense only (no fusion) | 0.6910 |
| combsum | 0.6570 |
| **RRF, best k tried (10)** | 0.6192 |
| RRF, k=60 (the canonical default) | 0.6103 |
| sparse only | 0.5422 |

Reciprocal Rank Fusion is the standard recommendation for combining retrievers, and it
appears as such in this repo's own docstrings. On this corpus it was the **worst fusion
method measured at every `k` tried, and worse than not fusing at all** — dense-only beats
the best RRF configuration by 7 points.

The reason is legible: RRF discards score magnitude and keeps only rank. When the two
arms are comparably strong that is a robustness win. Here dense (0.691) is far stronger
than sparse (0.542), and the dense arm's *margins* carry real information about how
confident it is. Throwing them away to average ranks with a weaker retriever costs more
than it buys.

The lexical arm still earns its 10% weight: sparse-only reaches 0.542 on its own and
rescues exact-token matches the embedding model misses. The right question was never
"fuse or not" but "how much".

---

## 6. Calibrating the auto-approve threshold

Artifact: `benchmarks/results/exp_calibration_combined.json`, n = 793.

| Threshold | Coverage | Precision on auto-approved |
|---|---|---|
| 0.75 (previous default) | 59.0% | 0.863 |
| 0.80 | 53.1% | 0.926 |
| **0.85 (current default)** | **42.7%** | **0.947** |
| 0.86 | 39.6% | 0.959 |

At the old 0.75 default, roughly one in seven auto-approved mappings was wrong, silently.
Auto-approving a wrong mapping is far more expensive than sending a field to a human, so
the default now targets ~95% precision on the approved slice and accepts that most fields
get reviewed.

**The trap that caught this twice during tuning:** improving retrieval shifts the whole
score distribution *upward*, which pushes more candidates over a fixed bar and can
therefore *lower* auto-approve precision. A retrieval win can register as a quality
regression at the decision layer. Re-run `exp_calibration.py` after any change to the
model, the fusion weights, or the query representation.

---

## 7. Reranking: real gain, real cost, and a surprise

Artifact: `benchmarks/results/exp_rerank_combined.json`.

| | P@1 | MRR@10 | Throughput |
|---|---|---|---|
| first stage only | 0.691 | 0.771 | — |
| + `ms-marco-MiniLM-L-6-v2` | **0.747** | 0.809 | 18.1 queries/sec |

+5.5 points, recovering 20.7% of the headroom between first-stage P@1 and the shortlist's
recall ceiling — at roughly an order of magnitude less throughput. Hence: **rerankers are
off by default**, and documented as an opt-in with the cost stated.

The surprise: **the larger rerankers measured worse.** `BAAI/bge-reranker-base` scored
4.7 points *below* the 22M-parameter MiniLM-L-6, and `ms-marco-MiniLM-L-12-v2` also
underperformed the L-6. Model size is not a proxy for reranking quality on a given
corpus. (Only the L-6 run survives in the committed artifact; re-run `exp_rerank.py`
with the default `--rerankers` list to regenerate the comparison.)

---

## 8. Hot-path work

These are micro-benchmarks measured while fixing the code. They have **no committed
artifacts** — reproduce before relying on them.

| Change | Effect |
|---|---|
| `InMemoryVectorStore` re-normalised the entire corpus matrix on **every query** | 68.9 ms → 3.2 ms at 50k × 768, and a 153 MB per-query allocation removed |
| Edit distance: pure-Python DP → `rapidfuzz` | ~145× faster, bit-identical output |
| Embedding one text at a time vs batching | ~128 texts/sec at batch 1 vs ~1691 at batch 128 |
| BM25 backend comparison: `rank_bm25` vs `bm25s` at 50k docs | 38.7 ms/query vs 0.12 ms/query — **not adopted**; the shipped retriever still uses `rank_bm25` |

The vector-store bug is worth dwelling on: it was invisible at benchmark scale (793
entries) and catastrophic at production scale. Correctness benchmarks on small corpora
do not surface complexity bugs.

---

## 9. Component work that is real but unproven end-to-end

These are implemented and unit-tested. None of them is exercised by the accuracy
benchmark, and none has an end-to-end result.

**ColBERT MaxSim reranking.** Pre-computing document token embeddings at index time
turns 274.0 ms per query at 100 candidates into 2.93 ms — 93.6× on average latency
(`suite_003_real_20251209_162900.json`). But the same artifact records top-5 ranking
agreement with the plain bi-encoder of **100%** on its sample: MaxSim did not change the
ranking at all. That is a latency result, not an accuracy result. For measured reranking
accuracy, see §7.

**INT8 quantization.** Measured 1.26×–2.93× depending on batch size (1.27× at batch 32),
on a machine **without VNNI** — the instruction set INT8 inference benefits most from.
The artifact's own verdict is `overall_pass: false`. The previously published "1.68×
speedup, 3.07% accuracy loss" appears in no artifact and is retracted.

**Caches (L1 LRU, Redis, semantic content cache).** The benchmark scripts print results
and exit; they write no files. The "56.99% hit rate" and "99.3% cost reduction" figures
have no artifact behind them, and one cited run ID (`run_20251209_062xxx`) is a literal
placeholder. Separately, a cache's hit rate is a property of the workload — that run
configured a 60% repetition rate and then reported the resulting hit rate as a system
capability.

**Incremental updates.** Change-detection throughput of 447K–476K entries/sec on 50,000
synthetic entries is a genuine measurement. The accompanying "99.9% savings at a 0.1%
change rate" is not — it is `100 − change_rate`, i.e. the definition of incremental
updating, restated.

**ModernBERT.** Deferred. On the quality probe it scored similar pairs marginally higher
(0.679 → 0.721) but dissimilar pairs *much* higher (0.111 → 0.401), so its separation is
substantially worse, and it is slower on CPU.

**Type projections and graph matching.** Their reported numbers (`schema_mrr` 0.9706,
graph `mrr` 0.477) come from the same 17/20-field toy set as the retracted 100% claim.
Treat as experimental.

---

## 10. What this cost us to learn

1. **A benchmark that cannot fail is worse than no benchmark**, because it manufactures
   confidence. The 17-pair toy set reported perfection while 787 of 793 real queries were
   returning zero sparse hits.
2. **Test that the system is good, not that it runs.** An earlier audit found that 11 of
   11 injected accuracy-destroying mutations survived the entire test suite. The tests
   asserted shapes and types. `tests/unit/test_regression_guards.py` now asserts on
   measured quality; the same mutation exercise against it killed 5 of 5.
3. **Measure the pipeline, not a component in isolation.** Every component benchmark
   here passed while the assembled system was performing at half its potential, because
   the defect lived in how the components were composed.
4. **Best practice is a hypothesis, not a result.** RRF is the standard recommendation
   and was the worst option measured. The bigger reranker was worse than the smaller one.
   Type information in the query hurt. All three were reasonable priors and all three
   were wrong here.
5. **A number without an artifact is a rumour.** Everything in this repository that could
   not be traced to a file in `benchmarks/results/` turned out to be either
   unreproducible, mismeasured, or measuring something other than what it claimed.

---

## Reproducing everything in this document

```bash
# Place BIRD-SQL dev and the OMOP CDM v5.4 field-level CSV under data/raw/ first.
python benchmarks/datasets/build_benchmarks.py

python benchmarks/eval_pipeline.py  --benchmark combined --save   # §2
python benchmarks/exp_query_repr.py --benchmark combined          # §4
python benchmarks/exp_fusion.py     --benchmark combined          # §5
python benchmarks/exp_calibration.py --benchmark combined         # §6
python benchmarks/exp_rerank.py     --benchmark combined          # §7
```

All measurement was done on one Windows workstation, CPU only. See the environment block
in [BENCHMARK_REGISTRY.md](BENCHMARK_REGISTRY.md#environment).
