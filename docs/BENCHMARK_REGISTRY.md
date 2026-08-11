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

## Read this before comparing two rows: there are two corpus generations

The `combined` benchmark was **rebuilt** after a leakage fix in the OMOP split, and it
shrank from **793 labelled pairs to 688**. Before the fix, each OMOP entry's business name
was derived from its field name, and the query representation is also derived from the
field name — so query and gold label were token-identical and half the benchmark scored
string identity. Every headline moved down when that was fixed, and **two techniques
inverted sign**.

Artifacts in this file therefore belong to one of two generations, and **a number from one
generation must never be differenced against a number from the other**:

| Artifact | Corpus | Generation |
|---|---|---|
| `eval_pipeline_combined.json` | n = 688 | post-fix |
| `eval_pipeline_bird.json` / `eval_pipeline_omop.json` | n = 361 / 327 | post-fix |
| `exp_query_repr_combined.json` | n = 688 | post-fix |
| `exp_calibration_combined.json` | n = 688 | post-fix |
| `exp_rerank_combined.json` | n = 688 | post-fix |
| `exp_encoder_batch_size.json` | fhir, n = 1556 | not affected (different corpus) |
| **`exp_fusion_combined.json`** | **n = 793** | **PRE-FIX — not re-run** |

The fusion table below is internally consistent with its artifact and the artifact is
stale. That is called out again in place. Everything under grades B and C measures latency
or embedding geometry and is unaffected by the label fix.

Every number in this document was re-checked against the artifact named beside it on
**2026-08-11**. Where an earlier revision of this file disagreed, the earlier number is
recorded as a retraction rather than deleted.

---

## A — End-to-end accuracy

### EVAL-PIPELINE — `NexusMatcher` on the labelled benchmark

- **Script:** `benchmarks/eval_pipeline.py`
- **Artifact:** `benchmarks/results/eval_pipeline_combined.json`
- **Data:** `data/benchmarks/combined/` — 688 labelled query→entry pairs against 688
  dictionary entries, built by `benchmarks/datasets/build_benchmarks.py` from BIRD-SQL dev
  `database_descriptions` (361) and OHDSI OMOP CDM v5.4 field-level spec (327).
- **Leakage control:** dictionary entries indexed with `logical_name` blanked, so the
  source system's technical column name is not in the corpus. Retrieval works from
  business name + human definition only.
- **Config:** `BAAI/bge-small-en-v1.5`, CPU, sparse retrieval on, BGE query-instruction
  prefix on, `auto_approve_threshold` at the shipped 0.87.

| Metric | Value |
|---|---|
| P@1 | 0.5814 |
| P@5 | 0.8256 |
| MRR@10 | 0.6853 |
| Recall@1 / @5 / @10 | 0.5814 / 0.8256 / 0.8779 |
| Throughput | 364.2 fields/sec |
| Index build (688 entries) | 2.067 s |
| Decisions | AUTO_APPROVE 85, REVIEW 603, REJECT 0 |
| Auto-approve precision at that operating point | 0.9529 |

Per split, same configuration:

| Metric | bird | omop |
|---|---|---|
| P@1 | 0.6011 | 0.5749 |
| P@5 | 0.7895 | 0.8869 |
| MRR@10 | 0.6822 | 0.7055 |
| Throughput | 528.4 fields/sec | 397.7 fields/sec |

**The split ordering reversed, and that is the finding, not a defect to tune away.** An
earlier revision of this file reported **bird 0.490 / omop 0.819** and explained the gap by
OMOP's descriptive prose. Post-fix the descriptive split is the *harder* one. The old
ordering was the leak: OMOP's business names were the field names. Do not "restore" it.

`eval_pipeline_omop.json` records `auto_approve_precision: null` and `AUTO_APPROVE 0` — at
threshold 0.87 the OMOP split auto-approves nothing at all, so it contributes no precision
measurement. The combined figure of 0.9529 is 81 correct out of 85, all from BIRD-like
fields.

Reproduce:

```bash
python benchmarks/datasets/build_benchmarks.py
python benchmarks/eval_pipeline.py --benchmark combined --save
```

---

## A — Ablations and calibration

### EXP-QUERY-REPR — what the query text should contain

- **Script:** `benchmarks/exp_query_repr.py`
- **Artifact:** `benchmarks/results/exp_query_repr_combined.json` (n = 688, post-fix)

| Variant | P@1 | P@5 | MRR@10 | R@50 |
|---|---|---|---|---|
| `raw` — bare field name | 0.3605 | 0.6047 | 0.4614 | 0.8372 |
| `underscores` | 0.3663 | 0.6061 | 0.4652 | 0.8387 |
| `split` | 0.3735 | 0.6105 | 0.4721 | 0.8372 |
| `abbrev` | 0.3765 | 0.6308 | 0.4822 | 0.8590 |
| **`context` — + parent path** | **0.5596** | 0.8096 | 0.6674 | 0.9491 |
| `type` — + scalar type words | 0.3532 | 0.5756 | 0.4464 | 0.8081 |
| `full_no_type` | 0.5451 | **0.8270** | 0.6653 | **0.9520** |
| `full` | 0.5305 | 0.8038 | 0.6475 | 0.9390 |

Findings, each stated with the contrast it comes from — a delta is a claim about *two*
rows and the repository has quoted three different numbers for this one:

- **Parent-path context is worth +19.9 points of P@1** over `raw`, the bare field name,
  and **+19.3** over `underscores`, which is what the code produced before context
  enrichment shipped. Either is defensible; say which. It remains by a wide margin the
  largest single accuracy factor in the pipeline.
- **Appending scalar type words costs 2.0 points** (`split` 0.3735 → `type` 0.3532) and
  **1.5 points** when context and expansion are already present (`full_no_type` 0.5451 →
  `full` 0.5305). It is off by default.
- `context` has the best P@1; `full_no_type` has the best P@5 and R@50. The shipped
  configuration optimises P@1.

**Retraction.** An earlier revision of this file reported this table on the 793-pair
pre-fix corpus, where `context` read 0.6910 and the type-word cost was quoted as 2.1
points. Those figures are superseded by the numbers above, not merely re-rounded.

### EXP-FUSION — combining dense and sparse

- **Script:** `benchmarks/exp_fusion.py`
- **Artifact:** `benchmarks/results/exp_fusion_combined.json`
- **PRE-LEAKAGE-FIX. n = 793.** This is the one A-grade artifact that has not been re-run
  since the corpus was rebuilt. The table below reproduces it exactly; do not difference
  any of it against a 688-pair number, and re-run before quoting `fusion_alpha` as
  re-validated.

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

The RRF finding is a large, one-directional gap and is unlikely to invert on the smaller
corpus; the `0.9`-versus-`0.8` margin is 1.6 points and easily could. Treat the ordering
as durable and the exact optimum as unconfirmed.

### EXP-CALIBRATION — auto-approve threshold

- **Script:** `benchmarks/exp_calibration.py`
- **Artifact:** `benchmarks/results/exp_calibration_combined.json` (n = 688,
  `min_gap` = 0.10). Contains a threshold sweep from 0.50 to 0.87 in 0.01 steps.

| Threshold | Coverage | Auto-approve precision | n auto |
|---|---|---|---|
| 0.50 (floor) | 51.0% | 0.7407 | 351 |
| ≤ 0.75 | 50.3% | 0.7514 | 346 |
| 0.79 | 45.1% | 0.7935 | 310 |
| 0.80 | 41.3% | 0.8521 | 284 |
| 0.85 | 20.8% | 0.9161 | 143 |
| 0.86 | 14.2% | 0.9388 | 98 |
| **0.87 (shipped default)** | **12.4%** | **0.9529** | **85** |

Lowest threshold reaching a given precision target, per the artifact's `recommended`
block: 0.95 → threshold 0.87; 0.90 → threshold 0.85; 0.85 → threshold 0.80; 0.80 →
threshold 0.80.

**The sweep stops at 0.87**, which is also the shipped default and the 0.95-precision
answer. So "0.87 is the lowest threshold reaching 0.95 precision" is true of the range
measured; nothing here says what 0.88 or 0.90 would buy. Extend the sweep before claiming
a ceiling.

**Two corrections to earlier revisions of this file.** It labelled **0.85** as the default
— the shipped default is **0.87** (`MatchingConfig.auto_approve_threshold`), and 0.85 is
the 0.90-precision point. And it stated that the 0.85 and 0.80 precision targets are
"already cleared by the floor": on the rebuilt corpus the floor is 0.7407, so neither is.

**Warning recorded with the measurement:** these numbers move with the retriever.
Improving retrieval shifts the score distribution upward, pushes more candidates over a
fixed bar, and can *lower* auto-approve precision. Re-run this experiment after any
change to the model, the fusion weights, or the query representation — including turning
on governed abbreviation expansion (see EXP-GOVERNED-ABBREV).

### EXP-RERANK — cross-encoder reranking

- **Script:** `benchmarks/exp_rerank.py`
- **Artifact:** `benchmarks/results/exp_rerank_combined.json` (n = 688, post-fix)
- **Setup:** dense first stage (`context` query representation + BGE prefix), then
  cross-encoder rerank of the shortlist. Shortlist recall ceiling 0.8648.

| Stage | P@1 | P@5 | MRR@10 | Throughput |
|---|---|---|---|---|
| first stage only | 0.5596 | 0.8096 | 0.6674 | — |
| + `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0.5465 | 0.8241 | 0.6630 | 87.3 queries/sec (7.88 s for 688 queries) |

**The cross-encoder now LOSES accuracy at rank 1.** The artifact records
`gain_p_at_1 = -0.0131` and `headroom_recovered = -0.0429`: it costs 1.3 points of P@1
and 0.4 points of MRR@10, while gaining 1.5 points of P@5. Rerankers are **off by
default**, and the reason is now accuracy as well as throughput cost.

**Retraction, and the reason it matters most on this row.** An earlier revision reported a
**+0.0555 P@1 gain recovering 20.7% of the headroom**, at 18.1 queries/sec over 793
queries. Every part of that is superseded. The gain did not shrink — it *inverted*, and it
inverted for a comprehensible reason: on the pre-fix corpus half the OMOP pairs were
solvable by string identity, and a cross-encoder is very good at confirming string
identity. It was being credited for recovering matches the benchmark should never have
posed. This is the single clearest example in the repository of why a technique's measured
gain must be re-derived after any change to the labels.

The throughput figure moved by 4.8x (18.1 → 87.3 queries/sec) across the same two
revisions. That is a toolchain and corpus-size change, not a speedup anyone engineered;
do not quote it as one.

Measured but **not preserved in the committed artifact** (the saved run contains only
the L-6 entry): `BAAI/bge-reranker-base` scored 4.7 points *below* the MiniLM-L-6, and
`ms-marco-MiniLM-L-12-v2` also underperformed the L-6. Re-run
`python benchmarks/exp_rerank.py --benchmark combined` with the default `--rerankers`
list to regenerate the comparison. Until then, treat the "bigger rerankers are worse"
finding as reported-but-unarchived.

### EXP-GOVERNED-ABBREV — what a caller-supplied abbreviation catalog is worth

- **Script:** `benchmarks/exp_governed_abbrev.py`
- **Artifacts:** `benchmarks/results/exp_governed_abbrev_combined.json`, `benchmarks/results/exp_governed_abbrev_fhir.json`
- **NOT COMMITTED.** Both artifacts and the script exist in the working tree and are
  untracked, so on a fresh clone every number in this section is unverifiable and
  `scripts/check_doc_numbers.py` will report it as such rather than as agreement. Commit
  them or delete this section; do not leave it in this state. The script also imports a
  third-party `acronymkit` for its expansion arm, which was **evaluated and rejected** —
  see the note at the end of this row.
- **How to read this row:** the *shape* of the trade is the result. The absolute recovery
  figures are an upper bound from a synthetic construction, and the caveat below is not
  optional reading.

The measured conditions, dense retrieval, both corpora, paired exact McNemar on hit@1:

| Condition | combined P@1 | fhir P@1 |
|---|---|---|
| original — un-abbreviated names | 0.5596 | 0.2442 |
| abbrev — every governed token rewritten to its short form | 0.0945 | 0.0103 |
| governed_full — expanded through the complete catalog | 0.5596 | 0.2442 |
| guessing_expander — expanded through the bundled generic dictionary | 0.1163 | 0.0167 |
| keep_both_full — expansion *appended* rather than substituted | 0.4680 | 0.1889 |
| guessing_on_original — generic dictionary on un-abbreviated names | 0.5480 | 0.2423 |

**Abbreviation is the largest single effect measured anywhere in this repository.** A fully
abbreviated schema costs **46.5 points of P@1 on combined and 23.4 on FHIR**
(`abbreviation_gap_p_at_1`), against the +19.9 that parent-path context is worth. Both
gaps are overwhelming under McNemar (combined: 331 queries lost against 11 gained).

Recovery tracks **catalog coverage** close to linearly, and tolerates **staleness** far
better than it tolerates absence. Five seeds per point; the value is the mean fraction of
the abbreviation gap recovered.

| Degraded catalog | combined recovery | fhir recovery |
|---|---|---|
| 75% of rows present | 72% | 68% |
| 50% of rows present | 46% | 38% |
| 25% of rows present | 22% | 15% |
| 5% of rows wrong | 91% | 92% |
| 10% of rows wrong | 82% | 81% |
| 25% of rows wrong | 64% | 62% |
| 50% of rows wrong | 31% | 30% |
| 75% of rows wrong | 3% | 5% |
| 100% of rows wrong | −15% | −4% |

Read: a catalog that is *incomplete* pays roughly in proportion; a catalog that is *wrong*
is forgiving up to about 25% and breaks even at about 75%. At 100% wrong the combined
corpus lands 7.1 points of P@1 **below leaving the abbreviations alone** — expansion is
not a free hedge, it is a bet on the catalog. The 75%-wrong row is the crossover: its
McNemar p against the un-expanded baseline runs from 0.0007 to 0.60 across seeds, i.e.
indistinguishable from doing nothing.

`keep_both_full` is the obvious hedge — append the expansion instead of substituting it —
and it is **not** free: it costs 9.2 points against substitution on a complete catalog
(0.5596 → 0.4680) and only overtakes it past 75% wrong. It buys insurance at a price this
model charges for longer queries, the same effect EXP-QUERY-REPR measures for type words.

> **The caveat that matters most, and the reason this row is not a headline.**
>
> These recovery figures come from a **synthetic** experiment. The abbreviation was
> generated *from the gold text* by a mechanical rule (`build_standard`), and the catalog
> is the exact inverse of that rule. Expanding therefore does not recover *meaning*, it
> reconstructs the original *string*. Measured directly, with the shipped
> `AbbreviationExpander` and the full catalog, the expanded query is caselessly identical
> to the original for **683 of 688** queries on combined and **1556 of 1556** on FHIR. The
> "100% of the gap recovered" headline is `f⁻¹(f(x)) == x`, and it is worth exactly
> nothing as an estimate of anyone's corpus.
>
> The **coverage and wrong-rate curves are informative** — they vary a real quantity
> (which rows are present, which rows are wrong) and produce a real, ordered response. The
> **absolute damage figure and the 100% recovery are an upper bound**, and the damage
> figure additionally uses a devowel-to-four-characters scheme harsher than any human
> naming standard. This repository has already published one benchmark whose headline was
> a leak; the docs are where that gets caught, so it is written down here rather than in
> a summary somewhere.

**What is untested, by anyone.** *Colliding short forms.* `build_standard` is **injective
by construction** — one short form, one long form, guaranteed. Real approved-abbreviation
catalogs are not: `ST` is State in one column and Street in the next, and no catalog row
can be right in both places. Nothing in this experiment, or in the shipped expander (which
does a single exact dictionary lookup per token), models that at all. It is the first
thing to measure on a real catalog and the most likely reason these numbers would not
transfer.

**Also recorded:** a third-party `acronymkit` library was evaluated as a replacement for
the shipped `AbbreviationExpander` and **rejected**. Fed the *same* catalog, the two
produce identical query text on 688/688 and 1556/1556 queries, and McNemar between them is
`b=0, c=0, p=1` on both corpora at every degradation point tested. The measured
contribution of the dependency is exactly zero; the capability already ships. The
20:1 "governed beats guessing" result really compares *catalogs*, not libraries, and this
repository can already load a catalog — see
[docs/guides/governed_abbreviations.md](guides/governed_abbreviations.md).

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

The same artifact also carries an ONNX **FP32** leg, which the table above omits and which
changes the reading: at batch 32 it runs 11.28 ms, so most of the "INT8 speedup" against
sentence-transformers is the ONNX runtime, not the quantization. INT8 over ONNX FP32 at
batch 32 is 1.14×.

**Corrections to earlier documentation.** Previous revisions claimed a "1.68× INT8
speedup" with "3.07% accuracy loss". Neither number appears in any artifact. The
batch-32 figures those docs cited (12.5 ms → 9.85 ms) are a **1.27×** speedup, not
1.68×. No accuracy field was ever recorded — the artifact carries only
`accuracy_pass: false`, which is the opposite of a validated 3.07% loss. Speedup here is
also strongly batch-size dependent and was measured on a machine without VNNI, the
instruction set INT8 inference benefits most from.

### EXP-ENCODER-BATCH — what the encoder `batch_size` default should be

- **Script:** `benchmarks/exp_encoder_batch_size.py`
- **Artifact:** `benchmarks/results/exp_encoder_batch_size.json`
- **Data:** full FHIR corpus, 4598 entries × 1556 queries, 252,168 real tokens.
- **Component, not pipeline:** dense retrieval through `BundledOnnxProvider` only — no
  BM25, no fusion, no reranking. The P@1 figures below are **encoder-only on the full
  corpus** and are not this system's accuracy; see EVAL-PIPELINE for that.
- **Outcome:** default changed **512 → 32** on 2026-08-11.

**Structure (exact, deterministic — no machine state involved).** `_plan_batches` caps a
batch by `MAX_BATCH_TOKENS` *and* by rows, so `batch_size` only binds on the short end of
the length-sorted order.

| batch_size | batches | padded tokens | padding ratio | batches actually row-capped |
|---|---|---|---|---|
| 16 | 386 | 254,742 | 1.0102 | 383 |
| **32** | **196** | **255,979** | **1.0151** | **187** |
| 64 | 107 | 257,095 | 1.0195 | 82 |
| 128 | 72 | 258,547 | 1.0253 | 25 |
| 256 | 65 | 259,518 | 1.0291 | 2 |
| 512 (old default) | 65 | 259,281 | 1.0282 | **0** |
| 1024 | 65 | 259,281 | 1.0282 | 0 |
| 4096 | 65 | 259,281 | 1.0282 | 0 |

**The old default of 512 was not a cap.** No batch on this corpus reaches 512 rows, so
512, 1024 and 4096 produce byte-identical batch plans and byte-identical embeddings.

**Correction.** A padding ratio of **2.230x for 512** has been quoted in this repo. It is
wrong for the current encoder — it belongs to the pre-token-budget fixed-window batching
(`tests/unit/infrastructure/test_bundled_onnx_batching.py` calls it "where the 2.2x came
from"). Measured on today's code the 32-vs-512 padding gap is **1.29% of tokens**, not
2.1×, so padding is *not* the mechanism behind the speedup.

**Cost.** Interleaved, best-of-3, each block beside the noise band measured on identical
code in the same session (H-007). Taken at 11.7% → 25.3% CPU busy.

| threads | band | 16 | 32 | 64 | 512 |
|---|---|---|---|---|---|
| 1 | 0.7% | 1.074× | 1.056× | 1.030× | 1.000× |
| 8 (the shipped cap) | 6.2% | 1.274× | **1.386×** | 1.294× | 1.000× |

**32 rather than 16** because the regimes disagree and H-003 requires a batch-scheduling
knob to win at 1 thread *and* at the library default: 16 is 1.8 points better at one
thread and 11.2 points **worse** at the thread count that ships. Treat the ~5% 1-thread
margin as the durable figure — it reproduced at 4.8% and 5.6% in two quiet windows. The
8-thread margin swings 10.6–38.6% with machine state; do not quote it.

**Accuracy: no batch size is distinguishable from any other.** Paired, full corpus, exact
McNemar against 512.

| batch_size | P@1 | R@5 | MRR | queries whose rank moved | gained@1 | lost@1 | McNemar p |
|---|---|---|---|---|---|---|---|
| 16 | 0.2879 | 0.5373 | 0.4049 | 883 | 47 | 37 | 0.3261 |
| **32** | 0.2796 | 0.5392 | 0.4005 | 886 | 45 | 48 | **0.8358** |
| 64 | 0.2763 | 0.5424 | 0.3977 | 882 | 35 | 43 | 0.4282 |
| 128 | 0.2783 | 0.5398 | 0.4010 | 879 | 38 | 43 | 0.6570 |
| 256 | 0.2847 | 0.5386 | 0.4024 | 840 | 38 | 33 | 0.6353 |
| 512 | 0.2815 | 0.5334 | 0.3996 | — | — | — | — |

int8 inference is genuinely not batch-invariant — **886 of 1556 queries change rank**
between 32 and 512 — but the churn is symmetric and every size measured is inconclusive.
Among the five that batch differently from 512 at all, p runs 0.33 … 0.84; 1024 and 4096
are p = 1 by construction. **No accuracy claim is made for this change**, in either
direction. The 1.67-point P@1 "regression" once attributed to `batch_size` came from a
300-query fixture and does not survive a paired test on the full corpus.

**One run was discarded as UNMEASURABLE and kept in the artifact anyway** (key
`cost_UNMEASURABLE_busy_machine`). Taken earlier the same day at 36–88% CPU busy, its
bands were 15.3% / 16.4% / 48.1% and *every* comparison was inconclusive, including ones
the quiet run resolves cleanly. Its point estimates agree in direction with the quiet run
at all three thread counts; nothing in it may be quoted. It also carries the only
32-intra-op-thread leg measured, where per-trial spread reached 135% — H-003's saturated
box, and the reason the shipped thread count is capped at 8.

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
| 200 | 494.00 ms avg | 3.13 ms avg |

At 100 candidates that is **93.6× on average latency** (274.04 / 2.93), or 86× if you
divide cold average by warm p95. Earlier docs quoted both "93.7×" and "86×" as if they
were different results; they are the same measurement compared two different ways. State
which one you mean. (An earlier revision of this file left the warm 200-candidate cell
empty; the artifact has it, at 3.13 ms average.)

Ranking agreement with the bi-encoder at top-5 was 100% on the sample used
(`ranking_comparison.overlap_pct = 100.0`) — i.e. on that sample MaxSim did not change
the ranking at all. This is a latency benchmark, **not** evidence that MaxSim reranking
improves accuracy. For measured reranking accuracy see EXP-RERANK above, where the
cross-encoder is now a 1.3-point loss.

### SUITE-007 — ModernBERT embedding quality probe

- **Artifacts:** `benchmarks/results/suite_007_modernbert_20251209_1644*.json`

| Model | Avg similarity, similar pairs | Avg similarity, dissimilar pairs | Separation |
|---|---|---|---|
| baseline | 0.679 | 0.111 | 0.568 |
| ModernBERT | 0.721 | 0.401 | 0.320 |

ModernBERT scores similar pairs slightly higher but dissimilar pairs *much* higher, so
its separation is materially worse on this probe. Combined with being slower on CPU
(24.6 ms vs 3.7 ms at batch 1), this is why ModernBERT was not adopted.

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
> retracted.**
>
> Recorded here so the number is not resurrected. The suite is retained as a smoke test
> for the graph-matching and type-projection code paths, not as an accuracy benchmark.

The end-to-end figure that replaces it is the one in EVAL-PIPELINE at the top of this
file. It is deliberately not repeated here: this section names the SUITE-008 artifacts, so
a copy of the pipeline number written under this heading would be checked against the wrong
file by `scripts/check_doc_numbers.py`, and one number in one place is the rule that keeps
the remaining recorded mismatches from growing.

Also in these artifacts: `gap009.mrr = 0.4775` and `gap009.precision_at_1 = 0.2941`
(graph matching alone, well below the semantic baseline on the same toy set) and `gap008`
type-projection numbers (`schema_mrr` 0.9706, `test_accuracy` 0.89–0.905, varying between
the two runs). All are on the same 17/20-field toy set and should not be quoted as system
accuracy.

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
| Embedding batch size | ~128 texts/sec at batch 1 vs ~1691 at batch 128. **Superseded** for the bundled encoder by EXP-ENCODER-BATCH, which has an artifact. Note it does not contradict that row: batch 1 versus batch 128 is a different question from where the optimum sits among 16–4096, and the bundled encoder now caps batches by tokens rather than rows. |

### Fixed defects that had inflated or destroyed earlier measurements

These are dated records of past states, measured on the **793-pair pre-fix** corpus. They
are kept because the mechanism is the lesson; none of the numbers is current.

| Defect | Effect while present (793-pair corpus) |
|---|---|
| `AbbreviationExpander` collapsed enriched natural-language queries into a single camelCase mega-token | Production path measured dense P@1 0.309 and **BM25 P@1 0.005, with 787 of 793 queries returning zero BM25 hits**. After the fix: dense 0.636, BM25 0.531, zero zero-hit queries. |
| Missing BGE query-instruction prefix | −5.3 points P@1 (0.438 vs 0.491 at the time) |
| OMOP business name derived from the field name | Inflated combined P@1 to 0.715 and inverted two technique verdicts. See the corpus-generation note at the top of this file. |

---

## Benchmarks with artifacts but no row here

`benchmarks/results/` contains committed artifacts that this file does not describe:
`exp_scale_combined.json`, `exp_alias_combined.json`, `exp_alias_scale.json`,
`exp_alias_bird.json`, `exp_alias_omop.json`, `exp_encoders_combined.json`,
`exp_instance_bird.json`, `exp_instance_signal_bird.json`, `exp_finetune_holdout.json`,
`exp_finetune_transfer.json`, `exp_rerank_bird.json`, `eval_combined_business_desc.json`,
`perf_baseline.json`, `perf_opt1.json`, `perf_opt2.json` and `optimization_ledger.jsonl`.

Numbers from those files are quoted elsewhere in the repository — README's encoder table,
`MatchingConfig.dictionary_alias_count`, the 100k-entry scale claim — and are therefore
outside this registry's coverage. Recorded so the gap is visible rather than implied to be
zero; writing those rows is a separate piece of work.

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

EXP-ENCODER-BATCH was measured on the same Windows machine but a later toolchain —
Python 3.13.4, numpy 2.3.5, **onnxruntime 1.28.0** (SUITE-002-REAL used 1.23.2). It uses
the bundled int8 ONNX encoder and no torch at all. Its own artifact records this block,
so check there rather than assuming this one applies.

**Timings on this machine need their noise band quoted beside them, always.** Identical
code measured 0.7% spread idle and 48.1% at 88% CPU busy in a single day (H-007). A
throughput number with no band is not a result here, and a run taken above ~25% busy
should be recorded as UNMEASURABLE rather than averaged in.

---

## SUITE-004c and SUITE-005 — artifact exists, but read the caveat

### SUITE-004c — context enrichment throughput

- **Artifact:** `benchmarks/results/suite_004c_context_enrichment_20251209_131426.json`
- 40 fields, 103,721 fields/sec, P50 8.47 µs, P95 16.90 µs, 100% depth coverage at all
  depths, 1.78 average hierarchy tokens.
- This measures the **speed and coverage** of string enrichment on 40 synthetic fields.
  The *accuracy* value of context enrichment is measured properly by EXP-QUERY-REPR
  (+19.9 points P@1 over the bare field name) — cite that instead.
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

## Adding a benchmark

1. Write the script under `benchmarks/`.
2. Make it write a JSON artifact into `benchmarks/results/` with a stable filename — an
   unpersisted benchmark is a D-grade claim the moment the terminal scrolls.
3. **Commit the artifact.** An artifact that exists only in your working tree is a D-grade
   claim from everyone else's point of view; see EXP-GOVERNED-ABBREV for what that looks
   like in this file.
4. Record the environment, and the corpus size, inside the artifact.
5. Add a row here, with its grade and its artifact filename.
6. If it does not drive `NexusMatcher` against labelled data, do not describe its output
   as system accuracy.
7. If the labelled data itself changed, **re-run everything** and re-derive every delta.
   Two techniques in this file inverted sign the last time that happened.
