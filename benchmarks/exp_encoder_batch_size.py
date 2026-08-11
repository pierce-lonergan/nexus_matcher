"""
benchmarks.exp_encoder_batch_size | Layer: BENCHMARK
What the encoder's `batch_size` default is actually worth, measured three ways.

Why this script exists
----------------------
`BundledOnnxProvider._encode(texts, batch_size=...)` has had its default argued about
twice in this repo, and both arguments were made from a throughput table alone. Both were
wrong in a way a throughput table cannot show:

  * a claimed 1.67-point P@1 regression from batch_size turned out to be an artifact of a
    300-query fixture -- on the full 1556-query corpus every paired comparison was
    inconclusive (p = 0.43 .. 0.84);
  * a claimed "32 is fastest" rested on single timings taken while the box was busy, and
    H-007 puts this machine's band at 30.6% loaded against 0.9% idle.

So this measures three separate things and keeps them separate, because they answer
different questions and only one of them is noisy:

  STRUCTURE (exact, deterministic, no machine involved)
      How many session.run calls each batch_size produces on the real corpus, and how
      much of the work in them is padding. `_plan_batches` caps a batch by
      MAX_BATCH_TOKENS *and* by rows, so `batch_size` only binds on the SHORT end of the
      length-sorted order -- entries where the token budget would otherwise allow
      hundreds of rows. This is arithmetic; it is the same on any machine and cannot be
      noise.

  COST (noisy, needs a calibrated band -- H-007)
      Wall time to encode the corpus, interleaved across batch sizes inside each trial so
      machine drift hits every candidate equally, best-of-N reported with the spread
      beside it. A batch size is only "faster" if it beats the band measured on IDENTICAL
      code, which this script measures first and prints.

  ACCURACY (paired, exact test -- M3 / H-007's quality exemption)
      int8 ONNX inference is NOT batch-invariant: which texts share a batch changes the
      dequantisation scale, so embeddings differ in the last bits and rankings can move.
      That is a real mechanism, so it gets a real test -- per-query correctness vectors on
      the FULL corpus and an exact McNemar against the incumbent. Not a delta of two
      scalars, which is what produced the retracted 1.67-point claim.

H-003 is binding here and is not optional: `batch_size` is a batch-scheduling knob, so any
verdict has to hold at 1 thread AND at the shipped default. The 24-thread GEMM collapse in
docs/HAZARDS.md is exactly this shape -- a batching change that only won because of how
many threads the box happened to be dispatching.

CLI
---
    python benchmarks/exp_encoder_batch_size.py --structure     # exact, seconds, no timing
    python benchmarks/exp_encoder_batch_size.py --cost          # timing sweep + noise band
    python benchmarks/exp_encoder_batch_size.py --accuracy      # paired McNemar vs incumbent
    python benchmarks/exp_encoder_batch_size.py --all --save
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    # eval_harness is a script in this directory, not an installed package; every other
    # benchmark in here does the same thing for the same reason.
    sys.path.insert(0, str(BENCH_DIR))

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "exp_encoder_batch_size.json"

# The default this experiment was run to adjudicate, and the baseline every candidate is
# compared against. Deliberately pinned at 512 rather than read from the shipped constant:
# 2026-08-11 moved the shipped default to 32 ON THE STRENGTH OF THIS ARTIFACT, and
# re-baselining the artifact onto its own conclusion would erase the comparison that
# justified it. If you re-open the question, set this to whatever ships at that point.
INCUMBENT = 512

# Powers of two spanning "smaller than any batch the token budget would build" to "so
# large the row cap never binds at all". 4096 is included precisely because it is
# equivalent to removing the cap: MAX_BATCH_TOKENS is 4096, so no batch can ever reach
# 4096 rows unless every member is a single token. If 512 and 4096 measure the same, the
# row cap is doing nothing at 512, which is a finding about the knob itself.
BATCH_SIZES: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 4096)

# Timing is the expensive part, so it sweeps the ends and the incumbent rather than all
# eight. 32 is the value the earlier "32 is fastest" claim named; 4096 is the no-cap case.
COST_BATCH_SIZES: tuple[int, ...] = (32, 64, 512, 4096)

# H-003: 1 thread and the shipped default (min(8, cpu_count)). A batching win that exists
# at one and not the other is a property of the scheduler, not of the batch size.
COST_THREADS: tuple[int, ...] = (1, 8)

COST_TRIALS = 3
NOISE_REPEATS = 4

BENCHMARK = "fhir"


# =============================================================================
# CORPUS
# =============================================================================


def load_corpus(limit: int | None = None) -> tuple[list[str], list[str], list[int]]:
    """
    The real FHIR corpus as the two text lists the encoder actually sees, plus gold rows.

    Text construction mirrors tests/regression/test_accuracy_floor.py exactly -- documents
    are "business_name description", queries are "parent_path field_name doc" with the
    field name de-underscored. Matching it matters: this script's accuracy numbers are
    only interpretable next to that gate's if both encode the same strings.
    """
    from eval_harness import Dataset

    ds = Dataset.load(BENCHMARK, limit=limit)
    position = {e.id: i for i, e in enumerate(ds.entries)}

    docs = [f"{e.business_name} {e.description}" for e in ds.entries]
    queries = [
        f"{q.parent_path} {q.field_name.replace('__', ' ').replace('_', ' ')} "
        f"{getattr(q, 'doc', '') or ''}".strip()
        for q in ds.queries
    ]
    gold = [position[q.gold_id] for q in ds.queries]
    return docs, queries, gold


def _provider(num_threads: int | None = None) -> Any:
    from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
        BundledOnnxProvider,
    )

    return BundledOnnxProvider(num_threads=num_threads)


# =============================================================================
# STRUCTURE  (exact -- no machine state involved)
# =============================================================================


def structure(docs: Sequence[str], queries: Sequence[str]) -> list[dict[str, Any]]:
    """
    Batch count and padding ratio per batch_size. Deterministic: same answer everywhere.

    Padding ratio is padded token count / real token count, i.e. the multiple of the
    minimum possible work the encoder is actually asked to do. It is the one part of this
    question that needs no noise band, so it is measured first and separately -- if two
    batch sizes produce identical structure, no timing difference between them can be
    real, and that is a cheap way to catch a noise claim before paying for it.
    """
    from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
        QUERY_INSTRUCTION,
        BundledOnnxProvider,
    )

    provider = _provider()
    provider._load()
    tok = provider._tokenizer

    texts = list(docs) + [QUERY_INSTRUCTION + q for q in queries]
    lengths = [len(e.ids) for e in tok.encode_batch(texts)]
    order = np.argsort(lengths, kind="stable")
    real_tokens = sum(lengths)

    rows: list[dict[str, Any]] = []
    for bs in BATCH_SIZES:
        batches = BundledOnnxProvider._plan_batches(order, lengths, bs)
        padded = sum(len(b) * max(lengths[i] for i in b) for b in batches)
        sizes = [len(b) for b in batches]
        rows.append(
            {
                "batch_size": bs,
                "n_batches": len(batches),
                "padded_tokens": padded,
                "real_tokens": real_tokens,
                "padding_ratio": padded / real_tokens,
                "rows_per_batch_median": int(statistics.median(sizes)),
                "rows_per_batch_max": max(sizes),
                "batches_capped_by_rows": sum(1 for s in sizes if s == bs),
            }
        )
    return rows


def render_structure(rows: Sequence[dict[str, Any]]) -> str:
    out = [
        "\nSTRUCTURE -- exact, deterministic, identical on any machine",
        f"  corpus: {BENCHMARK}, {rows[0]['real_tokens']:,} real tokens",
        f"  {'batch_size':>10} {'batches':>9} {'padded tok':>12} {'pad ratio':>10} "
        f"{'rows med':>9} {'rows max':>9} {'row-capped':>11}",
        f"  {'-' * 10} {'-' * 9} {'-' * 12} {'-' * 10} {'-' * 9} {'-' * 9} {'-' * 11}",
    ]
    for r in rows:
        mark = "  <- incumbent" if r["batch_size"] == INCUMBENT else ""
        out.append(
            f"  {r['batch_size']:>10} {r['n_batches']:>9} {r['padded_tokens']:>12,} "
            f"{r['padding_ratio']:>10.4f} {r['rows_per_batch_median']:>9} "
            f"{r['rows_per_batch_max']:>9} {r['batches_capped_by_rows']:>11}{mark}"
        )
    return "\n".join(out)


# =============================================================================
# COST  (noisy -- calibrated band first, H-007)
# =============================================================================


def _time_encode(provider: Any, docs: Sequence[str], queries: Sequence[str], bs: int) -> float:
    t0 = time.perf_counter()
    provider.embed_documents(docs, bs)
    provider.embed_queries(queries, bs)
    return time.perf_counter() - t0


def _cpu_busy(seconds: float = 1.0) -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    return float(psutil.cpu_percent(interval=seconds))


def cost(
    docs: Sequence[str],
    queries: Sequence[str],
    *,
    batch_sizes: Sequence[int] = COST_BATCH_SIZES,
    threads: Sequence[int] = COST_THREADS,
    trials: int = COST_TRIALS,
    noise_repeats: int = NOISE_REPEATS,
) -> dict[str, Any]:
    """
    Interleaved timing sweep, preceded by a noise band on IDENTICAL code.

    The band is measured by running the INCUMBENT batch size `noise_repeats` times and
    taking the full spread over the median. Anything the sweep produces that is smaller
    than that band is this machine, not the knob (M1). Interleaving -- every batch size
    inside every trial -- is what stops a quiet minute being attributed to whichever
    candidate happened to run during it; measuring all trials of one candidate and then
    all trials of the next is how the 715 -> 520 false regression was produced.
    """
    out: dict[str, Any] = {"threads": {}, "trials": trials, "noise_repeats": noise_repeats}
    for n_threads in threads:
        provider = _provider(num_threads=n_threads)
        provider.embed_documents(["warmup text"], INCUMBENT)

        busy_before = _cpu_busy()
        band_samples = [
            _time_encode(provider, docs, queries, INCUMBENT) for _ in range(noise_repeats)
        ]
        band = (max(band_samples) - min(band_samples)) / statistics.median(band_samples)

        samples: dict[int, list[float]] = {bs: [] for bs in batch_sizes}
        for _ in range(trials):
            for bs in batch_sizes:
                samples[bs].append(_time_encode(provider, docs, queries, bs))
        busy_after = _cpu_busy()

        base_best = min(samples[INCUMBENT])
        rows = []
        for bs in batch_sizes:
            vals = samples[bs]
            best = min(vals)
            rows.append(
                {
                    "batch_size": bs,
                    "best_seconds": best,
                    "median_seconds": statistics.median(vals),
                    "spread_relative": (max(vals) - min(vals)) / statistics.median(vals),
                    "speedup_vs_incumbent": base_best / best,
                    "samples": vals,
                }
            )
        out["threads"][str(n_threads)] = {
            "noise_band_relative": band,
            "noise_samples": band_samples,
            "cpu_busy_before": busy_before,
            "cpu_busy_after": busy_after,
            "rows": rows,
        }
    return out


def render_cost(result: dict[str, Any]) -> str:
    lines = [f"\nCOST -- interleaved, best-of-{result['trials']}, band from identical code"]
    for n_threads, blk in result["threads"].items():
        band = blk["noise_band_relative"]
        shown = ", ".join(f"{v:.3f}" for v in blk["noise_samples"])
        lines.append(
            f"\n  {n_threads} intra-op thread(s)  "
            f"noise band {band:.1%} over {result['noise_repeats']} runs of IDENTICAL code "
            f"(cpu busy {blk['cpu_busy_before']:.0f}% -> {blk['cpu_busy_after']:.0f}%)"
        )
        lines.append(f"    identical-code samples: {shown}")
        lines.append(
            f"    {'batch_size':>10} {'best s':>9} {'median s':>10} {'spread':>8} "
            f"{'vs 512':>8}  verdict"
        )
        for r in blk["rows"]:
            delta = abs(r["speedup_vs_incumbent"] - 1.0)
            if r["batch_size"] == INCUMBENT:
                verdict = "incumbent"
            elif delta <= band:
                verdict = f"INCONCLUSIVE (|{delta:.1%}| <= band {band:.1%})"
            else:
                verdict = "faster" if r["speedup_vs_incumbent"] > 1 else "slower"
            lines.append(
                f"    {r['batch_size']:>10} {r['best_seconds']:>9.3f} "
                f"{r['median_seconds']:>10.3f} {r['spread_relative']:>7.1%} "
                f"{r['speedup_vs_incumbent']:>7.3f}x  {verdict}"
            )
    return "\n".join(lines)


# =============================================================================
# ACCURACY  (paired, exact McNemar)
# =============================================================================


def exact_mcnemar_p(n_gained: int, n_lost: int) -> float:
    """
    Two-sided exact McNemar p for paired binary outcomes.

    Re-derived here rather than imported from optimization_ledger so this script can run
    without loading the ledger's full measurement stack; the two agree by construction
    (same exact binomial tail) and tests/... pins the ledger's copy.
    """
    n = n_gained + n_lost
    if n == 0:
        return 1.0
    k = min(n_gained, n_lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def _rank_vectors(
    provider: Any, docs: Sequence[str], queries: Sequence[str], gold: Sequence[int], bs: int
) -> np.ndarray:
    doc_v = provider.embed_documents(docs, bs)
    qry_v = provider.embed_queries(queries, bs)
    sims = qry_v @ doc_v.T
    order = np.argsort(-sims, axis=1)
    return np.array([int(np.where(order[i] == gold[i])[0][0]) + 1 for i in range(len(gold))])


def accuracy(
    docs: Sequence[str],
    queries: Sequence[str],
    gold: Sequence[int],
    *,
    batch_sizes: Sequence[int] = BATCH_SIZES,
) -> dict[str, Any]:
    """
    Per-query rank vectors at each batch size, and an exact paired test against 512.

    On the FULL corpus, never a subset. The retracted 1.67-point claim came from a
    300-query slice: at that size a handful of near-ties flipping is several points of
    P@1, and H-005 says near-ties are the normal case here (gold 0.7657 vs best wrong
    0.7633). The full 1556 queries is the smallest set on which this question has an
    answer.
    """
    provider = _provider()
    provider.embed_documents(["warmup text"], INCUMBENT)

    ranks = {bs: _rank_vectors(provider, docs, queries, gold, bs) for bs in batch_sizes}
    base = ranks[INCUMBENT]
    base_c1 = (base == 1).astype(int)

    rows = []
    for bs in batch_sizes:
        r = ranks[bs]
        c1 = (r == 1).astype(int)
        gained = int(np.sum((c1 == 1) & (base_c1 == 0)))
        lost = int(np.sum((c1 == 0) & (base_c1 == 1)))
        rows.append(
            {
                "batch_size": bs,
                "p_at_1": float((r == 1).mean()),
                "r_at_5": float((r <= 5).mean()),
                "mrr": float((1.0 / r).mean()),
                "identical_ranking_vs_incumbent": bool(np.array_equal(r, base)),
                "queries_with_moved_rank": int(np.sum(r != base)),
                "gained_at_1": gained,
                "lost_at_1": lost,
                "mcnemar_p": exact_mcnemar_p(gained, lost),
            }
        )
    return {"n_queries": len(gold), "n_entries": len(docs), "rows": rows}


def render_accuracy(result: dict[str, Any]) -> str:
    lines = [
        f"\nACCURACY -- full corpus, {result['n_entries']} entries x "
        f"{result['n_queries']} queries, paired vs batch_size={INCUMBENT}",
        f"  {'batch_size':>10} {'P@1':>8} {'R@5':>8} {'MRR':>8} {'ranks moved':>12} "
        f"{'+1':>4} {'-1':>4} {'McNemar p':>10}",
        f"  {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 12} {'-' * 4} {'-' * 4} {'-' * 10}",
    ]
    for r in result["rows"]:
        mark = "  <- incumbent" if r["batch_size"] == INCUMBENT else ""
        lines.append(
            f"  {r['batch_size']:>10} {r['p_at_1']:>8.4f} {r['r_at_5']:>8.4f} "
            f"{r['mrr']:>8.4f} {r['queries_with_moved_rank']:>12} "
            f"{r['gained_at_1']:>4} {r['lost_at_1']:>4} {r['mcnemar_p']:>10.4f}{mark}"
        )
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def _environment() -> dict[str, Any]:
    import onnxruntime as ort

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--trials", type=int, default=COST_TRIALS)
    # The 1-thread leg costs several minutes on its own, so it can be run separately from
    # the shipped-default leg. H-003 still needs BOTH before any verdict.
    ap.add_argument("--threads", type=int, nargs="+", default=list(COST_THREADS))
    # A sweep that stops at its own winner cannot tell "this is the optimum" from "this is
    # the smallest value I tried". Widen it when the best result sits at an end of the range.
    ap.add_argument("--cost-sizes", type=int, nargs="+", default=list(COST_BATCH_SIZES))
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    want = {
        "structure": args.structure or args.all,
        "cost": args.cost or args.all,
        "accuracy": args.accuracy or args.all,
    }
    if not any(want.values()):
        ap.error("pick at least one of --structure / --cost / --accuracy / --all")

    docs, queries, gold = load_corpus()
    print(f"corpus: {len(docs)} entries, {len(queries)} queries", flush=True)

    artifact: dict[str, Any] = {"environment": _environment(), "benchmark": BENCHMARK}

    if want["structure"]:
        rows = structure(docs, queries)
        artifact["structure"] = rows
        print(render_structure(rows), flush=True)

    if want["accuracy"]:
        acc = accuracy(docs, queries, gold)
        artifact["accuracy"] = acc
        print(render_accuracy(acc), flush=True)

    if want["cost"]:
        sizes = sorted({*args.cost_sizes, INCUMBENT})
        c = cost(docs, queries, trials=args.trials, threads=args.threads, batch_sizes=sizes)
        artifact["cost"] = c
        print(render_cost(c), flush=True)

    if args.save:
        RESULTS_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"\nwrote {RESULTS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
