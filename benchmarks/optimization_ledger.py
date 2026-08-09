"""
benchmarks.optimization_ledger | Layer: BENCHMARK
The bar an optimization has to clear before anyone is allowed to call it a win.

perf_harness.py tells you how fast the code is. eval_pipeline.py tells you how right it
is. Neither tells you whether a CHANGE helped, and that is a different question with its
own failure modes -- four of which have already cost real time in this repo. This module
exists to make those four mistakes structurally impossible rather than merely discouraged.

  M1  Calling noise a win.
      Timings on a developer laptop swing double digits between identical runs. This repo
      has a recorded example: benchmarks/results/perf_baseline.json vs perf_opt2.json show
      index throughput "regressing" 715 -> 520 entries/sec, which was machine state, not
      code. So the ledger CALIBRATES ITS OWN NOISE FLOOR by measuring identical code
      several times and recording the spread of the exact statistic it later compares. Any
      delta inside that band is INCONCLUSIVE. It is never a win, no matter how good the
      percentage looks.

  M2  Winning on speed while losing accuracy.
      Every record carries guarded metrics with explicit tolerances (see DEFAULT_GUARDS).
      Trip one and the verdict is REGRESSION regardless of the speedup, because a wrong
      mapping that gets auto-approved applies the wrong PII label to a production column.

  M3  Treating paired data as unpaired.
      Quality is measured on the SAME queries before and after, so the per-query outcomes
      are paired and a two-sample test throws away most of the signal. Quality deltas get
      a paired bootstrap over queries (effect size + 95% CI) and an exact paired test on
      the correctness vectors -- exact McNemar for binary metrics, exact sign-flip
      permutation otherwise. A bare delta with no interval is not reportable here.

  M4  Unreproducible results.
      Every record carries git SHA, dirty-tree flag, platform, CPU count, Python version,
      the full MatchingConfig, the seed, and a CPU-busy sample taken at measurement time.

The asymmetry is deliberate and is the whole design
---------------------------------------------------
Claiming a WIN requires evidence: the target must move further than the calibrated noise
band. Flagging a REGRESSION does not: a guard trips on the point estimate alone. The null
hypothesis is "your change hurt accuracy", and you have to buy your way out of it. A guard
that demanded p < 0.05 before failing would wave through every real loss too small to
prove -- and on this benchmark a real 1-point P@1 loss is exactly that size.

How to use this (you, about to optimize something)
--------------------------------------------------
    from benchmarks.optimization_ledger import (
        calibrate_noise, measure_all, record, compare, load_records, leaderboard,
    )

    # 1. ONCE per machine per session, before you change anything. This is the number
    #    that decides what counts as real. It takes a few minutes; skipping it is how
    #    people end up shipping noise.
    band, repeats = calibrate_noise(repeats=3)
    print(band.render())

    # 2. Baseline the code as it is now.
    base = record(measure_all(label="baseline"), noise=band)

    # ... make your change ...

    # 3. Measure the change the same way, then ask.
    cand = record(measure_all(label="inverted-bm25-postings"), noise=band)
    print(compare(base, cand, target="match_fields_per_sec").render())

    # 4. See where it sits.
    print(leaderboard())

`target` names the metric you were trying to improve. Naming it is required for a WIN --
"it got faster somewhere" is not a result. If you are refactoring and expect no change,
pass no target and the best you can earn is NEUTRAL, which is still worth having: it means
you proved you broke nothing.

Before you trust ANY of it, check your guards actually fire. `simulate_candidate()` builds
a fake candidate from a real record so you can watch REGRESSION happen on demand; the
`--demo` subcommand does exactly that and is the fastest way to see the whole thing work.

CLI
---
    python benchmarks/optimization_ledger.py --demo                 # prove it works
    python benchmarks/optimization_ledger.py --calibrate            # just the noise band
    python benchmarks/optimization_ledger.py --record "my change"   # measure + append
    python benchmarks/optimization_ledger.py --leaderboard
    python benchmarks/optimization_ledger.py --compare BASE_ID CAND_ID --target p_at_1
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    # eval_harness / perf_harness are scripts in this directory, not an installed package;
    # eval_pipeline.py does the same thing for the same reason.
    sys.path.insert(0, str(BENCH_DIR))

LEDGER_PATH = REPO_ROOT / "benchmarks" / "results" / "optimization_ledger.jsonl"
LEDGER_FORMAT_VERSION = 1

# One seed for everything resampled. Recorded in provenance, because a bootstrap CI you
# cannot regenerate is a number you cannot check.
DEFAULT_SEED = 20260809

DEFAULT_BENCHMARK = "fhir"
DEFAULT_SCALES: tuple[tuple[int, int], ...] = ((1000, 300), (5000, 300))
DEFAULT_TRIALS = 3

# Metrics where a bigger number is better. Everything else (latency, memory) is inverted
# when we render "did this improve?", so a reader never has to remember which is which.
HIGHER_IS_BETTER = {
    "p_at_1",
    "r_at_5",
    "r_at_10",
    "mrr_at_10",
    "auto_approve_precision",
    "auto_approve_coverage",
    "match_fields_per_sec",
    "index_entries_per_sec",
}

QUALITY_METRICS = (
    "p_at_1",
    "r_at_5",
    "r_at_10",
    "mrr_at_10",
    "auto_approve_precision",
    "auto_approve_coverage",
)

COST_METRICS = (
    "match_fields_per_sec",
    "index_entries_per_sec",
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_p99",
    "peak_memory_mb",
)


# =============================================================================
# GUARDS
# =============================================================================


@dataclass(frozen=True)
class Guard:
    """
    A metric that a speedup is not allowed to spend.

    `tolerance` is the worst delta that still passes, signed in the metric's own units
    (or as a fraction of the baseline when `relative`). It is generous on purpose: these
    are not "is it significant" tests, they are "how much are you allowed to lose before
    someone has to look at this".
    """

    metric: str
    tolerance: float
    relative: bool = False
    higher_is_better: bool = True
    why: str = ""

    def evaluate(self, baseline: float, candidate: float) -> tuple[bool, float]:
        """Return (tripped, the delta that was tested) for one baseline/candidate pair."""
        delta = candidate - baseline
        if self.relative:
            if baseline == 0:
                return False, 0.0
            delta = delta / abs(baseline)
        tripped = delta < self.tolerance if self.higher_is_better else delta > self.tolerance
        return tripped, delta

    def describe(self) -> str:
        unit = "x baseline" if self.relative else "abs"
        direction = "must not fall below" if self.higher_is_better else "must not rise above"
        return f"{self.metric} {direction} {self.tolerance:+.4g} ({unit})"


DEFAULT_GUARDS: tuple[Guard, ...] = (
    Guard(
        "p_at_1",
        -0.005,
        why="the headline accuracy. Below this the speedup was paid for with correctness.",
    ),
    Guard(
        "auto_approve_precision",
        -0.010,
        why=(
            "the only metric with a blast radius outside the benchmark: an auto-approved "
            "wrong match applies the wrong protection level to a real column, unreviewed."
        ),
    ),
    Guard(
        "r_at_10",
        -0.010,
        why=(
            "the ceiling for anything downstream. Once the gold entry leaves the top 10, "
            "no reranker, threshold or human shortlist can get it back."
        ),
    ),
    Guard(
        "peak_memory_mb",
        0.10,
        relative=True,
        higher_is_better=False,
        why="trading 10%+ of the footprint for speed changes what hardware this runs on.",
    ),
)


# =============================================================================
# PROVENANCE  (M4)
# =============================================================================


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _cpu_busy_percent() -> float | None:
    """
    How loaded the machine was when this was measured.

    Not decorative: the false 715 -> 520 entries/sec "regression" in this repo's history
    was machine state, and there was no field in the result file that could have shown it.
    """
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    return float(psutil.cpu_percent(interval=0.3))


def provenance(seed: int, config: dict[str, Any] | None) -> dict[str, Any]:
    """Everything needed to re-run this measurement and get the same answer."""
    dirty = bool(_git("status", "--porcelain"))
    return {
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": dirty,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": _cpu_count(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "seed": seed,
        "cpu_busy_percent_at_start": _cpu_busy_percent(),
        "config": config or {},
        "ledger_format_version": LEDGER_FORMAT_VERSION,
    }


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 0


# =============================================================================
# PAIRED STATISTICS  (M3)
# =============================================================================


@dataclass
class PairedResult:
    """Effect size for one metric, with the interval that says whether to believe it."""

    metric: str
    baseline: float
    candidate: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float | None
    test: str
    n_pairs: int
    n_discordant: int

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def paired_bootstrap_ci(
    before: Sequence[float],
    after: Sequence[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> tuple[float, float, float]:
    """
    Bootstrap the before/after difference by resampling QUERIES, not observations.

    Resampling query indices once and applying them to both vectors is what makes this
    paired: a query that is hard for both systems moves in or out of every resample
    together, so its shared difficulty cancels instead of inflating the interval. The
    unpaired version of this on 1500 queries gives an interval several times too wide,
    which is how a real accuracy loss gets waved through as "not significant".

    `statistic` defaults to the mean, which is the right thing for a rate like P@1. Pass
    a callable for metrics that are not means -- auto-approve precision is a ratio whose
    DENOMINATOR also moves, so it must be recomputed inside each resample.
    """
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    if b.shape != a.shape:
        raise ValueError(f"paired vectors must align: {b.shape} vs {a.shape}")
    n = b.size
    if n == 0:
        return 0.0, 0.0, 0.0

    stat = statistic or (lambda x, y: float(np.mean(y) - np.mean(x)))
    point = stat(b, a)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    draws = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        take = idx[i]
        draws[i] = stat(b[take], a[take])

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_precision_ci(
    flag_before: Sequence[float],
    correct_before: Sequence[float],
    flag_after: Sequence[float],
    correct_after: Sequence[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """
    Paired bootstrap CI for the change in auto-approve precision.

    Precision is correct/auto-approved, and BOTH move: a change can raise precision purely
    by auto-approving less. Resampling queries and recomputing the whole ratio on each
    resample is the only version that keeps those two effects attached to each other.
    Resamples where a side auto-approves nothing are dropped -- precision is undefined
    there, and the count of drops is what tells you the estimate is thin.
    """
    fb = np.asarray(flag_before, dtype=np.float64)
    cb = np.asarray(correct_before, dtype=np.float64)
    fa = np.asarray(flag_after, dtype=np.float64)
    ca = np.asarray(correct_after, dtype=np.float64)
    n = fb.size
    if n == 0:
        return 0.0, 0.0, 0.0

    def ratio(flag: np.ndarray, correct: np.ndarray) -> float:
        d = flag.sum()
        return float(correct.sum() / d) if d else float("nan")

    point = ratio(fa, ca) - ratio(fb, cb)

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_resamples):
        take = rng.integers(0, n, size=n)
        val = ratio(fa[take], ca[take]) - ratio(fb[take], cb[take])
        if math.isfinite(val):
            draws.append(val)
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(draws), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def exact_mcnemar_p(n_gained: int, n_lost: int) -> float:
    """
    Two-sided exact McNemar p for paired BINARY outcomes.

    Only the discordant queries carry information: the ones both systems get right, and
    the ones both get wrong, say nothing about which system is better. Under the null the
    discordant queries split 50/50, so this is an exact binomial tail -- no normal
    approximation, valid at any count, which matters because a 2-point move on this
    benchmark is roughly 30 discordant queries.
    """
    n = n_gained + n_lost
    if n == 0:
        return 1.0
    k = min(n_gained, n_lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def exact_sign_flip_p(
    diffs: Sequence[float],
    *,
    max_exact_pairs: int = 20,
    n_resamples: int = 20_000,
    seed: int = DEFAULT_SEED,
) -> tuple[float, str]:
    """
    Two-sided paired permutation test on per-query differences (sign-flip).

    Enumerates every sign assignment when the number of non-zero differences is small
    enough to be exact; falls back to a seeded Monte-Carlo permutation above that, which
    is reported by name so nobody mistakes an approximation for an exact result.
    """
    d = np.asarray([x for x in diffs if x != 0], dtype=np.float64)
    n = d.size
    if n == 0:
        return 1.0, "sign-flip (no discordant pairs)"

    observed = abs(float(d.sum()))
    if n <= max_exact_pairs:
        count = 0
        total = 1 << n
        for mask in range(total):
            signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
            if abs(float((d * signs).sum())) >= observed - 1e-12:
                count += 1
        return count / total, f"exact sign-flip permutation (2^{n} assignments)"

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, n))
    stats = np.abs(signs @ d)
    p = float((np.sum(stats >= observed - 1e-12) + 1) / (n_resamples + 1))
    return p, f"Monte-Carlo sign-flip permutation ({n_resamples} draws, seed={seed})"


def paired_compare_metric(
    metric: str,
    before: Sequence[float],
    after: Sequence[float],
    *,
    seed: int = DEFAULT_SEED,
    n_resamples: int = 10_000,
) -> PairedResult:
    """Effect size + 95% CI + an exact paired p-value for one per-query metric."""
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    diffs = a - b
    gained = int(np.sum(diffs > 0))
    lost = int(np.sum(diffs < 0))

    point, lo, hi = paired_bootstrap_ci(b, a, n_resamples=n_resamples, seed=seed)

    binary = bool(np.all(np.isin(b, (0.0, 1.0))) and np.all(np.isin(a, (0.0, 1.0))))
    if binary:
        p = exact_mcnemar_p(gained, lost)
        test = f"exact McNemar (paired binary, {gained} gained / {lost} lost)"
    else:
        p, test = exact_sign_flip_p(diffs, seed=seed)

    return PairedResult(
        metric=metric,
        baseline=float(np.mean(b)) if b.size else 0.0,
        candidate=float(np.mean(a)) if a.size else 0.0,
        delta=point,
        ci_low=lo,
        ci_high=hi,
        p_value=p,
        test=test,
        n_pairs=int(b.size),
        n_discordant=gained + lost,
    )


# =============================================================================
# QUALITY  (paired, per-query)
# =============================================================================


@dataclass
class QualityMetrics:
    """
    Aggregate quality plus the per-query vectors the aggregates were computed from.

    The vectors are the point. Without them a later comparison can only diff two scalars,
    which throws away the pairing and makes an honest interval impossible (M3). They cost
    a few KB per record; that is the cheapest part of this file.
    """

    benchmark: str
    corpus: str  # "real" | "synthetic"
    n_queries: int
    n_entries: int
    p_at_1: float
    r_at_5: float
    r_at_10: float
    mrr_at_10: float
    auto_approve_precision: float
    auto_approve_coverage: float
    query_ids: list[str] = dc_field(default_factory=list)
    correct_at_1: list[int] = dc_field(default_factory=list)
    hit_at_5: list[int] = dc_field(default_factory=list)
    hit_at_10: list[int] = dc_field(default_factory=list)
    rr_at_10: list[float] = dc_field(default_factory=list)
    auto_approved: list[int] = dc_field(default_factory=list)
    auto_correct: list[int] = dc_field(default_factory=list)
    seconds: float = 0.0
    notes: list[str] = dc_field(default_factory=list)

    def vector(self, metric: str) -> list[float]:
        return {
            "p_at_1": [float(x) for x in self.correct_at_1],
            "r_at_5": [float(x) for x in self.hit_at_5],
            "r_at_10": [float(x) for x in self.hit_at_10],
            "mrr_at_10": list(self.rr_at_10),
            "auto_approve_coverage": [float(x) for x in self.auto_approved],
        }[metric]


def _load_dataset(benchmark: str, limit: int | None) -> tuple[Any, str, list[str]]:
    """Load a labelled corpus, or say so and fall back rather than invent numbers."""
    from eval_harness import BENCH_ROOT, Dataset

    notes: list[str] = []
    path = BENCH_ROOT / benchmark
    if path.exists():
        return Dataset.load(benchmark, limit=limit), "real", notes

    notes.append(
        f"FHIR corpus absent at {path}; degraded to the synthetic generator. "
        f"Synthetic queries are derived from the same tokens as their gold entry, so "
        f"absolute scores are NOT comparable to a real-corpus record and compare() will "
        f"refuse to diff across the two."
    )
    return _synthetic_dataset(limit or 400), "synthetic", notes


def _synthetic_dataset(n: int) -> Any:
    """A labelled corpus from perf_harness's generator, for machines with no benchmarks."""
    from eval_harness import Dataset, DictEntry, Query
    from perf_harness import _make_glossary

    raw = _make_glossary(max(n * 3, 600))
    entries = [
        DictEntry(
            id=e.id,
            business_name=e.business_name,
            logical_name="",
            description=e.definition,
            data_type=str(e.data_type.value),
            domain=e.domain,
        )
        for e in raw
    ]
    queries = [
        Query(
            id=f"q::{e.id}",
            field_name=raw[i].logical_name,
            field_path=f"{raw[i].domain}.{raw[i].logical_name}",
            data_type=str(raw[i].data_type.value),
            parent_path=raw[i].domain,
            gold_id=e.id,
        )
        for i, e in enumerate(entries[:n])
    ]
    return Dataset(name="synthetic", entries=entries, queries=queries)


def default_matcher_factory(results_per_field: int = 10) -> Callable[[], Any]:
    """
    The shipped configuration -- bundled int8 ONNX encoder, BM25 on, calibrated thresholds.

    `results_per_field` is raised to 10 because R@10 and MRR@10 cannot be computed from a
    top-5 result list. That is a measurement setting, not a tuning change: it widens what
    is REPORTED, and the ranking at 1 and 5 is unaffected.
    """

    def factory() -> Any:
        from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher

        return NexusMatcher.from_config(MatchingConfig(results_per_field=results_per_field))

    return factory


def measure_quality(
    matcher_factory: Callable[[], Any] | None = None,
    benchmark: str = DEFAULT_BENCHMARK,
    *,
    limit: int | None = None,
) -> QualityMetrics:
    """
    Per-query correctness vectors plus P@1, R@5, R@10, MRR@10 and the auto-approve pair.

    Leakage control matches eval_pipeline: `logical_name` is blanked before indexing, so
    the source system's technical column name is never in the corpus and retrieval has to
    work from the business name and human definition. Indexing it inflates every number
    here to near-ceiling and has done so before.
    """
    from eval_pipeline import to_data_type

    from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
    from nexus_matcher.shared.types.base import MatchDecision

    factory = matcher_factory or default_matcher_factory()
    ds, corpus, notes = _load_dataset(benchmark, limit)
    matcher = factory()

    per_field = getattr(matcher._config, "results_per_field", 0)
    if per_field < 10:
        notes.append(
            f"matcher returns only {per_field} results per field; R@10 and MRR@10 are "
            f"truncated at {per_field} and will read low for a measurement reason, not a "
            f"code reason."
        )

    entries = [
        DictionaryEntry(
            id=e.id,
            business_name=e.business_name,
            logical_name="",
            definition=e.description,
            data_type=to_data_type(e.data_type),
            domain=e.domain,
        )
        for e in ds.entries
    ]
    fields = [
        SchemaField(
            name=q.field_name,
            data_type=to_data_type(q.data_type),
            full_path=q.field_path,
            parent_path=q.parent_path,
        )
        for q in ds.queries
    ]

    matcher._embedding_provider.embed_documents(["warmup text"])
    t0 = time.perf_counter()
    matcher._index_dictionary(entries)
    results = matcher._match_fields(fields)
    seconds = time.perf_counter() - t0

    # _match_fields documents "exactly one entry per field, in input order" as its
    # contract, so positional alignment is safe and avoids re-deriving the result key.
    ranked = list(results.values())
    if len(ranked) != len(fields):
        raise RuntimeError(f"matcher returned {len(ranked)} result sets for {len(fields)} fields")

    return _score_quality(ds, ranked, MatchDecision, corpus, seconds, notes)


def _score_quality(
    ds: Any,
    ranked: list[Any],
    decision_enum: Any,
    corpus: str,
    seconds: float,
    notes: list[str],
) -> QualityMetrics:
    """Turn one run's raw match results into per-query vectors and their aggregates."""
    query_ids: list[str] = []
    correct1: list[int] = []
    hit5: list[int] = []
    hit10: list[int] = []
    rr10: list[float] = []
    auto: list[int] = []
    auto_ok: list[int] = []

    for q, matches in zip(ds.queries, ranked, strict=True):
        rank = next(
            (i for i, m in enumerate(matches, 1) if m.dictionary_entry.id == q.gold_id),
            None,
        )
        query_ids.append(q.id)
        correct1.append(1 if rank == 1 else 0)
        hit5.append(1 if rank is not None and rank <= 5 else 0)
        hit10.append(1 if rank is not None and rank <= 10 else 0)
        rr10.append(1.0 / rank if rank is not None and rank <= 10 else 0.0)

        approved = bool(matches) and matches[0].decision == decision_enum.AUTO_APPROVE
        auto.append(1 if approved else 0)
        auto_ok.append(1 if approved and rank == 1 else 0)

    n = len(query_ids) or 1
    n_auto = sum(auto)
    return QualityMetrics(
        benchmark=ds.name,
        corpus=corpus,
        n_queries=len(query_ids),
        n_entries=len(ds.entries),
        p_at_1=sum(correct1) / n,
        r_at_5=sum(hit5) / n,
        r_at_10=sum(hit10) / n,
        mrr_at_10=sum(rr10) / n,
        auto_approve_precision=(sum(auto_ok) / n_auto) if n_auto else 0.0,
        auto_approve_coverage=n_auto / n,
        query_ids=query_ids,
        correct_at_1=correct1,
        hit_at_5=hit5,
        hit_at_10=hit10,
        rr_at_10=rr10,
        auto_approved=auto,
        auto_correct=auto_ok,
        seconds=round(seconds, 3),
        notes=notes,
    )


# =============================================================================
# COST  (repeated trials; median AND min AND IQR)
# =============================================================================


@dataclass
class CostStat:
    """
    One cost metric at one scale, summarised three ways because one way lies.

    median : what to compare on. Robust to a single scheduler hiccup.
    best   : the fastest observed trial (or lowest, for latency and memory). The closest
             thing to the machine's true cost -- interference only ever makes a run slower,
             never faster, so the extreme in the good direction is the least contaminated.
    iqr    : the spread. This is the honesty field: a large IQR means the machine was busy
             and NOTHING measured on it should be believed, regardless of the median.
    """

    median: float
    best: float
    worst: float
    iqr: float
    values: list[float]

    @property
    def iqr_relative(self) -> float:
        return self.iqr / abs(self.median) if self.median else 0.0


@dataclass
class CostAtScale:
    entries: int
    fields: int
    trials: int
    stats: dict[str, CostStat]

    @property
    def key(self) -> str:
        return f"entries={self.entries},fields={self.fields}"


def _summarise(values: Sequence[float], higher_is_better: bool) -> CostStat:
    vals = [float(v) for v in values]
    ordered = sorted(vals)
    if len(ordered) >= 4:
        q1, q3 = np.percentile(np.asarray(ordered), [25, 75])
        iqr = float(q3 - q1)
    else:
        # With 2-3 trials the quartiles are interpolation artefacts; the full range is the
        # honest statement of what was seen.
        iqr = float(ordered[-1] - ordered[0])
    return CostStat(
        median=float(statistics.median(vals)),
        best=float(max(vals) if higher_is_better else min(vals)),
        worst=float(min(vals) if higher_is_better else max(vals)),
        iqr=iqr,
        values=vals,
    )


def measure_cost(
    matcher_factory: Callable[[], Any] | None = None,
    scales: Sequence[tuple[int, int]] = DEFAULT_SCALES,
    *,
    trials: int = DEFAULT_TRIALS,
    progress: bool = False,
) -> list[CostAtScale]:
    """
    Throughput, latency percentiles, index rate and peak memory over repeated trials.

    Delegates the actual timing to perf_harness.measure so there is exactly one definition
    of "fields per second" in this repo. Trials are repeated because a single timing is
    not a measurement -- it is one sample from a distribution whose width is the entire
    subject of calibrate_noise().

    Peak memory is tracemalloc's high-water mark. perf_harness notes that tracemalloc
    misses the numpy/onnxruntime buffers that dominate the real footprint, so RSS is the
    truer number -- but RSS drifts with whatever else the process has touched and is far
    too noisy to guard on. The allocator number is stable enough to be a guard; RSS is
    recorded alongside it for context.
    """
    from perf_harness import measure

    out: list[CostAtScale] = []
    for entries_n, fields_n in scales:
        rows = []
        for t in range(trials):
            if progress:
                print(f"    cost trial {t + 1}/{trials} @ {entries_n} entries", flush=True)
            rows.append(measure(entries_n, fields_n, matcher_factory=matcher_factory))
        stats = {
            "match_fields_per_sec": _summarise([r.match_fields_per_sec for r in rows], True),
            "index_entries_per_sec": _summarise([r.index_entries_per_sec for r in rows], True),
            "latency_ms_p50": _summarise([r.latency_ms_p50 for r in rows], False),
            "latency_ms_p95": _summarise([r.latency_ms_p95 for r in rows], False),
            "latency_ms_p99": _summarise([r.latency_ms_p99 for r in rows], False),
            "peak_memory_mb": _summarise([r.peak_tracemalloc_mb for r in rows], False),
            "rss_delta_mb": _summarise([r.rss_delta_mb or 0.0 for r in rows], False),
        }
        out.append(CostAtScale(entries=entries_n, fields=fields_n, trials=trials, stats=stats))
    return out


# =============================================================================
# NOISE CALIBRATION  (M1)
# =============================================================================


@dataclass
class NoiseBand:
    """
    What this machine does to identical code, measured rather than guessed.

    `relative[metric]` is the full observed spread across repeats divided by the median --
    i.e. how far apart two runs of the SAME code landed. A candidate delta smaller than
    that is not a result; it is this machine.
    """

    repeats: int
    floor: float
    seed: int
    relative: dict[str, float] = dc_field(default_factory=dict)
    absolute: dict[str, float] = dc_field(default_factory=dict)
    samples: dict[str, list[float]] = dc_field(default_factory=dict)
    notes: list[str] = dc_field(default_factory=list)

    def band(self, metric: str, scope: str = "") -> float:
        """
        The band for one metric: the WORST spread it showed at ANY scale, never one scale's.

        Measured here on 2026-08-09, and the reason this is not `relative[scope::metric]`:
        the three calibration runs happened to fall in a calm stretch at 5000 entries and
        put that scale's band at 3.5% -- then an independent run of the SAME code moved
        match_fields_per_sec +12.6% there. At 1000 entries the same metric had measured
        30.6%. Scheduler interference is a property of the machine and the metric, not of
        the corpus size, so a scale calibrated during a quiet minute must not be allowed
        to certify the next fluctuation as a win. Erring toward INCONCLUSIVE is the whole
        point; the reverse error is mistake M1.
        """
        del scope  # kept in the signature so callers read as scope-aware; see above
        spreads = [v for k, v in self.relative.items() if k.split("::")[-1] == metric]
        return max([self.floor, *spreads])

    def render(self) -> str:
        lines = [
            f"Noise floor from {self.repeats} runs of IDENTICAL code "
            f"(floor={self.floor:.1%}, seed={self.seed})",
            f"  {'metric':<46} {'spread':>9} {'band used':>10}  samples",
            f"  {'-' * 46} {'-' * 9} {'-' * 10}  {'-' * 30}",
        ]
        for key in sorted(self.relative):
            vals = self.samples.get(key, [])
            shown = ", ".join(f"{v:.4g}" for v in vals)
            lines.append(
                f"  {key:<46} {self.relative[key]:>8.1%} "
                f"{self.band(key.split('::')[-1]):>10.1%}  {shown}"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def calibrate_noise(
    matcher_factory: Callable[[], Any] | None = None,
    *,
    repeats: int = 3,
    scales: Sequence[tuple[int, int]] = DEFAULT_SCALES,
    trials: int = DEFAULT_TRIALS,
    benchmark: str = DEFAULT_BENCHMARK,
    quality_limit: int | None = None,
    floor: float = 0.03,
    seed: int = DEFAULT_SEED,
    progress: bool = True,
) -> tuple[NoiseBand, list[Measurement]]:
    """
    Measure identical code `repeats` times and record the spread of every metric.

    The band is calibrated on the STATISTIC THAT IS LATER COMPARED (the median across
    trials), not on individual trials, because that is the number a verdict rests on.
    Calibrating on raw trials would overstate the band and hide real wins.

    `floor` exists because two runs can agree to 0.4% by luck, and a 0% band would then
    certify the next 1% fluctuation as a win. 3% is deliberately cheap insurance: it is
    well under the ~30% swing this repo's own history recorded, so it never masks a real
    optimization, and it makes a suspiciously tight calibration harmless.

    Returns the band and every full measurement, so the caller can reuse one as a baseline
    instead of paying for the same work twice.
    """
    factory = matcher_factory or default_matcher_factory()
    runs: list[Measurement] = []
    for i in range(repeats):
        if progress:
            print(f"  calibration run {i + 1}/{repeats} (identical code)", flush=True)
        runs.append(
            measure_all(
                matcher_factory=factory,
                label=f"noise-calibration-{i + 1}",
                scales=scales,
                trials=trials,
                benchmark=benchmark,
                quality_limit=quality_limit,
                seed=seed,
                progress=progress,
            )
        )

    band = NoiseBand(repeats=repeats, floor=floor, seed=seed)
    for metric in QUALITY_METRICS:
        vals = [getattr(m.quality, metric) for m in runs if m.quality]
        _add_band(band, metric, vals)
    for scale_i, scale in enumerate(scales):
        key_scope = f"entries={scale[0]},fields={scale[1]}"
        for metric in (*COST_METRICS, "rss_delta_mb"):
            vals = [m.cost[scale_i].stats[metric].median for m in runs if m.cost]
            _add_band(band, metric, vals, scope=key_scope)

    quality_spread = max(
        (band.relative.get(m, 0.0) for m in QUALITY_METRICS),
        default=0.0,
    )
    if quality_spread == 0.0:
        band.notes.append(
            "quality metrics were bit-identical across repeats -- the pipeline is "
            "deterministic, so quality deltas are judged by the paired CI, not this band."
        )
    else:
        band.notes.append(
            f"quality is NOT deterministic on this machine (max spread {quality_spread:.2%}); "
            f"treat any quality delta smaller than that as unmeasurable."
        )
    return band, runs


def _add_band(band: NoiseBand, metric: str, values: Sequence[float], scope: str = "") -> None:
    key = f"{scope}::{metric}" if scope else metric
    vals = [float(v) for v in values]
    if not vals:
        return
    med = statistics.median(vals)
    spread = max(vals) - min(vals)
    band.samples[key] = vals
    band.absolute[key] = spread
    band.relative[key] = (spread / abs(med)) if med else 0.0


# =============================================================================
# RECORDS + LEDGER
# =============================================================================


@dataclass
class Measurement:
    """One complete, provenanced observation of one version of the code."""

    record_id: str
    label: str
    created_utc: str
    benchmark: str
    corpus: str
    provenance: dict[str, Any]
    quality: QualityMetrics | None = None
    cost: list[CostAtScale] | None = None
    noise: NoiseBand | None = None
    target_metric: str | None = None
    simulated: bool = False
    notes: list[str] = dc_field(default_factory=list)

    def scale_keys(self) -> list[str]:
        return [c.key for c in (self.cost or [])]

    def cost_stat(self, metric: str, scope: str) -> CostStat | None:
        for c in self.cost or []:
            if c.key == scope:
                return c.stats.get(metric)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "label": self.label,
            "created_utc": self.created_utc,
            "benchmark": self.benchmark,
            "corpus": self.corpus,
            "provenance": self.provenance,
            "quality": asdict(self.quality) if self.quality else None,
            "cost": [asdict(c) for c in self.cost] if self.cost else None,
            "noise": asdict(self.noise) if self.noise else None,
            "target_metric": self.target_metric,
            "simulated": self.simulated,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Measurement:
        quality = QualityMetrics(**d["quality"]) if d.get("quality") else None
        cost = None
        if d.get("cost"):
            cost = [
                CostAtScale(
                    entries=c["entries"],
                    fields=c["fields"],
                    trials=c["trials"],
                    stats={k: CostStat(**v) for k, v in c["stats"].items()},
                )
                for c in d["cost"]
            ]
        noise = NoiseBand(**d["noise"]) if d.get("noise") else None
        return cls(
            record_id=d["record_id"],
            label=d["label"],
            created_utc=d["created_utc"],
            benchmark=d["benchmark"],
            corpus=d["corpus"],
            provenance=d["provenance"],
            quality=quality,
            cost=cost,
            noise=noise,
            target_metric=d.get("target_metric"),
            simulated=d.get("simulated", False),
            notes=d.get("notes", []),
        )


def measure_all(
    matcher_factory: Callable[[], Any] | None = None,
    *,
    label: str,
    scales: Sequence[tuple[int, int]] = DEFAULT_SCALES,
    trials: int = DEFAULT_TRIALS,
    benchmark: str = DEFAULT_BENCHMARK,
    quality_limit: int | None = None,
    target_metric: str | None = None,
    seed: int = DEFAULT_SEED,
    progress: bool = False,
) -> Measurement:
    """Quality + cost + provenance for one version of the code, in one object."""
    factory = matcher_factory or default_matcher_factory()

    config: dict[str, Any] | None = None
    try:
        from dataclasses import asdict as _asdict

        config = _asdict(factory()._config)
    except (AttributeError, TypeError):  # pragma: no cover - non-standard factory
        config = None

    if progress:
        print(f"  measuring quality on '{benchmark}' ...", flush=True)
    quality = measure_quality(factory, benchmark, limit=quality_limit)

    if progress:
        print("  measuring cost ...", flush=True)
    cost = measure_cost(factory, scales, trials=trials, progress=progress)

    return Measurement(
        record_id=uuid.uuid4().hex[:12],
        label=label,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        benchmark=quality.benchmark,
        corpus=quality.corpus,
        provenance=provenance(seed, config),
        quality=quality,
        cost=cost,
        target_metric=target_metric,
        notes=list(quality.notes),
    )


def record(
    measurement: Measurement,
    *,
    noise: NoiseBand | None = None,
    target_metric: str | None = None,
    path: Path = LEDGER_PATH,
) -> Measurement:
    """
    Append one measurement to the ledger. Append-only: nothing here is ever rewritten.

    A ledger you can edit is a ledger you can talk yourself into editing, and the whole
    value of the history is that the embarrassing entries are still in it.
    """
    if noise is not None:
        measurement.noise = noise
    if target_metric is not None:
        measurement.target_metric = target_metric
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(measurement.to_dict(), default=_json_default) + "\n")
    return measurement


def _json_default(o: Any) -> Any:
    """
    Let numpy scalars into the ledger.

    Per-query vectors arrive from whatever the caller measured with, and anything that
    touched numpy hands back `np.int64`, which `json` refuses. Failing the WRITE after a
    multi-minute measurement is the worst possible moment to discover that, so the
    encoder unwraps them instead.
    """
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def load_records(path: Path = LEDGER_PATH) -> list[Measurement]:
    if not path.exists():
        return []
    return [
        Measurement.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_record(record_id: str, path: Path = LEDGER_PATH) -> Measurement:
    for r in load_records(path):
        if r.record_id.startswith(record_id):
            return r
    raise KeyError(f"no ledger record starting with {record_id!r} in {path}")


# =============================================================================
# COMPARISON + VERDICT
# =============================================================================


@dataclass
class MetricDelta:
    metric: str
    scope: str
    baseline: float
    candidate: float
    delta: float
    relative_delta: float
    higher_is_better: bool
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    test: str = ""
    noise_band: float | None = None
    beyond_noise: bool | None = None
    guarded: bool = False
    guard_tripped: bool = False
    guard_reason: str = ""
    # The guard wanted to fail, but the move is inside this machine's calibrated noise
    # for that metric -- so the guard is INOPERATIVE here, not satisfied. Failing on it
    # would be mistake M1 wearing M2's clothes; passing silently would be M2. It is
    # surfaced as a warning instead, because the answer is "measure on a quieter machine".
    guard_masked_by_noise: bool = False

    @property
    def improved(self) -> bool:
        return self.delta > 0 if self.higher_is_better else self.delta < 0


@dataclass
class Comparison:
    baseline_id: str
    candidate_id: str
    baseline_label: str
    candidate_label: str
    target_metric: str | None
    target_scope: str | None
    verdict: str
    reasons: list[str]
    deltas: list[MetricDelta]
    warnings: list[str]

    def render(self) -> str:
        head = (
            f"{'metric':<26} {'scope':<24} {'baseline':>11} {'candidate':>11} "
            f"{'delta':>10} {'95% CI':>21} {'noise':>7} {'flag':<10}"
        )
        lines = [
            f"\n{self.baseline_label}  ->  {self.candidate_label}",
            f"  baseline {self.baseline_id}   candidate {self.candidate_id}",
            f"  target: {self.target_metric or '(none declared)'}"
            + (f" @ {self.target_scope}" if self.target_scope else ""),
            "",
            head,
            "-" * len(head),
        ]
        for d in self.deltas:
            ci = (
                f"[{d.ci_low:+.4f}, {d.ci_high:+.4f}]"
                if d.ci_low is not None and d.ci_high is not None
                else ""
            )
            noise = f"{d.noise_band:.1%}" if d.noise_band is not None else ""
            if d.guard_tripped:
                flag = "GUARD-FAIL"
            elif d.guard_masked_by_noise:
                flag = "GUARD-BLIND"
            elif d.guarded:
                flag = "guard-ok"
            elif d.beyond_noise is False:
                flag = "in-noise"
            else:
                flag = ""
            lines.append(
                f"{d.metric:<26} {d.scope:<24} {d.baseline:>11.4f} {d.candidate:>11.4f} "
                f"{d.delta:>+10.4f} {ci:>21} {noise:>7} {flag:<10}"
            )
        lines.append("")
        lines.append(f"VERDICT: {self.verdict}")
        for r in self.reasons:
            lines.append(f"  - {r}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def _quality_deltas(
    base: Measurement,
    cand: Measurement,
    guards: dict[str, Guard],
    band: NoiseBand | None,
    seed: int,
    n_resamples: int,
) -> list[MetricDelta]:
    """Paired, per-query comparison of every quality metric (M3)."""
    out: list[MetricDelta] = []
    bq, cq = base.quality, cand.quality
    if bq is None or cq is None:
        return out

    for metric in QUALITY_METRICS:
        guard = guards.get(metric)
        if metric == "auto_approve_precision":
            point, lo, hi = paired_precision_ci(
                bq.auto_approved,
                bq.auto_correct,
                cq.auto_approved,
                cq.auto_correct,
                n_resamples=n_resamples,
                seed=seed,
            )
            pr = PairedResult(
                metric=metric,
                baseline=bq.auto_approve_precision,
                candidate=cq.auto_approve_precision,
                delta=point,
                ci_low=lo,
                ci_high=hi,
                p_value=None,
                test="paired bootstrap over queries (ratio recomputed per resample)",
                n_pairs=bq.n_queries,
                n_discordant=0,
            )
        else:
            pr = paired_compare_metric(
                metric,
                bq.vector(metric),
                cq.vector(metric),
                seed=seed,
                n_resamples=n_resamples,
            )

        d = MetricDelta(
            metric=metric,
            scope="quality",
            baseline=pr.baseline,
            candidate=pr.candidate,
            delta=pr.delta,
            relative_delta=(pr.delta / abs(pr.baseline)) if pr.baseline else 0.0,
            higher_is_better=True,
            ci_low=pr.ci_low,
            ci_high=pr.ci_high,
            p_value=pr.p_value,
            test=pr.test,
        )
        if band is not None:
            abs_band = band.absolute.get(metric, 0.0)
            d.noise_band = band.relative.get(metric, 0.0)
            d.beyond_noise = abs(pr.delta) > abs_band
        if guard is not None:
            _apply_guard(d, guard, pr.baseline, pr.candidate)
        out.append(d)
    return out


def _apply_guard(d: MetricDelta, guard: Guard, baseline: float, candidate: float) -> None:
    """Record whether a guard failed -- or whether this machine is too noisy to ask."""
    d.guarded = True
    tripped, tested = guard.evaluate(baseline, candidate)
    if not tripped:
        return

    unit = "x baseline" if guard.relative else ""
    where = "" if d.scope == "quality" else f" @ {d.scope}"
    if d.beyond_noise is False:
        d.guard_masked_by_noise = True
        d.guard_reason = (
            f"{d.metric}{where} moved {tested:+.4g}{unit}, past its {guard.tolerance:+.4g} "
            f"tolerance -- but that is INSIDE the {d.noise_band:.1%} noise band measured "
            f"for it on this machine. The guard is inoperative here, not satisfied: "
            f"re-measure with more trials or on a quieter machine before trusting either "
            f"verdict."
        )
        return

    d.guard_tripped = True
    d.guard_reason = (
        f"{d.metric}{where} moved {tested:+.4g}{unit}, past its {guard.tolerance:+.4g} "
        f"tolerance. {guard.why}"
    )


def _cost_deltas(
    base: Measurement,
    cand: Measurement,
    guards: dict[str, Guard],
    band: NoiseBand | None,
    statistic: str,
) -> list[MetricDelta]:
    """Per-scale cost comparison, judged against the calibrated noise band (M1)."""
    out: list[MetricDelta] = []
    for scope in base.scale_keys():
        if scope not in cand.scale_keys():
            continue
        for metric in COST_METRICS:
            b_stat = base.cost_stat(metric, scope)
            c_stat = cand.cost_stat(metric, scope)
            if b_stat is None or c_stat is None:
                continue
            b_val = getattr(b_stat, statistic)
            c_val = getattr(c_stat, statistic)
            higher = metric in HIGHER_IS_BETTER
            rel = ((c_val - b_val) / abs(b_val)) if b_val else 0.0

            d = MetricDelta(
                metric=metric,
                scope=scope,
                baseline=b_val,
                candidate=c_val,
                delta=c_val - b_val,
                relative_delta=rel,
                higher_is_better=higher,
                test=f"{statistic} of {len(b_stat.values)} vs {len(c_stat.values)} trials",
            )
            if band is not None:
                d.noise_band = band.band(metric, scope)
                d.beyond_noise = abs(rel) > d.noise_band
            guard = guards.get(metric)
            if guard is not None:
                _apply_guard(d, guard, b_val, c_val)
            out.append(d)
    return out


def compare(
    baseline_record: Measurement,
    candidate_record: Measurement,
    *,
    target: str | None = None,
    target_scope: str | None = None,
    guards: Sequence[Guard] = DEFAULT_GUARDS,
    noise: NoiseBand | None = None,
    statistic: str = "median",
    seed: int = DEFAULT_SEED,
    n_resamples: int = 10_000,
) -> Comparison:
    """
    Per-metric delta, CI, and a verdict of WIN / NEUTRAL / REGRESSION / INCONCLUSIVE.

    Verdicts, in the order they are decided:

      REGRESSION    any guard tripped, or the declared target moved measurably in the
                    wrong direction. Guards trip on the point estimate -- see the module
                    docstring for why that asymmetry is intentional. The one exception is
                    a guard whose breach is smaller than the calibrated noise for its own
                    metric: that is not a regression, it is an inoperative guard, and it
                    is raised as a warning instead of being silently passed or failed.
      WIN           the target improved by more than the calibrated noise band (cost) or
                    with a 95% CI that excludes zero (quality), and no guard tripped.
      INCONCLUSIVE  the target moved, but by less than this machine's noise. This is the
                    verdict that the 715 -> 520 entries/sec "regression" should have got.
      NEUTRAL       no target was declared and nothing tripped: a change verified not to
                    break anything, which is the correct result for a refactor.
    """
    guard_map = {g.metric: g for g in guards}
    band = noise or candidate_record.noise or baseline_record.noise
    warnings: list[str] = []

    if baseline_record.corpus != candidate_record.corpus:
        raise ValueError(
            f"refusing to compare a '{baseline_record.corpus}' record against a "
            f"'{candidate_record.corpus}' one -- the corpora are not the same task."
        )
    if baseline_record.benchmark != candidate_record.benchmark:
        raise ValueError(
            f"refusing to compare across benchmarks: "
            f"{baseline_record.benchmark!r} vs {candidate_record.benchmark!r}"
        )
    if band is None:
        warnings.append(
            "NO NOISE CALIBRATION on either record. Every cost delta below is unjudged: "
            "run calibrate_noise() before believing any of them."
        )
    for r in (baseline_record, candidate_record):
        if r.provenance.get("git_dirty"):
            warnings.append(
                f"{r.label}: measured on a DIRTY tree ({r.record_id}) -- not reproducible."
            )
        if r.simulated:
            warnings.append(f"{r.label}: SIMULATED record, not a measurement.")

    deltas = _quality_deltas(baseline_record, candidate_record, guard_map, band, seed, n_resamples)
    deltas += _cost_deltas(baseline_record, candidate_record, guard_map, band, statistic)
    warnings += [d.guard_reason for d in deltas if d.guard_masked_by_noise]

    target = target or candidate_record.target_metric
    verdict, reasons = _decide(deltas, target, target_scope, band)

    return Comparison(
        baseline_id=baseline_record.record_id,
        candidate_id=candidate_record.record_id,
        baseline_label=baseline_record.label,
        candidate_label=candidate_record.label,
        target_metric=target,
        target_scope=target_scope,
        verdict=verdict,
        reasons=reasons,
        deltas=deltas,
        warnings=warnings,
    )


def _pick_target(
    deltas: Sequence[MetricDelta], target: str, scope: str | None
) -> MetricDelta | None:
    """The target row, or the worst-improving scale when no scope is named."""
    hits = [d for d in deltas if d.metric == target and (scope is None or d.scope == scope)]
    if not hits:
        return None
    # No scope named -> judge on the least favourable scale, so a win at 1k entries cannot
    # paper over a loss at 30k. Optimizations in this repo routinely invert with scale.
    return min(hits, key=lambda d: d.relative_delta if d.higher_is_better else -d.relative_delta)


def _decide(
    deltas: Sequence[MetricDelta],
    target: str | None,
    target_scope: str | None,
    band: NoiseBand | None,
) -> tuple[str, list[str]]:
    """Turn the per-metric rows into one word, with the reason spelled out."""
    tripped = [d for d in deltas if d.guard_tripped]
    if tripped:
        reasons = [d.guard_reason for d in tripped]
        reasons.append(
            "A tripped guard is a REGRESSION whatever the speedup: the metric it protects "
            "is not currency."
        )
        return "REGRESSION", reasons

    if not target:
        return "NEUTRAL", [
            "No target metric declared, and no guard moved past its tolerance.",
            "This says the change broke nothing. It does not say the change helped.",
        ]

    row = _pick_target(deltas, target, target_scope)
    if row is None:
        return "INCONCLUSIVE", [f"target metric {target!r} is not present in both records."]

    where = f" @ {row.scope}" if row.scope != "quality" else ""
    if row.scope == "quality":
        significant = row.ci_low is not None and (row.ci_low > 0 or row.ci_high < 0)
        if not significant:
            return "INCONCLUSIVE", [
                f"{target} moved {row.delta:+.4f}, but the 95% CI "
                f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}] spans zero: the queries do not "
                f"agree on a direction.",
                f"paired test: {row.test}"
                + (f", p={row.p_value:.3g}" if row.p_value is not None else ""),
            ]
        if row.improved:
            return "WIN", [
                f"{target} improved {row.delta:+.4f}, 95% CI "
                f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}] excludes zero.",
                f"paired test: {row.test}"
                + (f", p={row.p_value:.3g}" if row.p_value is not None else ""),
                "No guard tripped.",
            ]
        return "REGRESSION", [
            f"{target} got measurably WORSE ({row.delta:+.4f}, CI "
            f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}] excludes zero) -- and it was the "
            f"metric this change was supposed to improve."
        ]

    if band is None:
        return "INCONCLUSIVE", [
            f"{target}{where} moved {row.relative_delta:+.1%}, but no noise band was "
            f"calibrated, so there is nothing to judge it against. Run calibrate_noise()."
        ]
    if not row.beyond_noise:
        return "INCONCLUSIVE", [
            f"{target}{where} moved {row.relative_delta:+.1%}, which is INSIDE this "
            f"machine's {row.noise_band:.1%} noise band for that metric.",
            "Identical code has already been observed to move this far. Nothing was shown.",
        ]
    if row.improved:
        return "WIN", [
            f"{target}{where} moved {row.relative_delta:+.1%}, beyond the "
            f"{row.noise_band:.1%} noise band ({row.test}).",
            "No guard tripped.",
        ]
    return "REGRESSION", [
        f"{target}{where} moved {row.relative_delta:+.1%} the WRONG WAY, beyond the "
        f"{row.noise_band:.1%} noise band -- and it was this change's own target."
    ]


# =============================================================================
# LEADERBOARD
# =============================================================================


def leaderboard(
    path: Path = LEDGER_PATH,
    *,
    metric: str = "match_fields_per_sec",
    scope: str | None = None,
    statistic: str = "median",
) -> str:
    """Ranked view of the ledger, with the accuracy each entry cost printed beside it."""
    records = load_records(path)
    if not records:
        return f"(ledger at {path} is empty)"

    scope = scope or (records[0].scale_keys() or [""])[0]
    rows: list[tuple[float, Measurement]] = []
    for r in records:
        stat = r.cost_stat(metric, scope)
        rows.append((getattr(stat, statistic) if stat else float("nan"), r))

    higher = metric in HIGHER_IS_BETTER
    rows.sort(key=lambda t: (math.isnan(t[0]), -t[0] if higher else t[0]))

    head = (
        f"{'#':>2} {'label':<30} {'sha':<9} {metric:>15} {'P@1':>7} {'auto-P':>7} "
        f"{'auto-cov':>8} {'peakMB':>7} {'IQR':>6}  flags"
    )
    lines = [
        f"\nOptimization ledger -- {len(records)} records, ranked by {metric} @ {scope}",
        head,
        "-" * len(head),
    ]
    for i, (val, r) in enumerate(rows, 1):
        q = r.quality
        stat = r.cost_stat(metric, scope)
        mem = r.cost_stat("peak_memory_mb", scope)
        flags = []
        if r.simulated:
            flags.append("SIMULATED")
        if r.provenance.get("git_dirty"):
            flags.append("dirty")
        if stat and stat.iqr_relative > 0.15:
            flags.append(f"noisy({stat.iqr_relative:.0%})")
        # Measured 2026-08-09: the baseline was taken at 49.5% CPU busy and an identical
        # re-run at 4.5%, and the throughput between them differed by more than any
        # optimization in this repo has ever produced. A row taken on a loaded machine is
        # not wrong, but nothing should be ranked against it without seeing this.
        busy = r.provenance.get("cpu_busy_percent_at_start")
        if isinstance(busy, (int, float)) and busy > 25:
            flags.append(f"busy({busy:.0f}%)")
        if r.corpus != "real":
            flags.append(r.corpus)
        lines.append(
            f"{i:>2} {r.label[:30]:<30} {str(r.provenance.get('git_sha', ''))[:8]:<9} "
            f"{val:>15.1f} "
            f"{(q.p_at_1 if q else float('nan')):>7.4f} "
            f"{(q.auto_approve_precision if q else float('nan')):>7.4f} "
            f"{(q.auto_approve_coverage if q else float('nan')):>8.4f} "
            f"{(mem.median if mem else float('nan')):>7.1f} "
            f"{(stat.iqr_relative if stat else float('nan')):>6.1%}  {' '.join(flags)}"
        )
    lines.append(
        "\n  Ranked on speed, but the accuracy columns are printed on the same row on "
        "purpose:\n  a fast row with a low P@1 is not above a slow row, it is disqualified."
    )
    return "\n".join(lines)


# =============================================================================
# SELF-TEST SUPPORT
# =============================================================================


def simulate_candidate(
    base: Measurement,
    *,
    label: str,
    speedup: float = 0.0,
    p_at_1_delta: float = 0.0,
    memory_growth: float = 0.0,
    seed: int = DEFAULT_SEED,
) -> Measurement:
    """
    Build a FAKE candidate from a real record, to check the guards actually fire.

    This is how you test a measurement system: you cannot wait for a real regression to
    turn up, and if you have never seen your guards fail you do not know they can. Every
    record produced here carries `simulated=True`, which the leaderboard prints and
    compare() warns about, so a fake can never quietly become a baseline.

    `p_at_1_delta` moves the gold entry from rank 1 to rank 2 on a random subset of
    queries -- the realistic shape of a small accuracy loss. R@5/R@10 are deliberately
    left intact, so this exercises the P@1 guard specifically rather than knocking every
    metric over at once.
    """
    d = json.loads(json.dumps(base.to_dict(), default=_json_default))
    clone = Measurement.from_dict(d)
    clone.record_id = uuid.uuid4().hex[:12]
    clone.label = label
    clone.created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    clone.simulated = True
    clone.notes = [
        f"SIMULATED from {base.record_id}: {speedup:+.0%} speed, "
        f"{p_at_1_delta:+.3f} P@1, {memory_growth:+.0%} memory."
    ]

    if clone.cost:
        for scale in clone.cost:
            for metric, stat in scale.stats.items():
                if metric in ("match_fields_per_sec", "index_entries_per_sec"):
                    factor = 1.0 + speedup
                elif metric.startswith("latency"):
                    factor = 1.0 / (1.0 + speedup) if speedup > -1 else 1.0
                elif metric in ("peak_memory_mb", "rss_delta_mb"):
                    factor = 1.0 + memory_growth
                else:
                    factor = 1.0
                stat.median *= factor
                stat.best *= factor
                stat.worst *= factor
                stat.iqr *= abs(factor)
                stat.values = [v * factor for v in stat.values]

    if clone.quality and p_at_1_delta:
        _perturb_p_at_1(clone.quality, p_at_1_delta, seed)
    return clone


def _perturb_p_at_1(q: QualityMetrics, delta: float, seed: int) -> None:
    """Demote (or promote) enough rank-1 hits to move P@1 by `delta`, keeping R@k intact."""
    rng = np.random.default_rng(seed)
    n = q.n_queries
    want = round(abs(delta) * n)
    if delta < 0:
        pool = [i for i, c in enumerate(q.correct_at_1) if c == 1]
    else:
        pool = [i for i, c in enumerate(q.correct_at_1) if c == 0 and q.hit_at_10[i] == 1]
    chosen = rng.choice(pool, size=min(want, len(pool)), replace=False) if pool else []

    for i in map(int, chosen):
        if delta < 0:
            q.correct_at_1[i] = 0
            q.rr_at_10[i] = 0.5  # slipped to rank 2: still in the top 5 and top 10
            if q.auto_approved[i]:
                q.auto_correct[i] = 0
        else:
            q.correct_at_1[i] = 1
            q.rr_at_10[i] = 1.0
            if q.auto_approved[i]:
                q.auto_correct[i] = 1

    n_auto = sum(q.auto_approved)
    q.p_at_1 = sum(q.correct_at_1) / n
    q.r_at_5 = sum(q.hit_at_5) / n
    q.r_at_10 = sum(q.hit_at_10) / n
    q.mrr_at_10 = sum(q.rr_at_10) / n
    q.auto_approve_precision = (sum(q.auto_correct) / n_auto) if n_auto else 0.0
    q.auto_approve_coverage = n_auto / n


# =============================================================================
# CLI
# =============================================================================


def _demo(args: argparse.Namespace) -> None:
    """Prove the ledger works by using it, not by describing it."""
    scales = [(1000, 300), (5000, 300)]
    print("\n" + "=" * 96)
    print("STEP 1-2  Calibrating the noise floor: identical code, measured repeatedly")
    print("=" * 96)
    band, runs = calibrate_noise(
        repeats=args.repeats,
        scales=scales,
        trials=args.trials,
        benchmark=args.benchmark,
        quality_limit=args.limit,
    )
    print("\n" + band.render())

    baseline = runs[0]
    baseline.label = "HEAD baseline"
    record(baseline, noise=band)
    print(f"\nRecorded baseline {baseline.record_id} @ {baseline.provenance['git_sha'][:8]}")

    print("\n" + "=" * 96)
    print("STEP 3  M1: an INDEPENDENT run of the SAME code, compared against the baseline")
    print("=" * 96)
    print(
        "  The band above came from the calibration runs, which include the baseline.\n"
        "  This candidate is a FRESH run that was not part of the calibration, so the\n"
        "  verdict is a prediction the band could have failed, not an identity.\n"
    )
    twin = measure_all(
        label="identical code, re-measured",
        scales=scales,
        trials=args.trials,
        benchmark=args.benchmark,
        quality_limit=args.limit,
        progress=True,
    )
    record(twin, noise=band)
    print(compare(baseline, twin, target="match_fields_per_sec").render())

    print("\n" + "=" * 96)
    print("STEP 4  M2: a candidate that is FASTER but 2 points worse on P@1")
    print("=" * 96)
    fake = simulate_candidate(
        baseline,
        label="faster, 2pts worse P@1",
        speedup=0.30,
        p_at_1_delta=-0.02,
    )
    record(fake, noise=band)
    print(compare(baseline, fake, target="match_fields_per_sec").render())

    print("\n" + "=" * 96)
    print("STEP 5  Leaderboard")
    print("=" * 96)
    print(leaderboard())


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimization ledger for nexus-matcher.")
    ap.add_argument("--demo", action="store_true", help="prove the ledger works, end to end")
    ap.add_argument("--calibrate", action="store_true", help="measure this machine's noise floor")
    ap.add_argument("--record", type=str, metavar="LABEL", help="measure HEAD and append")
    ap.add_argument("--leaderboard", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("BASE_ID", "CAND_ID"))
    ap.add_argument("--target", type=str, default=None, help="the metric you meant to improve")
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--limit", type=int, default=None, help="cap query count (faster, noisier)")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--repeats", type=int, default=3, help="identical runs for calibration")
    args = ap.parse_args()

    if args.demo:
        _demo(args)
        return
    if args.calibrate:
        band, _ = calibrate_noise(
            repeats=args.repeats,
            trials=args.trials,
            benchmark=args.benchmark,
            quality_limit=args.limit,
        )
        print("\n" + band.render())
        return
    if args.record:
        m = measure_all(
            label=args.record,
            trials=args.trials,
            benchmark=args.benchmark,
            quality_limit=args.limit,
            target_metric=args.target,
            progress=True,
        )
        record(m)
        print(f"Recorded {m.record_id} ({m.label})")
        return
    if args.compare:
        base = find_record(args.compare[0])
        cand = find_record(args.compare[1])
        print(compare(base, cand, target=args.target).render())
        return
    if args.leaderboard:
        print(leaderboard())
        return
    ap.print_help()


if __name__ == "__main__":
    main()
