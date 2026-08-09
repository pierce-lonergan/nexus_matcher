"""
H-008 -- absolute timings are not gateable on this machine, but complexity SHAPE is.

H-007 established the problem: measured here on identical code, match throughput spread
**0.9%** idle and **30.6%** at 49.5% CPU busy. A fixed fields/sec threshold is therefore
wrong in both directions -- too loose when the box is quiet to catch a real regression,
too tight when it is busy to avoid inventing one. This repo has already recorded a false
regression from exactly that, and then got the direction of the correction wrong too.

The way out is a RATIO BETWEEN TWO SCALES MEASURED IN THE SAME RUN. Whatever the machine
is doing to the 1k measurement it is also doing to the 30k measurement, so machine state
largely divides out and what is left is the shape of the cost curve -- a property of the
algorithm, not of the afternoon.

Re-pinned on this tree, 6 interleaved repeats while other agents were running:

    statistic              single trial        best-of-3
    throughput 30k / 1k    0.608 .. 0.882      0.856 .. 0.904
    p95 latency 30k / 1k   1.417 .. 1.712      1.590 .. 1.659
    index rate 30k / 1k    0.843 .. 0.995      0.871 .. 0.929

Best-of-3 cuts the throughput-ratio spread from 31.4% to 2.2%, which is why the gate
interleaves scales and takes the best trial at each rather than trusting one pass.

WHY THIS GATE EARNS ITS RUNTIME: the O(|q| x N) scan that shipped here once took match
throughput from 550 to 49.7 fields/sec. As an absolute number that is indistinguishable
from a busy afternoon at the moment it lands. As a shape it is a ratio of 0.09 against a
floor of 0.40 -- caught immediately, on any machine, in any state.

WHAT IT CANNOT SEE: a uniform slowdown. Halve the speed at every scale and the ratio does
not move. That is deliberate division of labour, not an oversight -- absolute cost is
judged by `compare()` against a calibrated noise band, shape is judged here, and the test
below pins the blindness so nobody mistakes this gate for the other one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))

ledger = pytest.importorskip("optimization_ledger", reason="benchmarks/ not importable")


# =============================================================================
# FIXTURES -- synthetic cost, so the decision rule can be exercised in milliseconds
# =============================================================================


def _scale(entries: int, *, fields_per_sec: float, p95: float, entries_per_sec: float):
    def stat(v: float, higher: bool):
        return ledger.CostStat(median=v, best=v, worst=v, iqr=0.0, values=[v, v, v])

    return ledger.CostAtScale(
        entries=entries,
        fields=300,
        trials=3,
        stats={
            "match_fields_per_sec": stat(fields_per_sec, True),
            "index_entries_per_sec": stat(entries_per_sec, True),
            "latency_ms_p50": stat(p95 * 0.8, False),
            "latency_ms_p95": stat(p95, False),
            "latency_ms_p99": stat(p95 * 1.2, False),
            "peak_memory_mb": stat(60.0, False),
        },
    )


def _healthy():
    """Today's tree, best-of-3, rounded down to the least favourable end of each range."""
    return [
        _scale(1000, fields_per_sec=423.1, p95=3.605, entries_per_sec=1126.5),
        _scale(30000, fields_per_sec=362.3, p95=5.897, entries_per_sec=981.0),
    ]


def _ratio(report, metric: str) -> float:
    return next(r.ratio for r in report.ratios if r.metric == metric)


# =============================================================================
# 1. THE BOUNDS ARE RATCHETS
# =============================================================================


def test_the_shape_bounds_may_tighten_and_may_never_loosen():
    """
    Pinned at the values chosen on 2026-08-09 with the headroom recorded in SHAPE_BOUNDS.
    Raising a floor or lowering a ceiling is fine and this test moves with it. Going the
    other way is how a gate becomes a decoration, and it fails here.
    """
    bounds = {b.metric: b for b in ledger.SHAPE_BOUNDS}

    assert bounds["match_fields_per_sec"].lower >= 0.40, (
        "the match-throughput shape floor was loosened below 0.40. Observed 0.856-0.904 "
        "best-of-3 on this tree, and the O(|q| x N) scan this catches sat at 0.09."
    )
    assert bounds["latency_ms_p95"].upper <= 4.0, (
        "the p95 latency shape ceiling was loosened above 4.0x. Observed 1.59-1.66."
    )
    assert bounds["index_entries_per_sec"].lower >= 0.40, (
        "the index-rate shape floor was loosened below 0.40. Observed 0.871-0.929."
    )


def test_every_bound_states_the_headroom_it_was_chosen_with():
    """
    A bound whose reasoning is not written down cannot be reviewed, and the next person
    to hit it will assume it was fitted to whatever passed that day.
    """
    for bound in ledger.SHAPE_BOUNDS:
        assert bound.why.strip(), f"{bound.metric} has no recorded reason"
        assert bound.lower is not None or bound.upper is not None, (
            f"{bound.metric} is bounded in neither direction, so it gates nothing"
        )


def test_the_gate_spans_a_thirty_fold_corpus_range():
    """
    Shape is only visible across scales far enough apart to separate the curves. 1k -> 5k
    would leave an O(N) term looking like noise; 1k -> 30k is the range the alias sign
    inversion and the linear-scan regression both showed up in.
    """
    small, large = min(s[0] for s in ledger.SHAPE_SCALES), max(s[0] for s in ledger.SHAPE_SCALES)
    assert large / small >= 20, (
        f"the shape gate spans only {large / small:.0f}x of corpus size. A linear term "
        f"hides comfortably inside a range that narrow."
    )


# =============================================================================
# 2. THE DECISION RULE, AGAINST THE DEFECTS IT EXISTS FOR
# =============================================================================


def test_todays_tree_shape_passes_on_recorded_numbers():
    """
    The control. If this fails, every other assertion here is vacuous -- the gate would be
    rejecting the code it was calibrated on.
    """
    report = ledger.shape_report(_healthy())
    assert report.verdict == "SHAPE-OK", report.render()
    assert _ratio(report, "match_fields_per_sec") == pytest.approx(0.856, abs=0.02)
    assert _ratio(report, "latency_ms_p95") == pytest.approx(1.636, abs=0.02)


def test_the_original_linear_scan_regression_is_caught():
    """
    The defect this gate is built from: the O(|q| x N) scan took match throughput from 550
    fields/sec to 49.7. Its shape ratio is 0.09 against a 0.40 floor, so it fails on any
    machine in any state -- which is the whole argument for gating on shape.
    """
    broken = [
        _scale(1000, fields_per_sec=550.0, p95=3.6, entries_per_sec=1100.0),
        _scale(30000, fields_per_sec=49.7, p95=44.0, entries_per_sec=1000.0),
    ]
    report = ledger.shape_report(broken)
    assert report.verdict == "SHAPE-BROKEN", report.render()
    failed = {r.metric for r in report.failures}
    assert "match_fields_per_sec" in failed, report.render()


def test_a_latency_blowup_alone_is_caught():
    """
    Throughput can stay respectable on the batched path while the per-field tail falls
    apart -- they are measured separately for exactly that reason (see perf_harness). A
    gate on throughput alone would pass this.
    """
    broken = [
        _scale(1000, fields_per_sec=423.0, p95=3.6, entries_per_sec=1100.0),
        _scale(30000, fields_per_sec=360.0, p95=36.0, entries_per_sec=1000.0),
    ]
    report = ledger.shape_report(broken)
    assert report.verdict == "SHAPE-BROKEN"
    assert {r.metric for r in report.failures} == {"latency_ms_p95"}


def test_a_quadratic_index_build_is_caught():
    broken = [
        _scale(1000, fields_per_sec=423.0, p95=3.6, entries_per_sec=1100.0),
        _scale(30000, fields_per_sec=360.0, p95=5.9, entries_per_sec=36.0),
    ]
    report = ledger.shape_report(broken)
    assert report.verdict == "SHAPE-BROKEN"
    assert {r.metric for r in report.failures} == {"index_entries_per_sec"}


def test_a_uniform_slowdown_is_invisible_here_and_that_is_stated():
    """
    The hole, pinned so it cannot be forgotten.

    Everything four times slower at BOTH scales leaves every ratio unchanged and this gate
    green. That is not a bug in the ratio, it is the price of contention immunity, and the
    absolute side of the question belongs to compare() with a calibrated noise band. A
    reader who believes this gate covers absolute cost will stop running the other one.
    """
    slow = [
        _scale(1000, fields_per_sec=423.1 / 4, p95=3.605 * 4, entries_per_sec=1126.5 / 4),
        _scale(30000, fields_per_sec=362.3 / 4, p95=5.897 * 4, entries_per_sec=981.0 / 4),
    ]
    report = ledger.shape_report(slow)
    assert report.verdict == "SHAPE-OK", (
        "the shape gate reacted to a uniform slowdown. That is not what it measures, and "
        "a shape gate that drifts into absolute timing inherits H-007's noise problem."
    )


def test_one_scale_is_refused_rather_than_reported():
    """A single scale can only produce an absolute timing, which this machine cannot gate."""
    with pytest.raises(ValueError, match="at least two scales"):
        ledger.shape_report(_healthy()[:1])


# =============================================================================
# 3. THE MEASUREMENT PROCEDURE ITSELF
# =============================================================================


def test_scales_are_interleaved_so_machine_state_cancels(monkeypatch):
    """
    The design property the whole gate rests on.

    Measuring 1k three times and then 30k three times produces a ratio that describes two
    different minutes. One contended trial here read 0.608 against a true 0.86 for exactly
    that reason. The order must be 1k, 30k, 1k, 30k, 1k, 30k.
    """
    import perf_harness

    seen: list[int] = []

    def fake_measure(entries_n, fields_n, matcher_factory=None, **kw):
        seen.append(entries_n)
        return perf_harness.Measurement(
            entries=entries_n,
            fields=fields_n,
            index_seconds=1.0,
            index_entries_per_sec=1000.0,
            match_seconds=1.0,
            match_fields_per_sec=400.0 if entries_n < 10000 else 350.0,
            latency_ms_p50=3.0,
            latency_ms_p95=4.0 if entries_n < 10000 else 6.0,
            latency_ms_p99=5.0,
            latency_ms_mean=3.5,
            peak_tracemalloc_mb=60.0,
        )

    monkeypatch.setattr(perf_harness, "measure", fake_measure)
    report = ledger.measure_shape(trials=3)

    assert seen == [1000, 30000, 1000, 30000, 1000, 30000], (
        f"scales were not interleaved: {seen}. All trials of one scale followed by all "
        f"trials of the other measures two different machine states and calls the "
        f"difference an algorithm."
    )
    assert report.verdict == "SHAPE-OK"


def test_the_best_trial_is_used_not_the_median(monkeypatch):
    """
    Interference only ever makes a run slower, so the extreme in the GOOD direction is the
    least contaminated estimate of what the machine can do at that scale.

    Set up as sustained load: two of three trials at 30k are hit, one gets through clean.
    Best-of-3 recovers the true 0.857 shape and passes. The median lands on a contended
    trial, reports 0.286, and fails a healthy tree -- a flaky gate, which gets deleted.
    """
    import perf_harness

    large_trials = iter([360.0, 120.0, 120.0])

    def fake_measure(entries_n, fields_n, matcher_factory=None, **kw):
        small = entries_n < 10000
        return perf_harness.Measurement(
            entries=entries_n,
            fields=fields_n,
            index_seconds=1.0,
            index_entries_per_sec=1000.0,
            match_seconds=1.0,
            match_fields_per_sec=420.0 if small else next(large_trials),
            latency_ms_p50=3.0,
            latency_ms_p95=3.6 if small else 5.9,
            latency_ms_p99=5.0,
            latency_ms_mean=3.5,
            peak_tracemalloc_mb=60.0,
        )

    monkeypatch.setattr(perf_harness, "measure", fake_measure)
    report = ledger.measure_shape(trials=3)
    assert _ratio(report, "match_fields_per_sec") == pytest.approx(360.0 / 420.0, abs=1e-6), (
        "the shape ratio was taken from a contended trial. Best-of-N at each scale is what "
        "makes this gate survive a busy machine."
    )
    assert report.verdict == "SHAPE-OK", report.render()


# =============================================================================
# 4. THE REAL TREE
# =============================================================================


def test_the_current_tree_is_in_shape():
    """
    The gate, run for real: 1k and 30k entries, three interleaved trials, ~100 seconds.

    Slow, and unmarked on purpose -- `pytest tests/hazards` is a CI step with no marker
    filter, and a shape gate that only runs when somebody remembers to ask for it is the
    hand-enforced state H-003 spent three occurrences escaping.

    Nothing here asserts an absolute throughput. Other agents are on this machine right
    now and any absolute number would be void; the ratios are what survives that.
    """
    report = ledger.measure_shape()
    assert report.verdict == "SHAPE-OK", report.render()
