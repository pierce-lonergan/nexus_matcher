"""
H-003 -- optimizations "fix" artifacts of their own measurement environment.

A small-corpus fallback was added to `search_batch` because the GEMM path lost at 793
entries. The hypothesis was cache behaviour. It was wrong:

     1 thread:   loop 14.7 ms   GEMM   7.4 ms   -> GEMM 2.00x FASTER
     4 threads:  loop 17.3 ms   GEMM   6.9 ms   -> GEMM 2.51x FASTER
    24 threads:  loop 16.5 ms   GEMM 176.1 ms   -> GEMM 0.09x

Identical FLOPs. The fallback won only where OpenBLAS spread a small GEMM across 24
threads on a saturated box -- which is exactly what 22 concurrent agents create. Shipping
it would have made every uncontended run permanently slower in exchange for looking good
on a machine nobody deploys to.

The rule was written down and applied BY HAND. Nothing enforced it, which is the same
status a hazard has when it is only a paragraph: it works until the person who remembers
it is not in the room. These tests make the sweep executable and make its ABSENCE fatal to
a WIN verdict, so the next threading change cannot be accepted on one measurement.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))

ledger = pytest.importorskip("optimization_ledger", reason="benchmarks/ not importable")

SCALE = "entries=1000,fields=300"


# =============================================================================
# FIXTURES
# =============================================================================


def _quality():
    n = 40
    correct = [1] * 20 + [0] * 20
    return ledger.QualityMetrics(
        benchmark="combined",
        corpus="real",
        n_queries=n,
        n_entries=4598,
        p_at_1=0.5,
        r_at_5=1.0,
        r_at_10=1.0,
        mrr_at_10=0.75,
        auto_approve_precision=0.95,
        auto_approve_coverage=1.0,
        corpus_digest="aaaaaaaaaaaaaaaa",
        query_ids=[f"q{i}" for i in range(n)],
        correct_at_1=correct,
        hit_at_5=[1] * n,
        hit_at_10=[1] * n,
        rr_at_10=[1.0 if c else 0.5 for c in correct],
        auto_approved=[1] * n,
        auto_correct=list(correct),
    )


def _cost(fields_per_sec: float):
    def stat(v: float):
        return ledger.CostStat(median=v, best=v, worst=v, iqr=0.0, values=[v, v, v])

    return [
        ledger.CostAtScale(
            entries=1000,
            fields=300,
            trials=3,
            stats={
                "match_fields_per_sec": stat(fields_per_sec),
                "index_entries_per_sec": stat(700.0),
                "latency_ms_p50": stat(6.0),
                "latency_ms_p95": stat(8.0),
                "latency_ms_p99": stat(9.0),
                "peak_memory_mb": stat(60.0),
                "rss_delta_mb": stat(160.0),
            },
        )
    ]


def _band():
    b = ledger.NoiseBand(repeats=3, floor=0.03, seed=1)
    b.relative[f"{SCALE}::match_fields_per_sec"] = 0.08
    b.samples[f"{SCALE}::match_fields_per_sec"] = [100.0, 108.0, 104.0]
    for metric in ("p_at_1", "r_at_10", "auto_approve_precision"):
        b.relative[metric] = 0.0
        b.absolute[metric] = 0.0
    return b


def _record(label: str, fields_per_sec: float, *, sweep=None):
    return ledger.Measurement(
        record_id=label[:12].replace(" ", "-"),
        label=label,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        benchmark="combined",
        corpus="real",
        provenance={"git_sha": "abc1234", "git_dirty": False, "seed": 1},
        quality=_quality(),
        cost=_cost(fields_per_sec),
        noise=_band(),
        thread_sweep=sweep,
    )


def _verdict(candidate_label: str, *, sweep=None) -> str:
    base = _record("baseline", 200.0)
    cand = _record(candidate_label, 400.0, sweep=sweep)
    return ledger.compare(base, cand, target="match_fields_per_sec").verdict


def _sweep(verdict_shape: str):
    """A sweep in one of the three shapes that matter, built through the real judge."""
    if verdict_shape == "everywhere":
        obs = [
            ledger.ThreadObservation(threads=1, baseline=100.0, candidate=200.0),
            ledger.ThreadObservation(threads=None, baseline=100.0, candidate=195.0),
        ]
    elif verdict_shape == "one-only":
        obs = [
            ledger.ThreadObservation(threads=1, baseline=100.0, candidate=101.0),
            ledger.ThreadObservation(threads=None, baseline=100.0, candidate=200.0),
        ]
    else:
        obs = [ledger.ThreadObservation(threads=None, baseline=100.0, candidate=200.0)]
    return ledger.judge_thread_sweep("match_fields_per_sec", obs, noise_band=0.03)


# =============================================================================
# 1. THE SWEEP IS ACTUALLY A SWEEP -- the env has to reach the child
# =============================================================================


def test_every_blas_backend_knob_is_pinned_not_just_omp():
    """
    numpy wheels link OpenBLAS on Linux and Windows and MKL or Accelerate elsewhere, and
    each backend reads its OWN variable. Pinning OMP_NUM_THREADS alone gives a run
    labelled "1 thread" that dispatched 24 -- the original failure with a label on it.
    """
    pinned = set(ledger.THREAD_ENV_VARS)
    for required in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert required in pinned, (
            f"{required} is not pinned by the sweep. The 24-thread collapse that started "
            f"this hazard was OpenBLAS, which does not read OMP_NUM_THREADS on every build."
        )


def test_thread_env_pins_one_thread_and_unpins_the_default():
    """
    `threads=None` must REMOVE the variables, not leave whatever the parent had. A sweep
    run from a shell that already exports OMP_NUM_THREADS=1 would otherwise measure one
    thread twice and call the agreement a result.
    """
    parent = {"PATH": "x", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    pinned = ledger.thread_env(1, parent)
    assert all(pinned[v] == "1" for v in ledger.THREAD_ENV_VARS)

    freed = ledger.thread_env(None, parent)
    leftover = [v for v in ledger.THREAD_ENV_VARS if v in freed]
    assert not leftover, (
        f"{leftover} survived into the 'library default' arm. Both arms of the sweep would "
        f"be the same condition, which is the failure this check exists to detect."
    )
    assert freed["PATH"] == "x", "unrelated environment was dropped"


def test_the_pinning_reaches_a_separate_interpreter():
    """
    In-process is not good enough and this proves the code does not do it: OpenBLAS reads
    these variables when it is loaded, at `import numpy`. Setting them after that changes
    nothing.
    """
    probe = "import json, os; print(json.dumps({k: os.environ.get(k) for k in os.environ}))"
    seen = ledger.run_under_threads(probe, threads=1)
    for var in ledger.THREAD_ENV_VARS:
        assert seen.get(var) == "1", f"{var} did not reach the child interpreter: {seen.get(var)}"

    freed = ledger.run_under_threads(probe, threads=None)
    for var in ledger.THREAD_ENV_VARS:
        assert var not in freed, f"{var} was still set in the library-default arm"


def test_a_sweep_whose_env_never_arrived_is_an_error_not_a_pass():
    """The silent version of this failure: the child ignored the setting and nobody looked."""
    with pytest.raises(RuntimeError, match="did not reach the child"):
        ledger._assert_threads_took_effect({"thread_env": {"OMP_NUM_THREADS": "24"}}, 1)


# =============================================================================
# 2. THE JUDGE -- the recorded numbers, replayed
# =============================================================================


def test_the_recorded_gemm_collapse_is_flagged_as_thread_dependent():
    """
    The hazard itself, executed, with the numbers from the occurrence.

    Baseline is the GEMM path, candidate is the small-corpus fallback that was proposed.
    Metric is milliseconds, so lower is better. The fallback loses at 1 thread and wins
    only at 24 -- and 24 threads on a saturated box was the machine, not the workload.
    """
    sweep = ledger.judge_thread_sweep(
        "match_ms",
        [
            ledger.ThreadObservation(threads=1, baseline=7.4, candidate=14.7),
            ledger.ThreadObservation(threads=4, baseline=6.9, candidate=17.3),
            ledger.ThreadObservation(threads=24, baseline=176.1, candidate=16.5),
        ],
        noise_band=0.03,
        higher_is_better=False,
    )
    assert sweep.verdict == "THREAD-DEPENDENT", (
        f"the small-corpus fallback came back {sweep.verdict!r}. It won at 24 threads and "
        f"lost at 1 and 4; accepting that is how an optimization for the measurement "
        f"conditions ships."
    )
    assert sweep.thread_dependent and not sweep.demonstrated


def test_a_win_at_one_thread_count_only_is_flagged_in_the_other_direction_too():
    """The mirror case: wins at the library default, flat at 1 thread. Same verdict."""
    sweep = _sweep("one-only")
    assert sweep.verdict == "THREAD-DEPENDENT", sweep.render()
    assert not sweep.demonstrated


def test_a_single_thread_setting_is_not_a_pass():
    """
    Measuring once and declaring victory is the original sin here. INCOMPLETE, not
    WIN-EVERYWHERE: the question was never asked.
    """
    sweep = _sweep("single")
    assert sweep.verdict == "INCOMPLETE"
    assert not sweep.demonstrated


def test_a_genuine_win_survives_the_sweep():
    """
    The opposite failure. A rule that rejects every threading change is as useless as one
    that accepts them, and is the likelier over-correction after an escape.
    """
    sweep = _sweep("everywhere")
    assert sweep.verdict == "WIN-EVERYWHERE" and sweep.demonstrated, sweep.render()


def test_a_change_inside_the_band_everywhere_is_not_a_win():
    sweep = ledger.judge_thread_sweep(
        "match_fields_per_sec",
        [
            ledger.ThreadObservation(threads=1, baseline=100.0, candidate=101.0),
            ledger.ThreadObservation(threads=None, baseline=100.0, candidate=99.5),
        ],
        noise_band=0.03,
    )
    assert sweep.verdict == "NO-WIN" and not sweep.demonstrated


# =============================================================================
# 3. THE RULE IS ENFORCED BY compare(), NOT BY WHOEVER REMEMBERS IT
# =============================================================================


def test_mechanism_detection_covers_the_vocabulary_this_repo_actually_uses():
    """
    The trigger is read from the change's own description, so the words that have
    described real changes here must all fire.
    """
    for text in (
        "small-corpus fallback in search_batch",
        "GEMM orientation fix",
        "switch dense retrieval to BLAS matmul",
        "raise onnxruntime intra-op thread count",
        "parallel index build",
        "re-tune batch scheduling",
    ):
        assert ledger.mechanism_needs_thread_sweep(text), (
            f"{text!r} was not recognised as thread-sensitive, so a win on it would be "
            f"accepted from a single measurement."
        )
    assert not ledger.mechanism_needs_thread_sweep("prune duplicate tokenisation")


def test_a_threading_change_cannot_be_a_win_without_a_sweep():
    """
    The enforcement, executed. Same +100% speedup, same clean guards; the only difference
    is that the change describes a mechanism H-003 covers.
    """
    neutral = _verdict("prune duplicate tokenisation")
    assert neutral == "WIN", f"the control case did not win, so this test proves nothing: {neutral}"

    threading = _verdict("small-corpus fallback in search_batch")
    assert threading == "INCONCLUSIVE", (
        f"a 100% speedup from a batch/BLAS mechanism was judged {threading!r} with no "
        f"thread sweep attached. That is exactly the evidence the small-corpus fallback "
        f"had, and it was wrong."
    )


def test_a_thread_dependent_sweep_does_not_rescue_the_win():
    """Attaching a sweep is not the requirement. PASSING it is."""
    verdict = _verdict("small-corpus fallback in search_batch", sweep=_sweep("one-only"))
    assert verdict == "INCONCLUSIVE", verdict


def test_the_same_change_wins_once_the_sweep_shows_it_everywhere():
    verdict = _verdict("small-corpus fallback in search_batch", sweep=_sweep("everywhere"))
    assert verdict == "WIN", (
        f"a threading change that demonstrated its win at 1 thread AND at the library "
        f"default was still judged {verdict!r}. The rule has become unpassable, which "
        f"means it will be deleted."
    )


def test_a_sweep_survives_the_round_trip_through_the_ledger(tmp_path):
    """
    The sweep is evidence, and evidence that is not written down is a conversation. If it
    does not survive serialisation, the next reader of the ledger sees a bare WIN.
    """
    path = tmp_path / "ledger.jsonl"
    rec = _record("small-corpus fallback in search_batch", 400.0, sweep=_sweep("everywhere"))
    ledger.record(rec, path=path)
    loaded = ledger.load_records(path)[0]
    assert loaded.thread_sweep is not None
    assert loaded.thread_sweep.verdict == "WIN-EVERYWHERE"
    assert [o.threads for o in loaded.thread_sweep.observations] == [1, None]


def test_the_default_sweep_asks_one_thread_and_the_library_default():
    """
    Two settings, and specifically THESE two: 1 thread is the uncontended truth, the
    library default is what a user actually runs. The collapse lived only in the second.
    """
    seen: list[int | None] = []

    def runner(threads):
        seen.append(threads)
        return ledger.ThreadObservation(threads=threads, baseline=100.0, candidate=200.0)

    sweep = ledger.measure_thread_sweep(runner)
    assert seen == [1, None], seen
    assert sweep.demonstrated


def test_the_real_runner_measures_both_arms_under_a_pinned_environment():
    """
    The runner end to end against the real matcher, at a deliberately tiny scale.

    This is not a performance claim -- other agents are on this machine and any absolute
    number here is void. It is a wiring claim: that `perf_thread_runner` starts children,
    that the pinning reaches them, and that both arms come back as finite numbers. Without
    it, the helper could be broken in a way no synthetic test would ever notice.
    """
    runner = ledger.perf_thread_runner(None, None, entries=200, fields=20, trials=1)
    obs = runner(1)
    assert obs.threads == 1
    assert obs.baseline > 0 and obs.candidate > 0


def test_this_process_is_not_already_pinned_to_one_thread():
    """
    A precondition, not a formality.

    If OMP_NUM_THREADS is exported in the ambient environment then the "library default"
    arm of every sweep run from here is secretly the 1-thread arm, both arms agree, and
    the sweep reports WIN-EVERYWHERE having measured one condition twice. That is a green
    check over the exact defect it was built for, so it has to fail loudly instead.
    """
    ambient = {v: os.environ[v] for v in ledger.THREAD_ENV_VARS if v in os.environ}
    assert not ambient, (
        f"BLAS threading is pinned in the ambient environment: {ambient}. Unset these "
        f"before measuring -- the library-default arm of a thread sweep is meaningless "
        f"while they are set."
    )
