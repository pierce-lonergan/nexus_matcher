"""
tests.unit.test_optimization_ledger | Layer: TEST
Guards for the four ways this repo has previously mis-read a benchmark.

The ledger is the thing that decides whether an optimization ships, so a bug in it is
worse than a bug in the optimization: it launders a wrong answer into a recorded one.
Every test here pins one specific wrong verdict. They are model-free and build their
records from vectors directly, so they run in milliseconds on every commit.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from optimization_ledger import (  # noqa: E402
    DEFAULT_GUARDS,
    CostAtScale,
    CostStat,
    Guard,
    Measurement,
    NoiseBand,
    QualityMetrics,
    compare,
    exact_mcnemar_p,
    exact_sign_flip_p,
    find_record,
    leaderboard,
    load_records,
    paired_bootstrap_ci,
    paired_compare_metric,
    record,
    simulate_candidate,
)

SCALE = "entries=1000,fields=300"


# =============================================================================
# FIXTURE BUILDERS
# =============================================================================


def _quality(correct: list[int], *, approved: list[int] | None = None) -> QualityMetrics:
    """A QualityMetrics whose aggregates are consistent with its per-query vectors."""
    n = len(correct)
    approved = approved if approved is not None else [1 if c else 0 for c in correct]
    auto_ok = [1 if (a and c) else 0 for a, c in zip(approved, correct, strict=True)]
    n_auto = sum(approved)
    return QualityMetrics(
        benchmark="fhir",
        corpus="real",
        n_queries=n,
        n_entries=4598,
        p_at_1=sum(correct) / n,
        r_at_5=1.0,
        r_at_10=1.0,
        mrr_at_10=sum(1.0 if c else 0.5 for c in correct) / n,
        auto_approve_precision=(sum(auto_ok) / n_auto) if n_auto else 0.0,
        auto_approve_coverage=n_auto / n,
        query_ids=[f"q{i}" for i in range(n)],
        correct_at_1=list(correct),
        hit_at_5=[1] * n,
        hit_at_10=[1] * n,
        rr_at_10=[1.0 if c else 0.5 for c in correct],
        auto_approved=list(approved),
        auto_correct=auto_ok,
    )


def _cost(fields_per_sec: float, *, peak_mb: float = 60.0) -> list[CostAtScale]:
    def stat(v: float) -> CostStat:
        return CostStat(median=v, best=v, worst=v, iqr=0.0, values=[v, v, v])

    return [
        CostAtScale(
            entries=1000,
            fields=300,
            trials=3,
            stats={
                "match_fields_per_sec": stat(fields_per_sec),
                "index_entries_per_sec": stat(700.0),
                "latency_ms_p50": stat(6.0),
                "latency_ms_p95": stat(8.0),
                "latency_ms_p99": stat(9.0),
                "peak_memory_mb": stat(peak_mb),
                "rss_delta_mb": stat(160.0),
            },
        )
    ]


def _measurement(
    label: str,
    correct: list[int],
    fields_per_sec: float,
    *,
    peak_mb: float = 60.0,
    approved: list[int] | None = None,
    corpus: str = "real",
) -> Measurement:
    m = Measurement(
        record_id=label.replace(" ", "-"),
        label=label,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        benchmark="fhir",
        corpus=corpus,
        provenance={"git_sha": "abc1234", "git_dirty": False, "seed": 1},
        quality=_quality(correct, approved=approved),
        cost=_cost(fields_per_sec, peak_mb=peak_mb),
    )
    m.quality.corpus = corpus
    return m


def _band(throughput_band: float = 0.08, floor: float = 0.03) -> NoiseBand:
    b = NoiseBand(repeats=3, floor=floor, seed=1)
    b.relative[f"{SCALE}::match_fields_per_sec"] = throughput_band
    b.samples[f"{SCALE}::match_fields_per_sec"] = [100.0, 108.0, 104.0]
    for metric in ("p_at_1", "r_at_10", "auto_approve_precision"):
        b.relative[metric] = 0.0
        b.absolute[metric] = 0.0
    return b


# =============================================================================
# M1  --  NOISE IS NOT A WIN
# =============================================================================


class TestNoiseIsNotAWin:
    """
    The recorded failure: index throughput "regressed" 715 -> 520 entries/sec between
    perf_baseline.json and perf_opt2.json, and the code change was blamed. It was machine
    state. Any verdict engine that reads a delta without a calibrated band will make that
    mistake again, so identical numbers inside the band must never come back as a result.
    """

    def test_identical_measurements_are_inconclusive(self):
        rng = np.random.default_rng(7)
        correct = list(rng.integers(0, 2, size=400).astype(int))
        base = _measurement("base", correct, 200.0)
        twin = _measurement("twin", correct, 200.0)

        c = compare(base, twin, target="match_fields_per_sec", noise=_band())

        assert c.verdict == "INCONCLUSIVE", c.render()

    def test_delta_inside_the_band_is_inconclusive_not_a_win(self):
        rng = np.random.default_rng(7)
        correct = list(rng.integers(0, 2, size=400).astype(int))
        base = _measurement("base", correct, 200.0)
        # +6%: a headline-worthy number, and smaller than the measured 8% noise band.
        cand = _measurement("cand", correct, 212.0)

        c = compare(base, cand, target="match_fields_per_sec", noise=_band(0.08))

        assert c.verdict == "INCONCLUSIVE"
        assert "noise band" in " ".join(c.reasons)

    def test_delta_beyond_the_band_is_a_win(self):
        rng = np.random.default_rng(7)
        correct = list(rng.integers(0, 2, size=400).astype(int))
        base = _measurement("base", correct, 200.0)
        cand = _measurement("cand", correct, 260.0)  # +30%, well clear of an 8% band

        c = compare(base, cand, target="match_fields_per_sec", noise=_band(0.08))

        assert c.verdict == "WIN", c.render()

    def test_band_has_a_floor_so_a_lucky_calibration_cannot_certify_noise(self):
        """
        Two calibration runs can agree to 0.4% by luck. Without a floor the band would
        then be ~0 and the next 1% fluctuation would be certified as a win -- exactly the
        mistake the calibration exists to prevent.
        """
        band = NoiseBand(repeats=2, floor=0.03, seed=1)
        band.relative[f"{SCALE}::match_fields_per_sec"] = 0.001

        assert band.band("match_fields_per_sec", SCALE) == pytest.approx(0.03)

        base = _measurement("base", [1] * 200, 200.0)
        cand = _measurement("cand", [1] * 200, 204.0)  # +2%, inside the 3% floor
        assert compare(base, cand, target="match_fields_per_sec", noise=band).verdict == (
            "INCONCLUSIVE"
        )

    def test_a_calm_scale_cannot_certify_a_win_the_noisy_scale_would_reject(self):
        """
        Observed 2026-08-09 while proving this module out: three calibration runs landed
        in a calm stretch at 5000 entries and put that scale's throughput band at 3.5%,
        then an INDEPENDENT run of identical code moved +12.6% there. The same metric at
        1000 entries had measured 30.6%. Taking the per-scale band would have certified
        that fluctuation as a win.
        """
        band = NoiseBand(repeats=3, floor=0.03, seed=1)
        band.relative["entries=1000,fields=300::match_fields_per_sec"] = 0.306
        band.relative["entries=5000,fields=300::match_fields_per_sec"] = 0.035

        assert band.band("match_fields_per_sec", "entries=5000,fields=300") == pytest.approx(0.306)

        base = _measurement("base", [1] * 100, 214.9)
        cand = _measurement("cand", [1] * 100, 241.9)  # +12.6%, the real observation
        for m, v in ((base, 214.9), (cand, 241.9)):
            m.cost = _cost(v)
            m.cost[0].entries = 5000

        c = compare(
            base,
            cand,
            target="match_fields_per_sec",
            target_scope="entries=5000,fields=300",
            noise=band,
        )

        assert c.verdict == "INCONCLUSIVE", c.render()

    def test_missing_calibration_is_inconclusive_and_says_so_loudly(self):
        base = _measurement("base", [1] * 200, 200.0)
        cand = _measurement("cand", [1] * 200, 400.0)  # a 2x speedup

        c = compare(base, cand, target="match_fields_per_sec", noise=None)

        assert c.verdict == "INCONCLUSIVE"
        assert any("NO NOISE CALIBRATION" in w for w in c.warnings)


# =============================================================================
# M2  --  SPEED IS NOT CURRENCY FOR ACCURACY
# =============================================================================


class TestGuardsBeatSpeed:
    """
    A change that is 30% faster and 2 points worse on P@1 is not a trade to be weighed --
    it is a regression. auto-approve precision is the one with a blast radius outside the
    benchmark: an auto-approved wrong match applies the wrong protection level to a real
    column with nobody in the loop.
    """

    def test_faster_but_worse_p_at_1_is_a_regression(self):
        rng = np.random.default_rng(11)
        correct = list(rng.integers(0, 2, size=1000).astype(int))
        base = _measurement("base", correct, 200.0)

        worse = list(correct)
        flipped = [i for i, c in enumerate(worse) if c == 1][:20]  # -2.0 points of P@1
        for i in flipped:
            worse[i] = 0
        cand = _measurement("faster but wrong", worse, 260.0)

        c = compare(base, cand, target="match_fields_per_sec", noise=_band(0.08))

        assert c.verdict == "REGRESSION", c.render()
        assert any("p_at_1" in r for r in c.reasons)

    def test_a_tiny_p_at_1_loss_inside_tolerance_is_not_a_regression(self):
        """The guard is a tolerance, not a ratchet: -0.002 must not block a real speedup."""
        correct = [1] * 500 + [0] * 500
        base = _measurement("base", correct, 200.0)
        worse = list(correct)
        worse[0] = 0  # -0.001, inside the -0.005 tolerance
        cand = _measurement("cand", worse, 300.0)

        assert compare(base, cand, target="match_fields_per_sec", noise=_band()).verdict == "WIN"

    def test_memory_growth_beyond_ten_percent_trips_its_guard(self):
        correct = [1] * 300
        base = _measurement("base", correct, 200.0, peak_mb=60.0)
        cand = _measurement("cand", correct, 400.0, peak_mb=70.0)  # +16.7%

        c = compare(base, cand, target="match_fields_per_sec", noise=_band())

        assert c.verdict == "REGRESSION"
        assert any("peak_memory_mb" in r for r in c.reasons)

    def test_a_guard_breach_smaller_than_the_noise_is_a_warning_not_a_regression(self):
        """
        If this machine's identical-code memory spread is 20%, a +12% observation cannot
        be told apart from noise. Failing on it is mistake M1 wearing M2's clothes;
        passing silently is M2. The honest answer is that the guard is inoperative here.
        """
        correct = [1] * 300
        base = _measurement("base", correct, 200.0, peak_mb=60.0)
        cand = _measurement("cand", correct, 400.0, peak_mb=67.0)  # +11.7%, tolerance +10%

        band = _band()
        band.relative[f"{SCALE}::peak_memory_mb"] = 0.20  # this machine is noisy on memory

        c = compare(base, cand, target="match_fields_per_sec", noise=band)
        mem = next(d for d in c.deltas if d.metric == "peak_memory_mb")

        assert c.verdict != "REGRESSION"
        assert mem.guard_masked_by_noise is True
        assert not mem.guard_tripped
        assert any("inoperative" in w for w in c.warnings), c.render()

    def test_a_guard_breach_beyond_the_noise_still_fails(self):
        """The masking rule must not become a way for a real regression to hide."""
        correct = [1] * 300
        base = _measurement("base", correct, 200.0, peak_mb=60.0)
        cand = _measurement("cand", correct, 400.0, peak_mb=90.0)  # +50%, band is 20%

        band = _band()
        band.relative[f"{SCALE}::peak_memory_mb"] = 0.20

        assert compare(base, cand, target="match_fields_per_sec", noise=band).verdict == (
            "REGRESSION"
        )

    def test_auto_approve_precision_guard_fires_on_its_own(self):
        """
        Precision can fall while P@1 holds: approve MORE fields and get the extra ones
        wrong. P@1 does not see it, and this is the metric that decides whether a PII
        label is applied unreviewed.
        """
        correct = [1] * 100 + [0] * 100
        base = _measurement("base", correct, 200.0, approved=[1] * 100 + [0] * 100)
        # Same ranking, but 40 of the wrong ones are now auto-approved too.
        cand = _measurement(
            "looser threshold", correct, 200.0, approved=[1] * 100 + [1] * 40 + [0] * 60
        )

        c = compare(base, cand, noise=_band())

        assert c.verdict == "REGRESSION"
        assert any("auto_approve_precision" in r for r in c.reasons)

    def test_guard_trips_on_the_point_estimate_not_on_significance(self):
        """
        Deliberate asymmetry: a WIN needs evidence, a guard failure does not. Requiring
        p<0.05 to FAIL would wave through every accuracy loss too small to prove, and on
        this benchmark a real 1-point loss is exactly that size.
        """
        correct = [1] * 500 + [0] * 500
        base = _measurement("base", correct, 200.0)
        worse = list(correct)
        # 12 queries lost, 6 gained: net -0.006, past the -0.005 tolerance, but with
        # discordant pairs in both directions McNemar puts it at p ~ 0.24.
        for i in range(12):
            worse[i] = 0
        for i in range(500, 506):
            worse[i] = 1
        cand = _measurement("cand", worse, 400.0)

        c = compare(base, cand, target="match_fields_per_sec", noise=_band())
        p1 = next(d for d in c.deltas if d.metric == "p_at_1")

        assert p1.p_value is not None and p1.p_value > 0.05, "premise: this loss is not significant"
        assert c.verdict == "REGRESSION"


# =============================================================================
# M3  --  PAIRED DATA NEEDS A PAIRED TEST
# =============================================================================


class TestPairedStatistics:
    """
    Before/after are measured on the SAME queries. Treating them as two independent
    samples throws away the pairing and inflates the interval several-fold, which turns a
    real accuracy loss into "not significant" and lets it ship.
    """

    def test_paired_interval_is_much_narrower_than_the_unpaired_one(self):
        rng = np.random.default_rng(3)
        # Query difficulty dominates: most queries are decided by the query, not the code.
        base_correct = rng.integers(0, 2, size=800).astype(float)
        cand_correct = base_correct.copy()
        cand_correct[:24] = 1 - cand_correct[:24]  # a small, real, systematic change

        _, lo, hi = paired_bootstrap_ci(base_correct, cand_correct, seed=1)
        paired_width = hi - lo

        rng2 = np.random.default_rng(5)
        unpaired = []
        for _ in range(2000):
            i = rng2.integers(0, 800, size=800)
            j = rng2.integers(0, 800, size=800)
            unpaired.append(float(cand_correct[j].mean() - base_correct[i].mean()))
        u_lo, u_hi = np.percentile(unpaired, [2.5, 97.5])

        assert paired_width < (u_hi - u_lo) / 2, (
            f"paired CI {paired_width:.4f} should be far narrower than unpaired "
            f"{u_hi - u_lo:.4f}; if it is not, the pairing is being discarded"
        )

    def test_identical_vectors_give_a_zero_width_interval(self):
        v = [1.0, 0.0, 1.0, 1.0, 0.0] * 40
        r = paired_compare_metric("p_at_1", v, v)

        assert r.delta == 0.0
        assert (r.ci_low, r.ci_high) == (0.0, 0.0)
        assert r.p_value == 1.0
        assert not r.ci_excludes_zero

    def test_mcnemar_uses_only_the_discordant_queries(self):
        """
        Hand-checked exact binomial values. Queries both systems agree on carry no
        information about which is better; counting them is what makes an approximate
        test look confident about nothing.
        """
        assert exact_mcnemar_p(0, 0) == 1.0
        assert exact_mcnemar_p(5, 5) == 1.0
        assert exact_mcnemar_p(12, 3) == pytest.approx(0.03515625)
        assert exact_mcnemar_p(0, 31) == pytest.approx(2 * 0.5**31)

    def test_sign_flip_is_exact_for_small_samples_and_named_when_it_is_not(self):
        p_small, how_small = exact_sign_flip_p([0.5, 0.5, 0.5, 0.5, 0.5])
        assert "exact" in how_small
        assert p_small == pytest.approx(2 / 32)  # only all-+ and all-- reach |sum|=2.5

        p_big, how_big = exact_sign_flip_p([0.1] * 60, seed=1)
        assert "Monte-Carlo" in how_big, "an approximation must never be reported as exact"
        assert 0.0 <= p_big <= 1.0

    def test_quality_verdict_reports_an_interval_not_a_bare_delta(self):
        rng = np.random.default_rng(2)
        correct = list(rng.integers(0, 2, size=600).astype(int))
        better = list(correct)
        for i in [i for i, c in enumerate(better) if c == 0][:60]:
            better[i] = 1

        c = compare(
            _measurement("base", correct, 200.0),
            _measurement("cand", better, 200.0),
            target="p_at_1",
            noise=_band(),
        )
        row = next(d for d in c.deltas if d.metric == "p_at_1")

        assert c.verdict == "WIN"
        assert row.ci_low is not None and row.ci_low > 0
        assert "CI" in " ".join(c.reasons)

    def test_a_quality_move_whose_ci_spans_zero_is_inconclusive(self):
        rng = np.random.default_rng(4)
        correct = list(rng.integers(0, 2, size=600).astype(int))
        noisy = list(correct)
        # Three queries better, two worse: a positive delta with no direction behind it.
        for i in [i for i, c in enumerate(noisy) if c == 0][:3]:
            noisy[i] = 1
        for i in [i for i, c in enumerate(noisy) if c == 1][:2]:
            noisy[i] = 0

        c = compare(
            _measurement("base", correct, 200.0),
            _measurement("cand", noisy, 200.0),
            target="p_at_1",
            noise=_band(),
        )

        assert c.verdict == "INCONCLUSIVE", c.render()


# =============================================================================
# M4  --  PROVENANCE AND THE LEDGER ITSELF
# =============================================================================


class TestLedgerAndProvenance:
    def test_records_append_and_round_trip_without_losing_the_vectors(self, tmp_path: Path):
        """
        The per-query vectors are what make a later comparison paired. A serialisation
        that drops them silently downgrades every future verdict to a scalar diff.
        """
        path = tmp_path / "ledger.jsonl"
        m = _measurement("first", [1, 0, 1, 1, 0] * 20, 200.0)
        record(m, path=path)
        record(_measurement("second", [1] * 100, 250.0), path=path)

        loaded = load_records(path)

        assert [r.label for r in loaded] == ["first", "second"]
        assert loaded[0].quality is not None
        assert loaded[0].quality.correct_at_1 == m.quality.correct_at_1
        assert loaded[0].cost_stat("match_fields_per_sec", SCALE).median == 200.0

    def test_appending_never_rewrites_an_existing_line(self, tmp_path: Path):
        path = tmp_path / "ledger.jsonl"
        record(_measurement("first", [1] * 10, 200.0), path=path)
        before = path.read_text(encoding="utf-8")
        record(_measurement("second", [1] * 10, 300.0), path=path)

        assert path.read_text(encoding="utf-8").startswith(before)

    def test_every_record_carries_what_is_needed_to_re_run_it(self, tmp_path: Path):
        from optimization_ledger import provenance

        p = provenance(seed=123, config={"auto_approve_threshold": 0.87})

        for key in (
            "git_sha",
            "git_dirty",
            "platform",
            "cpu_count",
            "python_version",
            "seed",
            "config",
        ):
            assert key in p, f"provenance is missing {key}: the record is not reproducible"
        assert p["seed"] == 123
        assert p["config"]["auto_approve_threshold"] == 0.87
        del tmp_path

    def test_find_record_accepts_a_short_id(self, tmp_path: Path):
        path = tmp_path / "ledger.jsonl"
        m = _measurement("only", [1] * 10, 200.0)
        record(m, path=path)

        assert find_record(m.record_id[:4], path=path).label == "only"

    def test_leaderboard_flags_simulated_and_dirty_rows(self, tmp_path: Path):
        path = tmp_path / "ledger.jsonl"
        real = _measurement("real run", [1] * 100, 200.0)
        record(real, path=path)
        record(simulate_candidate(real, label="fake", speedup=2.0), path=path)

        out = leaderboard(path, metric="match_fields_per_sec", scope=SCALE)

        assert "SIMULATED" in out, "a fabricated row must never look like a measurement"
        # Ranked on speed, so the fake 3x row sorts first -- and is labelled.
        assert out.index("SIMULATED") < out.index("real run")

    def test_comparing_a_synthetic_corpus_against_a_real_one_is_refused(self):
        """
        The synthetic fallback derives queries from the same tokens as their gold entry.
        Diffing it against the FHIR corpus compares two different tasks and would read as
        a colossal accuracy change.
        """
        real = _measurement("real", [1] * 100, 200.0, corpus="real")
        synth = _measurement("synthetic", [1] * 100, 200.0, corpus="synthetic")

        with pytest.raises(ValueError, match="refusing to compare"):
            compare(real, synth)


# =============================================================================
# VERDICT PLUMBING
# =============================================================================


class TestVerdictRules:
    def test_no_target_and_no_guard_trip_is_neutral_not_a_win(self):
        """A refactor that changes nothing has proved something, but not a win."""
        correct = [1] * 200
        c = compare(
            _measurement("base", correct, 200.0),
            _measurement("refactor", correct, 205.0),
            noise=_band(),
        )

        assert c.verdict == "NEUTRAL"

    def test_the_target_is_judged_on_its_worst_scale(self):
        """
        Optimizations in this repo routinely invert with corpus size -- dictionary
        aliasing is +1.9 P@1 at 688 entries and -18.8 at 30k. A win at the small scale
        must not be allowed to paper over a loss at the large one.
        """
        base = _measurement("base", [1] * 100, 200.0)
        cand = _measurement("cand", [1] * 100, 200.0)
        for m, small, big in ((base, 200.0, 100.0), (cand, 400.0, 70.0)):
            m.cost = _cost(small)
            big_scale = _cost(big)[0]
            big_scale.entries = 30000
            m.cost.append(big_scale)

        band = _band()
        band.relative["entries=30000,fields=300::match_fields_per_sec"] = 0.08

        c = compare(base, cand, target="match_fields_per_sec", noise=band)

        assert c.verdict == "REGRESSION", c.render()
        assert "30000" in " ".join(c.reasons)

    def test_a_target_that_moves_the_wrong_way_is_a_regression(self):
        c = compare(
            _measurement("base", [1] * 100, 200.0),
            _measurement("cand", [1] * 100, 140.0),
            target="match_fields_per_sec",
            noise=_band(0.08),
        )

        assert c.verdict == "REGRESSION"

    def test_simulated_candidates_are_flagged_in_the_comparison(self):
        real = _measurement("real", [1] * 200, 200.0)
        fake = simulate_candidate(real, label="fake", speedup=0.5)

        c = compare(real, fake, target="match_fields_per_sec", noise=_band())

        assert any("SIMULATED" in w for w in c.warnings)

    def test_simulate_candidate_moves_p_at_1_without_touching_recall(self):
        """
        The simulator has to model a REALISTIC loss -- gold slipping from rank 1 to rank 2
        -- or it would trip every guard at once and prove nothing about which one fired.
        """
        rng = np.random.default_rng(9)
        base = _measurement("base", list(rng.integers(0, 2, size=1000).astype(int)), 200.0)
        fake = simulate_candidate(base, label="fake", speedup=0.3, p_at_1_delta=-0.02)

        assert fake.quality.p_at_1 == pytest.approx(base.quality.p_at_1 - 0.02, abs=0.002)
        assert fake.quality.r_at_10 == base.quality.r_at_10
        assert fake.simulated is True

    def test_custom_guards_replace_the_defaults(self):
        strict = (Guard("p_at_1", -0.0001, why="test"),)
        correct = [1] * 999 + [0]
        worse = [1] * 998 + [0, 0]

        assert (
            compare(
                _measurement("base", correct, 200.0),
                _measurement("cand", worse, 400.0),
                target="match_fields_per_sec",
                guards=strict,
                noise=_band(),
            ).verdict
            == "REGRESSION"
        )

    def test_default_guard_tolerances_are_the_documented_ones(self):
        """These four numbers are the contract; a silent loosening is a silent regression."""
        by_metric = {g.metric: g for g in DEFAULT_GUARDS}

        assert by_metric["p_at_1"].tolerance == -0.005
        assert by_metric["auto_approve_precision"].tolerance == -0.010
        assert by_metric["r_at_10"].tolerance == -0.010
        assert by_metric["peak_memory_mb"].tolerance == 0.10
        assert by_metric["peak_memory_mb"].relative is True
        assert all(g.why for g in DEFAULT_GUARDS), "a guard with no stated reason cannot be judged"


def test_record_json_is_one_line_per_record(tmp_path: Path):
    """JSONL, not JSON: an append must never require reading or rewriting what is there."""
    path = tmp_path / "ledger.jsonl"
    record(_measurement("a", [1] * 5, 100.0), path=path)
    record(_measurement("b", [1] * 5, 100.0), path=path)

    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    assert len(lines) == 2
    assert all(json.loads(x)["record_id"] for x in lines)
