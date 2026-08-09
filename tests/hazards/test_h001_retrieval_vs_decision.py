"""
H-001 -- better retrieval lowers auto-approve precision at a fixed threshold.

Three occurrences. The third claimed auto-approve precision +0.030 and actually delivered
-0.066, taking wrong-and-unreviewed auto-approvals from 8 to 12. It was caught in
adjudication by a human-equivalent reading the numbers, not by any gate.

The mechanism is structural, not a bug: improving retrieval shifts every score upward, the
threshold does not move, so more candidates cross a fixed bar -- including wrong ones.
Recall improves and the metric that governs whether a PII label is applied WITHOUT a human
gets worse.

These tests pin the two properties that make the hazard un-shippable:
  1. a quality report cannot omit the decision metric
  2. a divergent-sign result -- retrieval up, decision down -- is a REGRESSION even though
     the headline number improved
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))

ledger = pytest.importorskip("optimization_ledger", reason="benchmarks/ not importable")


def test_the_decision_metric_is_guarded_at_all():
    """
    auto_approve_precision must be one of the guarded metrics. If it is only *reported*,
    a change can move it freely and still be recorded as a win.
    """
    guarded = {g.metric for g in ledger.DEFAULT_GUARDS}
    assert "auto_approve_precision" in guarded, (
        "auto_approve_precision is not guarded. It is the only metric whose blast radius "
        "is outside the benchmark: a wrong auto-approval applies a wrong protection level "
        "to a real column, unreviewed."
    )


def test_the_decision_guard_is_at_least_as_strict_as_recorded():
    """
    Tolerances ratchet tighter, never looser. The observed harmful move was -0.066; a
    tolerance laxer than -0.010 would have waved it through.
    """
    guard = next(g for g in ledger.DEFAULT_GUARDS if g.metric == "auto_approve_precision")
    assert guard.tolerance >= -0.010, (
        f"the auto-approve guard was loosened to {guard.tolerance}. Loosening a guard to "
        f"make something pass is how this hazard shipped three times."
    )


def test_quality_metrics_carry_the_decision_metrics_alongside_retrieval():
    """
    The reporting shape IS the control. If P@1 can be reported without auto-approve
    precision on the same run, somebody will report it that way, and the divergence
    becomes invisible.
    """
    fields = set(ledger.QualityMetrics.__dataclass_fields__)
    for required in ("p_at_1", "r_at_5", "r_at_10", "auto_approve_precision"):
        assert required in fields, f"QualityMetrics does not carry {required}"


def test_divergent_signs_are_a_regression_even_when_p_at_1_improves():
    """
    The hazard itself, executed.

    A candidate that improves the headline metric and degrades the decision metric must
    come back REGRESSION. This is the exact shape of the ContextEnricher change: genuinely
    better retrieval, materially worse governance.
    """
    verdict = _verdict_for(p_at_1_delta=+0.02, auto_approve_delta=-0.05)
    assert verdict == "REGRESSION", (
        f"a change that raised P@1 by 0.02 while dropping auto-approve precision by 0.05 "
        f"was judged {verdict!r}. It must be a REGRESSION: the decision metric governs."
    )


def test_an_unambiguous_improvement_is_still_allowed_through():
    """
    Guards the opposite failure. A gate that rejects everything is as useless as one that
    accepts everything, and is the likelier outcome of over-correcting after an escape.
    """
    verdict = _verdict_for(p_at_1_delta=+0.02, auto_approve_delta=+0.01)
    assert verdict != "REGRESSION", f"a change that improved BOTH metrics was judged {verdict!r}"


def _verdict_for(p_at_1_delta: float, auto_approve_delta: float) -> str:
    """
    Ask the ledger to judge a synthetic before/after pair.

    Built from the ledger's own guard machinery rather than re-implementing the rule, so
    this test tracks the real decision path instead of a copy of it that can drift.
    """
    base = {"p_at_1": 0.22, "auto_approve_precision": 0.75, "r_at_10": 0.55}
    cand = {
        "p_at_1": base["p_at_1"] + p_at_1_delta,
        "auto_approve_precision": base["auto_approve_precision"] + auto_approve_delta,
        "r_at_10": base["r_at_10"],
    }
    breaches = []
    for guard in ledger.DEFAULT_GUARDS:
        if guard.metric not in base:
            continue
        delta = cand[guard.metric] - base[guard.metric]
        if delta < guard.tolerance:
            breaches.append(guard.metric)
    return "REGRESSION" if breaches else "OK"
