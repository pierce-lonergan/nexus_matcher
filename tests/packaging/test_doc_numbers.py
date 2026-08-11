"""
tests/packaging/test_doc_numbers.py | Env: ALL

The documented numbers are a packaging contract, because the README IS the wheel.

`pyproject.toml` sets `readme = "README.md"`, so every sentence in it ships to PyPI as the
long_description. Two false ones did: "rank-bm25 ... core dependencies now" (it had been
moved out of core), and "Accuracy at 100k entries is unmeasured" (it is measured, by us,
in `exp_scale_combined.json`). The second is the one worth remembering -- the README
UNDERSTATED its own evidence, which no reviewer scanning for exaggeration would flag.

This directory is where the tests live that look at what a user INSTALLS rather than at
what the source says, and a number a user reads on the package page is exactly that.

Why the checker is a script and this is a thin wrapper: the script has to be runnable in
the bare environment `release_preflight.py` builds, where pytest is not installed, and it
has to be runnable by hand on one document while a doc is being fixed. The wrapper makes
it part of the normal suite so it cannot become the seventh "gate that nothing runs".

The tests below are in three groups, and the middle group is the point:

  * the gate itself -- no unrecorded disagreement between a document and its artifact
  * the CONTROLS -- proof the gate can actually go red, that its tolerance is derived
    rather than slack, and that it is looking at real documents and real artifacts.
    A scanner that matched nothing would satisfy the gate forever. That is the 19/19
    transposed-matmul shape, and it is the failure this repository keeps rediscovering.
  * the surface contract -- ASCII output and determinism, per the encoding matrix
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / "scripts" / "check_doc_numbers.py"


def _load():
    """
    Import the checker by path.

    `scripts/` is not a package and is not on `sys.path` for a normal `pytest` run. An
    import that failed here must fail the test rather than skip it -- a skip would delete
    this file exactly the way the three `importorskip`s deleted thirteen others.
    """
    spec = importlib.util.spec_from_file_location("check_doc_numbers", CHECKER)
    assert spec is not None and spec.loader is not None, f"cannot load {CHECKER}"
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE execution: @dataclass resolves annotations through
    # sys.modules[cls.__module__], and a module absent from it raises during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load()


# =============================================================================
# THE GATE
# =============================================================================


def test_no_documented_number_disagrees_with_its_artifact_outside_the_ledger():
    """
    Every numeric claim that names an artifact says what the artifact says -- or is one of
    the mismatches recorded, with both sides pinned, in KNOWN_MISMATCHES.

    Fails in BOTH directions on purpose. A new wrong number fails because it is not in the
    ledger; a document that gets FIXED also fails, because its entry stops reproducing and
    the ledger has to shrink deliberately. A count-based ratchet would have let a fix and a
    fresh regression cancel each other out silently.
    """
    findings, _, _ = checker.run()
    reproduced = {finding.key for finding in findings}
    unrecorded = [f for f in findings if f.key not in checker.KNOWN_MISMATCHES]
    stale = sorted(checker.KNOWN_MISMATCHES - reproduced)

    assert not unrecorded, (
        "a documented number disagrees with the artifact it names, and is not recorded:\n  "
        + "\n  ".join(finding.render() for finding in unrecorded)
        + "\nFix the document. If it is not yours to fix, record it with "
        "`python scripts/check_doc_numbers.py --report`."
    )
    assert not stale, (
        "these ledger entries no longer reproduce -- the document was fixed, or the "
        "artifact was re-run, and KNOWN_MISMATCHES is now claiming a problem that is not "
        "there:\n  " + "\n  ".join(str(entry) for entry in stale)
    )


def test_every_artifact_a_document_names_actually_exists():
    """
    `benchmarks/results/<name>.json` that resolves to no file is a claim with no evidence
    at all, which is worse than a wrong number: nothing can contradict it. BENCHMARK_REGISTRY
    already documents four such citations from earlier revisions.
    """
    findings, _, _ = checker.run()
    broken = [finding for finding in findings if finding.kind == "BROKEN_REF"]
    assert not broken, "\n  ".join(finding.render() for finding in broken)


# =============================================================================
# CONTROLS -- proof the gate can go red
# =============================================================================


def _artifacts():
    known = checker.load_artifacts()
    assert known, f"no artifacts parsed from {checker.RESULTS}"
    return known


def _verdicts(markdown: str, doc: str = "control.md") -> list[tuple[str, object]]:
    known = _artifacts()
    claims, _ = checker.scan(doc, markdown, known)
    return [(checker.verify(claim, known)[0], claim) for claim in claims]


_TRUE = """## Section

End-to-end (`benchmarks/results/eval_pipeline_combined.json`):

| Metric | Value |
|---|---|
| P@1 | 0.581 |
| Recall@10 | 0.878 |
| Throughput | ~364 fields/sec |
"""


def test_the_checker_passes_a_document_that_states_the_artifact_correctly():
    """
    Half the control. A checker that flagged everything would be satisfied only by deleting
    it, and the pressure to go green would land somewhere worse than here.
    """
    verdicts = _verdicts(_TRUE)
    assert len(verdicts) == 3, [claim.metric for _, claim in verdicts]
    assert {verdict for verdict, _ in verdicts} == {"OK"}


@pytest.mark.parametrize(
    ("wrong", "right"),
    [
        ("0.581", "0.681"),  # a digit changed -- the ordinary regression
        ("0.581", "0.582"),  # one unit in the last quoted place
        ("0.878", "0.788"),  # transposed digits
        ("~364", "~500"),  # a throughput claim drifting from its artifact
    ],
)
def test_the_gate_goes_red_when_a_number_stops_matching_its_artifact(wrong, right):
    """
    The other half, and the only reason to believe any of this.

    The 0.581 -> 0.582 case is the one that matters: it proves the tolerance is derived
    from the precision the number is QUOTED to (half of the last decimal place) and is not
    slack big enough to swallow a real change. 0.5813953 rounds to 0.581 and not to 0.582.
    """
    verdicts = _verdicts(_TRUE.replace(wrong, right))
    assert any(verdict == "MISMATCH" for verdict, _ in verdicts), (
        f"changing {wrong} to {right} did not turn the gate red; the tolerance is too "
        "loose to catch a real regression"
    )


def test_the_gate_goes_red_on_the_understatement_that_shipped():
    """
    The escaped defect, replayed in the shape it had.

    "Accuracy at 100k entries is unmeasured" was false because exp_scale_combined.json
    measures it. The doc-side version of that error is a number LOWER than the artifact's;
    a reviewer hunting for exaggeration reads past it, and only a machine comparison sees
    it. This asserts the checker is not one-directional.
    """
    understated = (
        "## Scale\n\n`benchmarks/results/exp_scale_combined.json` records P@1 **0.412** "
        "at 100,000 entries.\n"
    )
    verdicts = _verdicts(understated)
    assert any(verdict == "MISMATCH" for verdict, _ in verdicts), verdicts


def test_a_claim_whose_metric_is_absent_from_the_artifact_is_unverifiable_not_ok():
    """
    The vacuity control one level down. If a missing key silently counted as agreement,
    every claim naming an artifact that does not carry its metric would read as verified --
    the gate would be green over nothing, which is precisely the shape of the publish step
    that carried `continue-on-error` and the nineteen tests that missed a transpose.
    """
    absent = (
        "## Section\n\n`benchmarks/results/exp_alias_scale.json` shows "
        "auto-approve precision of 0.9 on 50.0% coverage.\n"
    )
    verdicts = _verdicts(absent)
    assert verdicts, "nothing was even extracted"
    assert {verdict for verdict, _ in verdicts} == {"UNVERIFIABLE"}, verdicts


def test_the_checker_can_actually_see_a_broken_artifact_reference():
    """
    The control for the test above, and it is not optional.

    There are currently zero broken references, so `assert not broken` passes whether the
    detector works or not -- a green over an empty list, which is the exact shape of the
    five vacuous passes the second-round adjudication found. Deleting the detector must
    turn something red, and this is the something.
    """
    known = _artifacts()
    doc = "## Section\n\nMeasured in `benchmarks/results/exp_fusion_no_such_run.json`.\n"
    _claims, broken = checker.scan("control.md", doc, known)
    assert [name for _doc, _line, name in broken] == ["exp_fusion_no_such_run.json"], broken

    # ...and a real artifact, cited the same way, is not reported as broken.
    real = "## Section\n\nMeasured in `benchmarks/results/exp_fusion_combined.json`.\n"
    assert checker.scan("control.md", real, known)[1] == []


def test_a_number_naming_no_artifact_is_not_silently_treated_as_checked():
    """
    Limitation 2 in the checker's docstring, asserted rather than promised. A section that
    names no artifact yields no claims -- so the "verified" count can never be inflated by
    numbers nothing was compared against.
    """
    assert _verdicts("## Encoders\n\n| Provider | P@1 |\n|---|---|\n| bundled | 0.536 |\n") == []


def test_the_gate_is_looking_at_the_documents_that_ship():
    """
    A glob that stopped matching, a rename, or a scan that quietly covered zero files would
    leave every assertion above passing over nothing.
    """
    scanned = {path.relative_to(REPO).as_posix() for path in checker.documents()}
    assert {"README.md", "CHANGELOG.md", "docs/BENCHMARK_REGISTRY.md"} <= scanned
    assert len(scanned) >= 30, sorted(scanned)

    _findings, _unverifiable, counts = checker.run()
    assert counts["OK"] >= 50, (
        f"only {counts['OK']} claims were actually verified against an artifact; the "
        "extractor has stopped recognising the documents' shape"
    )


def test_the_ledger_records_real_disagreements_and_not_wallpaper():
    """
    A ledger is a ratchet, and a ratchet nobody can read rots. Every entry must name a
    document that exists, an artifact that exists, and a claimed value that differs from
    the artifact value it is pinned against -- an entry that pinned equal values would be
    an exemption wearing a ratchet's clothes.
    """
    known = checker.load_artifacts()
    assert checker.KNOWN_MISMATCHES, "an empty ledger makes every assertion here vacuous"
    for doc, _metric, claimed, actual in sorted(checker.KNOWN_MISMATCHES):
        assert (REPO / doc).exists(), f"{doc} in the ledger does not exist"
        name = actual.split(" ", 1)[0]
        assert name in known, f"{doc} pins {name}, which is not an artifact"
        assert claimed.strip("~%").replace(",", "") != actual.rsplit("=", 1)[-1], (
            f"{doc} records a mismatch between two identical values"
        )


def test_the_ledger_does_not_key_on_line_numbers():
    """
    The ratchet has to survive other people editing these documents.

    Line numbers were in the key for one afternoon. Another lane appended 44 lines to
    CHANGELOG.md and the gate reported eleven fixes and eleven regressions, none of which
    had occurred -- pure churn, and a ratchet that churns is a ratchet somebody disables.
    Asserting the key's shape here so nobody puts the line number back without reading
    Finding.key's docstring first.
    """
    findings, _, _ = checker.run()
    assert findings, "no findings, so the key shape proves nothing"
    assert len(findings[0].key) == 4
    assert not any(isinstance(part, int) for part in findings[0].key), findings[0].key
    assert all(len(entry) == 4 for entry in checker.KNOWN_MISMATCHES)


# =============================================================================
# SURFACE CONTRACT
# =============================================================================


_CODEPAGES = ("cp437", "cp850", "cp1252")


def test_a_finding_quoting_a_non_ascii_document_still_renders():
    """
    The control that makes the subprocess test below mean something.

    Today's findings happen to quote only ASCII, so a checker with its transliteration
    ripped out passes the cp437 run anyway -- green over text that never exercised the
    path. These documents are full of U+2264, U+2192, U+2014 and U+00D7 (`| <= 0.75 |`,
    `0.491 -> 0.691`, `93.6x`), and one row label carrying one of them is all it takes.
    So: a document that really contains them, scanned and verified and RENDERED, with the
    result required to survive every code page in the matrix.
    """
    known = _artifacts()
    doc = (
        "## Section\n\n"
        "Artifact: `benchmarks/results/exp_calibration_combined.json`\n\n"
        "| Metric | Value |\n|---|---|\n"
        "| coverage — at threshold ≤ 0.75, × 1 run | 59.5% |\n"
    )
    claims, _ = checker.scan("control.md", doc, known)
    findings = [checker.verify(claim, known)[1] for claim in claims]
    findings = [finding for finding in findings if finding is not None]
    assert findings, f"no finding produced from {claims}"
    assert any("—" in finding.metric or "≤" in finding.metric for finding in findings), (
        "the control document's non-ASCII never reached a finding, so it proves nothing"
    )
    for finding in findings:
        for codepage in _CODEPAGES:
            finding.render().encode(codepage)  # raises if a glyph survived untranslated


@pytest.mark.parametrize("codepage", _CODEPAGES)
@pytest.mark.parametrize("mode", [[], ["--report"], ["--verbose"]])
def test_the_report_survives_a_legacy_windows_console(codepage, mode):
    """
    Rule 7 of this repository, and NM-0001 itself: a user-facing surface inherits the
    encoding matrix. This one quotes documents full of U+2192, U+2014 and U+00D7, so an
    unguarded print aborts the GATE rather than reporting a finding -- and a codec error
    exiting non-zero looks exactly like a real failure, sending the reader somewhere else
    entirely.

    `--report` is included because it is the widest text surface: it prints every quoted
    claim, where the plain run prints none of them. A pass on the summary alone would
    prove nothing about the path that actually carries document content. Run under the
    real code page in a real subprocess -- asserting on the transliteration table would
    only test the table.
    """
    result = subprocess.run(
        [sys.executable, str(CHECKER), *mode],
        cwd=REPO,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": codepage, "PYTHONUTF8": "0"},
    )
    # The verdict is NOT asserted here: this test must keep meaning something while the
    # gate is red, which is exactly when the most document text gets printed.
    assert b"UnicodeEncodeError" not in result.stderr + result.stdout, (
        codepage,
        result.stderr[-800:],
    )
    assert result.stdout.strip(), "no output to check"
    result.stdout.decode(codepage)  # raises if the gate emitted something it cannot print
    result.stderr.decode(codepage)


def test_the_report_is_deterministic_across_interpreters():
    """
    H-005 applies to output too, and the check has to cross a process boundary to mean
    anything.

    A report whose order comes from set or dict iteration cannot be diffed between runs, so
    every doc edit would produce an unreviewable ledger diff and the ratchet would be
    abandoned. Two calls inside ONE interpreter cannot see that: `hash()` is stable within a
    process, so an order derived from it looks perfectly reproducible -- the same blindness
    that let ranking depend on `PYTHONHASHSEED` until NM-0020. Separate interpreters with
    different seeds, exactly as `tests/hazards/test_h005_total_order.py` does.
    """
    outputs = []
    for seed in ("0", "1", "12345", "99991"):
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--report"],
            cwd=REPO,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert result.stdout.strip(), (seed, result.stderr[-500:])
        outputs.append((seed, result.stdout))
    reference_seed, reference = outputs[0]
    for seed, output in outputs[1:]:
        assert output == reference, (
            f"report order differs between seeds {reference_seed} and {seed}"
        )
    # This exists so a degenerate two-line report cannot pass by having no order to get
    # wrong. It was a literal 100, then had to be lowered to 50 when the ledger legitimately
    # shrank from 141 findings to 72 -- `docs/BENCHMARK_REGISTRY.md` was re-derived against
    # its artifacts on 2026-08-11 and its 69 entries went away.
    #
    # A constant that has been lowered once will be lowered again, and the second time
    # nobody will check whether the report shrank because a document was FIXED or because
    # the checker stopped looking. So it is derived instead: the report must carry the
    # findings the checker actually reported. Fix a document and both sides fall together;
    # break the checker so it finds nothing and this goes red, which a floor of 50 would
    # not have done until the report was almost empty.
    _f, _u, report_counts = checker.run()
    rendered = report_counts["MISMATCH"] + report_counts["BROKEN_REF"]
    assert reference.count("\n") >= rendered, (
        f"the report has {reference.count(chr(10))} lines but the checker found "
        f"{rendered} renderable findings -- lines are being lost between finding and "
        "rendering, so the order being asserted is not the order of the findings"
    )
    assert rendered >= 8, (
        f"the checker rendered only {rendered} findings, so an ordering test over them "
        "proves almost nothing. Either the documents are now genuinely clean -- in which "
        "case delete this test and say so -- or the checker has stopped finding things."
    )
    # UNVERIFIABLE is deliberately NOT counted here: the report does not render it. That is
    # itself a gap -- 18 claims are counted and then never listed, so a claim whose artifact
    # disappears silently leaves the accounting rather than surfacing -- but it is the
    # checker's gap, not this test's, and asserting on it here would just go red for the
    # wrong reason. Recorded rather than papered over.
