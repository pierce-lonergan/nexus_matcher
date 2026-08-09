"""
tests.meta.test_ci_has_teeth | Layer: META-GATE
A gate that cannot fail is not a gate. This is the gate on the gates.

The evidence
------------
2.0.0 shipped to PyPI with the only two commands that do real work crashing on any
non-UTF-8 Windows console, a console script installed without its dependencies, a
documented `-o` flag that was a total no-op, and a field-identity collision that silently
dropped a column's governance classification. CI was green throughout. The step meant to
catch it was:

    - name: Test CLI (if available)
      run: |
        pip install typer rich
        nexus-matcher --help || echo "CLI not available without full install"
      continue-on-error: true

Three independent reasons it could never fail:

  1. `pip install typer rich` installed the very dependencies whose absence WAS the bug
  2. it exercised `--help`, one of the few commands that still worked, rather than
     `match` or `sync`, which were the two that crashed
  3. `|| echo` swallowed the exit code, and `continue-on-error: true` meant even a
     non-zero exit could not turn the build red

Every one of those is mechanically detectable in the workflow YAML. So they are detected
here, and this test is what stops the pattern coming back the next time somebody wants a
red build to go away quickly.

Exceptions
----------
Some steps legitimately should not fail a build -- uploading coverage to a third-party
service, posting a comment. Those go in .ci-exceptions.yaml with a written reason, which
makes the decision visible and reviewable instead of invisible in a YAML file nobody
reads. An exception without a reason is rejected.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
EXCEPTIONS_FILE = REPO / ".ci-exceptions.yaml"

pytestmark = pytest.mark.skipif(not WORKFLOWS.exists(), reason="no .github/workflows")


def _yaml():
    return pytest.importorskip("yaml", reason="PyYAML needed to parse workflows")


def _workflow_files() -> list[Path]:
    return sorted(p for p in WORKFLOWS.iterdir() if p.suffix in (".yml", ".yaml"))


def _exceptions() -> dict[str, str]:
    """
    {"<workflow>::<job>::<step name>": reason}. A blank reason does not count.

    Deliberately keyed on the human-readable step name: an exception should be annoying
    enough to write that nobody adds one casually.
    """
    if not EXCEPTIONS_FILE.exists():
        return {}
    yaml = _yaml()
    raw = yaml.safe_load(EXCEPTIONS_FILE.read_text(encoding="utf-8")) or {}
    return {k: v for k, v in (raw.get("allowed_soft_steps") or {}).items() if str(v).strip()}


def _steps():
    """Yield (workflow, job, step_name, step_dict) for every step in every workflow."""
    yaml = _yaml()
    for path in _workflow_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_name, job in (data.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                if isinstance(step, dict):
                    yield path.name, job_name, step.get("name") or "<unnamed>", step


def _key(workflow: str, job: str, step: str) -> str:
    return f"{workflow}::{job}::{step}"


def test_there_are_workflows_to_check():
    """Guards the vacuous pass: no workflows means every assertion below is trivially true."""
    assert _workflow_files(), "no workflow files found -- this whole file would pass vacuously"


def test_no_step_carries_continue_on_error():
    """
    `continue-on-error: true` converts a gate into a decoration. Exactly this line is why
    a broken CLI reached PyPI.
    """
    allowed = _exceptions()
    offenders = [
        _key(w, j, n)
        for w, j, n, step in _steps()
        if step.get("continue-on-error") is True and _key(w, j, n) not in allowed
    ]
    assert not offenders, (
        "these steps cannot fail the build:\n  "
        + "\n  ".join(offenders)
        + "\nIf one genuinely should not gate, add it to .ci-exceptions.yaml WITH A REASON."
    )


@pytest.mark.parametrize(
    ("pattern", "what"),
    [
        (r"\|\|\s*true", "`|| true` discards the exit code"),
        (r"\|\|\s*echo", "`|| echo` turns a failure into a friendly message and exit 0"),
        (r"^\s*set\s+\+e", "`set +e` stops the shell failing on error"),
        (r"\|\|\s*:", "`|| :` is `|| true` wearing a hat"),
        (r"exit\s+0\s*$", "an unconditional `exit 0` at the end of a step"),
    ],
)
def test_no_step_swallows_its_exit_code(pattern, what):
    allowed = _exceptions()
    rx = re.compile(pattern, re.M)
    offenders = [
        f"{_key(w, j, n)}  ({what})"
        for w, j, n, step in _steps()
        if isinstance(step.get("run"), str)
        and rx.search(step["run"])
        and _key(w, j, n) not in allowed
    ]
    assert not offenders, "these steps cannot report failure:\n  " + "\n  ".join(offenders)


def test_no_step_installs_the_dependency_whose_absence_it_tests():
    """
    The subtlest of the three, and the one that made the other two survive review.

    A step named for testing a bare install that begins by pip-installing the optional
    dependencies is testing the opposite of what its name claims. It cannot be caught by
    looking for `continue-on-error`; it needs the name and the body read together.
    """
    allowed = _exceptions()
    bare_ish = re.compile(r"\b(bare|clean|minimal|no.extras|without|absence|base install)\b", re.I)
    installs = re.compile(r"pip install\s+(?!-r\b)(?!--upgrade pip\b)[a-zA-Z]", re.I)

    offenders = []
    for w, j, n, step in _steps():
        run = step.get("run")
        if not isinstance(run, str):
            continue
        if bare_ish.search(n) and installs.search(run) and _key(w, j, n) not in allowed:
            offenders.append(
                f"{_key(w, j, n)} installs packages while claiming to test their absence"
            )
    assert not offenders, "\n  ".join(offenders)


def test_gate_steps_do_not_run_unconditionally_on_failure():
    """
    `if: always()` on a GATE step means it runs even after the build has failed, which is
    right for uploading logs and wrong for a check whose verdict is supposed to matter.
    """
    allowed = _exceptions()
    gate_ish = re.compile(r"\b(test|lint|gate|check|verify|preflight|audit|museum)\b", re.I)
    offenders = [
        _key(w, j, n)
        for w, j, n, step in _steps()
        if str(step.get("if", "")).strip() in ("always()", "${{ always() }}")
        and gate_ish.search(n)
        and _key(w, j, n) not in allowed
    ]
    assert not offenders, "gate steps running under always():\n  " + "\n  ".join(offenders)


def test_every_exception_carries_a_reason_and_still_exists():
    """
    Exceptions rot two ways: someone adds one with an empty reason, or the step it excused
    is renamed and the exception silently starts excusing nothing while looking legitimate.
    """
    allowed = _exceptions()
    if not allowed:
        return
    real = {_key(w, j, n) for w, j, n, _ in _steps()}
    stale = sorted(set(allowed) - real)
    assert not stale, (
        "these .ci-exceptions.yaml entries no longer match any step (renamed or removed):\n  "
        + "\n  ".join(stale)
    )
