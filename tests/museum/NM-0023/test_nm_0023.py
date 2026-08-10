"""
NM-0023 -- a gate that exists but is not wired to everything it should cover.

CI ran `ruff check src/nexus_matcher tests`. Two other directories of real Python --
scripts/ and benchmarks/ -- were linted by nobody, and 33 errors accumulated there
unnoticed. The sibling case, NM-0022, was tests/regression existing but never executed by
any CI job.

Both are the same failure: a gate that is real, passes, and simply does not look at part
of the repo. Coverage of a subset reads exactly like coverage of the whole in a green
check mark.

This parses the workflow rather than trusting it, and is deliberately about SCOPE, not
about whether the lint passes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CI = REPO / ".github" / "workflows" / "ci.yml"

# Directories holding first-party Python that a linter should see. benchmarks/ is included
# deliberately: it is where the 33 errors accumulated.
CODE_DIRS = ("src/nexus_matcher", "tests", "scripts", "benchmarks")

pytestmark = pytest.mark.skipif(not CI.exists(), reason="ci.yml not present")


def _lint_commands() -> list[str]:
    text = CI.read_text(encoding="utf-8")
    return re.findall(r"ruff (?:check|format --check)[^\n]*", text)


def test_the_lint_job_exists_at_all():
    assert _lint_commands(), "no ruff invocation found in ci.yml"


@pytest.mark.parametrize("directory", CODE_DIRS)
def test_every_code_directory_is_linted(directory):
    """
    Each first-party Python directory must appear in some ruff invocation.

    Failing this does not mean the code is dirty -- it means nothing would tell you if it
    were, which is worse.
    """
    commands = " ".join(_lint_commands())
    assert directory in commands, (
        f"{directory}/ is not covered by any ruff command in ci.yml. Lint passing there "
        f"is currently unverified, which is how 33 errors accumulated in benchmarks/."
    )


def test_every_test_directory_is_executed_by_ci():
    """
    NM-0022's half: tests/regression existed, passed locally, and no CI job ran it.

    Any directory under tests/ holding test files must be named in some pytest invocation.
    """
    text = CI.read_text(encoding="utf-8")
    pytest_commands = " ".join(re.findall(r"pytest[^\n]*", text))
    test_dirs = sorted(
        p.name
        for p in (REPO / "tests").iterdir()
        if p.is_dir() and not p.name.startswith((".", "__")) and any(p.rglob("test_*.py"))
    )
    unrun = [d for d in test_dirs if f"tests/{d}" not in pytest_commands]
    assert not unrun, (
        f"these test directories are never executed by CI: {', '.join(unrun)}. "
        f"A test suite nothing runs is worth exactly nothing."
    )


def test_every_gate_runner_is_invoked_by_something():
    """
    The sixth occurrence of this defect class, and the reason it earned its own check.

    A gate can be unwired in more ways than a test directory. noxfile.py grew a `mutation`
    session -- the highest-severity hole in docs/DEFENSIBILITY.md, the only mechanical way
    to find tests invariant to the bug they claim to cover -- and `nox` appeared in no
    workflow and no Makefile. It was written, configured, ratcheted, documented, and run by
    nobody.

    Same shape as tests/regression, tests/museum, tests/properties and tests/packaging
    before it: real, passing, and invisible. Coverage of a subset reads exactly like
    coverage of the whole in a green check mark.
    """
    noxfile = REPO / "noxfile.py"
    if not noxfile.exists():
        return

    # Public sessions only. A leading underscore marks an internal helper that nox does
    # not expose and nothing is expected to invoke by name.
    sessions = {
        name
        for name in re.findall(
            r"^def ([a-z_][a-z0-9_]*)\(session", noxfile.read_text(encoding="utf-8"), re.M
        )
        if not name.startswith("_")
    }
    if not sessions:
        return

    invoked = ""
    for path in (REPO / ".github" / "workflows").glob("*.y*ml"):
        invoked += path.read_text(encoding="utf-8")
    makefile = REPO / "Makefile"
    if makefile.exists():
        invoked += makefile.read_text(encoding="utf-8")

    unrun = sorted(s for s in sessions if f"nox -s {s}" not in invoked and f"-s {s}" not in invoked)
    assert not unrun, (
        f"these nox sessions are invoked by no workflow and no Makefile: {', '.join(unrun)}. "
        f"A gate nothing runs is worth exactly nothing -- this is the sixth time in this "
        f"repository."
    )
