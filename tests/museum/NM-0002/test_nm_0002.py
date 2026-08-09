"""
NM-0002 -- pip put a command on PATH that could not start.

`[project.scripts]` declares `nexus-matcher` unconditionally, while `typer` and `rich`
shipped only in the `cli` extra. So `pip install nexus-matcher` created the console
script, the README told the user to run `nexus-matcher match customer.avsc` eight lines
later, and that command died on `import typer` at module scope -- a raw
`ModuleNotFoundError` traceback out of a command pip had just installed.

It never appeared in development because typer is in the `dev` install and in CI, so every
environment that ran the CLI already had it. Only a first-time user following the README
ever saw it. `tests/unit/test_entry_points.py` proved the script was DECLARED and that its
target resolved -- in an environment where it always would.

The observable symptom is therefore only visible on an install that does not have the
extras, so that is what this reproduces: a child interpreter with every extras-only module
hidden from the import system, in which each declared console script must import and its
attribute must resolve -- which is precisely what the generated console-script shim does.

Reading the requirement lists from pyproject rather than from the installed `.dist-info`
is deliberate: an editable install serves whatever metadata it was built with, so a
dependency promoted out of an extra still reads as optional there, and this test would
block a CORE module and fail for a reason that is not the defect.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PYPROJECT = REPO / "pyproject.toml"
DISTRIBUTION = "nexus-matcher"


def _load_toml(path: Path) -> dict:
    """tomllib is stdlib from 3.11; below that `tomli` is a declared core dependency."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - version-dependent
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _project() -> dict:
    return _load_toml(PYPROJECT)["project"]


def _distribution_name(requirement: str) -> str:
    """`uvicorn[standard]>=0.23.0` -> `uvicorn`, PEP 503 normalised."""
    return re.match(r"[A-Za-z0-9._-]+", requirement).group(0).lower().replace("_", "-")


@lru_cache(maxsize=1)
def _modules_shipped_only_by_extras() -> tuple[str, ...]:
    """Top-level modules a default `pip install nexus-matcher` does NOT get."""
    project = _project()
    core = {_distribution_name(r) for r in project["dependencies"]}
    optional = {
        _distribution_name(r) for reqs in project["optional-dependencies"].values() for r in reqs
    } - core
    optional.discard(DISTRIBUTION)  # the `full` extra just re-references this package

    # Distribution name != import name (python-multipart ships `multipart`). Anything not
    # installed here needs no blocking -- it is already absent.
    return tuple(
        sorted(
            module
            for module, dists in packages_distributions().items()
            for dist in dists
            if dist.lower().replace("_", "-") in optional
        )
    )


def _console_scripts() -> dict[str, str]:
    return dict(_project().get("scripts", {}))


def _run_on_a_bare_install(body: str) -> str:
    """Run `body` in a child where every extras-only module raises on import."""
    blocked = _modules_shipped_only_by_extras()
    preamble = f"""
import sys
BLOCKED = set({list(blocked)!r})


class _HideTheExtras:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError("No module named " + repr(fullname), name=fullname)
        return None


sys.meta_path.insert(0, _HideTheExtras())
for _mod in [m for m in sys.modules if m.partition(".")[0] in BLOCKED]:
    del sys.modules[_mod]
"""
    result = subprocess.run(
        [sys.executable, "-c", preamble + body],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        f"probe exited {result.returncode} on a simulated bare install\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr[-2000:]}"
    )
    marker = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    assert marker, f"probe printed no RESULT line; stdout was:\n{result.stdout}"
    return marker[-1][len("RESULT ") :]


def test_a_console_script_is_declared_at_all():
    """
    Without this the parametrized test below would pass by having nothing to run, which is
    the same shape of empty green as a lint job that lints no directories (NM-0023).
    """
    assert _console_scripts(), "pyproject declares no [project.scripts] entries"


def test_the_simulation_actually_hides_something():
    """
    Guards the guard. If nothing is blocked, every bare-install assertion here is a plain
    dev-environment run and proves nothing at all.
    """
    assert _modules_shipped_only_by_extras(), (
        "no extras-only module was found to block, so the 'bare install' below is just "
        "this environment"
    )


@pytest.mark.parametrize("script", sorted(_console_scripts()))
def test_every_console_script_starts_without_the_optional_extras(script):
    """
    The symptom, stated as the user meets it: the command pip installed must run.

    A console script's generated shim imports the module and calls the attribute, so
    resolving both under a bare install is exactly what happens when the user types the
    command. Anything a command needs at import time is a CORE dependency of this package
    by definition -- declaring the script unconditionally is the promise that it runs.
    """
    target = _console_scripts()[script]
    broken = json.loads(
        _run_on_a_bare_install(
            "import importlib, json\n"
            f"module_name, _, attribute = {target!r}.partition(':')\n"
            "broken = {}\n"
            "try:\n"
            "    module = importlib.import_module(module_name)\n"
            "    if attribute:\n"
            "        getattr(module, attribute)\n"
            "except Exception as exc:\n"
            "    broken[module_name] = type(exc).__name__ + ': ' + str(exc)\n"
            "print('RESULT ' + json.dumps(broken))\n"
        )
    )
    assert not broken, (
        f"the `{script}` console script is installed by a bare `pip install "
        f"{DISTRIBUTION}` and cannot start there: {broken}. Whatever it imports belongs "
        f"in [project] dependencies, not in an extra."
    )
