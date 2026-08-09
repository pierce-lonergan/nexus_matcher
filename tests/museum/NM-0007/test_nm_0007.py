"""
NM-0007 -- the documented star-import died on a default install.

    >>> from nexus_matcher import *
    ModuleNotFoundError: No module named 'fastapi'

`__all__` is exactly the list `import *` walks, so one name whose dependency lives in an
extra takes the whole statement down. `create_app` was listed; fastapi ships only in
`[api]`. The user is told about a package they never mentioned, from a line they never
wrote, so it reads as "nexus_matcher is broken" rather than "you skipped an extra".

Why the existing reachability test could not see it: it walks `__all__` in the dev venv,
where every extra is installed, so the one name that needed one resolved perfectly. A test
whose environment guarantees the condition it is checking for is not a test of anything.
This entry therefore runs on a simulated bare install -- a child interpreter with every
extras-only module hidden from the import system.

`__all__` is a promise that a name imports. A name that imports only when an extra happens
to be present does not belong in it; `_OPTIONAL_EXPORTS` plus `__dir__` is where such a
name goes, which is why the last test here pins discoverability as well. Removing the name
from `__all__` and also hiding it would trade one defect for another.

Read the requirement lists from pyproject rather than the installed `.dist-info`: an
editable install serves whatever metadata it was built with, so a dependency promoted out
of an extra still reads as optional there, and blocking a CORE module would fail this test
for a reason that is not the defect.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import packages_distributions
from pathlib import Path

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


def _distribution_name(requirement: str) -> str:
    """`uvicorn[standard]>=0.23.0` -> `uvicorn`, PEP 503 normalised."""
    return re.match(r"[A-Za-z0-9._-]+", requirement).group(0).lower().replace("_", "-")


@lru_cache(maxsize=1)
def _modules_shipped_only_by_extras() -> tuple[str, ...]:
    """Top-level modules a default `pip install nexus-matcher` does NOT get."""
    project = _load_toml(PYPROJECT)["project"]
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


def test_fastapi_is_genuinely_hidden_from_the_probe():
    """
    Guards the guard, and names the specific module this defect was about.

    If `fastapi` ever stopped being classified as extras-only -- because it moved into the
    core dependencies, or because the classification broke -- the star-import test below
    would pass on a machine that simply has fastapi, which is every machine here.
    """
    assert "fastapi" in _modules_shipped_only_by_extras(), (
        "fastapi is not being blocked, so the bare-install simulation below is just this "
        "environment, in which the defect cannot reproduce"
    )


def test_the_documented_star_import_works_without_any_extra():
    """The symptom, verbatim: the line from the README on a plain `pip install`."""
    assert _run_on_a_bare_install("from nexus_matcher import *\nprint('RESULT ok')\n") == "ok"


def test_every_name_in_all_resolves_without_the_extras():
    """
    The general rule behind the symptom, so the next optional export cannot repeat it.

    Star-import fails on the FIRST such name, and reports only that one. Walking every
    entry reports all of them at once, and keeps this useful when a second extras-only
    name is added.
    """
    broken = json.loads(
        _run_on_a_bare_install(
            "import json, nexus_matcher\n"
            "broken = {}\n"
            "for name in nexus_matcher.__all__:\n"
            "    try:\n"
            "        getattr(nexus_matcher, name)\n"
            "    except Exception as exc:\n"
            "        broken[name] = type(exc).__name__ + ': ' + str(exc)\n"
            "print('RESULT ' + json.dumps(broken))\n"
        )
    )
    assert not broken, (
        f"declared in __all__ but needs an extra to import: {broken}. __all__ is the list "
        f"`import *` walks, so each of these takes the whole statement down on a default "
        f"install."
    )


def test_a_missing_extra_is_reported_as_a_missing_extra():
    """
    Asking for the optional name directly must still work, and must say what to install.

    The fix is not "make create_app disappear". It is reachable through `__getattr__`, and
    the error a bare install gets has to name the extra rather than surfacing whatever the
    deepest import happened to raise.
    """
    message = _run_on_a_bare_install(
        "import nexus_matcher\n"
        "try:\n"
        "    nexus_matcher.create_app\n"
        "except ModuleNotFoundError as exc:\n"
        "    print('RESULT ' + str(exc).replace(chr(10), ' '))\n"
        "else:\n"
        "    print('RESULT resolved-without-the-extra')\n"
    )
    assert "create_app" in message, message
    assert "pip install nexus-matcher[api]" in message, message


def test_dropping_the_name_from_all_did_not_hide_it():
    """
    The trade this fix must not have made.

    Deleting a name from `__all__` is a one-character way to make star-import green while
    making the export invisible to `dir()`, `help()` and REPL completion for the users who
    DO have the extra. `__dir__` carries discoverability instead, and this pins it.
    """
    import nexus_matcher

    listing = dir(nexus_matcher)
    assert "create_app" in listing, "create_app vanished from dir() instead of being fixed"
    missing = [name for name in nexus_matcher.__all__ if name not in listing]
    assert not missing, f"in __all__ but missing from dir(): {missing}"
