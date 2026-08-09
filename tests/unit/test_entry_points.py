"""
tests.unit.test_entry_points | Layer: TEST
Every declared entry point must actually resolve.

A broken entry point is not a cosmetic problem: plugin discovery iterates the group and
calls `load()`, so a single dangling target raises `ModuleNotFoundError` for every
consumer of the package. This repo shipped five of them at once -- pointing at
`schema_parsers.csv_headers`, `dictionary_loaders.csv`, `dictionary_loaders.database`,
`vector_stores.faiss` and `embedding_providers.openai`, none of which exist -- and
nothing in the suite noticed.

Note on staleness: entry-point metadata lives in the installed `.dist-info`, not in
`pyproject.toml`. An editable install does NOT refresh it when pyproject changes, so a
fix can look applied in source while the environment still serves the old table. If this
test fails right after editing pyproject, reinstall before assuming the source is wrong:

    pip install -e . --no-deps
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import distribution, packages_distributions
from pathlib import Path

import pytest

DISTRIBUTION = "nexus-matcher"
PLUGIN_PREFIX = "nexus_matcher."
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _entry_points():
    return list(distribution(DISTRIBUTION).entry_points)


def _plugin_entry_points():
    return [ep for ep in _entry_points() if ep.group.startswith(PLUGIN_PREFIX)]


def test_distribution_is_installed():
    assert _entry_points(), f"{DISTRIBUTION} exposes no entry points at all"


@pytest.mark.parametrize("ep", _entry_points(), ids=lambda ep: f"{ep.group}:{ep.name}")
def test_entry_point_loads(ep):
    """Resolve the target module AND attribute, exactly as plugin discovery does."""
    try:
        loaded = ep.load()
    except Exception as exc:
        # Catching broadly is deliberate: plugin discovery fails on ANY exception the
        # target raises, so the test must reproduce that, not just ImportError.
        pytest.fail(
            f"entry point {ep.group}:{ep.name} -> {ep.value} failed to load: "
            f"{type(exc).__name__}: {exc}"
        )
    assert loaded is not None


def test_every_plugin_group_is_non_empty():
    """A declared group with no members means a plugin category silently disappeared."""
    groups: dict[str, int] = {}
    for ep in _plugin_entry_points():
        groups[ep.group] = groups.get(ep.group, 0) + 1

    assert groups, "no nexus_matcher.* plugin groups are declared"
    empty = [g for g, n in groups.items() if n == 0]
    assert not empty, f"plugin groups with no entries: {empty}"


def test_console_script_is_declared():
    scripts = [ep for ep in _entry_points() if ep.group == "console_scripts"]
    assert any(ep.name == "nexus-matcher" for ep in scripts), (
        f"the nexus-matcher CLI is not declared; found {[e.name for e in scripts]}"
    )


# =============================================================================
# PUBLIC API SURFACE
# =============================================================================


def test_every_declared_export_is_reachable():
    """
    Everything in __all__ must actually resolve.

    __init__ resolves names lazily through PEP 562 __getattr__, so a name can sit in
    __all__ while its branch is missing and nothing fails until a user types it. Several
    modules written for this release (ingest, FlattenedAvroParser, BundledOnnxProvider)
    were unreachable from the top level for exactly that reason -- present in the package,
    invisible to anyone who did not already know the deep module path.

    This checks the DEV environment, which has the extras installed, so it cannot see a
    name that resolves only because an extra happens to be present. That blind spot is
    what let create_app sit in __all__ while breaking star-import on a default install;
    the bare-install tests below are the ones that close it.
    """
    import nexus_matcher

    missing = [name for name in nexus_matcher.__all__ if not hasattr(nexus_matcher, name)]
    assert not missing, f"declared in __all__ but unreachable: {missing}"


def test_importing_the_package_does_not_load_an_inference_runtime():
    """
    `import nexus_matcher` must stay cheap.

    The bundled encoder is the default provider, so it is tempting to import it eagerly.
    Doing so would pull onnxruntime into every process that merely wants a DictionaryEntry
    and roughly quadruple import time. The lazy __getattr__ is what prevents that, and
    this test is what stops someone "simplifying" it away.
    """
    import subprocess
    import sys

    code = (
        "import sys, nexus_matcher;"
        "assert 'onnxruntime' not in sys.modules, 'onnxruntime imported eagerly';"
        "assert 'torch' not in sys.modules, 'torch imported eagerly';"
        "print('lazy')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert "lazy" in result.stdout


def test_the_documented_three_line_api_resolves():
    """The README's headline usage must work from the top-level namespace."""
    import nexus_matcher

    assert callable(nexus_matcher.build_index)
    assert callable(nexus_matcher.sync)
    assert nexus_matcher.GlossaryIndex is not None


# =============================================================================
# THE PUBLIC API SURFACE ON A *BARE* INSTALL
# =============================================================================
#
# Everything above runs in the dev venv, which has every extra installed. That is how
# `create_app` -- importable only with nexus-matcher[api] -- sat in __all__ through a
# release: the reachability test passed here and would have failed on any machine that
# ran plain `pip install nexus-matcher`. The tests below simulate that machine by hiding
# the extras' modules from the import system in a subprocess.


@lru_cache(maxsize=1)
def _modules_shipped_only_by_extras() -> tuple[str, ...]:
    """
    Top-level modules a default `pip install nexus-matcher` does NOT get.

    Read from pyproject, not from the installed .dist-info: an editable install serves
    whatever metadata it was built with, so a dependency promoted from an extra into the
    core list (openpyxl and rank-bm25 both were) still reads as optional there. Blocking
    a core module would make these tests fail for a reason that is not the defect.
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - version-dependent
        tomllib = pytest.importorskip("tomli", reason="no TOML parser on this interpreter")

    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]

    def _name(requirement: str) -> str:
        # "uvicorn[standard]>=0.23.0" -> "uvicorn", then PEP 503 normalisation.
        return re.match(r"[A-Za-z0-9._-]+", requirement).group(0).lower().replace("_", "-")

    core = {_name(r) for r in project["dependencies"]}
    optional = {_name(r) for reqs in project["optional-dependencies"].values() for r in reqs} - core
    optional.discard(DISTRIBUTION)  # the `full` extra just re-references this package

    # dist name != import name (python-multipart ships `multipart`). Anything not
    # installed needs no blocking -- it is already absent.
    return tuple(
        sorted(
            module
            for module, dists in packages_distributions().items()
            for dist in dists
            if dist.lower().replace("_", "-") in optional
        )
    )


def _run_on_a_bare_install(body: str) -> str:
    """Run `body` in a subprocess where every extras-only module raises on import."""
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


def test_star_import_works_without_the_optional_extras():
    """
    `from nexus_matcher import *` must not need an extra.

    __all__ is the list star-import walks, so one name whose dependency lives in an extra
    takes the whole statement down. Shipped that way: create_app was listed, fastapi ships
    only in [api], and a default install answered the documented star-import with
    `ModuleNotFoundError: No module named 'fastapi'` -- a package the user never mentioned,
    raised from a line they never wrote, which reads as a broken package rather than a
    missing extra.
    """
    assert _run_on_a_bare_install("from nexus_matcher import *\nprint('RESULT ok')\n") == "ok"


def test_every_declared_export_is_reachable_without_the_optional_extras():
    """
    Same promise as test_every_declared_export_is_reachable, held to a bare install.

    __all__ is a promise that a name imports. A name that imports only when an extra is
    present is not covered by that promise and belongs in _OPTIONAL_EXPORTS instead, where
    __dir__ still surfaces it. This is the assertion that would have caught create_app.
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
    assert not broken, f"declared in __all__ but needs an extra to import: {broken}"


def test_a_missing_extra_is_reported_as_an_extra_not_as_a_stray_package():
    """
    The lazy loader must name the extra to install, for every optional export.

    Before, asking for an optional export on a bare install produced whatever the deepest
    import happened to raise -- `No module named 'fastapi'`. Nothing in that mentions
    nexus_matcher, the attribute asked for, or the one command that fixes it, so the user
    has to know the package-to-extra mapping to act on it.
    """
    message = _run_on_a_bare_install(
        "import nexus_matcher\n"
        "try:\n"
        "    nexus_matcher.create_app\n"
        "except ModuleNotFoundError as exc:\n"
        "    print('RESULT ' + str(exc))\n"
        "else:\n"
        "    print('RESULT resolved-without-the-extra')\n"
    )
    assert "pip install nexus-matcher[api]" in message, message
    assert "create_app" in message, message


def test_optional_exports_stay_discoverable():
    """
    Dropping a name from __all__ must not hide it.

    The reason not to just delete create_app from __all__ is discoverability for people
    who DO have [api]. __dir__ carries that instead: dir() and REPL tab-completion list
    the optional exports, which plain __all__ membership never did anyway -- a module's
    default dir() reports its __dict__, and a lazily-resolved name is not in there.
    """
    import nexus_matcher

    listing = dir(nexus_matcher)
    assert "create_app" in listing
    for name in nexus_matcher.__all__:
        assert name in listing, f"{name} is in __all__ but missing from dir()"
