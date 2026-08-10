"""
tests/packaging/test_extras_graph.py | Env: ALL

The extras graph, checked in both directions.

An extra is a two-way promise. Forward: "everything the code can import optionally is
installable by name." Backward: "everything this extra installs, the code actually
imports." Both halves shipped broken in 2.0.1:

  - forward: seven of the fifteen runtime extras had no entry in `_EXTRA_FOR_MODULE`, so a
    missing `networkx` or `model2vec` surfaced as a raw ModuleNotFoundError naming a
    third-party package with no visible connection to this project. The user had to already
    know the package-to-extra mapping to act on the error.
  - backward: `parsers` installed fastavro, jsonschema and sqlparse, and `async` installed
    celery, while `grep -rn` over src/ found not one import of any of them. The three
    parsers those packages were supposedly for run on stdlib `json` and `re`. An extra that
    installs packages the code never loads is a lie about what the package needs, and it is
    the kind of lie that survives forever because installing too much never fails.

This file also holds the bare-install gate for `nexus_matcher.presentation.api`, which is
the same defect one level down: that subpackage imported fastapi at module scope, so
`import nexus_matcher.presentation.api` raised ModuleNotFoundError on a default install.

WHAT "REACHABLE" MEANS HERE, precisely, because the weaker reading would make this
worthless: an extra's dependency counts as used when some module under src/nexus_matcher
imports it. That is an import graph, not a call graph -- it cannot prove the importing
branch is ever executed. Proving that needs the call graph docs/DEFENSIBILITY.md tracks as
an open hole (85 unreferenced public methods). This check catches the failure that actually
happened -- a dependency imported by NOTHING -- and does not claim to catch more.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
SRC = REPO_ROOT / "src" / "nexus_matcher"
DISTRIBUTION = "nexus-matcher"

# Extras that install tooling, not runtime code. src/ must never import from these, so the
# reachability half of the graph does not apply to them.
_NON_RUNTIME_EXTRAS = frozenset({"dev", "docs"})

# Extras that exist only to aggregate other extras via `nexus-matcher[...]`.
_AGGREGATE_EXTRAS = frozenset({"full"})

# Distribution name -> the module(s) it makes importable, for the cases where the import
# name is NOT the normalised distribution name. Kept minimal and asserted stale-free by
# test_the_import_name_overrides_are_all_live: an override for a distribution nothing
# declares any more is dead weight that hides the next real mismatch.
_IMPORT_NAME_OVERRIDES = {
    "py-cpuinfo": ("cpuinfo",),
    "opentelemetry-api": ("opentelemetry",),
    "opentelemetry-sdk": ("opentelemetry",),
}

# Extra dependencies that are deliberately imported by nothing under src/, each with the
# reason it stays. This list may shrink, never grow: a new entry is a written decision, not
# a way to make this file green. Adding one without the reason is the failure this whole
# module exists to prevent.
_DELIBERATELY_UNIMPORTED = {
    ("sparse", "rank-bm25"): (
        "BM25Retriever runs on a numpy inverted index and imports nothing. rank-bm25 is the "
        "reference implementation tests/unit/infrastructure/test_bm25_vectorized.py pins the "
        "built-in scores against, and the `sparse` extra name is kept resolving for anyone "
        "who pinned it. Both facts are recorded on the extra in pyproject.toml."
    ),
}


# -----------------------------------------------------------------------------
# THE STDLIB FILTER, MADE VERSION-INDEPENDENT
# -----------------------------------------------------------------------------
# `sys.stdlib_module_names` is a property of the RUNNING interpreter, and pyproject
# classifies 3.10, 3.11, 3.12 and 3.13 as supported. Filtering with it directly makes this
# file give DIFFERENT ANSWERS on different legs of the same CI matrix: `import tomllib` in
# match_schema.py is invisible on 3.13 and a third-party import with no extra on 3.10, so
# test_every_optional_import_in_src_maps_to_an_extra passed here and would have failed on
# the 3.10 leg -- a red build with nothing wrong in the tree, which is how a real gate gets
# loosened by whoever has to make the branch green.
#
# So the filter is the set of modules that are stdlib on EVERY supported interpreter:
# anything whose membership moved inside the support window is subtracted, and has to be
# accounted for explicitly below instead. `_STDLIB` therefore contains no version-dependent
# name at all, which test_the_stdlib_filter_is_the_same_on_every_supported_python asserts.

# Module -> the (major, minor) it became stdlib in. Below that version the import is a
# third-party one and needs something declared to satisfy it.
_STDLIB_SINCE = {"tomllib": (3, 11)}

# ...and the distribution that provides an equivalent on the older interpreters. tomli is a
# core dependency carrying a `python_version < '3.11'` marker; the pair is checked, not
# assumed, by test_a_stdlib_module_missing_on_an_older_python_has_a_declared_backport.
_BACKPORT_FOR = {"tomllib": "tomli"}

# Stdlib on 3.10 and gone by 3.13: PEP 632 (distutils, 3.12), the 3.12 removals, and PEP
# 594's "dead batteries" (3.13). Listed so that an import of one of these is classified the
# same way on every leg -- as NOT stdlib -- rather than as stdlib on the older ones only.
_STDLIB_REMOVED_BY_3_13 = frozenset(
    {
        "binhex",  # 3.11
        "asynchat",  # 3.12
        "asyncore",  # 3.12
        "distutils",  # 3.12, PEP 632
        "imp",  # 3.12
        "smtpd",  # 3.12
        "lib2to3",  # 3.13
        # PEP 594, all removed in 3.13
        "aifc",
        "audioop",
        "cgi",
        "cgitb",
        "chunk",
        "crypt",
        "imghdr",
        "mailcap",
        "msilib",
        "nis",
        "nntplib",
        "ossaudiodev",
        "pipes",
        "sndhdr",
        "spwd",
        "sunau",
        "telnetlib",
        "uu",
        "xdrlib",
    }
)

_VERSION_DEPENDENT_STDLIB = frozenset(_STDLIB_SINCE) | _STDLIB_REMOVED_BY_3_13

# Stdlib on every Python this package supports. Same set on 3.10 and on 3.13.
_STDLIB = frozenset(sys.stdlib_module_names) - _VERSION_DEPENDENT_STDLIB


# =============================================================================
# PYPROJECT / IMPORT-GRAPH HELPERS
# =============================================================================


def _toml_parser():
    """
    The TOML parser, resolved exactly the way the shipped package resolves it.

    Deliberately NOT `pytest.importorskip`. Every test in this file parses pyproject.toml
    through here, so a skip at this line disables the ENTIRE FILE -- the whole extras graph,
    both directions, plus the bare-install gate on the api subpackage -- and CI stays green
    with a healthy-looking count. That is the failure mode this file exists to prevent, one
    level up: an absence that never fails.

    A missing parser is a BROKEN INSTALL, not an unsupported environment.
    `[project.dependencies]` declares `tomli>=2.0.0; python_version < '3.11'` as a core
    dependency for exactly this reason, and
    src/nexus_matcher/application/use_cases/match_schema.py::_load_matching_config falls
    back the same way so `NexusMatcher.from_config("matching.toml")` works on 3.10. So the
    absence is reported as a failure, naming the one command that fixes it.
    """
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:  # pragma: no cover - only on Python < 3.11
        pass
    try:
        import tomli

        return tomli
    except ModuleNotFoundError as exc:  # pragma: no cover - only on a broken 3.10 install
        raise ModuleNotFoundError(
            f"no TOML parser on Python {sys.version_info.major}.{sys.version_info.minor}: "
            "tomllib is stdlib from 3.11, and tomli -- declared as a core dependency for "
            "python_version < '3.11' -- is not installed. Every packaging contract in this "
            "file reads pyproject.toml, so this is a hard failure rather than a skip: "
            "skipping would silently disable the whole file. Fix: pip install tomli",
            name="tomli",
        ) from exc


def _load_pyproject() -> dict:
    return _toml_parser().loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _distribution_name(requirement: str) -> str:
    """'uvicorn[standard]>=0.23.0' -> 'uvicorn', PEP 503 normalised."""
    return re.match(r"[A-Za-z0-9._-]+", requirement).group(0).lower().replace("_", "-")


def _import_names(distribution: str) -> tuple[str, ...]:
    """The module names a distribution makes importable."""
    override = _IMPORT_NAME_OVERRIDES.get(distribution)
    if override is not None:
        return override
    return (distribution.replace("-", "_"),)


def _runtime_extras() -> dict[str, list[str]]:
    """Extra -> the distributions it installs. Tooling and aggregate extras excluded."""
    project = _load_pyproject()
    out: dict[str, list[str]] = {}
    for extra, requirements in project.get("optional-dependencies", {}).items():
        if extra in _NON_RUNTIME_EXTRAS or extra in _AGGREGATE_EXTRAS:
            continue
        out[extra] = [
            name
            for name in map(_distribution_name, requirements)
            if name != DISTRIBUTION  # an extra re-referencing this package adds no module
        ]
    return out


def _core_modules() -> frozenset[str]:
    """Modules a bare `pip install nexus-matcher` already provides."""
    project = _load_pyproject()
    return frozenset(
        module
        for requirement in project["dependencies"]
        for module in _import_names(_distribution_name(requirement))
    )


def _third_party_imports_in_src() -> dict[str, list[str]]:
    """Top-level third-party module -> the 'path:line' sites importing it, from the AST.

    Reads the source rather than importing it: an import inside `if TYPE_CHECKING`, inside a
    function, or behind a try/except ImportError is exactly the optional-dependency idiom
    this package is built on, and none of those are visible to a runtime importer.

    Filtered with `_STDLIB`, never with `sys.stdlib_module_names` -- see the note there.
    """
    sites: dict[str, list[str]] = {}

    def record(module: str, path: Path, lineno: int) -> None:
        root = module.partition(".")[0]
        if root in _STDLIB or root == "nexus_matcher":
            return
        sites.setdefault(root, []).append(f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno}")

    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    record(alias.name, path, node.lineno)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                record(node.module, path, node.lineno)
    return sites


# =============================================================================
# THIS FILE CANNOT BE SILENTLY DISABLED
# =============================================================================


def _probe(preamble: str, body: str) -> str:
    """Run `preamble + body` in a fresh interpreter and return its RESULT line."""
    result = subprocess.run(
        [sys.executable, "-c", preamble + body],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"probe exited {result.returncode}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    marker = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    assert marker, f"probe printed no RESULT line; stdout was:\n{result.stdout}"
    return marker[-1][len("RESULT ") :]


def test_a_missing_toml_parser_fails_this_file_instead_of_skipping_it():
    """
    The gate on the gate: every test here parses pyproject.toml, so whatever `_toml_parser`
    does on a machine without one decides whether this file runs AT ALL.

    It used to `pytest.importorskip("tomli")`, which on the 3.10 leg of the CI matrix turned
    all thirteen packaging contracts into thirteen skips -- a green build in which nothing
    about the extras graph was checked and the count still looked healthy. This runs the
    real resolver with both parsers hidden and asserts the outcome is an ERROR, not a skip.

    pytest's skip is raised, not returned, and `Skipped` derives from BaseException rather
    than Exception -- so the probe catches BaseException and reports the class by name.
    Restoring the importorskip makes this print `Skipped` and turns it red.
    """
    preamble = (
        # pytest is imported BEFORE the blocker: pytest itself reads pyproject.toml with
        # tomllib, so hiding it first would break the harness instead of the subject.
        "import json, sys, importlib.util\n"
        "import pytest\n"
        "HIDDEN = {'tomllib', 'tomli'}\n"
        "\n"
        "class _NoTomlParser:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.partition('.')[0] in HIDDEN:\n"
        "            raise ModuleNotFoundError('No module named ' + repr(fullname), "
        "name=fullname)\n"
        "        return None\n"
        "\n"
        "sys.meta_path.insert(0, _NoTomlParser())\n"
        "for _name in [m for m in sys.modules if m.partition('.')[0] in HIDDEN]:\n"
        "    del sys.modules[_name]\n"
        f"spec = importlib.util.spec_from_file_location('probe_extras_graph', {str(__file__)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
    )
    body = (
        "try:\n"
        "    module._load_pyproject()\n"
        "except BaseException as exc:\n"
        "    print('RESULT ' + json.dumps([type(exc).__name__, str(exc)]))\n"
        "else:\n"
        "    print('RESULT ' + json.dumps(['NoError', 'parsed pyproject.toml anyway']))\n"
    )
    kind, message = json.loads(_probe(preamble, body))

    assert kind not in ("Skipped", "Exit", "NoError"), (
        f"with no TOML parser installed, reading pyproject.toml produced {kind}: {message}. "
        "A skip here disables every test in this file and CI stays green."
    )
    assert kind == "ModuleNotFoundError", (kind, message)
    assert "tomli" in message and "pip install tomli" in message, message


# =============================================================================
# FORWARD: every optional import is installable by name
# =============================================================================


def _backport_distribution(module: str) -> str | None:
    """The core distribution that stands in for `module` where it is not stdlib yet."""
    return _BACKPORT_FOR.get(module)


def _has_a_core_backport(module: str, core: frozenset[str]) -> bool:
    """True when a core dependency provides `module` on the interpreters that lack it."""
    distribution = _backport_distribution(module)
    return bool(distribution) and any(name in core for name in _import_names(distribution))


def test_every_optional_import_in_src_maps_to_an_extra():
    """
    Every third-party module src/ imports is either core or named by `_EXTRA_FOR_MODULE`.

    This is the gate on the hole DEFENSIBILITY recorded as "_EXTRA_FOR_MODULE covers 8 of
    15 extras". A module in neither set is one whose absence gives the user
    `ModuleNotFoundError: No module named 'model2vec'` -- a package name they never asked
    for, from a line they never wrote, with no command that fixes it.

    Modules that are stdlib only on the NEWER supported interpreters (see `_STDLIB_SINCE`)
    reach here as third-party, because that is what they are on 3.10. They are excused only
    by a declared backport -- the same answer on every leg of the matrix.
    """
    from nexus_matcher import _EXTRA_FOR_MODULE

    core = _core_modules()
    unmapped = {
        module: sites
        for module, sites in _third_party_imports_in_src().items()
        if module not in core
        and module not in _EXTRA_FOR_MODULE
        and not _has_a_core_backport(module, core)
    }
    assert not unmapped, (
        "optional imports with no extra to install them -- a missing one of these gives a "
        "raw ModuleNotFoundError naming a stranger package:\n"
        + "\n".join(f"  {m}: {s[:3]}" for m, s in sorted(unmapped.items()))
    )


def test_the_stdlib_filter_is_the_same_on_every_supported_python():
    """
    The classification above must not depend on which interpreter runs the suite.

    `sys.stdlib_module_names` is the running interpreter's. Using it directly makes
    `import tomllib` invisible on 3.11+ and an unmapped third-party import on 3.10, so the
    test above passes on one leg of the CI matrix and fails on another with nothing wrong
    in the tree. The failing leg is then the one somebody "fixes", and the whole file gets
    weakened to shut it up.

    This asserts the property that prevents it: the filter contains NO name whose stdlib
    membership moves anywhere inside the 3.10-3.13 support window. Reverting the filter to
    `sys.stdlib_module_names` turns this red on every interpreter -- `tomllib` on 3.11+,
    the PEP 594 removals on 3.10.
    """
    drifting = sorted(_STDLIB & _VERSION_DEPENDENT_STDLIB)
    assert not drifting, (
        "the stdlib filter contains modules that are stdlib on some supported Pythons and "
        f"not others, so this file answers differently per interpreter: {drifting}"
    )
    # ...and it is still a real stdlib set, not something emptied to satisfy the line above.
    assert {"json", "re", "sys", "pathlib", "importlib"} <= _STDLIB, (
        "the stdlib filter lost ordinary stdlib modules; every one of them would now be "
        "reported as an unmapped third-party import"
    )
    assert _STDLIB_SINCE, "no version-dependent stdlib module is tracked at all"


@pytest.mark.parametrize("module", sorted(_STDLIB_SINCE))
def test_a_stdlib_module_missing_on_an_older_python_has_a_declared_backport(module: str):
    """
    A module src/ imports that is stdlib only from 3.11 needs something declared for 3.10.

    `import tomllib` in match_schema.py is what makes `NexusMatcher.from_config(...)` work
    -- the line printed in both README and QUICKSTART. On 3.10 that import raises, and only
    the `tomli>=2.0.0; python_version < '3.11'` core dependency makes the fallback resolve.
    Dropping either the dependency or its marker takes a documented entry point back to
    ModuleNotFoundError on the oldest supported interpreter, silently, because no 3.11+
    machine can reproduce it.
    """
    if module not in _third_party_imports_in_src():
        pytest.fail(
            f"{module} is tracked in _STDLIB_SINCE but nothing under src/ imports it; "
            "drop the entry rather than carrying an exemption for nothing"
        )

    distribution = _backport_distribution(module)
    assert distribution, (
        f"{module} is not stdlib before {_STDLIB_SINCE[module]} and has no backport"
    )

    since = _STDLIB_SINCE[module]
    declared = [
        requirement
        for requirement in _load_pyproject()["dependencies"]
        if _distribution_name(requirement) == distribution
    ]
    assert declared, (
        f"{distribution} backports {module} for Python < {since[0]}.{since[1]} but is not a "
        f"core dependency, so a {since[0]}.{since[1] - 1} install has neither"
    )
    marker = declared[0].partition(";")[2].replace('"', "'").replace(" ", "")
    assert f"python_version<'{since[0]}.{since[1]}'" in marker, (
        f"{declared[0]!r} does not carry the `python_version < '{since[0]}.{since[1]}'` "
        f"marker, so {distribution} is either installed on interpreters that do not need "
        f"it or absent from the ones that do"
    )


def test_every_optional_extra_dependency_is_mapped_to_an_extra_that_ships_it():
    """
    Both directions of `_EXTRA_FOR_MODULE`, so it cannot drift out of step with pyproject.

    Forward: every distribution an extra installs, and that a bare install does NOT already
    provide, has its import name in the map -- this is what "covers 8 of 15" violated.
    Backward: the extra each module points at genuinely declares that distribution, so the
    install command in the error message actually installs the missing module.

    Distributions already in `[project.dependencies]` are exempt and deliberately absent
    from the map. typer, rich, rapidfuzz and openpyxl are all core AND named by an extra;
    answering their absence with "install nexus-matcher[cli]" would send a user with a
    broken core install chasing an extra they do not need.
    """
    from nexus_matcher import _EXTRA_FOR_MODULE

    core = _core_modules()
    extras = _runtime_extras()

    missing = [
        (extra, dist, module)
        for extra, dists in extras.items()
        for dist in dists
        for module in _import_names(dist)
        if module not in core and module not in _EXTRA_FOR_MODULE
    ]
    assert not missing, "extra dependencies with no entry in _EXTRA_FOR_MODULE:\n" + "\n".join(
        f"  [{e}] {d} -> import {m}" for e, d, m in sorted(missing)
    )

    ships: dict[str, set[str]] = {}
    for extra, dists in extras.items():
        for dist in dists:
            for module in _import_names(dist):
                ships.setdefault(module, set()).add(extra)

    misdirected = {
        module: (named, sorted(ships.get(module, ())))
        for module, named in _EXTRA_FOR_MODULE.items()
        if named not in ships.get(module, set())
    }
    assert not misdirected, (
        "_EXTRA_FOR_MODULE points at an extra that does not install the module -- the "
        "suggested `pip install` would not fix the error:\n"
        + "\n".join(
            f"  {m}: mapped to '{named}', actually shipped by {actual or 'no extra at all'}"
            for m, (named, actual) in sorted(misdirected.items())
        )
    )


def test_every_extra_that_ships_an_optional_distribution_has_a_module_mapped():
    """
    No runtime extra is invisible to the error message that names extras.

    Stated per-extra rather than per-distribution because that is the shape the hole had:
    `graph`, `observability`, `quantization`, `static-embeddings`, `colbert`, `sparse` and
    `accel` were absent from the map entirely, not partially.

    An extra with no requirements, or whose every requirement is already a core dependency,
    has no module that can go missing and is exempt -- and
    test_no_extra_installs_a_distribution_nothing_imports independently proves the empty
    ones really are empty rather than merely unread.
    """
    from nexus_matcher import _EXTRA_FOR_MODULE

    core = _core_modules()
    unrepresented = []
    for extra, dists in sorted(_runtime_extras().items()):
        optional_modules = {m for d in dists for m in _import_names(d) if m not in core}
        if not optional_modules:
            continue
        if not any(_EXTRA_FOR_MODULE.get(m) == extra for m in optional_modules):
            unrepresented.append((extra, sorted(optional_modules)))
    assert not unrepresented, (
        "extras no module maps to -- a missing dependency from these names no extra:\n"
        + "\n".join(f"  [{e}] ships {m}" for e, m in unrepresented)
    )


def test_the_import_name_overrides_are_all_live():
    """
    Every hand-written distribution->module override still corresponds to a declared
    dependency. A stale override is a silent wrong answer waiting: it keeps mapping a
    distribution nobody installs any more, and hides the day a real name stops matching.
    """
    project = _load_pyproject()
    declared = {_distribution_name(r) for r in project["dependencies"]}
    for requirements in project.get("optional-dependencies", {}).values():
        declared.update(map(_distribution_name, requirements))

    stale = sorted(set(_IMPORT_NAME_OVERRIDES) - declared)
    assert not stale, f"overrides for distributions nothing declares any more: {stale}"


# =============================================================================
# BACKWARD: every extra installs something the code imports
# =============================================================================


def test_no_extra_installs_a_distribution_nothing_imports():
    """
    The half nobody checks, because installing too much never fails.

    Verified by AST over src/ on 2026-08-09: `parsers` installed fastavro, jsonschema and
    sqlparse for three parsers that use stdlib `json` and `re`; `async` installed celery
    with no task queue anywhere in the tree; `loaders` installed pandas after ingest.py had
    replaced it with openpyxl; `cache` installed diskcache with no disk cache adapter; and
    `api` installed python-multipart for an API whose every route is a GET. Seven
    distributions, none of them importable from a single line of this package.
    """
    imported = set(_third_party_imports_in_src())
    dead = [
        (extra, dist)
        for extra, dists in sorted(_runtime_extras().items())
        for dist in dists
        if not any(module in imported for module in _import_names(dist))
        and (extra, dist) not in _DELIBERATELY_UNIMPORTED
    ]
    assert not dead, (
        "extras installing distributions nothing under src/ imports. Remove them, or add "
        "the pair to _DELIBERATELY_UNIMPORTED with the reason it stays:\n"
        + "\n".join(f"  [{e}] {d}" for e, d in dead)
    )


def test_the_deliberately_unimported_list_is_all_live():
    """
    An exemption outlives the thing it exempts, and then quietly excuses its replacement.
    Every entry must still name a real extra/distribution pair, and must still be genuinely
    unimported -- once src/ starts importing it, the exemption has to go.
    """
    extras = _runtime_extras()
    imported = set(_third_party_imports_in_src())
    for (extra, dist), reason in _DELIBERATELY_UNIMPORTED.items():
        assert extra in extras, f"exemption names extra '{extra}', which does not exist"
        assert dist in extras[extra], f"extra '{extra}' no longer declares '{dist}'"
        assert reason.strip(), f"exemption for [{extra}] {dist} carries no reason"
        assert not any(m in imported for m in _import_names(dist)), (
            f"[{extra}] {dist} IS imported by src/ now -- drop the exemption"
        )


# =============================================================================
# THE api SUBPACKAGE ON A BARE INSTALL
# =============================================================================


def _api_extra_modules() -> tuple[str, ...]:
    """The modules a bare install does not have, taken from the `api` extra itself."""
    core = _core_modules()
    return tuple(
        sorted(
            {
                module
                for dist in _runtime_extras()["api"]
                for module in _import_names(dist)
                if module not in core
            }
        )
    )


def _run_without_the_api_extra(body: str) -> str:
    """Run `body` in a subprocess where every module the `api` extra ships is absent."""
    blocked = _api_extra_modules()
    assert blocked, "the api extra ships no optional module; this probe would prove nothing"
    preamble = f"""
import sys
BLOCKED = set({list(blocked)!r})


class _HideTheApiExtra:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in BLOCKED:
            raise ModuleNotFoundError("No module named " + repr(fullname), name=fullname)
        return None


sys.meta_path.insert(0, _HideTheApiExtra())
for _mod in [m for m in sys.modules if m.partition(".")[0] in BLOCKED]:
    del sys.modules[_mod]
"""
    return _probe(preamble, body)


def _run_with_a_findable_api_extra(body: str, tmp_path: Path) -> str:
    """
    Run `body` in a subprocess where every module the `api` extra ships is FINDABLE.

    The stubs are empty packages on sys.path, not the real fastapi, and that is the point:
    this half of the contract must be checkable on a bare install, where the real extra is
    absent by definition. `api.__dir__` decides availability with `importlib.util.find_spec`,
    which consults the import system's finders and executes nothing, so an empty package is
    exactly as findable as the real one -- which is what makes the check possible without
    `pytest.importorskip("fastapi")` silently deleting it on every machine that has not
    installed the extra.
    """
    findable = _api_extra_modules()
    assert findable, "the api extra ships no optional module; this probe would prove nothing"
    stubs = tmp_path / "stubs"
    for module in findable:
        package = stubs / module
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(
            f"# stand-in for {module}, findable but never imported by this probe\n",
            encoding="utf-8",
        )
    preamble = f"import sys\nsys.path.insert(0, {str(stubs)!r})\n"
    return _probe(preamble, body)


def test_the_api_subpackage_imports_without_the_api_extra():
    """
    `import nexus_matcher.presentation.api` must not need the `api` extra.

    It did. The subpackage ran `from ...api.app import create_app, run_dev_server` at module
    scope, app.py imports fastapi at module scope, and fastapi ships only in [api] -- so on
    a bare install the import of a subpackage of an installed package raised
    `ModuleNotFoundError: No module named 'fastapi'`. NM-0007 was this defect at the top
    level; this is the same one a directory down, and it survived that fix.
    """
    assert (
        _run_without_the_api_extra("import nexus_matcher.presentation.api\nprint('RESULT ok')\n")
        == "ok"
    )


def test_the_api_subpackage_declares_nothing_that_needs_an_extra():
    """
    `__all__` is the list `import *` walks, so one name whose dependency lives in an extra
    takes the whole statement down -- the same reason create_app is not in the top-level
    `__all__`. Checked on a bare install rather than by reading the list, because the
    promise is about what imports, not about what is written down.

    Three assertions, because the obvious one is vacuous on its own. `api.__all__` is `[]`
    today -- deliberately, see the note on it in the subpackage -- so "every name in
    `__all__` resolves" quantifies over nothing and passes whatever the code does. The
    emptiness is therefore asserted as the decision it is, the lazy exports are asserted
    ABSENT from it by name, and the resolution loop is run once over a control list that
    does contain them, proving the loop reports a breakage rather than merely finding none.
    """
    report = json.loads(
        _run_without_the_api_extra(
            "import json\n"
            "import nexus_matcher.presentation.api as api\n"
            "\n"
            "def resolve(names):\n"
            "    broken = {}\n"
            "    for name in names:\n"
            "        try:\n"
            "            getattr(api, name)\n"
            "        except Exception as exc:\n"
            "            broken[name] = type(exc).__name__ + ': ' + str(exc)\n"
            "    return broken\n"
            "\n"
            "lazy = sorted(api._OPTIONAL_EXPORT_REQUIRES)\n"
            "print('RESULT ' + json.dumps({\n"
            "    'all': list(api.__all__),\n"
            "    'lazy': lazy,\n"
            "    'broken': resolve(api.__all__),\n"
            "    'control': resolve(list(api.__all__) + lazy),\n"
            "}))\n"
        )
    )

    assert not report["broken"], (
        f"declared in __all__ but needs an extra to import: {report['broken']}"
    )
    assert not set(report["all"]) & set(report["lazy"]), (
        "a lazily-resolved export is listed in __all__, so `from "
        "nexus_matcher.presentation.api import *` raises ModuleNotFoundError on a bare "
        f"install: {sorted(set(report['all']) & set(report['lazy']))}"
    )
    assert report["all"] == [], (
        "api.__all__ is no longer empty. That emptiness is a written decision -- __all__ is "
        "a promise that a name imports, and neither factory does without the `api` extra, "
        f"so the list is empty rather than conditional. Found {report['all']}. If a name "
        "that resolves on a bare install was added on purpose, update the note in "
        "src/nexus_matcher/presentation/api/__init__.py and this assertion together."
    )
    # The control: the same loop, over a list that DOES contain names needing the extra.
    # Without it the check above passes over an empty sequence and could never fail.
    assert sorted(report["control"]) == report["lazy"], (
        "the resolution loop did not report the lazy exports as broken on a bare install, "
        "so it cannot report anything as broken: "
        f"{report['control']} for {report['lazy']}"
    )


@pytest.mark.parametrize("name", ["create_app", "run_dev_server"])
def test_the_api_subpackage_names_the_extra_it_needs(name: str):
    """
    Asking for either factory without the extra must name the extra and the one command
    that fixes it, not just whatever the deepest import happened to raise.
    """
    message = _run_without_the_api_extra(
        "import nexus_matcher.presentation.api as api\n"
        "try:\n"
        f"    api.{name}\n"
        "except ModuleNotFoundError as exc:\n"
        "    print('RESULT ' + str(exc).replace(chr(10), ' '))\n"
        "else:\n"
        "    print('RESULT resolved-without-the-extra')\n"
    )
    assert "pip install nexus-matcher[api]" in message, message
    assert name in message, message


def test_the_api_subpackage_stays_introspectable_without_the_extra():
    """
    Nothing `__dir__` advertises may raise when fetched.

    dir() is not a display list: inspect.getmembers(), help(), pydoc and rlcompleter
    tab-completion all walk it and getattr() every entry, so one advertised name that
    raises takes all four down. The obvious fix -- an error type that is both
    ModuleNotFoundError and AttributeError, so getattr-based introspection skips it -- is
    not available: the two have incompatible C layouts and cannot be subclassed together
    (`TypeError: multiple bases have instance lay-out conflict`, verified 2026-08-09).
    So availability is decided in __dir__ instead, with find_spec, which consults the
    import system's finders without executing anything.
    """
    report = json.loads(
        _run_without_the_api_extra(
            "import inspect, io, json, pydoc\n"
            "import nexus_matcher.presentation.api as api\n"
            "out = {'raised': {}, 'advertised': sorted(n for n in dir(api))}\n"
            "for name in dir(api):\n"
            "    try:\n"
            "        getattr(api, name)\n"
            "    except Exception as exc:\n"
            "        out['raised'][name] = type(exc).__name__ + ': ' + str(exc)\n"
            "for label, call in (\n"
            "    ('getmembers', lambda: inspect.getmembers(api)),\n"
            "    ('pydoc', lambda: pydoc.render_doc(api)),\n"
            "):\n"
            "    try:\n"
            "        call()\n"
            "    except Exception as exc:\n"
            "        out['raised'][label] = type(exc).__name__ + ': ' + str(exc)\n"
            # help() is checked on its OUTPUT, not by catching. pydoc.doc() wraps its work
            # in `except (ImportError, ErrorDuringImport)` and writes str(value) into the
            # caller's buffer in place of the documentation, so a broken __dir__ makes
            # help() quietly render the import error INSTEAD of the module -- and raise
            # nothing at all. Probing it with try/except looks like coverage and can never
            # fire; verified by mutating __dir__ and watching only this form catch it.
            #
            # The signal is the absence of pydoc's structural 'NAME' section, not the
            # presence of error text. Searching for the error text matches this package's
            # own prose: the api __init__ docstring quotes
            # "ModuleNotFoundError: No module named 'fastapi'" while explaining the bug it
            # fixed, so that spelling reported correct code as broken -- it did exactly
            # that here before being tightened to this.
            "buf = io.StringIO()\n"
            "try:\n"
            "    pydoc.doc(api, output=buf)\n"
            "except Exception as exc:\n"
            "    out['raised']['help'] = type(exc).__name__ + ': ' + str(exc)\n"
            "else:\n"
            "    if 'NAME' not in buf.getvalue():\n"
            "        out['raised']['help'] = (\n"
            "            'rendered no NAME section; pydoc bailed out and wrote the import "
            "error to stdout instead. Buffer was: '\n"
            "            + repr(buf.getvalue()[:200])\n"
            "        )\n"
            "print('RESULT ' + json.dumps(out))\n"
        )
    )
    assert not report["raised"], (
        "introspection broke on a bare install -- dir() advertises a name that raises: "
        f"{report['raised']}"
    )
    for name in ("create_app", "run_dev_server"):
        assert name not in report["advertised"], (
            f"{name} is advertised without the extra that makes it resolvable"
        )


def test_the_api_subpackage_advertises_its_exports_when_the_extra_is_findable(tmp_path: Path):
    """
    The other side of the availability check, and it must run WITHOUT the extra installed.

    Hiding the names unconditionally is the cheap way to pass
    test_the_api_subpackage_stays_introspectable_without_the_extra while making the package
    undiscoverable: `dir()`, tab-completion and pydoc would never mention create_app even
    with fastapi installed. This is the test that catches that, and it was guarded by
    `pytest.importorskip("fastapi")` -- so on exactly the bare install whose bug it was
    written for, it did not run.

    It needs no real fastapi: `__dir__` decides with `find_spec`, so an empty stub package
    on sys.path exercises the same branch. Pair it with the blocked-import test above and
    both directions are covered on every machine.
    """
    advertised = json.loads(
        _run_with_a_findable_api_extra(
            "import json\n"
            "import nexus_matcher.presentation.api as api\n"
            "print('RESULT ' + json.dumps(sorted(dir(api))))\n",
            tmp_path,
        )
    )
    for name in ("create_app", "run_dev_server"):
        assert name in advertised, (
            f"{name} is findable in this probe but dir() does not advertise it, so nothing "
            "-- help(), pydoc, tab-completion -- can discover the API factories even with "
            f"the extra installed. dir() gave {advertised}"
        )


def test_the_api_factories_resolve_where_the_api_extra_is_really_installed():
    """
    The stub above proves what is ADVERTISED; only the real package proves it RESOLVES.

    A hard failure, not `pytest.importorskip("fastapi")`. Every documented environment has
    the extra -- CI installs `.[full,dev]` on all four matrix legs, requirements.txt (which
    start.sh builds its venv from) composes in `api`, and tests/unit/presentation/test_api.py
    already imports fastapi at module scope -- so an env without it is one that cannot check
    the api surface at all, and saying so out loud is the whole point. A skip here deletes
    the test for the exact defect it guards: `import nexus_matcher.presentation.api` raising
    ModuleNotFoundError, fixed by resolving both factories lazily.
    """
    if importlib.util.find_spec("fastapi") is None:
        pytest.fail(
            "fastapi is not installed, so the api extra's resolution path cannot be "
            "checked here. This is deliberately a failure and not a skip: the packaging "
            "contracts for the api subpackage would otherwise vanish silently on the one "
            "kind of install they were written for. Fix: pip install -e '.[api]' (or "
            "'.[full,dev]', which is what CI installs)."
        )

    from nexus_matcher.presentation import api

    advertised = dir(api)
    for name in ("create_app", "run_dev_server"):
        assert name in advertised, f"{name} is installable here but not advertised"
        assert callable(getattr(api, name)), f"{name} did not resolve to a callable"


def test_the_api_subpackage_still_rejects_names_it_does_not_have():
    """
    A lazy `__getattr__` that answers everything hides typos and breaks `hasattr`. An
    unknown name must still be a plain AttributeError, with or without the extra.
    """
    from nexus_matcher.presentation import api

    with pytest.raises(AttributeError):
        _ = api.create_app_typo
