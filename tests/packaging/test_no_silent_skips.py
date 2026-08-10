"""
tests/packaging/test_no_silent_skips.py | Env: ALL

No test in this directory may be deleted by its environment.

The packaging contracts are the only tests that look at what a user INSTALLS rather than at
what the source says. The second-round adjudication found five ways they were switching
themselves off:

  - `pytest.importorskip("tomli")` sat inside the helper every one of the thirteen tests in
    test_extras_graph.py calls, so the 3.10 leg of the CI matrix ran none of them
  - `pytest.importorskip("hatchling")` deleted the three wheel-metadata tests on any machine
    where the build backend was not importable -- which `pip install -e .` never leaves
  - `pytest.importorskip("fastapi")` deleted the export test for the bare-install bug that
    had just been fixed, on precisely the bare install it was written for

None of those turned anything red. The count read twenty, the build read green, and the
checks were absent. A test that silently does not run is worse than a missing one, because
the gap is invisible: nobody adds a test for something that already appears to be covered.

So the skips are gone, and this file makes them stay gone. It is a check on the SHAPE of
the source, which is what a reviewer cannot reliably eyeball -- an importorskip four call
levels down reads like a helper, not like a deletion.

WHAT IS STILL ALLOWED, and why it is not a loophole: `@pytest.mark.skipif(<condition>,
reason=...)` with a real runtime condition. test_links.py uses one for the opt-in live-URL
check, which is off unless NEXUS_LINK_NETWORK=1 -- a test that is REQUESTED rather than
one that quietly vanishes. `skipif(True, ...)` and a skipif with no reason are both treated
as unconditional and rejected.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parent

# Dotted expressions that remove a test without a runtime condition. `pytest.fail` is
# deliberately absent: failing loudly is the remedy this file exists to force.
_UNCONDITIONAL = frozenset(
    {
        "pytest.importorskip",
        "pytest.skip",
        "pytest.xfail",
        "pytest.mark.skip",
        "pytest.mark.xfail",
    }
)

_CONDITIONAL = frozenset({"pytest.mark.skipif", "pytest.mark.xfail_if"})


def _dotted(node: ast.AST) -> str:
    """'pytest.mark.skip' for the attribute chain, '' for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _findings(source: str, label: str) -> list[str]:
    """Every place `source` removes a test with no runtime condition."""
    tree = ast.parse(source, filename=label)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted in _UNCONDITIONAL:
                found.append(f"{label}:{node.lineno} {dotted}")
        elif isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if dotted not in _CONDITIONAL:
                continue
            keywords = {kw.arg for kw in node.keywords}
            if "reason" not in keywords:
                found.append(f"{label}:{node.lineno} {dotted} with no reason")
            condition = node.args[0] if node.args else None
            if isinstance(condition, ast.Constant) and condition.value:
                found.append(f"{label}:{node.lineno} {dotted} on a constant condition")
    return sorted(found)


def _packaging_sources() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(PACKAGING.glob("*.py"))}


def test_no_packaging_test_removes_itself_without_a_condition():
    """
    The gate. Reintroducing any of the three importorskips names the file and the line.

    Includes this file, so the scanner cannot exempt itself -- the dotted names above are
    string literals, which is not an attribute expression and is not what is matched.
    """
    findings = [
        finding
        for name, source in _packaging_sources().items()
        for finding in _findings(source, name)
    ]
    assert not findings, (
        "these packaging tests delete themselves with no runtime condition, so their "
        "absence looks identical to their success:\n  "
        + "\n  ".join(findings)
        + "\nMake the missing thing a failure instead -- see _toml_parser() in "
        "test_extras_graph.py and _wheel_metadata() in test_requirements_derivation.py "
        "for the two shapes this took."
    )


def test_the_scanner_can_actually_see_a_skip():
    """
    The control. A scanner that matched nothing would pass the test above forever, which is
    the same vacuity one level up -- and is exactly how the five holes above stayed open.

    Each case is the real shape that was found in this directory, not a synthetic one.
    """
    cases = {
        'import pytest\ntomllib = pytest.importorskip("tomli")\n': "pytest.importorskip",
        'import pytest\ndef test_x():\n    pytest.importorskip("fastapi")\n': "pytest.importorskip",
        'import pytest\ndef test_x():\n    pytest.skip("later")\n': "pytest.skip",
        "import pytest\n@pytest.mark.skip\ndef test_x():\n    pass\n": "pytest.mark.skip",
        'import pytest\n@pytest.mark.xfail(reason="flaky")\ndef test_x():\n    pass\n': "pytest.mark.xfail",
        'import pytest\n@pytest.mark.skipif(True, reason="always")\ndef test_x():\n    pass\n': (
            "constant condition"
        ),
        "import pytest\n@pytest.mark.skipif(NO_NET)\ndef test_x():\n    pass\n": "no reason",
    }
    for source, expected in cases.items():
        findings = _findings(source, "<case>")
        assert findings, f"the scanner missed {expected} in:\n{source}"
        assert any(expected in finding for finding in findings), (findings, expected)


def test_the_scanner_still_allows_a_real_opt_in():
    """
    The other half of the control. A scanner that flagged everything would be satisfied only
    by deleting the opt-in live-URL check in test_links.py, which is a request rather than a
    disappearance -- and the pressure to "just make it pass" would land somewhere worse.
    """
    allowed = (
        "import os\n"
        "import pytest\n"
        'NETWORK = os.environ.get("NEXUS_LINK_NETWORK") == "1"\n'
        '@pytest.mark.skipif(not NETWORK, reason="set NEXUS_LINK_NETWORK=1")\n'
        "def test_x():\n"
        "    pass\n"
    )
    assert _findings(allowed, "<case>") == []


def test_there_are_packaging_files_with_tests_to_scan():
    """
    Guards the vacuous pass one more level down: an empty directory, a glob that stops
    matching, or a rename would make the gate above pass over nothing at all.
    """
    sources = _packaging_sources()
    assert len(sources) >= 5, f"only {sorted(sources)} found under {PACKAGING}"
    assert Path(__file__).name in sources, "the scan does not include itself"
    assert "conftest.py" in sources, (
        "conftest.py is not scanned, and a skip added there would apply to every test in "
        "this directory"
    )

    for name, source in sources.items():
        if not name.startswith("test_"):
            continue  # conftest.py holds hooks, not tests; it is scanned above, not counted
        functions = [
            node.name
            for node in ast.walk(ast.parse(source, filename=name))
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        assert functions, f"{name} defines no test functions; the scan of it proves nothing"


def _packaging_conftest(pytestconfig: pytest.Config):
    """The conftest for this directory, as pytest actually registered it."""
    for plugin in pytestconfig.pluginmanager.get_plugins():
        path = getattr(plugin, "__file__", "") or ""
        if Path(path).resolve().parent == PACKAGING and Path(path).name == "conftest.py":
            return plugin
    return None


def test_the_runtime_skip_guard_is_registered(pytestconfig: pytest.Config):
    """
    The source scan above cannot see a skip raised by a fixture, a plugin, or library code
    several frames below a helper -- which is the shape all three importorskips had. The
    conftest hook covers that, and is worth nothing if it is not loaded, so its registration
    is asserted here rather than assumed.
    """
    conftest = _packaging_conftest(pytestconfig)
    assert conftest is not None, (
        "tests/packaging/conftest.py is not registered, so a skip raised anywhere other "
        "than in the source scanned above would go unnoticed"
    )
    assert hasattr(conftest, "pytest_sessionfinish") and hasattr(conftest, "is_allowed_skip")


def test_the_runtime_skip_guard_exempts_only_the_declared_opt_in(pytestconfig: pytest.Config):
    """
    The control on the exemption. A guard that waved everything through would pass
    test_the_runtime_skip_guard_is_registered and catch nothing -- the same vacuity, moved
    into the conftest where it is even less visible.
    """
    conftest = _packaging_conftest(pytestconfig)
    assert conftest is not None

    opt_in = "tests/packaging/test_links.py::test_every_url_in_the_readme_returns_200"
    assert conftest.is_allowed_skip(opt_in, "set NEXUS_LINK_NETWORK=1 to check live URLs")

    # ...but not that same test skipped for some other reason, and not anything else.
    assert not conftest.is_allowed_skip(opt_in, "could not import fastapi")
    assert not conftest.is_allowed_skip(
        "tests/packaging/test_extras_graph.py::test_every_optional_import_in_src_maps_to_an_extra",
        "no TOML parser on this interpreter",
    )
    assert not conftest.is_allowed_skip(
        "tests/packaging/test_requirements_derivation.py::test_the_wheel_would_declare_the_version",
        "could not import 'hatchling'",
    )
