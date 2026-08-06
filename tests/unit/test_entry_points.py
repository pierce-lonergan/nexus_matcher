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

from importlib.metadata import distribution

import pytest

DISTRIBUTION = "nexus-matcher"
PLUGIN_PREFIX = "nexus_matcher."


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
