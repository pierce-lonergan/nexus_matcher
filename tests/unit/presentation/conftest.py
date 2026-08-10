"""
tests.unit.presentation.conftest | Layer: TEST
Fixtures shared by the CLI test modules.

## Relationships
# USED_BY → tests/unit/presentation/test_cli_regressions :: CLI rendering defects
# USED_BY → tests/unit/presentation/test_cli_json_governance :: DX-002

The test doubles themselves live in `_fakes.py` so both this file and the test modules
can import them by name; conftest holds only the wiring.
"""

from __future__ import annotations

import pytest

from nexus_matcher.application.use_cases.match_schema import MatchingConfig
from nexus_matcher.domain.models.entities import MatchResult

from ._fakes import FakeMatcher


@pytest.fixture
def cli_inputs(tmp_path):
    """Schema and dictionary files that merely have to exist (typer checks `exists=True`)."""
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object", "properties": {}}', encoding="utf-8")
    dictionary = tmp_path / "dictionary.csv"
    dictionary.write_text("id,business_name\ndict_042,Customer Email Address\n", encoding="utf-8")
    return schema, dictionary


@pytest.fixture
def install_matcher(monkeypatch):
    """Install a FakeMatcher returning whatever results the test wants."""

    def _install(
        results: dict[str, list[MatchResult]] | None = None,
        config: MatchingConfig | None = None,
    ) -> FakeMatcher:
        matcher = FakeMatcher(results, config)
        monkeypatch.setattr("nexus_matcher.presentation.cli.main._get_matcher", lambda: matcher)
        return matcher

    return _install


@pytest.fixture
def fake_matcher(install_matcher):
    """The default single-field result, for tests that do not care about the payload."""
    return install_matcher()
