"""
tests.unit.presentation.test_cli | Layer: TEST
Unit tests for CLI commands.

## Relationships
# TESTS → presentation/cli/main :: CLI commands
"""

import pytest
from typer.testing import CliRunner

from nexus_matcher.presentation.cli.main import app

# Pin the terminal width. Typer renders help through Rich, which wraps and truncates to
# the terminal, so an unpinned runner makes every help-scraping assertion depend on the
# environment it happens to run in. A test that passes locally and fails in CI for that
# reason costs more to diagnose than it ever catches.
runner = CliRunner(env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"})


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_version_flag(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "NexusMatcher" in result.stdout
        assert "version" in result.stdout.lower()

    def test_help_flag(self):
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "match" in result.stdout
        assert "sync" in result.stdout
        assert "api" in result.stdout

    def test_info_command(self):
        """Test info command."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "NexusMatcher" in result.stdout
        assert "Deployment Modes" in result.stdout


class TestMatchCommand:
    """Test match command."""

    def test_match_missing_dictionary(self, tmp_path):
        """Test match without dictionary fails gracefully."""
        # Create a dummy schema file
        schema_file = tmp_path / "schema.avsc"
        schema_file.write_text('{"type": "record", "name": "Test", "fields": []}')

        result = runner.invoke(app, ["match", str(schema_file)])
        # Should fail because -d is required
        assert result.exit_code != 0

    def test_match_nonexistent_schema(self):
        """Test match with nonexistent schema."""
        result = runner.invoke(
            app,
            [
                "match",
                "nonexistent.avsc",
                "-d",
                "dictionary.xlsx",
            ],
        )
        assert result.exit_code != 0


class TestSyncCommand:
    """Test sync command."""

    def test_sync_nonexistent_file(self):
        """Test sync with nonexistent file."""
        result = runner.invoke(app, ["sync", "nonexistent.xlsx"])
        assert result.exit_code != 0


class TestAPICommand:
    """Test api command help."""

    def test_api_help_runs(self):
        """`api --help` must succeed."""
        result = runner.invoke(app, ["api", "--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize(
        "argv",
        [
            # Value-taking options need their value, or the option swallows "--help"
            # as its argument and the command tries to start a real server.
            ["--host", "127.0.0.1", "--help"],
            ["--port", "9999", "--help"],
            ["--reload", "--help"],
        ],
        ids=["--host", "--port", "--reload"],
    )
    def test_api_accepts_expected_options(self, argv):
        """
        Verify the options BEHAVIOURALLY -- that the CLI accepts them -- rather than by
        introspecting Click, and rather than by scraping rendered help.

        Both of those were tried and both were wrong:

        * Scraping `result.stdout` for "--host" depends on Rich's wrapping and the
          terminal width. Passed locally, failed on all four Python versions in CI.
        * Introspecting `command.params` / `get_params(ctx)` returns an EMPTY list on
          typer 0.27.1 (CI resolved that from an unpinned `typer[all]>=0.9.0`, while this
          machine had 0.20.0). Typer builds parameters lazily at invocation time, so no
          Click-level introspection sees them -- yet the CLI works perfectly: the options
          are accepted and unknown flags are rejected. The introspection test would have
          reported a breakage that does not exist.

        Invoking the command is the only check that tests what a user actually gets and
        holds across typer versions.
        """
        result = runner.invoke(app, ["api", *argv])
        assert result.exit_code == 0, f"api rejected {argv}: {result.output[:300]}"

    def test_api_rejects_unknown_option(self):
        """The counterpart: accepting everything would make the test above vacuous."""
        result = runner.invoke(app, ["api", "--definitely-not-a-real-flag"])
        assert result.exit_code != 0
