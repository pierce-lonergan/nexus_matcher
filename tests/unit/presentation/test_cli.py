"""
tests.unit.presentation.test_cli | Layer: TEST
Unit tests for CLI commands.

## Relationships
# TESTS → presentation/cli/main :: CLI commands
"""

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
        """`api --help` must succeed. What it LOOKS like is asserted separately."""
        result = runner.invoke(app, ["api", "--help"])
        assert result.exit_code == 0

    def test_api_exposes_expected_options(self):
        """
        Assert the command's declared INTERFACE, not its rendered help text.

        This previously scraped `result.stdout` for "--host". That passed locally and
        failed on all four Python versions in CI, because Typer renders help through Rich
        and the layout depends on the Rich version and terminal width -- the option names
        can be wrapped or truncated out of the visible text while the options themselves
        are perfectly well defined. Scraping the rendering tests Typer and Rich, not this
        CLI. Introspecting the parameters tests what we actually own, and fails for a real
        reason if an option is ever removed or renamed.
        """
        import click
        import typer

        command = typer.main.get_command(app)
        api_cmd = command.commands["api"]  # type: ignore[attr-defined]

        declared = {
            opt for param in api_cmd.params if isinstance(param, click.Option) for opt in param.opts
        }
        for expected in ("--host", "--port", "--reload"):
            assert expected in declared, f"{expected} missing; api declares {sorted(declared)}"
