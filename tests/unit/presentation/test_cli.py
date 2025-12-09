"""
tests.unit.presentation.test_cli | Layer: TEST
Unit tests for CLI commands.

## Relationships
# TESTS → presentation/cli/main :: CLI commands
"""

import pytest
from typer.testing import CliRunner

from nexus_matcher.presentation.cli.main import app

runner = CliRunner()


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
        result = runner.invoke(app, [
            "match", "nonexistent.avsc",
            "-d", "dictionary.xlsx",
        ])
        assert result.exit_code != 0


class TestSyncCommand:
    """Test sync command."""

    def test_sync_nonexistent_file(self):
        """Test sync with nonexistent file."""
        result = runner.invoke(app, ["sync", "nonexistent.xlsx"])
        assert result.exit_code != 0


class TestAPICommand:
    """Test api command help."""

    def test_api_help(self):
        """Test api --help."""
        result = runner.invoke(app, ["api", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--reload" in result.stdout
