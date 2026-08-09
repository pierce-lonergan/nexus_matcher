"""
NM-0003 -- the CLI's own documented example was a no-op that reported success.

    nexus-matcher match schema.json -d dictionary.csv -o results.json

`--format` defaulted to `table`, and only the json and csv branches ever looked at
`--output`. So this command loaded the dictionary, matched the schema, printed a table,
wrote no file, said nothing about not writing one, and exited 0. Silent data loss dressed
up as success -- and the exit code is what a script checks.

The tests assert the OBSERVABLE contract -- a file exists at the requested path and holds
the results -- not how the format is decided. `--format` could be renamed or the inference
rules rewritten and these would still be the right assertions.

The unknown-extension case is here for a specific reason: the natural way to "fix" this is
to infer json from `.json` and csv from `.csv`, which quietly leaves `-o results.txt`
writing nothing at all. That is the same bug with a smaller blast radius.
"""

from __future__ import annotations

import csv
import json
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from nexus_matcher.presentation.cli.main import app

# Rich wraps and truncates to the terminal, so an unpinned runner makes output-dependent
# assertions depend on the environment they run in.
runner = CliRunner(env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"})

FIELD = "customer.email"
BUSINESS_NAME = "Customer Email Address"


class _Entry:
    id = "field_001"
    business_name = BUSINESS_NAME
    logical_name = "cust_email"

    class data_type:
        value = "STRING"


class _Scores:
    semantic_score = 0.90
    lexical_score = 0.80
    type_compatibility_score = 1.0


class _Match:
    rank = 1
    dictionary_entry = _Entry()
    final_confidence = 0.91
    score_breakdown = _Scores()

    class decision:
        value = "AUTO_APPROVE"


class _Stats:
    total_rows = 1
    valid_entries = 1
    skipped_rows = 0
    error_rows = 0
    success_rate = 1.0
    errors: ClassVar[list[str]] = []


class _Matcher:
    """
    Stands in for the model-loading matcher.

    The defect lives entirely in the CLI's output-writing path, so stubbing retrieval
    keeps every line under test real while making the test hermetic.
    """

    def load_dictionary(self, path):
        return _Stats()

    def match_schema(self, path):
        return {FIELD: [_Match()]}


@pytest.fixture
def cli_inputs(tmp_path):
    """Files that only have to exist -- typer checks `exists=True` before the command runs."""
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object", "properties": {}}', encoding="utf-8")
    dictionary = tmp_path / "dictionary.csv"
    dictionary.write_text("id,business_name\nfield_001,Customer Email Address\n", encoding="utf-8")
    return schema, dictionary


@pytest.fixture
def fake_matcher(monkeypatch):
    monkeypatch.setattr("nexus_matcher.presentation.cli.main._get_matcher", _Matcher)


def _match(cli_inputs, *extra):
    schema, dictionary = cli_inputs
    return runner.invoke(app, ["match", str(schema), "-d", str(dictionary), *extra])


def test_the_documented_example_writes_its_file(tmp_path, cli_inputs, fake_matcher):
    """
    The exact command from `match --help`, and the exact thing it failed to do.

    Asserting the file merely exists would be satisfied by an empty file, so this also
    reads the results back out of it.
    """
    out = tmp_path / "results.json"

    result = _match(cli_inputs, "-o", str(out))

    assert result.exit_code == 0, result.output
    assert out.exists(), (
        f"`-o {out.name}` produced no file and the command still exited 0. "
        f"Output was:\n{result.output}"
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload[FIELD][0]["dictionary_entry"]["business_name"] == BUSINESS_NAME


def test_a_csv_output_path_produces_csv(tmp_path, cli_inputs, fake_matcher):
    """A file whose contents do not match the name the user chose is its own defect."""
    out = tmp_path / "results.csv"

    result = _match(cli_inputs, "-o", str(out))

    assert result.exit_code == 0, result.output
    assert out.exists(), f"`-o {out.name}` produced no file. Output was:\n{result.output}"
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert rows and rows[0]["business_name"] == BUSINESS_NAME, out.read_text(encoding="utf-8")


def test_an_unmappable_extension_still_writes_something(tmp_path, cli_inputs, fake_matcher):
    """
    The near-miss fix, pinned so it cannot come back.

    Inferring the format from the extension and falling back to `table` for anything
    unrecognised reinstates the original bug for `-o results.txt`: still no file, still
    exit 0. Every branch that has an output path must write to it.
    """
    out = tmp_path / "results.txt"

    result = _match(cli_inputs, "-o", str(out))

    assert result.exit_code == 0, result.output
    assert out.exists(), (
        f"`-o {out.name}` produced no file. An extension we cannot map is still a request "
        f"for a file. Output was:\n{result.output}"
    )
    assert BUSINESS_NAME in out.read_text(encoding="utf-8")


def test_an_explicit_format_still_wins(tmp_path, cli_inputs, fake_matcher):
    """
    The counterpart: anything scripted against the old flag keeps its behaviour.

    Without this, "honour --output" could be implemented by making the extension
    authoritative, which would silently change the output of existing pipelines.
    """
    out = tmp_path / "results.json"

    result = _match(cli_inputs, "-o", str(out), "--format", "csv")

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8").startswith("field_path,rank,business_name")


def test_an_unknown_format_is_still_rejected(tmp_path, cli_inputs, fake_matcher):
    """Making `--format` optional must not make it permissive."""
    result = _match(cli_inputs, "--format", "yaml")
    assert result.exit_code != 0, result.output
