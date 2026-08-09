"""
NM-0009 -- an empty schema was reported as an arithmetic fault inside the tool.

The summary line divides auto-approvals by the field count:

    Summary: {auto_approved}/{total_fields} fields auto-approved ({...:.1f}%)

A schema that parses to zero fields -- an empty file, a truncated download, a format the
parser did not recognise -- made that denominator zero, and the generic handler turned the
`ZeroDivisionError` into:

    Error: division by zero

Which is a wholly ordinary user mistake presented as a crash in the matcher. Nothing in it
names the schema, the field count, or anything the user could act on, so the natural next
step is to file a bug against this library rather than to look at the input file.

The tests below are about the DIAGNOSTIC, not about the arithmetic. A guard that silences
the exception and prints nothing would still leave the user with a mystery, so the message
must reach them and must state the fact they can go and check: the schema produced no
fields. The exit code is asserted non-zero separately, because a script that pipes results
somewhere has to be able to tell "nothing matched" from "matched nothing because the file
was empty".
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from typer.testing import CliRunner

from nexus_matcher.presentation.cli.main import app

runner = CliRunner(env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"})


class _Stats:
    total_rows = 1
    valid_entries = 1
    skipped_rows = 0
    error_rows = 0
    success_rate = 1.0
    errors: ClassVar[list[str]] = []


class _MatcherWithAnEmptySchema:
    """
    Parses the schema to zero fields -- the state an empty or unrecognised file produces.

    Stubbed because the defect is in the CLI's summary line, not in any parser; every line
    under test is the shipped one.
    """

    def load_dictionary(self, path):
        return _Stats()

    def match_schema(self, path):
        return {}


@pytest.fixture
def cli_inputs(tmp_path):
    """An empty schema file, and a dictionary that only has to exist."""
    schema = tmp_path / "schema.json"
    schema.write_text("", encoding="utf-8")
    dictionary = tmp_path / "dictionary.csv"
    dictionary.write_text("id,business_name\nfield_001,Customer Email Address\n", encoding="utf-8")
    return schema, dictionary


@pytest.fixture
def result(cli_inputs, monkeypatch):
    monkeypatch.setattr(
        "nexus_matcher.presentation.cli.main._get_matcher", _MatcherWithAnEmptySchema
    )
    schema, dictionary = cli_inputs
    return runner.invoke(app, ["match", str(schema), "-d", str(dictionary)])


def test_an_empty_schema_is_not_reported_as_arithmetic(result):
    """The symptom exactly as it reached the user."""
    assert "division by zero" not in result.output.lower(), (
        f"an empty schema is still reported as a division fault:\n{result.output}"
    )
    assert "ZeroDivisionError" not in result.output, result.output


# Ways of saying "your schema parsed to nothing". A set rather than one literal because
# the exact wording is not the contract; SAYING IT is. Add to this list if the message is
# reworded -- do not delete the check.
_SAYS_THERE_WERE_NO_FIELDS = ("no fields", "0 fields", "zero fields", "empty")


def test_the_message_says_the_schema_produced_no_fields(result):
    """
    A guard that swallows the error silently would pass the test above and help nobody.

    Note what is NOT asserted here: that the word "schema" appears. It always does -- the
    progress spinner prints "Matching schema..." on the way past -- so that assertion is
    satisfied even by `Error: division by zero`, which is precisely the useless message
    this entry exists to keep out. The discriminating fact is the field COUNT, because
    that is the one thing the user can go and check.
    """
    lowered = result.output.lower()
    assert any(phrase in lowered for phrase in _SAYS_THERE_WERE_NO_FIELDS), (
        f"the output never tells the user their schema parsed to zero fields, so there is "
        f"nothing in it to act on:\n{result.output}"
    )


def test_the_exit_code_reports_failure(result):
    """
    Zero fields matched is not success, and a script has to be able to tell.

    Exiting 0 here means a pipeline that matched nothing at all looks identical to one
    that worked, which is the same silent-success class as NM-0003 and NM-0004.
    """
    assert result.exit_code != 0, f"matching a schema with no fields exited 0:\n{result.output}"


def test_a_normal_schema_still_reports_its_summary(cli_inputs, monkeypatch):
    """
    The counterpart, so the guard cannot be "fixed" by removing the summary entirely.

    Without this, deleting the percentage line would turn every test above green while
    taking away the one number the command exists to report.
    """

    class _Entry:
        id = "field_001"
        business_name = "Customer Email Address"
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

    class _Matcher(_MatcherWithAnEmptySchema):
        def match_schema(self, path):
            return {"customer.email": [_Match()]}

    monkeypatch.setattr("nexus_matcher.presentation.cli.main._get_matcher", _Matcher)
    schema, dictionary = cli_inputs

    result = runner.invoke(app, ["match", str(schema), "-d", str(dictionary)])

    assert result.exit_code == 0, result.output
    assert "1/1" in result.output, result.output
