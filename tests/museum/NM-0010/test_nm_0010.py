"""
NM-0010 -- the UI rewrote the data it was asked to display.

A glossary entry called

    [deprecated] Customer Email

reached the screen as

    Customer Email

because Rich parses table cell contents as markup, and `[deprecated]` looks exactly like
a style tag. The prefix is not decoration: it is the one signal telling a reviewer not to
map new fields onto that entry. Losing it is worse than a crash, because the output is
still perfectly plausible -- nothing anywhere suggests a character was dropped.

The same hazard applies to the field-path cell, whose contents come from the user's
schema, so both are covered here. What is NOT covered is the confidence and decision
cells: those brackets are markup this CLI writes deliberately, and escaping them would
break the colouring. The rule is about the ORIGIN of a value, not about the character.

Every assertion is on what the user sees. `escape()` is one way to satisfy them; a
`Text()` cell or a different renderer would be another, and this test survives that.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from typer.testing import CliRunner

from nexus_matcher.presentation.cli.main import app

# Rich truncates to the terminal, so an unpinned runner would make "the value survived"
# depend on the window the test happened to run in.
runner = CliRunner(env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"})

# Both values look like markup and are both DATA: one arrives from the glossary, the other
# from the schema being matched.
#
# The field path is a SQL Server identifier, which is how that dialect quotes names and so
# how it appears in every DDL export this tool reads. It is chosen over the more obvious
# `orders[0].sku` because Rich only reads `[` as a tag when what follows starts with a
# letter, `#`, `/` or `@` -- an array INDEX passes through untouched and would make this
# test pass against the broken code. Verified against Rich directly: `orders[0].sku`
# survives an unescaped cell, `[dbo].[customer].[email_address]` does not.
BUSINESS_NAME = "[deprecated] Customer Email"
FIELD_PATH = "[dbo].[customer].[email_address]"


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
    """Stands in for the model-loading matcher; the defect is purely in the rendering."""

    def load_dictionary(self, path):
        return _Stats()

    def match_schema(self, path):
        return {FIELD_PATH: [_Match()]}


@pytest.fixture
def cli_inputs(tmp_path):
    """Files that only have to exist -- typer checks `exists=True` before the command runs."""
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object", "properties": {}}', encoding="utf-8")
    dictionary = tmp_path / "dictionary.csv"
    dictionary.write_text("id,business_name\nfield_001,Customer Email Address\n", encoding="utf-8")
    return schema, dictionary


@pytest.fixture
def result(cli_inputs, monkeypatch):
    monkeypatch.setattr("nexus_matcher.presentation.cli.main._get_matcher", _Matcher)
    schema, dictionary = cli_inputs
    return runner.invoke(app, ["match", str(schema), "-d", str(dictionary)])


def test_the_command_still_succeeds(result):
    """
    An unknown style tag can also make Rich raise rather than silently drop the text, so
    "it crashed" is the other face of this defect and is pinned here.
    """
    assert result.exit_code == 0, result.output


def test_a_bracketed_business_name_reaches_the_screen_intact(result):
    """
    The symptom: the deprecation marker vanished on the way to the terminal.

    Asserting the WHOLE value, not just the bracket, so a rendering that keeps the
    brackets and loses the word inside them still fails.
    """
    assert BUSINESS_NAME in result.output, (
        f"{BUSINESS_NAME!r} was rewritten on its way to the screen. What was printed:\n"
        f"{result.output}"
    )


def test_a_bracketed_field_path_reaches_the_screen_intact(result):
    """
    The same hazard on the other data-bearing cell.

    Field paths come from the user's schema, and SQL Server quotes every identifier in
    square brackets -- so this is not a contrived value, it is what a DDL export of a
    Microsoft catalogue produces for an ordinary column.
    """
    assert FIELD_PATH in result.output, (
        f"{FIELD_PATH!r} was rewritten on its way to the screen. What was printed:\n{result.output}"
    )


def test_the_decision_and_confidence_are_still_styled_not_escaped(result):
    """
    The counterpart, so this cannot be "fixed" by escaping every cell.

    The brackets around a style name are markup the CLI writes on purpose. Escaping them
    would print `[green]87.00%[/green]` at the user, which is the same class of defect --
    output that does not mean what it says -- pointed the other way.
    """
    assert "AUTO_APPROVE" in result.output, result.output
    assert "[green]" not in result.output, (
        f"style tags are being shown as literal text rather than applied:\n{result.output}"
    )
