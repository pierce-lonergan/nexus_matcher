"""
NM-0025 -- the governance payload was missing from the one interface a script can read.

`nexus-matcher match -f json` emitted the dictionary entry as
`{id, business_name, logical_name, data_type}`. No `protection_level`. The stated use case
of this library is that a matched field "inherits that entry's classification", and the
single field that use case rests on was absent from the only documented machine-readable
surface. The `scores` block carried 3 of 5 components and no weights, so the emitted
numbers could not reproduce the emitted confidence: an auditor could not check the
arithmetic from the file.

A fresh-eyes agent given a real governance task abandoned the CLI and rebuilt on the
Python API. docs/research/fresh-eyes.md, DX-002.

Why it escaped: the CLI tests carried hand-written stand-ins for the dictionary entry and
the score breakdown that declared exactly the attributes the writer read -- four and
three. A stub that omits what the code omits is a mirror, not an oracle, so no test in
this repository could have noticed. The doubles are built from real domain objects now.

This file is deliberately self-contained: the museum runner executes it on its own.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from nexus_matcher.application.use_cases.match_schema import MatchingConfig
from nexus_matcher.domain.models.entities import DictionaryEntry, MatchResult, SchemaField
from nexus_matcher.presentation.cli.main import app
from nexus_matcher.shared.types.base import (
    DataType,
    MatchDecision,
    PerformanceMetrics,
    ProtectionLevel,
    ScoreBreakdown,
)

runner = CliRunner(env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"})

CONFIG = MatchingConfig()
# Five different values, so a component paired with the wrong weight cannot hide.
SIGNALS = (0.9012, 0.5, 0.4211, 0.8, 0.25)
# Hand-computed against the shipped weights (0.70/0.05/0.05/0.05/0.15):
#   0.630840 + 0.025000 + 0.021055 + 0.040000 + 0.037500 = 0.754395
EXPECTED_CONFIDENCE = 0.754395


class _Stats:
    total_rows = 1
    valid_entries = 1
    skipped_rows = 0
    error_rows = 0
    success_rate = 1.0
    errors: list[str] = []  # noqa: RUF012 - a stub attribute, not shared state


class _Matcher:
    _config = CONFIG

    def load_dictionary(self, path):
        return _Stats()

    def match_schema(self, path):
        sem, lex, edit, type_, domain = SIGNALS
        entry = DictionaryEntry(
            id="dict_042",
            business_name="Customer Email Address",
            logical_name="cust_email",
            definition="The email address used to contact a customer.",
            data_type=DataType.STRING,
            protection_level=ProtectionLevel.PII,
            domain="billing",
        )
        return {
            "customer.email": [
                MatchResult(
                    schema_field=SchemaField(
                        name="email", data_type=DataType.STRING, full_path="customer.email"
                    ),
                    dictionary_entry=entry,
                    rank=1,
                    final_confidence=EXPECTED_CONFIDENCE,
                    score_breakdown=ScoreBreakdown(
                        fused_retrieval_score=sem,
                        lexical_score=lex,
                        edit_distance_score=edit,
                        type_compatibility_score=type_,
                        domain_score=domain,
                    ),
                    decision=MatchDecision.REVIEW,
                    performance=PerformanceMetrics(latency_ms=1.0),
                )
            ]
        }


def _emit(tmp_path, monkeypatch) -> dict:
    monkeypatch.setattr("nexus_matcher.presentation.cli.main._get_matcher", _Matcher)
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object", "properties": {}}', encoding="utf-8")
    dictionary = tmp_path / "dictionary.csv"
    dictionary.write_text("id,business_name\ndict_042,Customer Email Address\n", encoding="utf-8")

    result = runner.invoke(app, ["match", str(schema), "-d", str(dictionary), "-f", "json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["customer.email"][0]


def test_the_classification_reaches_the_machine_readable_output(tmp_path, monkeypatch):
    """
    The defect in one assertion. Without this field a governance script cannot do the one
    thing this library exists for, and has no way to tell that it cannot.
    """
    match = _emit(tmp_path, monkeypatch)

    assert match["dictionary_entry"]["protection_level"] == "PII", (
        "the JSON output does not carry the dictionary entry's classification, so a "
        "field cannot inherit it -- the stated use case is impossible from the CLI"
    )


def test_a_reviewer_can_tell_which_entry_this_is(tmp_path, monkeypatch):
    """
    `definition` and `domain`. The business name alone does not distinguish "Customer
    Email Address" in billing from the same name in marketing, and those two entries can
    carry different classifications.
    """
    entry = _emit(tmp_path, monkeypatch)["dictionary_entry"]

    assert entry["definition"] == "The email address used to contact a customer."
    assert entry["domain"] == "billing"


def test_the_confidence_can_be_recomputed_from_the_file(tmp_path, monkeypatch):
    """
    The other half of the defect: 3 of 5 components and no weights, so the numbers in the
    file could not add up to the confidence in the file.

    This is the check an auditor performs -- multiply, add, compare -- and nothing but the
    file's own contents feeds it.
    """
    match = _emit(tmp_path, monkeypatch)

    assert match["scores"].keys() == match["weights"].keys(), (
        "a component with no weight, or a weight with no component, leaves the emitted "
        "confidence unverifiable"
    )
    recomputed = sum(match["scores"][key] * match["weights"][key] for key in match["weights"])

    assert abs(recomputed - match["confidence"]) < 1e-5
    # Absolute, not just self-consistent: a file whose numbers agree with each other and
    # disagree with the matcher would pass the comparison above (H-004).
    assert match["confidence"] == EXPECTED_CONFIDENCE
