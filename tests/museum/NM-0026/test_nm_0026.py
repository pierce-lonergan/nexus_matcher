"""
NM-0026 -- the machine-readable output was not machine-readable when redirected.

    nexus-matcher match customer.avsc -d dictionary.csv -f json > results.json

produced

    <spinner frame> Matching schema...
    { ...the document... }

    Summary: 0/3 fields auto-approved (0.0%)

Rich's Progress and `rich.print` both default to stdout, and nothing in the `match`
command had ever said otherwise, so the payload shared its channel with the human status
output. The most obvious way anyone would script this CLI yielded a file `json.loads`
rejects on its first character.

Found while fixing NM-0025 -- the two shipped together, on the same surface, in 2.0.1.

Why it escaped: every CLI test asserted that some substring APPEARED in the output.
`"Summary" in result.output` passes just as happily when the summary is sitting inside a
JSON document. None of them ever parsed what the command emitted, and an in-process
CliRunner would not have shown it anyway -- this needs a real pipe.

The fix is a redirect, not a deletion: status moves to stderr only when stdout is
carrying a payload, so a user still watches progress and still gets told what happened.
Both halves are asserted below, because getting the summary off stdout by throwing it
away would satisfy the first assertion and cost the user their feedback.

Self-contained on purpose: the museum runner executes this file on its own, and the
child below is the only honest way to get a non-TTY stdout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# The shipped CLI in full; only the matcher is replaced, because that is the part that
# would load an embedding model. Everything it builds is a real domain object.
CHILD = """
import sys

from nexus_matcher.application.use_cases.match_schema import MatchingConfig
from nexus_matcher.domain.models.entities import DictionaryEntry, MatchResult, SchemaField
from nexus_matcher.presentation.cli import main as cli
from nexus_matcher.shared.types.base import (
    DataType,
    MatchDecision,
    PerformanceMetrics,
    ProtectionLevel,
    ScoreBreakdown,
)

CONFIG = MatchingConfig()


class _Stats:
    total_rows = 1
    valid_entries = 1
    skipped_rows = 0
    error_rows = 0
    success_rate = 1.0
    errors = []


def _result(path):
    entry = DictionaryEntry(
        id="dict_042",
        business_name="Customer Email Address",
        logical_name="cust_email",
        definition="The email address used to contact a customer.",
        data_type=DataType.STRING,
        protection_level=ProtectionLevel.PII,
        domain="billing",
    )
    return MatchResult(
        schema_field=SchemaField(name="email", data_type=DataType.STRING, full_path=path),
        dictionary_entry=entry,
        rank=1,
        final_confidence=0.754395,
        score_breakdown=ScoreBreakdown(
            fused_retrieval_score=0.9012,
            lexical_score=0.5,
            edit_distance_score=0.4211,
            type_compatibility_score=0.8,
            domain_score=0.25,
        ),
        decision=MatchDecision.REVIEW,
        performance=PerformanceMetrics(latency_ms=1.0),
    )


class _Matcher:
    _config = CONFIG

    def load_dictionary(self, path):
        return _Stats()

    def match_schema(self, path):
        return {"customer.email": [_result("customer.email")]}


cli._get_matcher = lambda: _Matcher()
cli.app()
"""


def _run(tmp_path):
    """Run the real CLI with stdout on a PIPE, which is what a redirect actually gives it."""
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object", "properties": {}}', encoding="utf-8")
    dictionary = tmp_path / "dictionary.csv"
    dictionary.write_text("id,business_name\ndict_042,Customer Email Address\n", encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["COLUMNS"] = "80"
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"

    return subprocess.run(
        [
            sys.executable,
            str(script),
            "match",
            str(schema),
            "-d",
            str(dictionary),
            "-f",
            "json",
        ],
        capture_output=True,
        cwd=str(tmp_path),
        env=env,
        timeout=300,
        check=False,
    )


def test_redirected_json_output_is_a_json_document(tmp_path):
    """
    The defect in one assertion, and the one nobody had ever made: parse it.

    A substring check passes just as well on a document with a spinner frame glued to the
    front of it, which is why fifteen CLI tests saw nothing wrong.
    """
    proc = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")

    document = json.loads(proc.stdout.decode("utf-8"))  # this is the test

    assert document["customer.email"][0]["dictionary_entry"]["protection_level"] == "PII"


def test_the_user_still_gets_their_status_output(tmp_path):
    """
    The counterpart. Deleting the summary would make the assertion above pass and would
    quietly take away the feedback a person running this command relies on.
    """
    proc = _run(tmp_path)

    assert "Summary" in proc.stderr.decode("utf-8", "replace")
