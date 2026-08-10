"""
tests.unit.presentation._legacy_console | Layer: TEST
Run the real CLI in a child process whose stdout genuinely uses a chosen code page.

## Relationships
# USED_BY → tests/unit/presentation/test_cli_regressions :: NM-0001, the encoding matrix
# USED_BY → tests/unit/presentation/test_cli_json_governance :: DX-002 on that matrix

It has to be a real subprocess. `PYTHONIOENCODING` can only be set before the interpreter
starts, and the only honest way to exercise a cp437 stdout is to have one. The child is
the shipped CLI in full; only the matcher is replaced, because that is the part that would
load an embedding model.

The child is also the non-TTY path by construction: `capture_output=True` gives it a pipe,
not a terminal, which is what a script redirecting `-f json` into a file actually gets.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Everything the child builds is a REAL domain object imported from the installed package,
# for the reason given in _fakes.py: a stub declaring exactly the fields the CLI reads
# cannot notice a field the CLI fails to read.
CHILD_SOURCE = """
import codecs
import os
import sys

# Refuse to run rather than pass vacuously: if the code page did not take effect this
# would be an ordinary UTF-8 run and would pin down nothing at all.
_wanted = codecs.lookup(sys.argv[1]).name
if codecs.lookup(sys.stdout.encoding).name != _wanted:
    sys.stderr.write("ENCODING-NOT-APPLIED:" + str(sys.stdout.encoding) + "\\n")
    raise SystemExit(97)

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
SIGNALS = (0.9012, 0.5, 0.4211, 0.8, 0.25)
WEIGHTS = (
    CONFIG.semantic_weight,
    CONFIG.lexical_weight,
    CONFIG.edit_distance_weight,
    CONFIG.type_weight,
    CONFIG.domain_weight,
)
CONFIDENCE = min(max(sum(s * w for s, w in zip(SIGNALS, WEIGHTS)), 0.0), 1.0)


def _result(field_path, suffix=""):
    entry = DictionaryEntry(
        id="dict_042" + suffix,
        business_name=os.environ.get("NEXUS_TEST_BUSINESS_NAME", "Customer Email Address"),
        logical_name="cust_email",
        definition=os.environ.get(
            "NEXUS_TEST_DEFINITION", "The email address used to contact a customer."
        ),
        data_type=DataType.STRING,
        protection_level=ProtectionLevel.PII,
        domain=os.environ.get("NEXUS_TEST_DOMAIN", "customer"),
    )
    sem, lex, edit, type_, domain = SIGNALS
    return MatchResult(
        schema_field=SchemaField(
            name=field_path.rsplit(".", 1)[-1],
            data_type=DataType.STRING,
            full_path=field_path,
        ),
        dictionary_entry=entry,
        rank=1,
        final_confidence=CONFIDENCE,
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


class _Stats:
    total_rows = 2
    valid_entries = 2
    skipped_rows = 0
    error_rows = 0
    success_rate = 1.0
    errors = []


class _Matcher:
    _config = CONFIG

    def load_dictionary(self, path):
        return _Stats()

    def match_schema(self, path):
        # Inserted in NON-alphabetical order on purpose: a sorted-key claim that is only
        # ever handed already-sorted input asserts nothing.
        if os.environ.get("NEXUS_TEST_MULTI") == "1":
            return {
                "zeta.updated_at": [_result("zeta.updated_at", "-z")],
                "alpha.account_id": [_result("alpha.account_id", "-a")],
                "mid.customer_email": [_result("mid.customer_email", "-m")],
            }
        path = os.environ.get("NEXUS_TEST_FIELD_PATH", "customer.email")
        return {path: [_result(path)]}


cli._get_matcher = lambda: _Matcher()
sys.argv = [sys.argv[0]] + sys.argv[2:]
cli.app()
"""


def run_child(
    tmp_path,
    argv,
    *,
    encoding="cp437",
    columns="80",
    env_extra=None,
    hashseed=None,
) -> subprocess.CompletedProcess:
    """Run the real CLI in a child whose stdout genuinely uses `encoding`. Returns raw bytes."""
    script = tmp_path / "legacy_console_child.py"
    script.write_text(CHILD_SOURCE, encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = encoding
    env["PYTHONUTF8"] = "0"  # UTF-8 mode would otherwise hand the child a utf-8 stdout
    env["COLUMNS"] = columns
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    if hashseed is not None:
        env["PYTHONHASHSEED"] = str(hashseed)
    for key, value in (env_extra or {}).items():
        env[key] = value

    return subprocess.run(
        [sys.executable, str(script), encoding, *argv],
        capture_output=True,
        cwd=str(tmp_path),
        env=env,
        timeout=300,
        check=False,  # the exit code is the assertion
    )


def run_on_legacy_codepage(
    tmp_path,
    argv,
    encoding="cp437",
    columns="80",
    field_path=None,
    env_extra=None,
):
    """As `run_child`, decoded for text assertions. Returns (returncode, text)."""
    extra = dict(env_extra or {})
    if field_path is not None:
        extra["NEXUS_TEST_FIELD_PATH"] = field_path

    proc = run_child(tmp_path, argv, encoding=encoding, columns=columns, env_extra=extra)
    text = (proc.stdout + b"\n" + proc.stderr).decode(encoding, errors="replace")
    assert proc.returncode != 97, f"code page {encoding} never took effect: {text}"
    return proc.returncode, text
