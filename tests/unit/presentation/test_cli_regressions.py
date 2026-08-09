"""
tests.unit.presentation.test_cli_regressions | Layer: TEST
Regression tests for the CLI defects fixed for the 2.0.1 re-cut.

## Relationships
# TESTS → presentation/cli/main :: match, sync, api, info
"""

from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from nexus_matcher.presentation.cli.main import app

# Same pinning rationale as test_cli.py: Rich wraps and truncates to the terminal, so an
# unpinned runner makes every output assertion depend on the environment it runs in.
runner = CliRunner(env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"})


# =============================================================================
# FAKE MATCHER
# =============================================================================
#
# The defects under test all live in the CLI's rendering and output-writing path. The
# real matcher loads embedding models, so stubbing it keeps these tests hermetic and
# fast while leaving every line of the code under test real.


class _FakeEntry:
    id = "field_001"
    business_name = "Customer Email Address"
    logical_name = "cust_email"

    # Lower-cased to mirror the DataType enum's `.value` access shape.
    class data_type:
        value = "STRING"


class _FakeScores:
    semantic_score = 0.90
    lexical_score = 0.80
    type_compatibility_score = 1.0


class _FakeMatch:
    rank = 1
    dictionary_entry = _FakeEntry()
    final_confidence = 0.91
    score_breakdown = _FakeScores()

    # Lower-cased to mirror the MatchDecision enum's `.value` access shape.
    class decision:
        value = "AUTO_APPROVE"


class _FakeStats:
    total_rows = 2
    valid_entries = 2
    skipped_rows = 0
    error_rows = 0
    success_rate = 1.0
    errors: ClassVar[list[str]] = []


class _FakeMatcher:
    def load_dictionary(self, path):
        return _FakeStats()

    def match_schema(self, path):
        return {"customer.email": [_FakeMatch()]}


@pytest.fixture
def cli_inputs(tmp_path):
    """Schema and dictionary files that merely have to exist (typer checks `exists=True`)."""
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object", "properties": {}}', encoding="utf-8")
    dictionary = tmp_path / "dictionary.csv"
    dictionary.write_text("id,business_name\nfield_001,Customer Email Address\n", encoding="utf-8")
    return schema, dictionary


@pytest.fixture
def fake_matcher(monkeypatch):
    """Replace the model-loading matcher with an in-memory stand-in."""
    monkeypatch.setattr("nexus_matcher.presentation.cli.main._get_matcher", _FakeMatcher)


# =============================================================================
# DEFECT 1 -- UnicodeEncodeError on non-UTF-8 Windows consoles
# =============================================================================

# Child program for the legacy-code-page tests. It has to be a real subprocess: the only
# way to exercise a cp437 stdout honestly is to have one, and `PYTHONIOENCODING` can only
# be set before the interpreter starts. Everything here is the shipped CLI except the
# matcher, which is stubbed for speed.
_LEGACY_CONSOLE_CHILD = """
import codecs
import os
import sys

# Refuse to run rather than pass vacuously: if the code page did not take effect this
# would be an ordinary UTF-8 run and would pin down nothing at all.
_wanted = codecs.lookup(sys.argv[1]).name
if codecs.lookup(sys.stdout.encoding).name != _wanted:
    sys.stderr.write("ENCODING-NOT-APPLIED:" + str(sys.stdout.encoding) + "\\n")
    raise SystemExit(97)

from nexus_matcher.presentation.cli import main as cli


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


class _Stats:
    total_rows = 2
    valid_entries = 2
    skipped_rows = 0
    error_rows = 0
    success_rate = 1.0
    errors = []


class _Matcher:
    def load_dictionary(self, path):
        return _Stats()

    def match_schema(self, path):
        return {os.environ.get("NEXUS_TEST_FIELD_PATH", "customer.email"): [_Match()]}


cli._get_matcher = lambda: _Matcher()
sys.argv = [sys.argv[0]] + sys.argv[2:]
cli.app()
"""


def _run_on_legacy_codepage(tmp_path, argv, encoding="cp437", columns="80", field_path=None):
    """Run the real CLI in a child whose stdout genuinely uses `encoding`."""
    script = tmp_path / "legacy_console_child.py"
    script.write_text(_LEGACY_CONSOLE_CHILD, encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = encoding
    env["PYTHONUTF8"] = "0"  # UTF-8 mode would otherwise hand the child a utf-8 stdout
    env["COLUMNS"] = columns
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    if field_path is not None:
        env["NEXUS_TEST_FIELD_PATH"] = field_path

    proc = subprocess.run(
        [sys.executable, str(script), encoding, *argv],
        capture_output=True,
        cwd=str(tmp_path),
        env=env,
        timeout=300,
        check=False,  # the exit code is the assertion
    )
    text = (proc.stdout + b"\n" + proc.stderr).decode(encoding, errors="replace")
    assert proc.returncode != 97, f"code page {encoding} never took effect: {text}"
    return proc.returncode, text


def test_the_chosen_codepages_really_reject_the_glyphs():
    """
    Guards the guards below: if these code pages ever gained the offending characters,
    every legacy-console test in this file would start passing for the wrong reason.
    """
    for encoding in ("cp437", "cp850", "cp1252"):
        codecs.lookup(encoding)
        with pytest.raises(UnicodeEncodeError):
            "⠸".encode(encoding)  # a frame of Rich's default "dots" spinner


@pytest.mark.parametrize("encoding", ["cp437", "cp850", "cp1252"])
def test_match_survives_legacy_console_codepage(tmp_path, cli_inputs, encoding):
    """
    `match` used to die with UnicodeEncodeError on any console whose code page cannot
    encode Rich's default Braille spinner (U+2838 and friends). The two commands that do
    real work were the only two that broke, and the user saw a bare codec error instead
    of their results.
    """
    schema, dictionary = cli_inputs
    code, text = _run_on_legacy_codepage(
        tmp_path, ["match", str(schema), "-d", str(dictionary)], encoding=encoding
    )
    assert "UnicodeEncodeError" not in text
    assert "codec can't encode" not in text
    assert code == 0, text
    assert "Summary" in text


@pytest.mark.parametrize("encoding", ["cp437", "cp850", "cp1252"])
def test_sync_survives_legacy_console_codepage(tmp_path, cli_inputs, encoding):
    """
    Same crash for `sync`, which additionally ends on a U+2713 check mark that cp437,
    cp850 and cp1252 all reject.
    """
    _, dictionary = cli_inputs
    code, text = _run_on_legacy_codepage(tmp_path, ["sync", str(dictionary)], encoding=encoding)
    assert "UnicodeEncodeError" not in text
    assert "codec can't encode" not in text
    assert code == 0, text
    assert "Sync Statistics" in text


@pytest.mark.parametrize("encoding", ["cp437", "cp850"])
def test_info_survives_legacy_console_codepage(tmp_path, encoding):
    """
    `info` was reported as unaffected, but that only holds on cp1252, which happens to
    have U+2022. On the true DOS code pages the bullet list killed it too.
    """
    code, text = _run_on_legacy_codepage(tmp_path, ["info"], encoding=encoding)
    assert "UnicodeEncodeError" not in text
    assert "codec can't encode" not in text
    assert code == 0, text
    assert "Deployment Modes" in text


def test_long_field_path_survives_legacy_console_codepage(tmp_path, cli_inputs):
    """
    The second instance of the same class, and the reason the spinner swap alone is not
    enough: Rich truncates a `no_wrap` column with U+2026, which cp437 also rejects. A
    field path long enough to be truncated on a narrow console used to abort the whole
    command after the matching work had already been paid for.
    """
    schema, dictionary = cli_inputs
    code, text = _run_on_legacy_codepage(
        tmp_path,
        ["match", str(schema), "-d", str(dictionary)],
        columns="40",
        field_path="customer.contact.preferences.email_address_primary",
    )
    assert "UnicodeEncodeError" not in text
    assert "codec can't encode" not in text
    assert code == 0, text


# =============================================================================
# DEFECT 2 -- `--output` silently ignored at the default `--format`
# =============================================================================


def test_output_path_is_honoured_without_explicit_format(tmp_path, cli_inputs, fake_matcher):
    """
    The CLI's own documented example, `match schema.json -d dictionary.csv -o results.json`,
    wrote no file at all and still exited 0: `--format` defaults to `table`, and only the
    json/csv branches ever looked at `--output`. Reporting success while dropping the
    user's results on the floor is silent data loss.
    """
    schema, dictionary = cli_inputs
    out = tmp_path / "results.json"

    result = runner.invoke(app, ["match", str(schema), "-d", str(dictionary), "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists(), f"-o was ignored; output was: {result.output}"
    assert json.loads(out.read_text(encoding="utf-8"))["customer.email"][0]["rank"] == 1


def test_output_extension_selects_csv(tmp_path, cli_inputs, fake_matcher):
    """A `.csv` output path must produce CSV, not JSON, when `--format` is not given."""
    schema, dictionary = cli_inputs
    out = tmp_path / "results.csv"

    result = runner.invoke(app, ["match", str(schema), "-d", str(dictionary), "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8").startswith("field_path,rank,business_name")


def test_output_with_unknown_extension_still_writes(tmp_path, cli_inputs, fake_matcher):
    """
    An extension we cannot map must still write something -- falling back to `table` and
    then writing nothing would reintroduce the exact silent-success bug.
    """
    schema, dictionary = cli_inputs
    out = tmp_path / "results.txt"

    result = runner.invoke(app, ["match", str(schema), "-d", str(dictionary), "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert "Customer Email Address" in out.read_text(encoding="utf-8")


def test_explicit_format_beats_the_extension(tmp_path, cli_inputs, fake_matcher):
    """An explicit `--format` must win, so scripted callers keep the behaviour they pinned."""
    schema, dictionary = cli_inputs
    out = tmp_path / "results.json"

    result = runner.invoke(
        app, ["match", str(schema), "-d", str(dictionary), "-o", str(out), "--format", "csv"]
    )

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8").startswith("field_path,rank,business_name")


def test_unknown_format_still_fails(tmp_path, cli_inputs, fake_matcher):
    """The counterpart: making `--format` optional must not make it permissive."""
    schema, dictionary = cli_inputs
    result = runner.invoke(app, ["match", str(schema), "-d", str(dictionary), "--format", "yaml"])
    assert result.exit_code != 0


# =============================================================================
# DEFECT 3 -- Rich markup swallowing square brackets
# =============================================================================


def test_missing_uvicorn_message_keeps_the_extra_name(monkeypatch):
    """
    The install hint said `pip install nexus-matcher`, which installs the CLI without the
    API extra and so leaves the user in exactly the state the message is about. Rich read
    the literal `[api]` as a style tag and dropped it.
    """
    monkeypatch.setitem(sys.modules, "uvicorn", None)

    result = runner.invoke(app, ["api"])

    assert result.exit_code == 1
    assert "nexus-matcher[api]" in result.output


def test_error_messages_keep_bracketed_detail(tmp_path, cli_inputs, monkeypatch):
    """
    Same class of bug, and the one that actually costs debugging time: loader errors name
    the offending columns in brackets, and Rich ate them, so the user was told a column
    was missing without being told which.
    """
    schema, dictionary = cli_inputs

    def _boom():
        raise ValueError("dictionary is missing columns [business_name, logical_name]")

    monkeypatch.setattr("nexus_matcher.presentation.cli.main._get_matcher", _boom)

    result = runner.invoke(app, ["match", str(schema), "-d", str(dictionary)])

    assert result.exit_code == 1
    assert "[business_name, logical_name]" in result.output


def test_unknown_format_message_keeps_brackets(tmp_path, cli_inputs, fake_matcher):
    """The rejected value is echoed back, so it must survive markup parsing verbatim."""
    schema, dictionary = cli_inputs

    result = runner.invoke(
        app, ["match", str(schema), "-d", str(dictionary), "--format", "[bogus]"]
    )

    assert result.exit_code == 1
    assert "[bogus]" in result.output


def test_bracketed_business_name_reaches_the_table(tmp_path, cli_inputs, monkeypatch):
    """
    Table cells are markup-parsed too, so a dictionary entry named like `[deprecated] Foo`
    lost its prefix on the way to the screen -- a data value quietly rewritten by the UI.
    """
    schema, dictionary = cli_inputs

    class _BracketEntry(_FakeEntry):
        business_name = "[deprecated] Customer Email"

    class _BracketMatch(_FakeMatch):
        dictionary_entry = _BracketEntry()

    class _BracketMatcher(_FakeMatcher):
        def match_schema(self, path):
            return {"customer.email": [_BracketMatch()]}

    monkeypatch.setattr("nexus_matcher.presentation.cli.main._get_matcher", _BracketMatcher)

    result = runner.invoke(app, ["match", str(schema), "-d", str(dictionary)])

    assert result.exit_code == 0, result.output
    assert "[deprecated]" in result.output


# =============================================================================
# DEFECT 4 -- wrong project URL in `info`
# =============================================================================


def test_info_points_at_the_real_repository():
    """`info` advertised a repository that is not this project's."""
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert "https://github.com/pierce-lonergan/nexus_matcher" in result.output
    assert "jpmc" not in result.output
