"""
NM-0001 -- the CLI crashed on the consoles it was most likely to be run from.

`nexus-matcher match` and `nexus-matcher sync` -- the only two commands that do real work
-- died with a bare `UnicodeEncodeError` and exit 1 on any Windows console whose code page
cannot encode Rich's default Braille spinner (U+2838 and friends). `info` went down too on
a true DOS code page, where U+2022 is absent. The crash landed AFTER the dictionary had
been loaded and the schema matched, so the user paid for the work and received a codec
error in place of results.

Nothing in the suite could see it: every CLI test drove typer's `CliRunner`, whose
captured stream is UTF-8. So this test refuses to run in-process. It starts a real child
interpreter with `PYTHONIOENCODING` set, and refuses to pass at all if the code page did
not take effect -- a legacy-console test that quietly ran on UTF-8 would be exactly the
kind of thing that reads as coverage and is not.

Two INDEPENDENT symptoms are asserted, because the fix has two halves and either half
going missing is a real regression:

  1. the command completes -- no codec error, exit 0            (both halves)
  2. what it prints is readable, not `\\u2022` escape text      (the glyph half alone)

Symptom 2 matters on its own: the error-handler backstop turns an unencodable character
into visible escape text rather than an exception, so losing the glyph fallback alone
would not crash -- it would just print gibberish, pass symptom 1, and look fine.

One test here is NOT a regression gate, and its name says so: the `test_premise_guard_*`
test asserts stdlib codec facts, so no change to `nexus_matcher` can make it fail. Nobody
should count it as coverage of the fix. It guards the PREMISE the real gates rest on --
that the code pages they run under genuinely cannot encode the glyphs -- and it fails on
an edit to THIS FILE that would otherwise make every gate in it pass without exercising
the fix at all.
"""

from __future__ import annotations

import codecs
import os
import subprocess
import sys

import pytest

# The real CLI, with only the model-loading matcher stubbed out. `PYTHONIOENCODING` can
# only be set before an interpreter starts, so this has to be a child process; and the
# first thing the child does is verify the code page actually applied.
_CHILD = """
import codecs
import os
import sys

wanted = codecs.lookup(sys.argv[1]).name
if codecs.lookup(sys.stdout.encoding).name != wanted:
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
        return {os.environ.get("NEXUS_FIELD_PATH", "customer.email"): [_Match()]}


cli._get_matcher = lambda: _Matcher()
sys.argv = [sys.argv[0]] + sys.argv[2:]
cli.app()
"""


# The code pages every test in this file runs the CLI under. These are the ONLY literals:
# the premise guard below and the parametrized gates read the same names, so re-pointing
# the tests at a code page that can encode the glyphs -- which would make every gate here
# pass while exercising none of the fix -- turns the premise guard red instead of quietly
# going green. Before this was shared, the guard held its own private copy and saw nothing.
_CP437 = "cp437"  # the DOS default: no Braille spinner frames, no U+2022 bullet
_DOS_CODE_PAGES = (_CP437, "cp850")  # true DOS pages -- U+2022 is absent from both
_LEGACY_CODE_PAGES = (*_DOS_CODE_PAGES, "cp1252")  # + Windows ANSI, which does carry U+2022


@pytest.fixture(scope="module")
def child_script(tmp_path_factory):
    script = tmp_path_factory.mktemp("nm0001") / "legacy_console_child.py"
    script.write_text(_CHILD, encoding="utf-8")
    return script


@pytest.fixture(scope="module")
def cli_inputs(tmp_path_factory):
    """Files that only have to exist -- typer checks `exists=True` before the command runs."""
    directory = tmp_path_factory.mktemp("nm0001-inputs")
    schema = directory / "schema.json"
    schema.write_text('{"type": "object", "properties": {}}', encoding="utf-8")
    dictionary = directory / "dictionary.csv"
    dictionary.write_text("id,business_name\nfield_001,Customer Email Address\n", encoding="utf-8")
    return schema, dictionary


def _run(child_script, argv, encoding=_CP437, columns="80", field_path=None):
    """Run the shipped CLI in a child whose stdout genuinely uses `encoding`."""
    # A code page the premise guard never checked could turn out to encode the glyphs, and
    # then this run proves nothing while still reporting a pass. Refuse rather than let a
    # gate go quietly vacuous; adding a code page means adding it to the guarded tuple.
    assert encoding in _LEGACY_CODE_PAGES, (
        f"{encoding} is not covered by the premise guard -- add it to _LEGACY_CODE_PAGES "
        f"so the guard proves it really rejects the glyphs before any gate runs under it"
    )
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = encoding
    env["PYTHONUTF8"] = "0"  # UTF-8 mode would hand the child a utf-8 stdout regardless
    env["COLUMNS"] = columns
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    if field_path is not None:
        env["NEXUS_FIELD_PATH"] = field_path

    proc = subprocess.run(
        [sys.executable, str(child_script), encoding, *argv],
        capture_output=True,
        cwd=str(child_script.parent),
        env=env,
        timeout=300,
        check=False,  # the exit code IS the assertion
    )
    text = (proc.stdout + b"\n" + proc.stderr).decode(encoding, errors="replace")
    assert proc.returncode != 97, f"code page {encoding} never took effect: {text}"
    return proc.returncode, text


def _assert_no_codec_failure(code, text):
    assert "UnicodeEncodeError" not in text, text
    assert "codec can't encode" not in text, text
    assert code == 0, f"the command exited {code} on a legacy code page:\n{text}"


def test_premise_guard_the_code_pages_under_test_reject_the_glyphs():
    """
    PREMISE GUARD, NOT A REGRESSION GATE. Nothing in `nexus_matcher` can turn this red.

    It asserts a fact about the stdlib codecs -- that every code page the tests below run
    under really cannot encode the glyphs -- and it is here because every one of those
    tests is worthless if that stops being true. Do not count it as coverage of NM-0001;
    the gates are the `*_completes_on_a_legacy_code_page` tests and the escape-text test.

    What it WOULD catch, and the reason it exists: an edit to this file that re-points the
    code page tuples above at something that can encode the glyphs -- `utf-8`, or a future
    code page that gained a Braille frame. Every gate below would then run on a stream that
    never had the problem, pass without exercising a single line of the fix, and report the
    same green as today. That edit makes this test fail, which is the only warning anyone
    would get. It reads the SAME tuples the gates are parametrized from, deliberately: when
    it kept a private copy of the code page names, the swap sailed straight past it.
    """
    for encoding in _LEGACY_CODE_PAGES:
        codecs.lookup(encoding)
        with pytest.raises(UnicodeEncodeError):
            "⠋".encode(encoding)  # frame 0 of Rich's default "dots" spinner
    for encoding in _DOS_CODE_PAGES:
        with pytest.raises(UnicodeEncodeError):
            "•".encode(encoding)  # the bullet `info` prints


@pytest.mark.parametrize("encoding", _LEGACY_CODE_PAGES)
def test_match_completes_on_a_legacy_code_page(child_script, cli_inputs, encoding):
    """The headline symptom: results, not a codec error, on a real DOS/ANSI console."""
    schema, dictionary = cli_inputs
    code, text = _run(
        child_script, ["match", str(schema), "-d", str(dictionary)], encoding=encoding
    )
    _assert_no_codec_failure(code, text)
    assert "Summary" in text, text


@pytest.mark.parametrize("encoding", _LEGACY_CODE_PAGES)
def test_sync_completes_on_a_legacy_code_page(child_script, cli_inputs, encoding):
    """`sync` additionally signs off with U+2713, which all three code pages reject."""
    _, dictionary = cli_inputs
    code, text = _run(child_script, ["sync", str(dictionary)], encoding=encoding)
    _assert_no_codec_failure(code, text)
    assert "Sync Statistics" in text, text


@pytest.mark.parametrize("encoding", _DOS_CODE_PAGES)
def test_info_completes_on_a_dos_code_page(child_script, encoding):
    """
    `info` was first reported as unaffected. That is only true on cp1252, which happens
    to carry U+2022; on the true DOS code pages the bullet list took it down as well.
    """
    code, text = _run(child_script, ["info"], encoding=encoding)
    _assert_no_codec_failure(code, text)
    assert "Deployment Modes" in text, text


def test_info_prints_readable_text_not_escape_sequences(child_script):
    """
    The second symptom, and the one a crash test cannot see.

    Retuning the stream to `backslashreplace` stops the exception but does NOT make the
    output right: an unencodable bullet becomes the literal text `\\u2022`. So a fix that
    kept only the error handler and dropped the glyph fallback would still exit 0 while
    showing the user escape codes where the panel used to be.
    """
    code, text = _run(child_script, ["info"], encoding=_CP437)
    _assert_no_codec_failure(code, text)
    assert "\\u" not in text, (
        f"`info` rendered unencodable characters as escape text instead of choosing "
        f"decoration this code page can encode:\n{text}"
    )


def test_match_survives_truncation_on_a_narrow_legacy_console(child_script, cli_inputs):
    """
    The half of the fix that picking safe glyphs does not cover.

    Rich truncates a `no_wrap` column with U+2026, which is ours to suffer and not ours
    to choose. A field path one character too long for the window therefore aborted the
    whole command on cp437 -- again after the matching work had been paid for, and with a
    codec error rather than any hint that a NAME was the problem.
    """
    schema, dictionary = cli_inputs
    code, text = _run(
        child_script,
        ["match", str(schema), "-d", str(dictionary)],
        encoding=_CP437,
        columns="40",
        field_path="customer.contact.preferences.email_address_primary",
    )
    _assert_no_codec_failure(code, text)
