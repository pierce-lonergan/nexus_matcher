"""
NM-0004 -- a flag that was declared, documented, and wired to nothing.

`sync` took `--output-dir/-o`, printed it in its own `--help` example, and then never
mentioned it again anywhere in the file. So:

    nexus-matcher sync dictionary.csv -o ./index

loaded the dictionary, built the index, threw it away, printed "Dictionary synced
successfully" and exited 0. The user's next run rebuilt everything from scratch and they
had no way to tell, because nothing about the output distinguishes "wrote the index" from
"did not".

Two symptoms are asserted, and they are opposites on purpose:

  1. when an index CAN be persisted, `-o DIR` leaves a file in DIR;
  2. when it CANNOT, `-o DIR` fails loudly instead of exiting 0.

The second is what makes the first a real gate. Only the sparse index has a `save()`;
dense vectors live in whatever vector store was wired in, and the in-memory one cannot
persist. "Write what can be written, and say plainly what was written" is a defensible
answer to that. "Exit 0 having written nothing" is the defect.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from nexus_matcher.presentation.cli.main import app
from nexus_matcher.shared.types.base import Result

runner = CliRunner(env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"})


class _Stats:
    total_rows = 2
    valid_entries = 2
    skipped_rows = 0
    error_rows = 0
    success_rate = 1.0
    errors: ClassVar[list[str]] = []


class _PersistableSparseIndex:
    """A sparse retriever that can save itself, like the shipped BM25 one."""

    def save(self, path):
        Path(path).write_bytes(b"NM-0004 sparse index")
        return Result.success(True)


class _Matcher:
    """
    Stands in for the model-loading matcher; the defect is entirely in the CLI.

    `_sparse_retriever` is the attribute `sync` reaches for, so the stub carries one.
    """

    _sparse_retriever = _PersistableSparseIndex()

    def load_dictionary(self, path):
        return _Stats()


class _MatcherWithNothingToPersist(_Matcher):
    _sparse_retriever = None


@pytest.fixture
def dictionary(tmp_path):
    """A file that only has to exist -- typer checks `exists=True` before the command runs."""
    path = tmp_path / "dictionary.csv"
    path.write_text("id,business_name\nfield_001,Customer Email Address\n", encoding="utf-8")
    return path


def _use(monkeypatch, matcher_class):
    monkeypatch.setattr("nexus_matcher.presentation.cli.main._get_matcher", matcher_class)


def test_sync_without_an_output_dir_still_works(dictionary, monkeypatch):
    """
    The baseline. Without it, "nothing was written" and "the command is broken" would be
    indistinguishable in the tests below.
    """
    _use(monkeypatch, _Matcher)

    result = runner.invoke(app, ["sync", str(dictionary)])

    assert result.exit_code == 0, result.output
    assert "Sync Statistics" in result.output


def test_the_documented_output_dir_example_leaves_a_file_behind(tmp_path, dictionary, monkeypatch):
    """
    The exact command from `sync --help`, and the exact thing it failed to do.

    The assertion is on the DIRECTORY's contents, not on a filename: what the index is
    called is an implementation detail, whereas "the directory the user named is still
    empty" is the defect.
    """
    _use(monkeypatch, _Matcher)
    out_dir = tmp_path / "index"

    result = runner.invoke(app, ["sync", str(dictionary), "-o", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert out_dir.is_dir(), (
        f"`-o {out_dir.name}` did not even create the directory. Output was:\n{result.output}"
    )
    written = [p for p in out_dir.iterdir() if p.is_file() and p.stat().st_size > 0]
    assert written, (
        f"`-o {out_dir.name}` wrote nothing and the command still exited 0 -- the index "
        f"was built and discarded. Output was:\n{result.output}"
    )


def test_sync_says_what_it_wrote(tmp_path, dictionary, monkeypatch):
    """
    Only the sparse index round-trips, so the report must not imply the whole index did.

    Naming the path in the output is what lets a user tell "wrote the index" from "did
    not" without going to look -- which is the confusion the original defect created.
    """
    _use(monkeypatch, _Matcher)
    out_dir = tmp_path / "index"

    result = runner.invoke(app, ["sync", str(dictionary), "-o", str(out_dir)])

    assert result.exit_code == 0, result.output
    written = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    assert any(name in result.output for name in written), (
        f"sync wrote {written} but its output never names any of them:\n{result.output}"
    )


def test_an_output_dir_that_cannot_be_honoured_fails_loudly(tmp_path, dictionary, monkeypatch):
    """
    The half that keeps the test above honest.

    If a configuration has no persistable index, the only two defensible answers are "fail
    and say so" and "write something". Exiting 0 with an empty directory is how the flag
    got to be decorative in the first place, so that outcome is pinned as a failure here.
    """
    _use(monkeypatch, _MatcherWithNothingToPersist)
    out_dir = tmp_path / "index"

    result = runner.invoke(app, ["sync", str(dictionary), "-o", str(out_dir)])

    wrote_something = out_dir.is_dir() and any(
        p.is_file() and p.stat().st_size > 0 for p in out_dir.iterdir()
    )
    assert result.exit_code != 0 or wrote_something, (
        f"`-o` was accepted, nothing was persisted, and the command reported success. "
        f"Output was:\n{result.output}"
    )
