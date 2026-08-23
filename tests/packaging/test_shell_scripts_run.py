"""
tests/packaging/test_shell_scripts_run.py | Env: ALL

A shell script with CRLF line endings is a script that does not run.

The evidence
------------
`scripts/publish.sh` was committed from Windows, checked out with CRLF because
`.gitattributes` said `* text=auto` and `core.autocrlf` defaults to true there, and died
on its 34th line with

    line 34: $'\r': command not found

before it uploaded anything. The release did not happen and the failure did not look like
a line-ending problem -- it looked like a syntax error in a script that had been read and
reviewed. Three other shell scripts in this repository were in the same state at the same
time, including the one that starts the fixture servers the Java integration tests need.

Git had been printing

    warning: in the working copy of 'scripts/publish.sh', LF will be replaced by CRLF

on every commit that touched one of these files, for the whole life of the branch. It is a
warning that scrolls past in a wall of identical warnings, and it was read as noise every
single time. That is the argument for a test: the signal existed and was useless, because
nothing turned it into a failure.

Why the shebang check as well
-----------------------------
The `\r` lands at the end of EVERY line, so the shebang breaks first: the kernel looks for
an interpreter literally named `/usr/bin/env bash\r` and reports "no such file or
directory" naming a path that plainly exists, which is a worse error message than the one
above. A file that is executable and has a shebang is a file something will try to run.

Scope
-----
Only what a POSIX tool reads directly. `.ps1` is deliberately not covered -- PowerShell is
content either way, and `.gitattributes` pins it to CRLF so a cross-platform edit does not
churn the diff.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Extensionless files a POSIX tool reads directly, checked by name.
_POSIX_BY_NAME = frozenset({"Makefile"})
_POSIX_SUFFIXES = frozenset({".sh", ".bash", ".mk"})


def _tracked() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return [REPO / rel for rel in out.split("\0") if rel]


def posix_read_files() -> list[Path]:
    return sorted(
        p
        for p in _tracked()
        if p.is_file() and (p.suffix in _POSIX_SUFFIXES or p.name in _POSIX_BY_NAME)
    )


def test_there_are_posix_scripts_to_check():
    """Guards the vacuous pass: an empty list satisfies every test below trivially."""
    found = posix_read_files()
    assert found, "no POSIX-read files found; the checks below would assert nothing"
    names = {p.name for p in found}
    assert "publish.sh" in names, (
        "publish.sh is not in the scanned set -- it is the script whose CRLF endings "
        f"cost a release, so its absence means the scan is looking in the wrong place: {names}"
    )


@pytest.mark.parametrize("path", posix_read_files(), ids=lambda p: p.name)
def test_no_carriage_returns(path: Path):
    """
    Not "the file parses" -- the presence of a single CR byte anywhere.

    A CRLF script can be syntactically perfect and still fail on its first line. Checking
    bytes rather than behaviour also means this test needs no shell, so it holds on a
    machine that has no bash at all.
    """
    body = path.read_bytes()
    carriage_returns = body.count(b"\r")
    assert carriage_returns == 0, (
        f"{path.relative_to(REPO).as_posix()} contains {carriage_returns} carriage return "
        "byte(s). bash reads the \\r as part of the command and fails with "
        "\"$'\\r': command not found\" on the first line that has one. "
        "`.gitattributes` pins these to `eol=lf`; if this is red, either that entry was "
        "removed or the file was written with CRLF and committed with --no-renormalize."
    )


@pytest.mark.parametrize("path", posix_read_files(), ids=lambda p: p.name)
def test_a_shebang_names_an_interpreter_that_could_exist(path: Path):
    """A trailing CR turns `#!/usr/bin/env bash` into a request for `bash\\r`."""
    first = path.read_bytes().split(b"\n", 1)[0]
    if not first.startswith(b"#!"):
        return
    assert not first.endswith(b"\r"), (
        f"{path.relative_to(REPO).as_posix()}'s shebang ends with a carriage return, so the "
        "kernel looks for an interpreter whose name ends in \\r and reports 'no such file "
        "or directory' for a path that exists."
    )


def test_gitattributes_pins_shell_scripts_to_lf():
    """
    The control. Byte-checking the files catches today's state; this catches the cause.

    Without an explicit `eol=lf`, `* text=auto` plus Windows' default `core.autocrlf=true`
    re-introduces CRLF on the next fresh clone, and every check above would pass in this
    working tree while failing on the machine that actually runs the release.
    """
    attributes = (REPO / ".gitattributes").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in attributes.splitlines() if ln.strip() and not ln.startswith("#")]
    assert any(ln.startswith("*.sh") and "eol=lf" in ln for ln in lines), (
        "`.gitattributes` does not pin `*.sh` to `eol=lf`. The byte checks above would "
        "still pass here and the scripts would still arrive broken on a fresh Windows "
        "clone, which is exactly how this shipped."
    )
