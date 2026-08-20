"""
scripts.museum_replay | Layer: GATE
Reintroduce every historical defect and prove the net still catches it.

The premise
-----------
A test is not a gate. A gate is a check somebody has WATCHED go red against a known-bad
artifact. This repo has now shipped two gates that could not fail:

  * the publish workflow's CLI step, which installed the very dependencies whose absence
    was the bug, exercised the one command that still worked, and carried
    `continue-on-error: true`
  * nineteen passing tests that did not notice a transposed matrix multiply, because they
    compared two implementations that were wrong in the same way

Both looked like coverage. Neither was. So every defect in tests/museum/ ships with a way
to put it BACK, and this script asserts that doing so turns the corresponding gate red.
An entry whose replay still passes is a hole in the net, and fails the build.

Entry format
------------
tests/museum/NM-XXXX/
    defect.yaml      id, symptom, severity, the gate that missed it, the gate that
                     catches it now
    replay.py        `apply(repo_root)` mutates the tree into the known-bad state and
                     returns nothing; the runner restores from its own snapshot
    test_nm_xxxx.py  the permanent regression test, asserting the OBSERVABLE SYMPTOM

Why replay.py and not replay.patch
----------------------------------
Roughly a third of these defects are not wrong lines in src/. NM-0014 was a stale
requirements.txt, NM-0015 was a version that disagreed with the changelog, NM-0023 was a
CI job that linted three directories out of four. A unified diff against src/ cannot
express those, and forcing them into that shape would make the museum quietly claim
coverage it does not have. A replay is any callable that produces the bad state.

Safety
------
The runner snapshots every file an entry touches BEFORE applying it and restores from that
snapshot in a finally block, then verifies the tree is byte-identical afterwards. It
refuses to run on a dirty tree for the files it intends to touch, because a crash midway
through would otherwise leave a deliberately-broken repo behind.

Usage
-----
    python scripts/museum_replay.py             # every entry
    python scripts/museum_replay.py NM-0005     # one entry
    python scripts/museum_replay.py --list
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MUSEUM = REPO / "tests" / "museum"


@dataclass
class Entry:
    id: str
    directory: Path
    symptom: str
    severity: str
    test_file: Path
    touches: list[str]

    @property
    def replay(self) -> Path:
        return self.directory / "replay.py"


def _parse_yaml(text: str) -> dict:
    """
    Minimal YAML reader for defect.yaml: flat `key: value` plus `key:` + `- item` lists.

    A dependency-free reader keeps the gate runnable in the bare environment the packaging
    tests create, where PyYAML is not installed. defect.yaml is deliberately kept flat so
    this stays honest rather than growing into a bad YAML parser.
    """
    data: dict = {}
    current_list: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("- ") and current_list:
            data[current_list].append(line.lstrip()[2:].strip().strip("\"'"))
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip("\"'")
        if value:
            data[key] = value
            current_list = None
        else:
            data[key] = []
            current_list = key
    return data


def discover() -> list[Entry]:
    entries: list[Entry] = []
    if not MUSEUM.exists():
        return entries
    for directory in sorted(p for p in MUSEUM.iterdir() if p.is_dir() and p.name.startswith("NM-")):
        meta_path = directory / "defect.yaml"
        if not meta_path.exists():
            raise SystemExit(f"{directory.name}: defect.yaml missing")
        meta = _parse_yaml(meta_path.read_text(encoding="utf-8"))
        tests = list(directory.glob("test_nm_*.py"))
        if len(tests) != 1:
            raise SystemExit(
                f"{directory.name}: expected exactly one test_nm_*.py, found {len(tests)}"
            )
        if meta.get("id") != directory.name:
            raise SystemExit(f"{directory.name}: defect.yaml id is {meta.get('id')!r}")
        entries.append(
            Entry(
                id=directory.name,
                directory=directory,
                symptom=meta.get("symptom", ""),
                severity=meta.get("severity", "UNKNOWN"),
                test_file=tests[0],
                touches=meta.get("touches", []),
            )
        )
    return entries


def _load_replay(entry: Entry):
    spec = importlib.util.spec_from_file_location(
        f"replay_{entry.id.replace('-', '_')}", entry.replay
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"{entry.id}: cannot load replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "apply"):
        raise SystemExit(f"{entry.id}: replay.py has no apply(repo_root)")
    return module


def _run_test(entry: Entry) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(entry.test_file),
            "-q",
            "--no-header",
            "--no-cov",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )


def _snapshot(paths: list[str]) -> dict[str, bytes | None]:
    """Record exact bytes (or absence) so restore is exact, not 'git checkout'-ish."""
    snap: dict[str, bytes | None] = {}
    for rel in paths:
        target = REPO / rel
        snap[rel] = target.read_bytes() if target.exists() else None
    return snap


def _restore(snapshot: dict[str, bytes | None]) -> None:
    for rel, blob in snapshot.items():
        target = REPO / rel
        if blob is None:
            if target.exists():
                target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)


def replay_one(entry: Entry, verbose: bool = False) -> tuple[bool, str]:
    """
    Apply the defect, run its gate, and require the gate to FAIL.

    Returns (ok, detail). ok means the museum entry did its job: green before, red during,
    green after.
    """
    if not entry.touches:
        return False, "defect.yaml declares no `touches:` list, so nothing can be restored"

    before = _run_test(entry)
    if before.returncode != 0:
        return False, "the gate is ALREADY red on a clean tree -- fix that first"

    snapshot = _snapshot(entry.touches)
    module = _load_replay(entry)
    try:
        module.apply(REPO)
        during = _run_test(entry)
    except Exception as exc:
        # A replay that no longer applies is itself a hole, and the commonest way for one
        # to rot: the code it patches gets refactored and the anchor stops matching. It
        # must be REPORTED as a hole, not allowed to abort the sweep and hide every entry
        # after it.
        _restore(snapshot)
        return False, f"replay could not be applied ({type(exc).__name__}: {exc})"
    finally:
        _restore(snapshot)

    after = _run_test(entry)
    if after.returncode != 0:
        return False, "the tree did not restore cleanly -- the gate is still red afterwards"

    if during.returncode == 0:
        tail = (during.stdout or "")[-400:]
        return False, f"REPLAY DID NOT TRIP THE GATE -- this defect could ship again.\n{tail}"

    return True, "green -> red -> green"


_LOCK = REPO / ".museum-replay.lock"


@contextlib.contextmanager
def _exclusive_run() -> Iterator[None]:
    """Refuse to start if another replay is already mutating the tree.

    This runner injects a defect into real source, runs pytest, and restores from a byte
    snapshot. That is safe alone and corrupting in parallel: two runs interleave, one
    restores a snapshot it took while the other's defect was applied, and the tree is left
    holding an injected defect nobody typed. Several are pure DELETIONS -- NM-0025 removes
    a line rather than adding a marker -- so nothing greps for evidence and a commit taken
    mid-run ships a re-introduced museum defect with every gate green.

    That is not hypothetical: three concurrent lanes hit it in one session, one had to
    `git checkout` a file it did not own to remove another's injection, and each saw
    disjoint false HOLEs.

    An advisory lock file, not a mutex: this is a developer tool, and the failure it must
    prevent is two well-meaning runs, not an adversary. O_EXCL makes the check atomic, and
    a stale lock names the pid that left it so a human can judge rather than guess.
    """
    try:
        fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        held = _LOCK.read_text(encoding="utf-8").strip() or "an unrecorded process"
        raise SystemExit(
            f"another museum replay is running (started by {held}). Two replays "
            "cannot share a working tree: each injects a defect into real source and "
            "restores from its own snapshot, so interleaving them leaves the tree "
            "holding a defect nobody typed. "
            f"If nothing is running, delete {_LOCK.name} and re-run."
        ) from None
    try:
        os.write(fd, f"pid {os.getpid()}".encode())
        os.close(fd)
        yield
    finally:
        _LOCK.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", nargs="*", help="only these entries (e.g. NM-0005)")
    ap.add_argument("--list", action="store_true", help="list entries and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    entries = discover()
    if args.ids:
        wanted = {i.upper() for i in args.ids}
        entries = [e for e in entries if e.id in wanted]
        missing = wanted - {e.id for e in entries}
        if missing:
            raise SystemExit(f"unknown entries: {', '.join(sorted(missing))}")

    if args.list:
        for e in entries:
            print(f"  {e.id}  [{e.severity:8}] {e.symptom[:88]}")
        print(f"\n  {len(entries)} entries")
        return 0

    if not entries:
        print("No museum entries found. That is itself a hole -- seed tests/museum/.")
        return 1

    print(f"\nMuseum replay: {len(entries)} entries")
    print("=" * 78)
    failures: list[tuple[Entry, str]] = []
    # Held across the WHOLE loop, not per entry: two runs alternating entry-by-entry
    # corrupt the tree just as thoroughly as two running the same one.
    with _exclusive_run():
        for entry in entries:
            ok, detail = replay_one(entry, args.verbose)
            mark = "PASS" if ok else "HOLE"
            print(f"  {mark}  {entry.id}  {entry.symptom[:60]}")
            if not ok:
                print(f"        {detail.splitlines()[0]}")
                failures.append((entry, detail))

    print("=" * 78)
    print(f"  {len(entries) - len(failures)} replayed and caught, {len(failures)} holes")
    if failures:
        print("\nHOLES IN THE NET -- these defects can ship again:")
        for entry, detail in failures:
            print(f"\n  {entry.id}: {entry.symptom}")
            for line in detail.splitlines():
                print(f"      {line}")
        return 1
    print("\n  every historical defect turns its gate red")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
