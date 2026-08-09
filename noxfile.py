"""
noxfile | Layer: GATE
Mutation testing: one command, a measured ratchet, and every survivor triaged.

## Relationships
# READS  → setup.cfg :: [mutmut], [mutmut.scope], [mutmut.group:*] -- the whole scope
# WRITES → docs/mutation_survivors.md :: nothing; it READS it, to check triage is complete
# USED_BY → docs/DEFENSIBILITY.md :: closes the "no mutation testing" hole

## Why this exists
Every other gate in this repository assumes somebody chose the right thing to assert.
Mutation testing is the only mechanical check on that assumption: it breaks the code on
purpose, one edit at a time, and asks whether any test notices. A test that passes against
both the correct and the broken implementation is worth nothing, and this is the only tool
here that can say so without a human guessing where to look. Nineteen tests in this repo
agreed with each other while both sides were wrong (H-004); that is exactly the shape of
defect this finds.

## Why the run happens in a copy of the tree
mutmut 2.x mutates files IN PLACE -- it writes the broken source into src/, runs the
tests, restores it, and does that once per mutant. On a shared machine that hands anyone
else running pytest a deliberately-broken source file thousands of times in a row, and a
hard kill mid-mutant leaves the tree broken. So the session copies the tree to a scratch
directory outside the repository and mutates that. The consequence for you: `nox -s
mutation` is safe to start while other work is in flight, and it can never leave a mutated
file behind in your checkout.

## Sessions
    nox -s mutation                  the full scoped run, then the ratchets
    nox -s mutation -- bm25          one group only (no ratchet -- partial scope)
    nox -s mutation -- --clean       discard the cached results and re-run everything
    nox -s verify-mutation-scope     prove each group's test list is not missing a killer
"""

from __future__ import annotations

import ast
import configparser
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import nox

REPO = Path(__file__).resolve().parent
SETUP_CFG = REPO / "setup.cfg"
PYPROJECT = REPO / "pyproject.toml"
SURVIVORS_DOC = REPO / "docs" / "mutation_survivors.md"

# The tree is copied here rather than mutated in place -- see the module docstring. Kept
# at a fixed path (not a fresh temp dir) so .mutmut-cache survives between runs and an
# unchanged file is not re-tested.
WORKSPACE = Path(
    os.environ.get("NEXUS_MUTATION_WORKSPACE")
    or (Path(os.environ.get("TEMP") or "/tmp") / "nexus-matcher-mutation")
)

# Copied into the workspace. tests/ imports from benchmarks/ and scripts/ in a few places,
# and pytest needs pyproject.toml for [tool.pytest.ini_options]; setup.cfg goes along so a
# bare `mutmut` invocation inside the workspace sees the same scope this session uses.
WORKSPACE_TREES = ("src", "tests", "benchmarks", "scripts")
WORKSPACE_FILES = ("pyproject.toml", "setup.cfg")

# mutmut's own statuses. Only these four are mutants that were actually decided; SKIPPED
# and UNTESTED are not, and counting them either way would move the score for reasons
# that have nothing to do with the tests.
KILLED = "ok_killed"
SURVIVED = "bad_survived"
TIMEOUT = "bad_timeout"
SUSPICIOUS = "ok_suspicious"
DECIDED = (KILLED, SURVIVED, TIMEOUT, SUSPICIOUS)

# `ok_suspicious` is a mutant the tests DID kill; mutmut only flags it because the run took
# more than test_time_multiplier x the baseline. On a box running other work that threshold
# is crossed for reasons that have nothing to do with the mutant -- this repo measured a
# 30% throughput band under load (H-007) -- so scoring it as "not killed" would make a
# correctness ratchet move with the machine's load. An assertion fired; it counts.
KILLED_STATUSES = (KILLED, SUSPICIOUS)

nox.options.sessions = ["mutation"]


# =============================================================================
# CONFIG
# =============================================================================


class Group:
    """One (files to mutate, command that must kill the mutants) pair from setup.cfg."""

    def __init__(self, name: str, paths: list[str], tests: list[str]) -> None:
        self.name = name
        self.paths = paths
        self.tests = tests

    def runner(self, base: str) -> str:
        """The base runner from [mutmut], pointed at this group's tests only."""
        return f"{base} {' '.join(self.tests)}"


def _load_config() -> tuple[configparser.ConfigParser, list[Group], float, int]:
    """
    Read the scope out of setup.cfg, and refuse to run if mutmut would not read it too.

    mutmut prefers [tool.mutmut] in pyproject.toml and only falls back to setup.cfg when
    that section is absent. If someone adds it, every value in setup.cfg stops applying
    to a bare `mutmut run` while this session keeps honouring it -- two tools with two
    different ideas of the scope, and no error. Fail loudly instead.
    """
    if not SETUP_CFG.exists():
        raise RuntimeError(f"{SETUP_CFG} is missing; mutation scope is defined there")
    if PYPROJECT.exists() and "[tool.mutmut]" in PYPROJECT.read_text(encoding="utf-8"):
        raise RuntimeError(
            "pyproject.toml now has a [tool.mutmut] section. mutmut reads that INSTEAD of "
            "setup.cfg, so the scope in setup.cfg would be silently ignored. Move the "
            "settings into one file or the other, not both."
        )

    cfg = configparser.ConfigParser()
    cfg.read(SETUP_CFG, encoding="utf-8")

    groups = []
    for name in cfg.get("mutmut.scope", "groups").split():
        section = f"mutmut.group:{name}"
        if not cfg.has_section(section):
            raise RuntimeError(f"[mutmut.scope] lists {name!r} but [{section}] is missing")
        groups.append(
            Group(
                name=name,
                paths=cfg.get(section, "paths").split(),
                tests=cfg.get(section, "tests").split(),
            )
        )

    floor = cfg.getfloat("mutmut.scope", "score_floor")
    budget = cfg.getint("mutmut.scope", "survivors_documented")
    return cfg, groups, floor, budget


# =============================================================================
# WORKSPACE
# =============================================================================


def _sync_workspace(session: nox.Session) -> None:
    """Refresh the scratch copy of the tree, keeping any existing .mutmut-cache."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for tree in WORKSPACE_TREES:
        source = REPO / tree
        if not source.exists():
            continue
        target = WORKSPACE / tree
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    for name in WORKSPACE_FILES:
        if (REPO / name).exists():
            shutil.copy2(REPO / name, WORKSPACE / name)
    session.log(f"workspace: {WORKSPACE}")


def _workspace_env() -> dict[str, str]:
    """
    Environment for everything that runs inside the workspace.

    PYTHONPATH puts the COPY of src/ ahead of the editable install's path entry, so the
    tests import the mutated file and not the pristine one in the repository. Getting this
    wrong would make every mutant survive, which is why the session asserts the import
    resolves into the workspace before it runs anything.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(WORKSPACE / "src"), str(WORKSPACE / "benchmarks")])
    # mutmut prints emoji in its legend and results; the Windows console default is cp1252
    # and raises UnicodeEncodeError on them.
    env["PYTHONIOENCODING"] = "utf-8"
    # Mutants in a scoring path can trivially produce a NaN or an infinite loop over a
    # numpy array; keep BLAS single-threaded so a runaway mutant cannot take the box down.
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    # NOT optional, and not a speed setting. CPython invalidates a .pyc by comparing the
    # source's mtime IN WHOLE SECONDS and its byte length. mutmut writes a mutant, runs the
    # tests, and restores the original -- often inside one second -- so a mutation that
    # preserves the file's length (`==` -> `!=`, `True` -> `None`) can leave the next run
    # importing the PREVIOUS mutant's bytecode. That silently mislabels mutants in both
    # directions. Observed while building this: a survivor reproduced by hand read as
    # killed until bytecode writing was turned off. mutmut sets this for its own test
    # subprocesses; it is set here too so every process this session starts is covered.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _assert_workspace_isolated(session: nox.Session, env: dict[str, str]) -> None:
    """Prove the tests will import the copy. If they import the repo, nothing is measured."""
    resolved = subprocess.run(
        [sys.executable, "-c", "import nexus_matcher; print(nexus_matcher.__file__)"],
        cwd=WORKSPACE,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if Path(resolved).resolve().parent.parent != (WORKSPACE / "src").resolve():
        session.error(
            "nexus_matcher resolves to "
            f"{resolved}, not the workspace copy. Every mutant would survive because the "
            "tests would import unmutated source. Check PYTHONPATH and any .pth files."
        )


# =============================================================================
# RESULTS
# =============================================================================


def _cache_path() -> Path:
    return WORKSPACE / ".mutmut-cache"


def _read_results() -> list[dict]:
    """
    Read every decided mutant straight out of mutmut's sqlite cache.

    `mutmut results` is meant for humans -- emoji, headings, grouping -- and parsing it
    would break the first time that formatting changed. The cache schema is three tables
    and has not moved in years.
    """
    cache = _cache_path()
    if not cache.exists():
        return []
    connection = sqlite3.connect(cache)
    try:
        rows = connection.execute(
            """
            SELECT SourceFile.filename, Line.line_number, Line.line, Mutant."index",
                   Mutant.status, Mutant.id
            FROM Mutant
            JOIN Line ON Line.id = Mutant.line
            JOIN SourceFile ON SourceFile.id = Line.sourcefile
            """
        ).fetchall()
    finally:
        connection.close()

    return [
        {
            "filename": filename.replace("\\", "/"),
            # mutmut stores line_number 0-based; every other tool here is 1-based.
            "line_number": line_number + 1,
            "line": line,
            "index": index,
            "status": status,
            "pk": pk,
        }
        for filename, line_number, line, index, status, pk in rows
        if status in DECIDED
    ]


def key_of(mutant: dict) -> str:
    """Stable identity for a survivor: file, line, and which mutation of that line."""
    return f"{mutant['filename']}:{mutant['line_number']}:{mutant['index']}"


# Below this many elements a literal is far more likely to be logic (a two-element tuple
# return, an empty accumulator) than reference data.
TABLE_MIN_ENTRIES = 4


def _literal_size(value) -> int | None:
    """Element count if this node is a literal container, else None."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        if value.func.id in {"frozenset", "set", "dict", "tuple", "list"} and value.args:
            return _literal_size(value.args[0])
        return None
    if isinstance(value, ast.Dict):
        return len(value.keys)
    if isinstance(value, (ast.Set, ast.List, ast.Tuple)):
        return len(value.elts)
    return None


def _table_lines(source: str) -> set[int]:
    """
    1-based line numbers occupied by literal lookup tables, at any scope.

    A mutant inside one of these edits a single ROW of reference data -- one abbreviation,
    one type-pair score -- not a decision. It is still a real gap, but a different KIND of
    gap from an unasserted branch, and mixing the two makes the headline score a function
    of how long the abbreviation dictionary happens to be. Both numbers are reported.

    Function-local tables count: `_infer_domain_from_path` builds its 18-entry pattern map
    inside the method body.
    """
    lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        size = _literal_size(node.value)
        if size is not None and size >= TABLE_MIN_ENTRIES:
            lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


def _print_report(
    mutants: list[dict], score: float, counts: dict[str, int], elapsed: float
) -> None:
    """The run's numbers: headline score, logic-only score, and a per-file breakdown."""
    print()
    print("=" * 78)
    print(f"  mutation score  {score:.1f}%   over {sum(counts.values())} decided mutants")
    for status in DECIDED:
        print(f"    {status:<14} {counts[status]}")
    logic = [m for m in mutants if not m["in_table"]]
    if logic:
        logic_killed = sum(m["status"] in KILLED_STATUSES for m in logic)
        print(
            f"  logic only      {100.0 * logic_killed / len(logic):.1f}%   "
            f"{logic_killed}/{len(logic)}  (rows of lookup literals excluded)"
        )
    # Timing is reported for orientation only. This box runs other work concurrently and
    # H-007 measured a 30% throughput band under load, so treat any number here as
    # unmeasured until it is re-run idle.
    print(f"  wall clock      {elapsed / 60:.1f} min  (NOT a measurement -- see H-007)")
    print("=" * 78)

    by_file: dict[str, list[int]] = {}
    for mutant in mutants:
        entry = by_file.setdefault(mutant["filename"], [0, 0, 0])
        entry[0] += 1
        entry[1] += mutant["status"] in KILLED_STATUSES
        entry[2] += mutant["status"] not in KILLED_STATUSES and mutant["in_table"]
    for filename in sorted(by_file):
        total, killed, table_survivors = by_file[filename]
        print(
            f"  {100.0 * killed / total:5.1f}%  {killed:4d}/{total:<4d}  "
            f"{table_survivors:4d} table-row survivors  {filename}"
        )
    print()


def _tag_tables(mutants: list[dict]) -> None:
    """Mark each mutant `in_table`, reading each source file once."""
    cache: dict[str, set[int]] = {}
    for mutant in mutants:
        filename = mutant["filename"]
        if filename not in cache:
            path = WORKSPACE / filename
            source = (path if path.exists() else REPO / filename).read_text(encoding="utf-8")
            cache[filename] = _table_lines(source)
        mutant["in_table"] = mutant["line_number"] in cache[filename]


def _score(mutants: list[dict]) -> tuple[float, dict[str, int]]:
    """
    Killed / decided, as a percentage.

    A `bad_timeout` counts as NOT killed. Unlike a suspicious result it carries no evidence
    that any assertion fired -- the command never finished -- so calling it a kill would
    credit the tests for a hang. It is also the one status load can manufacture, which is
    why the session warns rather than quietly scoring when any appear.
    """
    counts = dict.fromkeys(DECIDED, 0)
    for mutant in mutants:
        counts[mutant["status"]] += 1
    total = sum(counts.values())
    killed = sum(counts[status] for status in KILLED_STATUSES)
    return (100.0 * killed / total if total else 0.0), counts


def _documented_keys() -> set[str]:
    """
    Survivor keys already triaged in docs/mutation_survivors.md.

    Only cells shaped exactly like `path.py:line:index` count. The document has other
    tables whose first cell is also in backticks -- the per-file score table, the
    hand-reproduction table -- and letting those through would inflate the count of things
    this gate believes are triaged.
    """
    if not SURVIVORS_DOC.exists():
        return set()
    shape = re.compile(r"^\| `([^`]+\.py:\d+:\d+)` \|")
    return {
        match.group(1)
        for match in (
            shape.match(line.strip())
            for line in SURVIVORS_DOC.read_text(encoding="utf-8").splitlines()
        )
        if match
    }


# =============================================================================
# SESSIONS
# =============================================================================


def _undecided_count() -> int:
    """How many generated mutants mutmut has not reached a verdict on yet."""
    cache = _cache_path()
    if not cache.exists():
        return 0
    connection = sqlite3.connect(cache)
    try:
        return connection.execute(
            'SELECT count(*) FROM Mutant WHERE status = "untested"'
        ).fetchone()[0]
    finally:
        connection.close()


def _kill_tree(pid: int) -> None:
    """Kill a process and its children. mutmut runs pytest through a shell on Windows."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        subprocess.run(["pkill", "-9", "-P", str(pid)], capture_output=True, check=False)


def _run_group(
    session: nox.Session,
    group: Group,
    tests_dir: str,
    base_runner: str,
    env: dict[str, str],
    stall_minutes: float,
    max_restarts: int,
) -> None:
    """
    Run mutmut over one group's files with that group's test command.

    Restarts on a STALL rather than on a clock. mutmut 2.4.5 deadlocks on Windows: both of
    its processes drop to 0% CPU with no pytest child running and never recover -- seen
    once in a 1600-mutant run, 245 mutants from the end. A wall-clock timeout would have to
    be set long enough for a legitimate slow run and would therefore be useless; "no mutant
    has been decided in N minutes" is the actual symptom. Every verdict is already in the
    cache, so a restarted mutmut resumes where the wedged one stopped.

    A stall is REPORTED, not swallowed. If a restart makes no progress at all the session
    fails rather than looping.
    """
    command = [
        sys.executable,
        "-m",
        "mutmut",
        "run",
        "--paths-to-mutate",
        ":".join(group.paths),
        "--tests-dir",
        tests_dir,
        "--runner",
        group.runner(base_runner),
        "--simple-output",
        "--no-progress",
    ]
    for attempt in range(max_restarts + 1):
        undecided_at_start = _undecided_count()
        process = subprocess.Popen(command, cwd=WORKSPACE, env=env)
        last_seen = _undecided_count()
        last_change = time.monotonic()
        stalled = False
        while process.poll() is None:
            time.sleep(30)
            current = _undecided_count()
            if current != last_seen:
                last_seen, last_change = current, time.monotonic()
            elif time.monotonic() - last_change > stall_minutes * 60:
                session.warn(
                    f"{group.name}: mutmut has decided nothing for {stall_minutes} minutes "
                    f"with {current} mutants left. This is the known mutmut 2.4.5 Windows "
                    "deadlock; killing it and resuming from the cache."
                )
                _kill_tree(process.pid)
                stalled = True
                break
        process.wait()

        if not stalled and process.returncode & 1:
            session.error(f"mutmut failed while running group {group.name!r}")
        if _undecided_count() == 0:
            # mutmut's exit code is a bit field: 1 fatal, 2 survivors, 4 timeout, 8 slow.
            # Survivors are the POINT of the run, so only the fatal bit is an error; the
            # ratchets decide whether the survivors are acceptable.
            return
        if not stalled:
            # mutmut exited SUCCESSFULLY and still left mutants undecided. Seen for real:
            # after the cache's recorded source hash stopped matching the file, 15 of 464
            # mutants became unreachable and every further run exited 14 without touching
            # them. Retrying is right -- a normal partial run finishes them -- and the
            # no-progress check below is what stops it looping on the corrupt case.
            session.warn(
                f"{group.name}: mutmut exited cleanly with {_undecided_count()} mutants "
                "still undecided. Retrying; if the count does not move, the cache is stale "
                "for this file and needs `nox -s mutation -- --clean`."
            )
        if attempt and _undecided_count() == undecided_at_start:
            session.error(
                f"{group.name}: a restart decided nothing at all, with "
                f"{_undecided_count()} mutants left. That is not the deadlock -- the cache "
                "no longer matches the source. Re-run with `-- --clean`."
            )
        session.warn(f"{group.name}: restart {attempt + 1} of {max_restarts}")
    session.error(
        f"{group.name}: still {_undecided_count()} undecided mutants after "
        f"{max_restarts} restarts. Scoring a partial group would understate the tests."
    )


def _ratchet_failures(score: float, floor: float, survivors: list[dict], budget: int) -> list[str]:
    """Every ratchet this run breaks. Empty means the run passes."""
    failures = []
    if score < floor:
        failures.append(
            f"mutation score {score:.1f}% is below the ratchet floor {floor}%. "
            "The floor may only rise (docs/DEFENSIBILITY.md); lowering it is a written, "
            "measured decision, not a fix."
        )
    if len(survivors) > budget:
        failures.append(
            f"{len(survivors)} survivors, budget is {budget}. The budget may only fall."
        )

    documented = _documented_keys()
    untriaged = [m for m in survivors if key_of(m) not in documented]
    if untriaged:
        failures.append(
            f"{len(untriaged)} survivor(s) are not triaged in {SURVIVORS_DOC.name}. "
            "A survivor nobody has looked at is a hole nobody has looked at."
        )
        # Printed as finished table rows so triaging one is reading it and writing the last
        # cell, not hand-formatting markdown. `TABLE` means the mutant edits a row of a
        # lookup literal; anything else is a decision nothing asserted on.
        print(f"\nRows to triage in {SURVIVORS_DOC.as_posix()}:\n")
        for mutant in untriaged:
            kind = "TABLE" if mutant["in_table"] else "LOGIC"
            source_line = mutant["line"].strip().replace("|", r"\|")[:80]
            print(f"| `{key_of(mutant)}` | {kind} | `{source_line}` | TRIAGE ME |")
    return failures


@nox.session(python=False, name="mutation")
def mutation(session: nox.Session) -> None:
    """Run the scoped mutation suite and enforce the ratchets."""
    try:
        import mutmut  # noqa: F401
    except ImportError:
        session.error(
            "mutmut is not installed. `pip install mutmut==2.4.5` -- pinned exactly, see "
            "the note in setup.cfg about 3.x refusing to start on Windows."
        )
    try:
        import pytest_timeout  # noqa: F401
    except ImportError:
        # Without the plugin the runner's --timeout becomes "unrecognized arguments", the
        # test command fails for EVERY mutant, and mutmut reports a perfect 100% score.
        # That is the same defect shape as a CI step that installs the dependency whose
        # absence it is meant to detect. Refuse rather than measure nothing.
        session.error(
            "pytest-timeout is not installed, but the runner in setup.cfg passes "
            "--timeout. Without it every mutant would be reported killed. "
            "`pip install pytest-timeout`."
        )

    cfg, groups, floor, budget = _load_config()
    base_runner = cfg.get("mutmut", "runner").replace(
        "python -m pytest", f'"{sys.executable}" -m pytest', 1
    )

    posargs = list(session.posargs)
    clean = "--clean" in posargs
    if clean:
        posargs.remove("--clean")
    selected = [g for g in groups if g.name in posargs] if posargs else groups
    if posargs and not selected:
        session.error(f"no such group(s): {' '.join(posargs)}")

    _sync_workspace(session)
    env = _workspace_env()
    _assert_workspace_isolated(session, env)
    if clean and _cache_path().exists():
        _cache_path().unlink()

    started = time.monotonic()
    for group in selected:
        session.log(f"--- {group.name}: {len(group.paths)} path(s) ---")
        _run_group(
            session,
            group,
            cfg.get("mutmut", "tests_dir"),
            base_runner,
            env,
            stall_minutes=cfg.getfloat("mutmut.scope", "stall_minutes"),
            max_restarts=cfg.getint("mutmut.scope", "max_restarts"),
        )
    elapsed = time.monotonic() - started

    # Only the groups just run. The cache is shared and persistent, so an unfiltered read
    # after `nox -s mutation -- bm25` would quietly fold in whatever the last full run
    # left behind and report it as this run's score.
    wanted = tuple(p.rstrip("/") for group in selected for p in group.paths)
    mutants = [m for m in _read_results() if m["filename"].startswith(wanted)]
    if not mutants:
        session.error("no mutants were decided -- the run produced nothing to score")
    score, counts = _score(mutants)
    survivors = sorted(
        (m for m in mutants if m["status"] not in KILLED_STATUSES),
        key=lambda m: (m["filename"], m["line_number"], m["index"]),
    )
    if counts[TIMEOUT]:
        session.warn(
            f"{counts[TIMEOUT]} mutant(s) TIMED OUT. A timeout is the one outcome a loaded "
            "machine can invent, and it is scored as not-killed -- re-run this idle before "
            "believing the number, and before moving the ratchet."
        )

    _tag_tables(mutants)
    _print_report(mutants, score, counts, elapsed)

    if {g.name for g in selected} != {g.name for g in groups}:
        session.warn("partial scope -- ratchets NOT enforced. Only a full run scores the ratchet.")
        return

    failures = _ratchet_failures(score, floor, survivors, budget)
    if failures:
        session.error("\n".join(failures))

    if score > floor + 2.0:
        session.warn(
            f"score {score:.1f}% is well above the floor {floor}% -- raise score_floor in "
            "setup.cfg so the gain cannot be given back silently."
        )
    session.log(f"mutation score {score:.1f}% >= floor {floor}%; all survivors triaged")


@nox.session(python=False, name="verify-mutation-scope")
def verify_mutation_scope(session: nox.Session) -> None:
    """
    Prove no group's test list is missing a test that would have killed its mutants.

    A survivor is only evidence about the TESTS if the tests that cover that line were in
    the command. Each group here runs a subset of the suite for speed, so a line covered
    by some test outside the subset would produce a survivor that says nothing except
    "wrong command". This compares, per target file, the lines a group's command executes
    against the lines the WHOLE suite executes: any line only the whole suite reaches is
    a false-survivor factory and the session fails.

    Runs in the repository, not the workspace: nothing here mutates anything, and the full
    suite contains tests that assert about the repository's own layout (packaging, museum,
    CI) which cannot collect anywhere else. Coverage data files are still written to the
    workspace so the checkout stays clean.
    """
    _, groups, _, _ = _load_config()
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    targets = sorted({p for group in groups for p in _expand(group.paths)})
    whole = _executed_lines(session, "whole", ["tests"], targets, must_be_green=False)

    problems = []
    for group in groups:
        mine = _expand(group.paths)
        got = _executed_lines(session, group.name, group.tests, targets)
        for target in mine:
            only_whole = whole.get(target, set()) - got.get(target, set())
            marker = "OK  " if not only_whole else "GAP "
            print(
                f"  {marker}{group.name:<16} {target}: {len(only_whole)} line(s) reached "
                f"only by the full suite"
            )
            if only_whole:
                print(f"        {sorted(only_whole)}")
                problems.append(f"{group.name} / {target}")

    if problems:
        session.error(
            "these groups run a test command that does not reach every covered line of "
            "their own targets, so their survivors are not trustworthy: " + ", ".join(problems)
        )
    session.log("every group's command reaches every line the full suite reaches")


def _expand(paths: list[str]) -> list[str]:
    """Turn the configured paths (files or directories) into a flat list of module paths."""
    files: list[str] = []
    for entry in paths:
        candidate = REPO / entry
        if candidate.is_dir():
            # __init__.py is in [tool.coverage.run] omit, so it never appears in a
            # coverage report and cannot be compared. It also generates no mutants in the
            # current scope; if that changes, this check goes blind to it.
            files += [
                str(p.relative_to(REPO)).replace("\\", "/")
                for p in sorted(candidate.rglob("*.py"))
                if p.name != "__init__.py"
            ]
        else:
            files.append(entry)
    return files


def _executed_lines(
    session: nox.Session,
    label: str,
    pytest_args: list[str],
    targets: list[str],
    must_be_green: bool = True,
) -> dict[str, set[int]]:
    """
    Line numbers actually executed in each target file by the given pytest command.

    A FAILING test still executes its lines, and its coverage still belongs in the
    reference set -- leaving it out would shrink the reference and make a genuine gap read
    as "OK", which is the one direction this check must never err in. A test that could
    not be COLLECTED executed nothing, so that is fatal either way.

    `must_be_green` is False only for the whole-suite reference. For a group's own command
    a red result is fatal: mutmut decides a mutant by whether the command exits non-zero,
    so a command that is already red marks every mutant killed.
    """
    data_file = WORKSPACE / f".covscope-{label}"
    json_file = WORKSPACE / f".covscope-{label}.json"
    for stale in WORKSPACE.glob(f".covscope-{label}*"):
        stale.unlink()
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            f"--data-file={data_file}",
            "--source=nexus_matcher",
            "-m",
            "pytest",
            *pytest_args,
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    # pytest exit codes: 0 ok, 1 tests failed, 2 interrupted (collection error),
    # 3 internal error, 4 usage error, 5 nothing collected.
    if run.returncode not in (0, 1) or (must_be_green and run.returncode != 0):
        print((run.stdout or "")[-3000:])
        session.error(
            f"the '{label}' coverage run exited {run.returncode}; refusing to compare "
            "against a reference set that is missing rows."
        )
    if run.returncode == 1:
        failed = [line for line in (run.stdout or "").splitlines() if line.startswith("FAILED ")]
        session.warn(
            f"'{label}' has {len(failed)} failing test(s). Their coverage is still counted "
            "-- a red test would kill mutants once it is green, so leaving it out would "
            "under-report the gap. Failures are NOT in this lane:\n  " + "\n  ".join(failed)
        )
    # --fail-under=0 overrides [tool.coverage.report] fail_under, which otherwise makes
    # this exit 2 for a subset run. Nothing is being weakened: the project's coverage gate
    # is `make test-cov`; this call only extracts which lines ran.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "json",
            f"--data-file={data_file}",
            "-o",
            str(json_file),
            "--quiet",
            "--fail-under=0",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(json_file.read_text(encoding="utf-8"))
    executed: dict[str, set[int]] = {}
    for filename, info in report["files"].items():
        normalised = filename.replace("\\", "/")
        for target in targets:
            if normalised.endswith(target):
                executed[target] = set(info["executed_lines"])
    return executed
