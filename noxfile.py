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

## Why the population is pinned, not just the score
A mutation score is a fraction, and both ratchets in setup.cfg are satisfied by shrinking
the denominator: a smaller scope still has a score, and it has strictly fewer survivors.
So `scope_mutants` pins the file list AND the per-file mutant count that was measured, and
this session refuses to start when the configured paths no longer expand to that list and
refuses to score when the run did not reach a verdict on those counts. The consequence for
you: "the score went up" can never mean "we measured less", and a run cut short reports
TRUNCATED instead of a number.

## Sessions
    nox -s mutation                  the full scoped run, then the ratchets
    nox -s mutation -- bm25          one group only (no ratchet -- partial scope)
    nox -s mutation -- --clean       discard the cached results and re-run everything
    nox -s mutation -- --score-only  re-check the ratchets against the cached verdicts
    nox -s verify-mutation-scope     prove each group's test list is not missing a killer
"""

from __future__ import annotations

import ast
import configparser
import hashlib
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


class Scope:
    """
    Everything [mutmut.scope] pins: the groups, the two ratchets, and the POPULATION.

    `mutants` is the third ratchet and the one that makes the other two mean anything --
    {source file: how many mutants the pinned run decided}. Without it, `score_floor` and
    `survivors_documented` are both satisfied by deleting a path from the scope, because a
    smaller population still has a score and has fewer survivors.
    """

    def __init__(
        self,
        cfg: configparser.ConfigParser,
        groups: list[Group],
        floor: float,
        budget: int,
        mutants: dict[str, int],
    ) -> None:
        self.cfg = cfg
        self.groups = groups
        self.floor = floor
        self.budget = budget
        self.mutants = mutants

    @property
    def files(self) -> list[str]:
        """Every file that must be mutated, pinned in setup.cfg."""
        return sorted(self.mutants)

    @property
    def total_mutants(self) -> int:
        return sum(self.mutants.values())

    def pinned_for(self, groups: list[Group]) -> dict[str, int]:
        """
        The pinned counts for just these groups' files.

        Normalises through `_scope_files`, the same way `_assert_scope_pinned` built the
        list it checked `mutants` against -- so this cannot raise KeyError for a path
        spelled with a trailing slash or a backslash in setup.cfg.
        """
        return {f: self.mutants[f] for group in groups for f in _scope_files(group.paths)}


def _scope_files(paths: list[str]) -> list[str]:
    """The configured paths (files or directories) as a sorted, normalised file list."""
    return sorted({p.replace("\\", "/").rstrip("/") for p in _expand(paths)})


def _pinned_mutants(cfg: configparser.ConfigParser) -> dict[str, int]:
    """Parse [mutmut.scope] scope_mutants -- one `<count> <path>` per line."""
    pinned: dict[str, int] = {}
    for raw in cfg.get("mutmut.scope", "scope_mutants").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        count, _, path = line.partition(" ")
        path = path.strip().replace("\\", "/")
        if not count.isdigit() or not path:
            raise RuntimeError(f"scope_mutants line {raw.strip()!r} is not '<count> <path>'")
        pinned[path] = int(count)
    if not pinned:
        raise RuntimeError(
            "[mutmut.scope] scope_mutants is empty, so the population is unpinned and the "
            "score could be computed over anything at all."
        )
    return pinned


def _assert_scope_pinned(cfg: configparser.ConfigParser, groups: list[Group]) -> dict[str, int]:
    """
    Refuse to run at all unless the scope is exactly the population that was measured.

    This is the fix for a ratchet that could be satisfied by measuring less. `score_floor`
    and `survivors_documented` are both fractions of, or counts within, whatever happened
    to be mutated -- delete a path and both improve. Three descriptions of the scope have
    to agree before anything runs:

      * each group's `paths`            -- what `nox -s mutation` actually mutates
      * [mutmut] `paths_to_mutate`      -- what a bare `mutmut run` would mutate
      * [mutmut.scope] `scope_mutants`  -- what was measured when the ratchets were set

    Growing the scope fails here too, on purpose: the new file's mutant count and both
    ratchets have to be re-measured in the same edit or the three numbers stop describing
    the same population.
    """
    pinned = _pinned_mutants(cfg)
    grouped = _scope_files([p for group in groups for p in group.paths])
    native = _scope_files(cfg.get("mutmut", "paths_to_mutate").split(":"))

    problems = []
    if native != grouped:
        problems.append(
            "[mutmut] paths_to_mutate and the union of the group paths do not name the "
            "same files. A bare `mutmut run` uses the first, this session uses the second."
            f"\n      only in paths_to_mutate: {sorted(set(native) - set(grouped))}"
            f"\n      only in the groups:      {sorted(set(grouped) - set(native))}"
        )
    dropped = sorted(set(pinned) - set(grouped))
    if dropped:
        problems.append(
            f"the scope SHRANK -- {dropped} is pinned in scope_mutants but is no longer "
            "mutated. Measuring less is not a way to satisfy score_floor or "
            "survivors_documented; both of them improve when the scope shrinks."
        )
    added = sorted(set(grouped) - set(pinned))
    if added:
        problems.append(
            f"the scope grew -- {added} is mutated but not pinned in scope_mutants. "
            "Widening is a good change and it still stops here: record the new file's "
            "mutant count, and re-measure score_floor and survivors_documented in the "
            "same edit, so all three keep describing the same population."
        )
    absent = sorted(f for f in pinned if not (REPO / f).exists())
    if absent:
        problems.append(
            f"pinned scope file(s) do not exist: {absent}. A path that resolves to nothing "
            "generates no mutants and would otherwise read as a clean run."
        )
    if problems:
        raise RuntimeError(
            "the mutation scope no longer matches the population pinned in setup.cfg "
            "[mutmut.scope]:\n    - " + "\n    - ".join(problems)
        )
    return pinned


def _load_config() -> Scope:
    """
    Read the scope out of setup.cfg, and refuse to run if mutmut would not read it too.

    mutmut prefers [tool.mutmut] in pyproject.toml and only falls back to setup.cfg when
    that section is absent. If someone adds it, every value in setup.cfg stops applying
    to a bare `mutmut run` while this session keeps honouring it -- two tools with two
    different ideas of the scope, and no error. Fail loudly instead.

    Every session goes through here, so the scope pin below is not something a caller can
    forget to ask for.
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

    return Scope(
        cfg=cfg,
        groups=groups,
        floor=cfg.getfloat("mutmut.scope", "score_floor"),
        budget=cfg.getint("mutmut.scope", "survivors_documented"),
        mutants=_assert_scope_pinned(cfg, groups),
    )


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


def _decided_by_file() -> dict[str, int]:
    """
    How many mutants the cache holds a VERDICT for, per source file.

    Completeness is measured against the pinned counts in setup.cfg, not against mutmut's
    own `untested` rows, because those rows cannot answer the question. mutmut writes them
    only when it (re)registers a file, and it skips registration entirely when the file's
    hash has not changed -- so on every resumed run the untested count reads 0 from the
    first second, whether or not there is work left to do. The old watchdog watched exactly
    that number: it never moved, so it declared healthy runs deadlocked, killed them, saw
    "0 undecided", and scored the truncated population as if it were the whole thing. A
    silently partial mutation score is the worst output this session can produce, because
    it looks like evidence.

    (The query it used also said `status = "untested"`. SQLite resolves a double-quoted
    token as an IDENTIFIER first and only falls back to a string literal when no column of
    that name exists, so that comparison would have silently become a column reference the
    day the schema grew one. There is no status literal in SQL here at all now -- the
    tally is done in Python against the DECIDED tuple this module already defines.)
    """
    cache = _cache_path()
    if not cache.exists():
        return {}
    connection = sqlite3.connect(cache)
    try:
        rows = connection.execute(
            """
            SELECT SourceFile.filename, Mutant.status
            FROM Mutant
            JOIN Line ON Line.id = Mutant.line
            JOIN SourceFile ON SourceFile.id = Line.sourcefile
            """
        ).fetchall()
    finally:
        connection.close()

    decided: dict[str, int] = {}
    for filename, status in rows:
        if status in DECIDED:
            name = filename.replace("\\", "/")
            decided[name] = decided.get(name, 0) + 1
    return decided


def _missing_verdicts(pinned: dict[str, int]) -> dict[str, tuple[int, int]]:
    """{file: (decided, pinned)} for every pinned file the cache is short on."""
    decided = _decided_by_file()
    return {
        filename: (decided.get(filename, 0), expected)
        for filename, expected in pinned.items()
        if decided.get(filename, 0) < expected
    }


def _stale_sources(files: list[str]) -> list[str]:
    """
    Pinned files whose cached verdicts were recorded against different source bytes.

    mutmut stores sha256 of each file it registered. A normal run syncs the repository
    into the workspace before it starts, so this is a no-op there. It is the guard that
    stops `--score-only` from being a way around measuring: a cache full of verdicts about
    yesterday's source has the right SHAPE -- every pinned mutant decided -- and says
    nothing about the code in the tree today.
    """
    cache = _cache_path()
    if not cache.exists():
        return []
    connection = sqlite3.connect(cache)
    try:
        recorded = {
            filename.replace("\\", "/"): digest
            for filename, digest in connection.execute("SELECT filename, hash FROM SourceFile")
        }
    finally:
        connection.close()
    stale = []
    for filename in files:
        source = REPO / filename
        if filename not in recorded or not source.exists():
            continue  # absence is reported by the completeness check, with a better message
        if hashlib.sha256(source.read_bytes()).hexdigest() != recorded[filename]:
            stale.append(filename)
    return stale


def _verdict_fingerprint() -> str:
    """
    A value that changes the moment mutmut records ANY verdict, for any mutant.

    This is the liveness signal for the deadlock watchdog, and it has to move in every
    case where real work is happening. A count of decided mutants is not enough: when the
    TESTS change but the source does not, mutmut re-tests the entire population and the
    count never moves, even over hours. `tested_against_hash` is rewritten on every
    re-test, so folding it in covers that case. Under the actual deadlock -- both mutmut
    processes at 0% CPU with no pytest child -- nothing is written and this stands still.
    """
    cache = _cache_path()
    if not cache.exists():
        return "no-cache"
    connection = sqlite3.connect(cache)
    try:
        digest = hashlib.blake2b(digest_size=16)
        for pk, status, tested_against in connection.execute(
            "SELECT id, status, tested_against_hash FROM Mutant ORDER BY id"
        ):
            digest.update(f"{pk}\0{status}\0{tested_against}\n".encode())
        return digest.hexdigest()
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
    pinned: dict[str, int],
    stall_minutes: float,
    max_restarts: int,
) -> None:
    """
    Run mutmut over one group's files with that group's test command.

    Restarts on a STALL rather than on a clock. mutmut 2.4.5 deadlocks on Windows: both of
    its processes drop to 0% CPU with no pytest child running and never recover -- seen
    once in a 1600-mutant run, 245 mutants from the end. A wall-clock timeout would have to
    be set long enough for a legitimate slow run and would therefore be useless; "nothing
    has been written to the cache in N minutes" is the actual symptom. Every verdict is
    already in the cache, so a restarted mutmut resumes where the wedged one stopped.

    Two things make that safe, and neither was true before:

    * The watchdog is only ARMED while this group still has mutants without a verdict. If
      the pinned population is already complete, mutmut is walking cached results and has
      nothing to write -- silence there is not a deadlock, and killing it is what used to
      cut healthy runs short.
    * "Done" means every pinned mutant has a verdict, not "mutmut's untested count is
      zero". That count is zero from the first second of any resumed run, so it agreed
      with every truncated run that it had finished.

    A run that ends short is REPORTED as TRUNCATED and the session fails; it is never
    handed on to be scored.
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

    def outstanding() -> int:
        """Pinned mutants in this group that still have no verdict."""
        return sum(want - got for got, want in _missing_verdicts(pinned).values())

    for attempt in range(max_restarts + 1):
        outstanding_at_start = outstanding()
        process = subprocess.Popen(command, cwd=WORKSPACE, env=env)
        fingerprint = _verdict_fingerprint()
        last_change = time.monotonic()
        stalled = False
        warned_idle = False
        while process.poll() is None:
            time.sleep(30)
            current = _verdict_fingerprint()
            if current != fingerprint:
                fingerprint, last_change = current, time.monotonic()
                continue
            if time.monotonic() - last_change <= stall_minutes * 60:
                continue
            if not outstanding():
                # Every pinned mutant already has a verdict, so mutmut is replaying the
                # cache and has nothing to write. Silence here is normal. Killing it was
                # what silently truncated healthy runs, so wait it out instead -- and say
                # so, because a run that sits here for long really has wedged.
                if not warned_idle:
                    session.warn(
                        f"{group.name}: nothing written to the cache for {stall_minutes} "
                        "minutes, but every pinned mutant already has a verdict -- mutmut "
                        "is replaying cached results, not deadlocked. Waiting."
                    )
                    warned_idle = True
                last_change = time.monotonic()
                continue
            session.warn(
                f"{group.name}: nothing written to the cache for {stall_minutes} minutes "
                f"with {outstanding()} pinned mutants still undecided. This is the known "
                "mutmut 2.4.5 Windows deadlock; killing it and resuming from the cache."
            )
            _kill_tree(process.pid)
            stalled = True
            break
        process.wait()

        if not stalled and process.returncode & 1:
            session.error(f"mutmut failed while running group {group.name!r}")
        left = outstanding()
        if left == 0:
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
                f"{group.name}: mutmut exited cleanly with {left} pinned mutants still "
                "undecided. Retrying; if the count does not move, the cache is stale for "
                "this file and needs `nox -s mutation -- --clean`."
            )
        if attempt and left == outstanding_at_start:
            session.error(
                f"TRUNCATED: {group.name} -- a restart decided nothing at all, with {left} "
                "pinned mutants left. That is not the deadlock: either the cache no longer "
                "matches the source (re-run with `-- --clean`) or scope_mutants in "
                "setup.cfg pins more mutants than this source now generates."
            )
        session.warn(f"{group.name}: restart {attempt + 1} of {max_restarts}")
    session.error(
        f"TRUNCATED: {group.name} still has {outstanding()} of its pinned "
        f"{sum(pinned.values())} mutants without a verdict after {max_restarts} restarts. "
        "Refusing to score a partial population -- a mutation score computed over less "
        "than the scope is not a weaker result, it is a misleading one."
    )


def _ratchet_failures(score: float, survivors: list[dict], decided: int, scope: Scope) -> list[str]:
    """Every ratchet this run breaks. Empty means the run passes."""
    floor, budget = scope.floor, scope.budget
    failures = []
    # The population ratchet, checked first because the other two are meaningless without
    # it: both of them improve when the scope shrinks, so a score over fewer mutants than
    # were pinned is not a smaller measurement, it is a different one wearing the same
    # number. (Per-file completeness is asserted before anything is scored; this is the
    # same fact stated where a reader looks for the ratchets.)
    if decided != scope.total_mutants:
        failures.append(
            f"{decided} mutants decided, but setup.cfg pins {scope.total_mutants} for this "
            "scope. score_floor and survivors_documented are both satisfied by measuring "
            "less, so the population is pinned too; re-measure all three together."
        )
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


def _assert_tools_present(session: nox.Session) -> None:
    """Refuse to measure anything with a toolchain that would fake a perfect score."""
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


def _assert_verdicts_are_evidence(session: nox.Session, pinned: dict[str, int]) -> None:
    """
    Refuse to score until the cached verdicts are about the whole scope and today's source.

    A partial run is not a weaker result than a complete one, it is a misleading one: it
    prints a percentage that looks exactly like evidence and is computed over an unstated
    subset. So is a complete-looking run whose verdicts describe source that has since
    changed. This runs for partial-scope invocations too -- the ratchets are skipped there,
    but the number printed still has to cover the whole of what was asked for.
    """
    stale = _stale_sources(sorted(pinned))
    if stale:
        session.error(
            "STALE -- refusing to score. The cached verdicts for these files were recorded "
            "against different source bytes than the ones in the tree now:\n  "
            + "\n  ".join(stale)
            + "\n  Run `nox -s mutation` so they are re-tested."
        )
    short = _missing_verdicts(pinned)
    if not short:
        return
    lines = "\n  ".join(
        f"{filename}: {got} of {want} decided" for filename, (got, want) in sorted(short.items())
    )
    session.error(
        "TRUNCATED -- refusing to score. These files have fewer verdicts in the cache "
        f"than setup.cfg pins:\n  {lines}\n  Re-run `nox -s mutation`; if the shortfall "
        "does not close, the cache is stale (`-- --clean`) or scope_mutants no longer "
        "matches the source."
    )


@nox.session(python=False, name="mutation")
def mutation(session: nox.Session) -> None:
    """Run the scoped mutation suite and enforce the ratchets."""
    _assert_tools_present(session)
    scope = _load_config()
    cfg, groups = scope.cfg, scope.groups
    base_runner = cfg.get("mutmut", "runner").replace(
        "python -m pytest", f'"{sys.executable}" -m pytest', 1
    )

    posargs = list(session.posargs)
    clean = "--clean" in posargs
    if clean:
        posargs.remove("--clean")
    # Re-check the ratchets against the verdicts already in the cache, without running
    # mutmut again. This is not a way to pass without measuring: the completeness check
    # below holds the cache to the population pinned in setup.cfg, so the only cache that
    # can be scored is one a real run produced.
    score_only = "--score-only" in posargs
    if score_only:
        posargs.remove("--score-only")
    if clean and score_only:
        session.error("--clean deletes the verdicts that --score-only exists to read")
    selected = [g for g in groups if g.name in posargs] if posargs else groups
    if posargs and not selected:
        session.error(f"no such group(s): {' '.join(posargs)}")

    started = time.monotonic()
    if score_only:
        session.warn(
            f"--score-only: mutmut is NOT being run. Scoring the verdicts already in "
            f"{_cache_path()}."
        )
    else:
        _sync_workspace(session)
        env = _workspace_env()
        _assert_workspace_isolated(session, env)
        if clean and _cache_path().exists():
            _cache_path().unlink()

        for group in selected:
            session.log(f"--- {group.name}: {len(group.paths)} path(s) ---")
            _run_group(
                session,
                group,
                cfg.get("mutmut", "tests_dir"),
                base_runner,
                env,
                scope.pinned_for([group]),
                stall_minutes=cfg.getfloat("mutmut.scope", "stall_minutes"),
                max_restarts=cfg.getint("mutmut.scope", "max_restarts"),
            )
    elapsed = time.monotonic() - started

    _assert_verdicts_are_evidence(session, scope.pinned_for(selected))

    # Only the groups just run. The cache is shared and persistent, so an unfiltered read
    # after `nox -s mutation -- bm25` would quietly fold in whatever the last full run
    # left behind and report it as this run's score.
    wanted = tuple(p.rstrip("/") for group in selected for p in group.paths)
    mutants = [m for m in _read_results() if m["filename"].startswith(wanted)]
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

    failures = _ratchet_failures(score, survivors, len(mutants), scope)
    if failures:
        session.error("\n".join(failures))

    if score > scope.floor + 2.0:
        session.warn(
            f"score {score:.1f}% is well above the floor {scope.floor}% -- raise score_floor "
            "in setup.cfg so the gain cannot be given back silently."
        )
    session.log(
        f"mutation score {score:.1f}% >= floor {scope.floor}% over the pinned "
        f"{scope.total_mutants} mutants; all survivors triaged"
    )


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
    scope = _load_config()
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    # The PINNED file list, not whatever the paths happen to expand to today. If the two
    # disagree _load_config has already refused to run.
    targets = scope.files
    whole = _executed_lines(session, "whole", ["tests"], targets, must_be_green=False)

    problems = []
    for group in scope.groups:
        got = _executed_lines(session, group.name, group.tests, targets)
        for target in _scope_files(group.paths):
            marker, detail = _reachability(target, whole, got)
            print(f"  {marker:<6} {group.name:<16} {target}: {detail}")
            if marker != "OK":
                problems.append(f"{group.name} / {target} ({marker})")

    if problems:
        session.error(
            "these groups run a test command that does not reach every covered line of "
            "their own targets, or were not measured at all, so their survivors are not "
            "trustworthy: " + ", ".join(problems)
        )
    session.log("every group's command reaches every line the full suite reaches")


def _reachability(
    target: str, whole: dict[str, set[int]], got: dict[str, set[int]]
) -> tuple[str, str]:
    """
    One target's verdict: OK, GAP, or NODATA.

    ABSENCE IS NOT AGREEMENT. The comparison used to be
    `whole.get(target, set()) - got.get(target, set())`, which is the empty set when the
    target is simply missing from the coverage map -- so a renamed path, a typo in
    setup.cfg, or a file coverage never measured printed `OK  0 line(s)` and the session
    passed. That is the same defect this whole session exists to prevent, one level up:
    a check that reports "no gap" when what it actually has is "no data". Every way of
    having nothing to compare is a failure now.
    """
    if target not in whole:
        return "NODATA", (
            "the whole-suite coverage run has no rows for this file at all -- it is not "
            "under --source=nexus_matcher, or the path in setup.cfg does not exist"
        )
    if not whole[target]:
        return "NODATA", (
            "the whole suite executes no line of this file, so there is no reference set "
            "to compare against and every mutant in it survives for lack of coverage"
        )
    if target not in got:
        return "NODATA", "this group's own coverage run has no rows for this file at all"
    only_whole = whole[target] - got[target]
    if only_whole:
        return "GAP", (
            f"{len(only_whole)} line(s) reached only by the full suite: {sorted(only_whole)}"
        )
    return "OK", f"0 of {len(whole[target])} covered line(s) reached only by the full suite"


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
    return _executed_from_report(json.loads(json_file.read_text(encoding="utf-8")), targets)


def _executed_from_report(report: dict, targets: list[str]) -> dict[str, set[int]]:
    """
    {target: lines coverage recorded as executed}, for the targets the report mentions.

    A target the report does not mention is ABSENT from the result rather than present
    with an empty set, and `_reachability` treats those two as different things -- see the
    note there. Split out from the run so the comparison can be exercised against a saved
    report without spending a whole test suite to produce one.
    """
    executed: dict[str, set[int]] = {}
    for filename, info in report["files"].items():
        normalised = filename.replace("\\", "/")
        for target in targets:
            if normalised.endswith(target):
                executed[target] = set(info["executed_lines"])
    return executed
