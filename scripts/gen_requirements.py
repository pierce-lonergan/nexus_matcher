"""
scripts.gen_requirements | Layer: PACKAGING
Generate requirements.txt FROM pyproject.toml, so the two cannot disagree.

Why this exists
---------------
H-006 occurrence 2: `requirements.txt` still carried both packaging defects that had
already been fixed in `pyproject.toml`, because the two files sat in different lanes and
nothing tied them together. Noticing drift later is not a fix -- the file is hand-written,
so it drifts again the moment anyone edits dependencies.

It had drifted again by 2026-08-09. The checked-in file was headed "v2.0.0", declared
`pandas` (replaced by openpyxl), declared `rank-bm25` as a core need (BM25 is built in and
imports nothing), and was missing `onnxruntime`, `tokenizers` and `tomli` -- the three
dependencies that make `pip install nexus-matcher` run the documented quickstart at all.
A user following start.sh got an environment that could not load the bundled encoder.

So requirements.txt is now a BUILD ARTIFACT. `pyproject.toml` is the single declaration;
this script renders it; `tests/packaging/test_requirements_derivation.py` regenerates and
diffs on every test run. Drift becomes impossible rather than merely noticed later.

Usage
-----
    python scripts/gen_requirements.py            # rewrite requirements.txt
    python scripts/gen_requirements.py --check    # exit 1 if it is stale, print the diff
    python scripts/gen_requirements.py --stdout   # print, write nothing
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"

# The core dependency table, spelled as a group name so composition below reads uniformly.
CORE = "<core>"


# =============================================================================
# COMPOSITION -- what requirements.txt is FOR
# =============================================================================
# requirements.txt is not "the core dependencies". Its consumers decide its contents:
# `start.sh` / `start.ps1` build a venv from it and then run BOTH pytest and
# `uvicorn nexus_matcher.presentation.api.app:app`, and the Dockerfile in
# docs/DEPLOYMENT.md does `pip wheel -r requirements.txt`. Narrowing it to the core
# dependencies would break every one of those without a single test noticing, which is
# the same cross-lane shape H-006 describes.
#
# So the composition is declared here, as a list of extras, and everything inside those
# extras follows automatically. A hand-picked subset of an extra is exactly the
# hand-maintenance that drifts, so there is no mechanism to express one.
COMPOSITION: dict[str, list[str]] = {
    "requirements.txt": [
        CORE,
        "parsers",  # start.sh serves the API, which parses uploaded Avro/JSON/DDL
        "loaders",  # the documented quickstart loads an .xlsx dictionary
        "sparse",  # the dev tests pin the built-in BM25 scores against rank-bm25
        "api",  # start.sh's whole purpose is `uvicorn ...presentation.api.app:app`
        "cli",  # redundant with core today; kept so a future core change cannot drop it
        "dev",  # start.sh runs the test suite before it starts anything
    ],
}

# Every extra NOT in a composition needs a reason here. A new extra that is neither
# included nor excused fails the test, so nobody can add one and leave the question of
# whether it belongs in requirements.txt unanswered.
EXCLUDED: dict[str, str] = {
    "accel": (
        "blake3 publishes no wheel for every Python this package classifies as supported, "
        "so a hard requirement on it turns `pip install -r requirements.txt` into a "
        "compiler error. The hand-written file commented blake3 out for exactly this "
        "reason. rapidfuzz -- the part of `accel` that is on the hot path -- is already a "
        "core dependency, so excluding the extra costs nothing measurable."
    ),
    "embeddings": "pulls torch; the bundled int8 encoder is the default and needs none of it",
    "quantization": "pulls torch; the export path is a developer tool, not a runtime need",
    "colbert": (
        "ragatouille is ResolutionImpossible on CPython 3.12+ (see the note on the extra "
        "in pyproject.toml). Putting it here would make a plain `make install` fail."
    ),
    "vector-stores": "qdrant/usearch are alternative backends; the default store is in-memory",
    "static-embeddings": "an alternative provider; the bundled encoder is the default",
    "graph": "graph_matcher guards its import and is inert without networkx",
    "observability": "metrics backends nothing in the default pipeline instantiates",
    "cache": "an optional out-of-process cache backend; the default caches are in-process",
    "async": "celery is an optional deployment shape, not a dependency of the library",
    "docs": "mkdocs builds the site; it is not needed to run or test anything",
    "full": "a meta-extra that unions the others; composing from it would smuggle in the above",
}

_HEADER = """\
# requirements.txt | GENERATED FILE -- DO NOT EDIT
#
# Rendered from pyproject.toml by scripts/gen_requirements.py.
# tests/packaging/test_requirements_derivation.py regenerates this file and diffs it on
# every test run, so an edit here is reverted by the next generation and fails the suite.
#
# Change a dependency in pyproject.toml, then run:
#     python scripts/gen_requirements.py
#
# This file exists because it drifted twice: once carrying two packaging defects already
# fixed in pyproject.toml, and once missing onnxruntime/tokenizers/tomli entirely, which
# left a start.sh install unable to load the bundled encoder.
#
# Composition: {composition}
# Deliberately excluded: {excluded}
"""


# =============================================================================
# READING THE DECLARATION
# =============================================================================


def _load_toml(path: Path) -> dict:
    """
    tomllib is stdlib from 3.11 only, and this package supports 3.10.

    Same fallback the package itself uses for `NexusMatcher.from_config`; without it this
    generator -- and therefore the test that calls it -- dies on the oldest interpreter in
    the support matrix.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - exercised only on 3.10
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(path.read_text(encoding="utf-8"))


def canonical_name(requirement: str) -> str:
    """
    PEP 503 normalised project name for a requirement string.

    `rank-bm25` and `rank_bm25` are the same project, and `uvicorn[standard]>=0.23` and
    `uvicorn>=0.23` are the same project declared two ways. Both shapes have to collapse
    to one key or the conflict detection below never fires.
    """
    head = re.split(r"[\[<>=!~;\s]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", head).lower()


class DerivationError(RuntimeError):
    """The declaration cannot be rendered into a coherent requirements file."""


_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*(?:\[([^\]]*)\])?\s*(.*)$")


def normalize_requirement(requirement: str) -> tuple:
    """
    A comparable form: (name, extras, specifier clauses, marker).

    Same requirement, three spellings -- `pydantic>=2.0.0,<3.0.0` in pyproject,
    `pydantic<3.0.0,>=2.0.0` in the wheel METADATA, and `tomli>=2.0.0; python_version <
    '3.11'` versus the double-quoted form the build backend emits. Comparing the raw
    strings reports all three as conflicts, which is how a real conflict would end up
    being ignored.
    """
    head, _, marker = requirement.partition(";")
    match = _REQUIREMENT.match(head)
    if not match:
        raise DerivationError(f"cannot parse requirement {requirement!r}")
    name, extras, specifiers = match.groups()
    return (
        re.sub(r"[-_.]+", "-", name).lower(),
        tuple(sorted(e.strip() for e in (extras or "").split(",") if e.strip())),
        tuple(sorted(s.strip().replace(" ", "") for s in specifiers.split(",") if s.strip())),
        re.sub(r"\s+", " ", marker.replace('"', "'")).strip(),
    )


_SELF_REFERENCE = re.compile(r"^nexus[-_]matcher\s*\[([^\]]+)\]", re.I)


def read_groups(pyproject: Path = PYPROJECT) -> dict[str, list[str]]:
    """{group name -> requirement strings}. `<core>` is project.dependencies."""
    data = _load_toml(pyproject)
    project = data.get("project") or {}
    groups: dict[str, list[str]] = {CORE: list(project.get("dependencies") or [])}
    for name, reqs in (project.get("optional-dependencies") or {}).items():
        groups[name] = list(reqs)
    return groups


def expand(
    group: str, groups: dict[str, list[str]], _seen: frozenset[str] = frozenset()
) -> list[str]:
    """
    Requirement strings for a group, with `nexus-matcher[a,b]` self-references resolved.

    `full` is declared as two self-referencing lines. Left unexpanded they would be
    written into requirements.txt verbatim, and `pip install -r` would then reinstall the
    package from PyPI over the local checkout -- silently testing a different version of
    the code than the one on disk.
    """
    if group in _seen:
        raise DerivationError(f"extra `{group}` references itself in a cycle")
    if group not in groups:
        raise DerivationError(f"pyproject.toml declares no group named `{group}`")
    out: list[str] = []
    for req in groups[group]:
        match = _SELF_REFERENCE.match(req)
        if match:
            for referenced in (part.strip() for part in match.group(1).split(",")):
                out.extend(expand(referenced, groups, _seen | {group}))
        else:
            out.append(req)
    return out


def compose(group_names: list[str], groups: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """
    Merge the named groups in order, deduplicating by project name.

    Two declarations of the same project with DIFFERENT text is the drift this file
    exists to stop -- `rapidfuzz>=3.0.0` in core and `rapidfuzz>=2.0.0` in an extra would
    resolve to whichever pip saw last. It raises rather than picking one.
    """
    chosen: dict[str, tuple[str, tuple]] = {}
    sections: list[tuple[str, list[str]]] = []
    for name in group_names:
        emitted: list[str] = []
        for req in expand(name, groups):
            key = canonical_name(req)
            text = re.sub(r"\s+", " ", req.strip())
            shape = normalize_requirement(text)
            if key in chosen:
                # Compared on the NORMALISED shape, not the raw text: `a>=1,<2` and
                # `a<2,>=1` are the same requirement and must not read as a conflict,
                # or the real conflicts get lost in the noise.
                if chosen[key][1] != shape:
                    raise DerivationError(
                        f"`{key}` is declared twice with different requirements: "
                        f"{chosen[key][0]!r} and {text!r}. Reconcile them in "
                        f"pyproject.toml -- pip would resolve this by accident."
                    )
                continue
            chosen[key] = (text, shape)
            emitted.append(text)
        sections.append((name, sorted(emitted, key=canonical_name)))
    return sections


# =============================================================================
# RENDERING
# =============================================================================

_SECTION_TITLE = {
    CORE: "CORE -- installed by a bare `pip install nexus-matcher`",
}


def render(filename: str = "requirements.txt", pyproject: Path = PYPROJECT) -> str:
    groups = read_groups(pyproject)
    names = COMPOSITION[filename]
    sections = compose(names, groups)

    lines = [
        _HEADER.format(
            composition=", ".join(names),
            excluded=", ".join(sorted(EXCLUDED)),
        ).rstrip()
    ]
    for name, reqs in sections:
        title = _SECTION_TITLE.get(name, f"EXTRA: {name}")
        lines.append("")
        lines.append("# " + "=" * 75)
        lines.append(f"# {title}")
        lines.append("# " + "=" * 75)
        if not reqs:
            # An extra whose every member was already pulled in by an earlier group. Say
            # so, rather than emitting a bare heading that reads like a truncation.
            lines.append("# (every requirement already listed above)")
            continue
        lines.extend(reqs)
    return "\n".join(lines) + "\n"


def missing_reasons(pyproject: Path = PYPROJECT) -> list[str]:
    """Extras that are neither composed in nor excused. See EXCLUDED."""
    groups = read_groups(pyproject)
    composed = {name for names in COMPOSITION.values() for name in names}
    return sorted(
        name for name in groups if name != CORE and name not in composed and name not in EXCLUDED
    )


def stale_reasons(pyproject: Path = PYPROJECT) -> list[str]:
    """EXCLUDED entries naming an extra that no longer exists -- silent rot."""
    groups = read_groups(pyproject)
    return sorted(set(EXCLUDED) - set(groups))


def diff(filename: str = "requirements.txt", repo: Path = REPO) -> str:
    """Unified diff of what is on disk against what pyproject says. Empty means clean."""
    target = repo / filename
    current = target.read_text(encoding="utf-8") if target.exists() else ""
    generated = render(filename, repo / "pyproject.toml")
    return "".join(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile=f"{filename} (on disk)",
            tofile=f"{filename} (generated from pyproject.toml)",
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if stale; write nothing")
    ap.add_argument("--stdout", action="store_true", help="print the file; write nothing")
    args = ap.parse_args()

    unexplained = missing_reasons()
    if unexplained:
        print(
            "pyproject.toml declares extras that are neither in COMPOSITION nor in "
            f"EXCLUDED: {', '.join(unexplained)}\n"
            "Decide whether requirements.txt needs them, and record the answer.",
            file=sys.stderr,
        )
        return 1
    stale = stale_reasons()
    if stale:
        print(f"EXCLUDED names extras that no longer exist: {', '.join(stale)}", file=sys.stderr)
        return 1

    status = 0
    for filename in COMPOSITION:
        if args.stdout:
            print(render(filename))
            continue
        delta = diff(filename)
        if args.check:
            if delta:
                print(f"{filename} is STALE:\n{delta}", file=sys.stderr)
                status = 1
            else:
                print(f"{filename} matches pyproject.toml")
            continue
        (REPO / filename).write_text(render(filename), encoding="utf-8", newline="\n")
        print(f"wrote {filename}" + (" (changed)" if delta else " (no change)"))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
