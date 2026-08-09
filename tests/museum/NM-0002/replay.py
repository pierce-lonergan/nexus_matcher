"""
Reintroduce NM-0002: move typer and rich back out of the core dependency list.

They stay declared in the `cli` extra, which is exactly the shipped state -- the point of
the defect was never that the packages were unavailable, it was that `pip install
nexus-matcher` did not pull them while `[project.scripts]` handed the user a command that
imports typer at module scope.

Anchoring notes
---------------
The requirement lines appear TWICE in pyproject: once in `[project] dependencies` and once
in the `cli` extra. So the mutation is scoped to the `dependencies = [...]` block, and
inside it the lines are matched by DISTRIBUTION NAME with the version specifier left as a
wildcard. Pinning the literal `"typer>=0.9.0"` would rot the first time somebody raises a
floor -- which is what turned NM-0016's original anchor into a hole.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "pyproject.toml"

# The core dependency array, from `dependencies = [` to the closing bracket at column 0.
DEPENDENCIES_RE = re.compile(r"^dependencies = \[\n(?P<body>.*?)^\]", re.M | re.S)

# One requirement line for typer or rich, whatever version specifier it carries.
REQUIREMENT_RE = re.compile(r"^[ \t]*\"(?:typer|rich)(?:\[[^\]]*\])?[^\"]*\",[ \t]*\n", re.M)


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")

    block = DEPENDENCIES_RE.search(text)
    if block is None:
        raise LookupError(f"NM-0002 replay: no `dependencies = [...]` array in {TARGET}")

    body = block.group("body")
    stripped, removed = REQUIREMENT_RE.subn("", body)
    if removed != 2:
        raise LookupError(
            f"NM-0002 replay: expected typer and rich in the core dependencies of "
            f"{TARGET}, removed {removed}"
        )

    rewritten = text[: block.start("body")] + stripped + text[block.end("body") :]
    path.write_text(rewritten, encoding="utf-8")
