"""
tests.packaging.test_architecture | Layer: GATE
The layering rules, enforced. A layering rule nobody has watched reject anything is a
diagram, not a constraint.

What is enforced
----------------
  1. `domain/` has no IMPORT-TIME dependency on application, infrastructure or
     presentation. Deferred (function-local or TYPE_CHECKING) cross-layer imports are
     allowed only by name, with a written reason, in `ALLOWED_DEFERRED` below.
  2. Nothing outside `presentation/api` imports fastapi or starlette.
  3. Nothing outside `presentation/cli` imports typer or rich.

Why ast and not import-linter
-----------------------------
import-linter is the right tool for a project that can take the dependency. Here it costs
more than it returns, for three reasons that are specific to this repo:

  * it would have to be declared in `pyproject.toml`, which is another lane's file this
    session -- and a check that lands without its dependency declared is the half-wired
    shape H-006 exists to stop;
  * `scripts/release_preflight.py` builds a BARE venv and this repo's gates are written to
    stay runnable there (see the dependency-free YAML reader in `scripts/museum_replay.py`
    for the same decision made once already);
  * the three contracts above are the whole ruleset, and expressing them over `ast` is
    ~80 lines with no configuration file to drift away from what it claims to enforce.

The one thing import-linter would add is transitive-chain reporting (`domain -> shared ->
infrastructure`). That is a real gap and is recorded as such in the report; contract 1
below catches only direct edges.

Vacuity
-------
`test_the_scanner_actually_sees_the_framework_imports` exists because contracts 2 and 3
pass trivially if the scanner ever stops finding imports at all -- a path typo, a rename
of `src/`, a swallowed SyntaxError. That is precisely the failure mode that let nineteen
tests miss a transposed matrix multiply: green because they were measuring nothing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "nexus_matcher"
PACKAGE_ROOT = "nexus_matcher"

# The layers, inner to outer. `shared` and `core` are deliberately absent: they are
# cross-cutting and every layer is allowed to use them.
INNER = "domain"
OUTER_LAYERS = ("application", "infrastructure", "presentation")

# Third-party frameworks that must not escape the one package that owns them.
# The value is the ONLY directory (relative to the package root) allowed to import it.
FRAMEWORK_OWNERS = {
    "fastapi": "presentation/api",
    "starlette": "presentation/api",  # fastapi's substrate; importing it is the same edge
    "typer": "presentation/cli",
    "rich": "presentation/cli",
}

# Cross-layer imports from `domain` that are DEFERRED -- inside a function body or under
# `if TYPE_CHECKING:` -- so they do not create an import-time dependency. Each needs a
# reason, so this cannot quietly become a list of everything the rule would otherwise
# reject. Import-TIME violations have no allowlist at all.
ALLOWED_DEFERRED = {
    (
        "domain/ports/dictionary_loader.py",
        "nexus_matcher.application.ingest",
    ): (
        "detect_column_mapping() shares application.ingest's column-alias table on "
        "purpose: two independent notions of 'which column is the business name' is how "
        "the loader path ended up rejecting files the ingest path read without "
        "complaint. The import is function-local, so domain still imports cleanly on its "
        "own."
    ),
}


# =============================================================================
# SCANNER
# =============================================================================


@dataclass(frozen=True)
class Edge:
    """One import, with enough context to name it in a failure message."""

    source: str  # posix path relative to the package root
    layer: str  # first path component: domain / application / ...
    target: str  # fully-qualified module being imported
    line: int
    import_time: bool  # False when function-local or under `if TYPE_CHECKING:`

    @property
    def root(self) -> str:
        return self.target.split(".")[0]


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PACKAGE_ROOT, *parts))


def _resolve_relative(path: Path, level: int, module: str | None) -> str:
    """
    `from ..application import x` in domain/ports/foo.py -> nexus_matcher.application.

    Relative imports are the shape a layering check misses most easily: they carry no
    package name, so a rule matching on the string "nexus_matcher.application" sees
    nothing at all.
    """
    parts = _module_name(path).split(".")
    if path.name != "__init__.py":
        parts = parts[:-1]  # a module's own name is not part of its package
    base = parts[: len(parts) - (level - 1)] if level > 1 else parts
    return ".".join([*base, module]) if module else ".".join(base)


def _edges_in(path: Path) -> list[Edge]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(PACKAGE).as_posix()
    layer = rel.split("/")[0]

    # An import executes at import time unless it sits inside a function body or under
    # `if TYPE_CHECKING:`. A class body executes at import time, so it does NOT count as
    # deferred -- getting that backwards would excuse a real import-time dependency.
    deferred: set[int] = set()
    for node in ast.walk(tree):
        in_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        under_type_checking = isinstance(node, ast.If) and _is_type_checking(node.test)
        if not (in_function or under_type_checking):
            continue
        for stmt in node.body:
            for inner in ast.walk(stmt):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    deferred.add(id(inner))

    edges: list[Edge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(Edge(rel, layer, alias.name, node.lineno, id(node) not in deferred))
        elif isinstance(node, ast.ImportFrom):
            target = (
                _resolve_relative(path, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            edges.append(Edge(rel, layer, target, node.lineno, id(node) not in deferred))
    return edges


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def all_edges() -> list[Edge]:
    edges: list[Edge] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        edges.extend(_edges_in(path))
    return edges


# =============================================================================
# VACUITY GUARDS -- these come first on purpose
# =============================================================================


def test_the_package_is_where_this_test_thinks_it_is():
    assert PACKAGE.is_dir(), f"{PACKAGE} does not exist -- every contract below is vacuous"
    for layer in (INNER, *OUTER_LAYERS):
        assert (PACKAGE / layer).is_dir(), f"layer `{layer}` is missing"


def test_the_scanner_actually_sees_the_framework_imports():
    """
    Contracts 2 and 3 are satisfied by a scanner that finds nothing. Pin the two imports
    that are known to exist, so a broken scan fails HERE instead of quietly passing there.
    """
    edges = all_edges()
    assert len(edges) > 200, f"only {len(edges)} imports found across the package -- scan is broken"
    roots = {(e.source, e.root) for e in edges}
    assert ("presentation/api/app.py", "fastapi") in roots, "the scanner no longer sees fastapi"
    assert ("presentation/cli/main.py", "typer") in roots, "the scanner no longer sees typer"
    assert ("presentation/cli/main.py", "rich") in roots, "the scanner no longer sees rich"


def test_relative_imports_are_resolved_not_dropped():
    """
    A relative import carries no package name. If `_resolve_relative` ever returned the
    bare module, `from ..application import x` inside domain/ would read as an import of
    `application` and match nothing -- contract 1 would pass while being violated.
    """
    fake = PACKAGE / "domain" / "ports" / "example.py"
    # level counts packages upward from the module's OWN package, and the module's own
    # name is not one of them -- an off-by-one here turns every relative import into a
    # module name that matches no contract.
    assert _resolve_relative(fake, 1, "base") == "nexus_matcher.domain.ports.base"
    assert _resolve_relative(fake, 2, "models") == "nexus_matcher.domain.models"
    assert _resolve_relative(fake, 3, "application.ingest") == "nexus_matcher.application.ingest"
    assert _resolve_relative(fake, 1, None) == "nexus_matcher.domain.ports"
    pkg_init = PACKAGE / "domain" / "__init__.py"
    assert _resolve_relative(pkg_init, 1, "models") == "nexus_matcher.domain.models"


# =============================================================================
# CONTRACT 1 -- the domain layer depends on nothing outward
# =============================================================================


def _domain_violations() -> list[Edge]:
    return [
        e
        for e in all_edges()
        if e.layer == INNER
        and e.target.startswith(f"{PACKAGE_ROOT}.")
        and e.target.split(".")[1] in OUTER_LAYERS
    ]


def test_domain_has_no_import_time_dependency_on_an_outer_layer():
    """
    The hard half of the contract, with no allowlist. An import-time edge from domain to
    infrastructure means `import nexus_matcher.domain` drags a database driver in, and the
    dependency inversion the hexagonal architecture is built on has been reversed.
    """
    bad = [e for e in _domain_violations() if e.import_time]
    assert not bad, "domain imports an outer layer at IMPORT TIME:\n  " + "\n  ".join(
        f"{e.source}:{e.line} -> {e.target}" for e in bad
    )


def test_every_deferred_domain_dependency_is_declared_with_a_reason():
    """
    Deferring an import hides the edge from `import nexus_matcher.domain` but does not
    remove it: the call still reaches upward at runtime. Allowed, named, and reasoned --
    never silent.
    """
    undeclared = [
        e
        for e in _domain_violations()
        if not e.import_time and (e.source, e.target) not in ALLOWED_DEFERRED
    ]
    assert not undeclared, (
        "domain reaches into an outer layer through a deferred import that nobody "
        "signed for:\n  "
        + "\n  ".join(f"{e.source}:{e.line} -> {e.target}" for e in undeclared)
        + "\nInvert the dependency, or add it to ALLOWED_DEFERRED with a reason."
    )


def test_no_allowlist_entry_has_gone_stale():
    """
    An allowlist entry excusing an import that no longer exists reads as a live exception
    and excuses nothing -- the same rot `.ci-exceptions.yaml` is checked for.
    """
    live = {(e.source, e.target) for e in _domain_violations() if not e.import_time}
    stale = sorted(set(ALLOWED_DEFERRED) - live)
    assert not stale, (
        "ALLOWED_DEFERRED excuses imports that are gone; delete them:\n  "
        + "\n  ".join(f"{src} -> {tgt}" for src, tgt in stale)
    )


# =============================================================================
# CONTRACTS 2 AND 3 -- frameworks stay inside the package that owns them
# =============================================================================


def test_no_framework_escapes_the_package_that_owns_it():
    """
    fastapi outside presentation/api, or typer/rich outside presentation/cli, is what made
    `pip install nexus-matcher` ship a library that could not be imported without the web
    stack. Deferred or not: a runtime import of typer from the application layer is still
    a hard dependency on a CLI toolkit for someone using this as a library.
    """
    offenders = [
        (e, owner)
        for e in all_edges()
        for framework, owner in FRAMEWORK_OWNERS.items()
        if e.root == framework and not e.source.startswith(owner + "/")
    ]
    assert not offenders, "framework imports outside their owning package:\n  " + "\n  ".join(
        f"{e.source}:{e.line} imports {e.root} (only {owner}/ may)" for e, owner in offenders
    )


def test_each_framework_contract_has_something_to_constrain():
    """
    A contract naming a package nobody imports is decoration. If fastapi or typer stops
    being used entirely, this fails and the rule gets deleted rather than left as a
    comforting no-op.
    """
    roots = {e.root for e in all_edges()}
    unused = sorted(f for f in FRAMEWORK_OWNERS if f not in roots)
    # starlette is imported transitively by fastapi, never directly -- it is a guard
    # against a future direct import, not a description of today's tree.
    unused = [f for f in unused if f != "starlette"]
    assert not unused, (
        f"FRAMEWORK_OWNERS constrains packages this codebase never imports: {unused}. "
        "Delete the rule or the dependency."
    )
