"""
tests/packaging/test_documented_routes.py | Env: ALL

The route table is a packaging contract, and it was wrong in the expensive direction.

`create_app()` registers `POST /api/v1/match`, `POST /api/v1/match/batch` and
`POST /api/v1/feedback`. Eleven sentences across README.md, QUICKSTART.md,
docs/API_REFERENCE.md, docs/ARCHITECTURE.md, docs/DEPLOYMENT.md and docs/PROJECT_STATE.md
said they do not exist -- not by omitting them, but by DENYING them, in the register this
repository reserves for being honest about what is missing. `grep -rn "api/v1/match"
--include=*.md .` returned zero hits while the routes were live. A caller who trusts these
documents (which is the point of writing them this way) concludes the feature is not there
and builds around it.

Eleven is the number this file measured; the two reviews that found the defect reported nine
and eleven SITES, disagreeing because they grepped. The gate below reported denials in six
documents on the tree as it stood, which is what makes the count checkable at all.

Omission and denial fail differently, so this file checks BOTH directions:

  * **Registered but undocumented** -- the next endpoint added silently repeats the defect.
  * **Documented as absent while registered** -- the defect that actually shipped. A doc
    that merely forgot a route is a gap; a doc that argues the route does not exist is a
    wrong answer delivered with confidence.

This closes the hole `scripts/check_doc_numbers.py` names in its own limitations and
declines to cover: "Claims about behaviour rather than measurement... 'there is no HTTP
matching endpoint'. Those are checked against pyproject, the config dataclass and the
router -- by other tests, or by nobody." It was nobody.

## Why the text is flattened before it is searched

Because a line-oriented grep undercounted the sites, twice, in two independent reviews.
`README.md` wrote "There is **no** HTTP matching endpoint" and the emphasis markers broke
the phrase; `docs/ARCHITECTURE.md` wrapped "The REST app serves health and / introspection
routes only." across two lines of a blockquote. Both are denials and neither is greppable.
So each document is stripped of emphasis and blockquote markers and collapsed to one line
before the patterns run -- a denial must not become invisible by being reformatted.

## What is deliberately NOT scanned, and why

A denial that is a dated record of a past state is not a defect; erasing it would be.

  * `CHANGELOG.md` is scanned only from `## [Unreleased]` to the first released heading.
    The 2.0.x retraction table is a true statement about what 2.0.0 falsely claimed, and the
    "Planned" list is what was planned at that release. Rewriting a released section to
    match today is falsifying the record, which is the opposite of this gate's purpose.
  * The dated narratives in `_HISTORICAL` -- a session log and two reviewer reports -- quote
    the state of the repository on the day they were written. A reviewer writing up the fix
    would trip the gate for describing the bug accurately.

Both exemptions are asserted to name files that exist, so neither can silently widen onto a
document that was renamed into it.

## A quoted denial is still a denial

This gate went red twice on the change that retracted the eleven sentences, both times on
the same mistake: retracting a claim by QUOTING it. "This row previously said X", where X
is the denial, flattens to the denial. That is not a false positive to be exempted -- a
reader skimming for whether the endpoint exists takes the same wrong answer from the quote
as from the original. Retract by describing, not by repeating; the control below pins the
behaviour so nobody "fixes" it later.

A third quoted retraction, written in the PAST tense ("previously said no matching endpoint
existed"), slipped through and was found by hand. Widening the patterns to catch tense is
not obviously right -- it makes the true sentence "2.0.0 shipped no matching endpoint"
unwriteable outside the changelog -- so it is left uncaught and written down here instead.
This gate is a floor under the documentation, not a proofreader for it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# The four documents a caller lands on before they ever read source. Each one carries a
# route table; all four denied the matching endpoint.
ROUTE_TABLE_DOCS = (
    "README.md",
    "QUICKSTART.md",
    "docs/API_REFERENCE.md",
    "docs/ARCHITECTURE.md",
)

# FastAPI's own OAuth2 redirect helper for the Swagger page. It is machinery for /docs, not
# a surface this service offers, and documenting it would be noise. Everything else the app
# registers -- including `/`, which is `include_in_schema=False` -- is a route a caller can
# call and must therefore appear in the tables.
INTERNAL_PATHS = frozenset({"/docs/oauth2-redirect"})

# Dated records of a past state; see the module docstring.
_HISTORICAL = (
    "docs/SESSION_LOG.md",
    "docs/research/fresh-eyes.md",
    "docs/research/friction-log.md",
    "benchmarks/dx_baseline.md",
)

_UNRELEASED = "## [Unreleased]"


# =============================================================================
# THE ROUTES, AND THE DOCUMENTS
# =============================================================================


def registered_routes() -> list[tuple[str, tuple[str, ...]]]:
    """
    Every route a live `create_app()` mounts, minus the Swagger machinery.

    `environ={}` so the enumeration is the same on a developer's machine as in CI: with
    NEXUS_API_DICTIONARY set, `create_app` would load a dictionary and a model to answer a
    question about the ROUTER, which registers its routes either way.

    HEAD and OPTIONS are dropped: Starlette adds HEAD to every GET route and CORS answers
    OPTIONS, so requiring them in a table would document the framework rather than the
    service.
    """
    from nexus_matcher.presentation.api.app import create_app

    app = create_app(configure_logs=False, environ={})
    found: list[tuple[str, tuple[str, ...]]] = []
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path is None or not methods or path in INTERNAL_PATHS:
            continue
        verbs = tuple(sorted(set(methods) - {"HEAD", "OPTIONS"}))
        if verbs:
            found.append((path, verbs))
    return sorted(found)


def tracked_markdown() -> list[str]:
    """
    Every markdown file under version control, as repo-relative posix paths.

    `git ls-files` rather than a glob: an untracked scratch document is not something this
    repository ships, and a doc that IS shipped cannot hide from it.
    """
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _scannable(doc: str, text: str) -> str:
    """The part of a document that makes a claim about TODAY. See the module docstring.

    The window is `[Unreleased]` PLUS the newest version section, because those are the
    two places a claim about today can live. The first cut stopped at the newest version
    heading, which was right while `[Unreleased]` held the staged work -- and silently
    wrong the moment that work was collected into `## [2.1.0]` to cut the release. The
    window fell to 86 of 52,787 characters, 0.16%, and a denial written into the staged
    section passed. A released section is history and must not be rewritten to agree with
    the router; a section for a version that has not shipped is not history yet.

    Anchored with a line-start regex rather than `str.find`, because prose in this file
    legitimately quotes the heading, and an unanchored search relocates the window to the
    quotation -- observed, and it silently passed a mutation it should have caught.
    """
    if doc != "CHANGELOG.md":
        return text
    opening = re.search(rf"^{re.escape(_UNRELEASED)}\s*$", text, re.M)
    assert opening is not None, (
        f"CHANGELOG.md no longer contains {_UNRELEASED!r} as a heading, so this gate is "
        "scanning nothing of it. Restore the heading or rewrite the scoping rule deliberately."
    )
    body = text[opening.end() :]
    headings = [m.start() for m in re.finditer(r"^## ", body, re.M)]
    return body if len(headings) < 2 else body[: headings[1]]


def flatten(text: str) -> str:
    """
    One lowercase line, with emphasis, code ticks and blockquote markers removed.

    Both undercounted greps died on formatting rather than on wording, so the comparison is
    made against text a reformat cannot change.
    """
    stripped = [line.lstrip("> \t").strip() for line in text.splitlines()]
    joined = " ".join(stripped).replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", joined).lower()


# =============================================================================
# THE DENIALS
# =============================================================================

# (pattern, the route it denies, the site it was written from). Every pattern is pinned in
# the controls below against the exact sentence that shipped, so a pattern that stops
# matching -- a typo, a widened regex, a rewrite -- goes red instead of going quiet.
_DENIALS: tuple[tuple[str, str, str], ...] = (
    (r"no http matching endpoint", "/api/v1/match", "README.md, QUICKSTART.md, ARCHITECTURE 4.1"),
    (r"matching over http is not implemented", "/api/v1/match", "README.md, API_REFERENCE.md"),
    (r"no matching endpoint exists", "/api/v1/match", "docs/API_REFERENCE.md"),
    (r"there is no post /match endpoint", "/api/v1/match", "docs/ARCHITECTURE.md"),
    (
        r"endpoints that do not exist[^.]{0,120}post /match",
        "/api/v1/match",
        "docs/ARCHITECTURE.md 2.1.1",
    ),
    (
        r"health and introspection (?:endpoints|routes) only",
        "/api/v1/match",
        "README.md, QUICKSTART.md, PROJECT_STATE.md, CHANGELOG.md",
    ),
    (
        r"deployed http service does not match schemas",
        "/api/v1/match",
        "docs/DEPLOYMENT.md preamble",
    ),
    (
        r"implement an http matching endpoint, or stop describing",
        "/api/v1/match",
        "docs/PROJECT_STATE.md outstanding work",
    ),
)


def denials_in(doc: str, text: str, registered: set[str]) -> list[str]:
    """Every denial in `text` of a route that `registered` says is live."""
    flat = flatten(_scannable(doc, text))
    return [
        f"{doc}: denies {route} -- matched {pattern!r} (originally at {site})"
        for pattern, route, site in _DENIALS
        if route in registered and re.search(pattern, flat)
    ]


# =============================================================================
# THE GATE
# =============================================================================


def test_every_registered_route_appears_in_every_route_table():
    """
    Direction one: registered but undocumented.

    A route must appear backticked, with its verb, on one line of all four tables. Same
    line, because a path mentioned in prose three sections away from a table of methods is
    not a route table -- and the tables are what a caller reads to decide what to call.
    """
    missing: list[str] = []
    contents = {doc: (REPO / doc).read_text(encoding="utf-8") for doc in ROUTE_TABLE_DOCS}
    for path, verbs in registered_routes():
        for doc, text in contents.items():
            lines = text.splitlines()
            for verb in verbs:
                if not any(f"`{path}`" in line and verb in line for line in lines):
                    missing.append(f"{doc} does not document {verb} {path}")

    assert not missing, (
        "these routes are registered by create_app() and absent from a documented route "
        "table:\n  " + "\n  ".join(missing)
    )


def test_no_document_denies_a_route_that_is_registered():
    """
    Direction two: documented as absent while registered. This is the one that shipped.

    Nine sites said the matching endpoint did not exist. Retracting them is a one-time edit;
    this assertion is what stops the next one, because the sentence is cheap to write and
    reads as diligence.
    """
    registered = {path for path, _ in registered_routes()}
    found: list[str] = []
    for doc in tracked_markdown():
        if doc in _HISTORICAL:
            continue
        found.extend(denials_in(doc, (REPO / doc).read_text(encoding="utf-8"), registered))

    assert not found, (
        "documentation denies an endpoint this application registers:\n  "
        + "\n  ".join(found)
        + "\nRetract the sentence. If the route really should not exist, delete the route."
    )


# =============================================================================
# CONTROLS -- proof the gate can go red
# =============================================================================

# The nine denial sites, verbatim as they shipped, each with the file it came from. This is
# the evidence the patterns above were derived from and the only reason to believe they
# work: after the retraction every pattern matches nothing in the tree, so without these the
# whole denial half would be green over an empty scan.
_RETRACTED: tuple[tuple[str, str], ...] = (
    (
        "README.md",
        "There is **no** HTTP matching endpoint, no dictionary CRUD endpoint, no cache "
        "endpoint,\nand no `/metrics` endpoint. Matching over HTTP is not implemented.",
    ),
    (
        "QUICKSTART.md",
        "**There is no HTTP matching endpoint.** Matching over HTTP is not implemented. Use "
        "the\nPython API or the CLI.",
    ),
    (
        "QUICKSTART.md",
        "The server exposes **health and introspection endpoints only**: `/`, `/health`,",
    ),
    (
        "docs/API_REFERENCE.md",
        "| `POST /match` | No matching endpoint exists. Matching over HTTP is not implemented. |",
    ),
    (
        "docs/ARCHITECTURE.md",
        "> - **There is no `POST /match` endpoint.** The REST app serves health and\n"
        ">   introspection routes only.",
    ),
    (
        "docs/ARCHITECTURE.md",
        "**Endpoints that do NOT exist**, despite appearing in earlier revisions of this "
        "table:\n`POST /match`, `POST /batch`, and all `/dictionary` CRUD routes.",
    ),
    (
        "docs/ARCHITECTURE.md",
        "### 4.1 Match Request Flow (DESIGN TARGET - no HTTP matching endpoint exists today)",
    ),
    (
        "docs/PROJECT_STATE.md",
        "- **HTTP matching.** The REST app serves health and introspection routes only.",
    ),
    (
        "docs/PROJECT_STATE.md",
        "4. Implement an HTTP matching endpoint, or stop describing the service as a matching API.",
    ),
    (
        "docs/DEPLOYMENT.md",
        "> - **The deployed HTTP service does not match schemas.** The FastAPI app serves "
        "`/`,\n>   `/health`, `/health/live`, `/health/ready`, `/health/startup`, `/docs`, "
        "`/redoc` and\n>   `/openapi.json` - nothing else.",
    ),
)


@pytest.mark.parametrize(("source", "sentence"), _RETRACTED, ids=[s for s, _ in _RETRACTED])
def test_the_scan_catches_every_denial_that_actually_shipped(source, sentence):
    """
    Replay each retracted site through the detector.

    Two of these are the reason the flattening exists: README's `**no**` and ARCHITECTURE's
    blockquote wrapped across two lines both survived a plain grep, which is why the first
    two reviews reported different counts of the same defect.
    """
    assert denials_in(source, sentence, {"/api/v1/match"}), (
        f"the detector no longer recognises the denial that shipped in {source}:\n{sentence}"
    )


def test_a_denial_is_only_a_defect_while_the_route_is_registered():
    """
    The gate reads the router, not a wish list. If the matching routes were removed, every
    sentence above would become true again and this file must go quiet rather than force a
    document to lie the other way.
    """
    _source, sentence = _RETRACTED[0]
    assert denials_in("README.md", sentence, {"/api/v1/match"})
    assert denials_in("README.md", sentence, set()) == []


def test_retracting_a_denial_by_quoting_it_still_counts_as_a_denial():
    """
    Deliberate, and the gate's most useful catch: it fired three times on the change that
    wrote the retractions, always on a draft that explained the old claim by repeating it.

    Formatting is gone by the time the patterns run, so "an earlier revision said <denial>"
    and "<denial>" are the same string, and a reader skimming for whether the endpoint
    exists takes the same wrong answer from both. Pinned so this reads as a decision rather
    than as a rough edge somebody later exempts.
    """
    quoted = (
        "This section previously said: **There is no HTTP matching endpoint.** That is retracted."
    )
    described = "This section previously denied that the endpoint existed. That is retracted."
    assert denials_in("README.md", quoted, {"/api/v1/match"})
    assert denials_in("README.md", described, {"/api/v1/match"}) == []


def test_the_replacement_wording_is_not_itself_read_as_a_denial():
    """
    The other half of the control, and the one that keeps the patterns narrow.

    A pattern broad enough to match any sentence containing "matching" and "HTTP" would make
    the endpoint impossible to document at all -- green only while the docs stay silent,
    which is how this started.
    """
    replacement = (
        "**Matching over HTTP is implemented.** `POST /api/v1/match` returns the protection "
        "class each field would inherit; `POST /api/v1/match/batch` takes a larger chunk, "
        "and `POST /api/v1/feedback` records a reviewer's verdict. Health and introspection "
        "routes are served alongside them."
    )
    assert denials_in("README.md", replacement, {"/api/v1/match"}) == []


def test_the_changelog_scan_covers_the_staged_release_and_stops_before_history():
    """
    The window must include the release being staged and exclude the ones that shipped.

    The 2.0.x retraction table states, correctly, that 2.0.0 shipped no matching endpoint.
    Widening the scan onto released sections would demand it be rewritten, which would make
    the changelog agree with the router by lying about history. But stopping at the FIRST
    heading excluded the staged release, which is where the live claims actually are once a
    release is collected -- so both bounds are asserted here, in both directions.
    """
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    scanned = _scannable("CHANGELOG.md", changelog)

    newest = re.search(r"^##\s*\[?([0-9]+\.[0-9]+\.[0-9]+)\]?", changelog, re.M)
    assert newest, "CHANGELOG.md has no released-version heading"
    assert newest.group(1) in scanned, (
        f"the staged release {newest.group(1)} is outside the scan window; a denial written "
        "there would pass. This is the 0.16%-coverage regression."
    )
    assert len(scanned) < len(changelog), "the window must not cover the whole file"

    # The second version heading down is history and must stay out. Matched as a HEADING:
    # the staged section legitimately mentions older versions in prose (2.1.0's security
    # entry names the 2.0.0 and 2.0.1 wheels), and a bare version-string search reads that
    # as the section itself.
    older = re.findall(r"^##\s*\[?([0-9]+\.[0-9]+\.[0-9]+)\]?", changelog, re.M)
    if len(older) > 1:
        assert f"## [{older[1]}]" not in scanned, (
            f"released section {older[1]} must stay out of the window"
        )

    staged_denial = (
        f"{_UNRELEASED}\n\n## [{newest.group(1)}] - 2026-08-10\n\n"
        "There is no HTTP matching endpoint.\n\n## [2.0.1] - 2026-08-09\n"
    )
    assert denials_in("CHANGELOG.md", staged_denial, {"/api/v1/match"}), (
        "a denial written into the STAGED release section must be caught"
    )
    assert denials_in(
        "CHANGELOG.md",
        f"{_UNRELEASED}\n\nThere is no HTTP matching endpoint.\n\n## [2.1.0] - 2026-08-10\n",
        {"/api/v1/match"},
    ), "a denial written into [Unreleased] must still be caught"


def test_the_changelog_window_is_not_relocated_by_prose_quoting_the_heading():
    """
    This file's own prose quotes `## [Unreleased]`, and an unanchored `str.find` locates
    the quotation rather than the heading -- moving the window into an arbitrary slice of
    the file. Observed: a mutation deleting the real heading stayed green because the
    search landed on a sentence mentioning it.
    """
    forged = (
        "# Changelog\n\nSee the section named ## [Unreleased] for staged work.\n\n"
        f"{_UNRELEASED}\n\nThere is no HTTP matching endpoint.\n\n## [1.0.0] - 2025-10-15\n"
    )
    assert denials_in("CHANGELOG.md", forged, {"/api/v1/match"}), (
        "the window was relocated by prose quoting the heading"
    )


def test_the_route_table_gate_can_see_a_missing_route():
    """
    The direction-one control. `assert not missing` passes whether the comparison works or
    not once the docs are fixed, which is the vacuous-green shape this repository keeps
    rediscovering; so: a table that documents the route, and one that does not.
    """
    documented = "| Method | Path |\n|---|---|\n| POST | `/api/v1/match` |\n"
    undocumented = "| Method | Path |\n|---|---|\n| POST | `/api/v1/match/batch` |\n"

    def documents(table: str, path: str, verb: str) -> bool:
        return any(f"`{path}`" in line and verb in line for line in table.splitlines())

    assert documents(documented, "/api/v1/match", "POST")
    assert not documents(undocumented, "/api/v1/match", "POST")
    # ...and the verb has to be on the same line as the path.
    assert not documents("| POST | x |\n| GET | `/api/v1/match` |\n", "/api/v1/match", "POST")


# =============================================================================
# THE GATE IS LOOKING AT REAL ROUTES AND REAL DOCUMENTS
# =============================================================================


def test_the_routes_come_from_a_live_application():
    """
    Every assertion above is over `registered_routes()`. An enumeration that quietly
    returned nothing -- a renamed attribute, a router that stopped being included -- would
    make both directions pass over an empty list.
    """
    routes = dict(registered_routes())
    assert routes["/api/v1/match"] == ("POST",)
    assert routes["/api/v1/match/batch"] == ("POST",)
    assert routes["/api/v1/feedback"] == ("POST",)
    assert "/health/ready" in routes and "/" in routes
    assert "/docs/oauth2-redirect" not in routes


def test_the_denial_scan_is_looking_at_the_documents_that_ship():
    """A `git ls-files` glob that stopped matching would leave the scan green over zero files."""
    tracked = tracked_markdown()
    assert {"README.md", "QUICKSTART.md", "docs/GOVERNANCE.md", "CHANGELOG.md"} <= set(tracked)
    assert len(tracked) >= 40, tracked
    for doc in _HISTORICAL:
        assert doc in tracked, f"{doc} is exempted from the denial scan and does not exist"


def test_the_documented_route_tables_exist_and_are_tables():
    """
    The four documents are named as strings, so a rename would turn direction one into a
    FileNotFoundError at best and a silently shortened loop at worst.
    """
    for doc in ROUTE_TABLE_DOCS:
        text = (REPO / doc).read_text(encoding="utf-8")
        assert "|---|" in text or "|--------|" in text, f"{doc} carries no markdown table"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# =============================================================================
# THE VERSION THE SERVICE REPORTS
# =============================================================================


def test_the_service_reports_the_version_it_actually_is():
    """
    `/`, `/health`, the OpenAPI `info.version` and every log record must name the build.

    They named "2.0.0" — a literal repeated in four places while `__version__` moved to
    2.1.0. Publishing that would have shipped a service identifying itself as a release
    that had been DELETED from PyPI, and `docs/API_REFERENCE.md` documented the wrong
    value as correct.

    `pyproject.toml` sets `dynamic = ["version"]` against `__init__.py`, so `__version__`
    is what the wheel is built with, and any second spelling is a copy that drifts. This
    asserts there is no second spelling — checking each surface separately, because they
    are wired independently and three of the four were wrong in the same way.
    """
    from fastapi.testclient import TestClient

    from nexus_matcher import __version__
    from nexus_matcher.presentation.api.app import create_app
    from nexus_matcher.shared.logging import service_version

    assert service_version() == __version__

    app = create_app(configure_logs=False)
    assert app.version == __version__, "OpenAPI info.version disagrees with the package"

    client = TestClient(app)
    for path in ("/", "/health"):
        body = client.get(path).json()
        reported = body.get("version")
        assert reported == __version__, (
            f"{path} reports version {reported!r}, package is {__version__!r}"
        )

    assert client.get("/openapi.json").json()["info"]["version"] == __version__


def test_no_service_surface_hardcodes_a_version_string():
    """
    The control for the test above: it can only see surfaces it knows to look at.

    A fifth copy added tomorrow, on a surface nobody listed, would pass every assertion up
    there and drift exactly as the first four did. So this looks for the SHAPE of the
    mistake rather than its instances.

    Scoped to the modules that REPORT the service's own version -- the API layer and the
    log processor -- and not to `src/` at large. That is not convenience: elsewhere a
    version literal is usually some OTHER thing's version, and the first cut proved it by
    flagging `shared/plugins.py`, where `version: str = "0.0.0"` is plugin metadata a
    subclass overrides and `version = "1.0.0"` is inside a docstring example. Both are
    correct. A gate that reports those gets muted, and a muted gate protects nothing.

    What this therefore does NOT cover, said out loud: a version literal on a new surface
    outside these paths. If the service grows one, add the path here.
    """
    surfaces = [
        *(REPO / "src" / "nexus_matcher" / "presentation" / "api").rglob("*.py"),
        REPO / "src" / "nexus_matcher" / "shared" / "logging.py",
    ]
    assert surfaces, "no service surfaces found; this check would pass over nothing"

    offenders = []
    for path in surfaces:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"""=\s*["']\d+\.\d+\.\d+["']""", line) and "version" in line.lower():
                offenders.append(f"{path.relative_to(REPO).as_posix()}:{n}: {line.strip()}")

    assert not offenders, (
        "a version literal on a surface that reports the service version -- this is the "
        "drift that made it report a deleted release:\n  " + "\n  ".join(offenders)
    )
