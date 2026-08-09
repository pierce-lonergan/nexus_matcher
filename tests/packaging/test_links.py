"""
tests.packaging.test_links | Layer: GATE
Every markdown link in this repository resolves. In the normal suite, not only at release.

Why this exists
---------------
README.md is the PyPI `long_description`. On pypi.org a relative link like
`[docs](docs/API_REFERENCE.md)` resolves against pypi.org and 404s, and in 2.0.0 that
killed the README's entire Documentation section for everyone arriving from the package
page -- twelve dead links, plus one in-page anchor (`#known-limits`) that named no heading
and scrolled nowhere.

A regression test for it was written during the 2.0.1 work and never shipped: it lived in
a scratchpad, so the only thing standing between the repo and a repeat was
`scripts/release_preflight.py`, which runs at release time. By then the README is already
what is about to be published.

What this adds over release_preflight
-------------------------------------
`check_readme_links` is not reimplemented here -- it is CALLED, which makes the release
gate itself exercised on every test run. On top of it:

  * **anchors**, which release_preflight does not look at, resolved against the real
    headings using GitHub's slug algorithm (the one that turns `Documentation — retractions`
    into `documentation--retractions`, doubling the hyphen where the em dash was);
  * **every other markdown file** -- QUICKSTART, CONTRIBUTING, CHANGELOG and all 33 files
    under docs/ -- where relative links ARE legal and must therefore point at something
    that exists, including cross-file anchors like `../BENCHMARK_REGISTRY.md#exp-rerank`;
  * **absolute GitHub URLs into this repository**, resolved against the working tree. That
    catches a renamed doc on the branch that renames it, which a live 200 check against
    `main` cannot do until after the merge.

The badge gap
-------------
The first regex written for this missed `[![alt](image)](target)`: the nested brackets
defeat a `[^]]+` label. Badges are exactly where a stale link hides, because they are
written once and never read again. `test_no_link_shape_is_silently_skipped` is the general
form of that lesson -- every `](` in the file must fall inside something the parser
matched, so the NEXT unhandled shape fails loudly instead of being silently uncovered.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

# The markdown this repository publishes. README first: it is the PyPI long_description
# and the only file with the no-relative-links rule.
ROOT_MARKDOWN = ("README.md", "QUICKSTART.md", "CONTRIBUTING.md", "CHANGELOG.md")

# Absolute URLs into this repository. A link of this shape is a promise that the path
# exists, and it is checkable offline.
SELF_URL = re.compile(
    r"https://github\.com/pierce-lonergan/nexus_matcher/(?:blob|tree)/[^/]+/(.*)",
    re.I,
)


# =============================================================================
# MARKDOWN PARSING
# =============================================================================

_FENCE = re.compile(r"^\s*(```|~~~)")

# A link, with the badge shape handled explicitly. The label alternation is
# `![alt](image)` OR any bracket-free text; a bare `![...]` image is excluded by the
# lookbehind so an image is never mistaken for a link.
_LINK = re.compile(r"(?<!!)\[(?:!\[[^\]]*\]\([^)]*\)|[^\[\]]*)\]\(\s*<?([^)\s>]+)>?[^)]*\)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.M)
_HTML_ANCHOR = re.compile(r'<a\s+(?:name|id)="([^"]+)"')


def strip_code(text: str) -> str:
    """
    Blank out fenced blocks and inline code, preserving line numbers.

    Not cosmetic. README.md's bash examples contain lines beginning `# 1. Build ...`,
    which read as level-1 headings, and a `#anchor` pointing at one of those would then
    "resolve" against a comment in a shell snippet. Line count is preserved so failure
    messages can name the real line.
    """
    out: list[str] = []
    fence = ""
    for line in text.splitlines():
        match = _FENCE.match(line)
        if match:
            if not fence:
                fence = match.group(1)
                out.append("")
                continue
            if line.strip().startswith(fence):
                fence = ""
                out.append("")
                continue
        out.append("" if fence else line)
    return re.sub(r"`[^`\n]*`", "", "\n".join(out))


def github_slug(title: str) -> str:
    """
    GitHub's heading -> anchor transform, as github-slugger implements it.

    The subtle part is the LAST step: each space becomes one hyphen, runs are NOT
    collapsed. `Documentation — retractions` loses the em dash and keeps both spaces
    around it, giving `documentation--retractions`. Collapsing them here reports five
    live links in docs/ as broken, which is how this was found.
    """
    slug = title.strip().lower()
    slug = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", slug)  # a link in a heading: keep the text
    slug = re.sub(r"[`*_~]", "", slug)  # inline emphasis markers do not survive
    slug = re.sub(r"[^\w\s\-]", "", slug, flags=re.UNICODE)
    return re.sub(r"\s", "-", slug.strip())


def anchors_of(path: Path) -> set[str]:
    """Every fragment `#x` that resolves in this file: headings plus explicit HTML anchors."""
    if not path.is_file():
        return set()
    text = strip_code(path.read_text(encoding="utf-8"))
    seen: dict[str, int] = {}
    found: set[str] = set()
    for _hashes, title in _HEADING.findall(text):
        base = github_slug(title)
        count = seen.get(base, 0)
        seen[base] = count + 1
        # GitHub disambiguates repeats with -1, -2, ... which is why two `### Fixed`
        # headings do not silently give one dead anchor.
        found.add(base if count == 0 else f"{base}-{count}")
    found.update(_HTML_ANCHOR.findall(text))
    return found


def links_in(path: Path) -> list[tuple[str, int]]:
    """[(target, line number)] for every markdown link outside code."""
    text = strip_code(path.read_text(encoding="utf-8"))
    return [(m.group(1), text[: m.start()].count("\n") + 1) for m in _LINK.finditer(text)]


def markdown_files() -> list[Path]:
    files = [REPO / name for name in ROOT_MARKDOWN if (REPO / name).is_file()]
    files.extend(sorted(DOCS.rglob("*.md")))
    return files


# =============================================================================
# VACUITY GUARDS
# =============================================================================


def test_there_is_markdown_to_check():
    """Every assertion below is trivially true over an empty file list."""
    files = markdown_files()
    assert len(files) >= 30, f"only {len(files)} markdown files found -- the scan is broken"
    assert (REPO / "README.md") in files
    total = sum(len(links_in(p)) for p in files)
    assert total >= 50, f"only {total} links found across {len(files)} files -- parser is broken"


def test_no_link_shape_is_silently_skipped():
    """
    The general form of the badge lesson.

    A regex that does not match a link shape does not report an error -- it reports
    nothing, and the file looks clean. So every `](` in the tree must fall inside a span
    the parser DID match. An unhandled shape fails here, by name and line, instead of
    quietly leaving links unchecked.
    """
    uncovered: list[str] = []
    for path in markdown_files():
        text = strip_code(path.read_text(encoding="utf-8"))
        spans = [(m.start(), m.end()) for m in _LINK.finditer(text)]
        spans += [(m.start(), m.end()) for m in _IMAGE.finditer(text)]
        for hit in re.finditer(r"\]\(", text):
            if not any(start <= hit.start() < end for start, end in spans):
                line = text[: hit.start()].count("\n") + 1
                snippet = text[max(0, hit.start() - 50) : hit.start() + 30].replace("\n", " ")
                uncovered.append(f"{path.relative_to(REPO).as_posix()}:{line}  ...{snippet}...")
    assert not uncovered, (
        "these link-ish constructs matched no pattern, so nothing checked them:\n  "
        + "\n  ".join(uncovered)
    )


def test_the_badge_shape_is_parsed():
    """
    Pinned against a synthetic fixture with hand-verified expected values, because the
    first regex written for this file silently returned nothing for exactly this input.
    """
    fixture = (
        "[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](docs/BADGE.md)\n"
        "![just an image](docs/NOT_A_LINK.png)\n"
        "[plain](docs/PLAIN.md)\n"
        "[titled](docs/TITLED.md 'hover text')\n"
        "[angled](<docs/ANGLED.md>)\n"
    )
    targets = [m.group(1) for m in _LINK.finditer(fixture)]
    assert targets == [
        "docs/BADGE.md",  # the badge TARGET, not its image
        "docs/PLAIN.md",
        "docs/TITLED.md",
        "docs/ANGLED.md",
    ], targets


def test_the_slug_matches_githubs():
    """
    Absolute expected values, not a comparison against a second implementation of the
    same idea -- an oracle built the same way would share the same mistake (H-004).
    Every case below was verified against a live anchor in this repository.
    """
    assert github_slug("Documentation — retractions") == "documentation--retractions"
    assert github_slug("8. Monitoring & Observability") == "8-monitoring--observability"
    assert github_slug("What is actually implemented") == "what-is-actually-implemented"
    assert github_slug("`code` in a heading") == "code-in-a-heading"
    assert github_slug("Dictionary aliasing: a gain that inverts at scale") == (
        "dictionary-aliasing-a-gain-that-inverts-at-scale"
    )


def test_code_fences_do_not_contribute_headings():
    """README's bash examples contain `# 1. Build ...`, which is not a heading."""
    readme_anchors = anchors_of(REPO / "README.md")
    assert "limitations" in readme_anchors, "real headings vanished -- stripping is too greedy"
    assert not any(a.startswith("1-build") for a in readme_anchors), (
        "a shell comment inside a fenced block was read as a heading"
    )


# =============================================================================
# THE README -- the PyPI long_description
# =============================================================================


def _release_preflight():
    """
    Import the release script as a module so its check runs here rather than only at
    release. Calling it, instead of copying it, means a regression in THAT gate fails
    THIS test -- one implementation, two occasions.
    """
    path = REPO / "scripts" / "release_preflight.py"
    spec = importlib.util.spec_from_file_location("release_preflight_under_test", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `Report` is a dataclass in a module using
    # `from __future__ import annotations`, so dataclasses resolves its string
    # annotations by looking the defining module up in sys.modules. Unregistered, that
    # lookup returns None and constructing Report() dies inside the stdlib.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_readme_has_no_relative_links_per_the_release_gate():
    """
    The release gate, run on every test run. A relative link here 404s on PyPI, where
    README.md is the long_description and resolves against pypi.org.
    """
    preflight = _release_preflight()
    report = preflight.Report()
    preflight.check_readme_links(report)
    assert not report.failed, [detail for _name, detail in report.failed]
    assert report.passed, "check_readme_links reported neither pass nor fail -- it did nothing"


def test_readme_has_no_relative_links_by_this_parser_too():
    """
    The same absolute property, seen by the parser that also handles angle-bracket
    destinations and link titles. Not an oracle for the other -- both are measured
    against `zero`, which is a fact about the file rather than about either parser.
    """
    relative = [
        f"README.md:{line} -> {target}"
        for target, line in links_in(REPO / "README.md")
        if not target.startswith(("http://", "https://", "#", "mailto:"))
    ]
    assert not relative, "relative links 404 on the PyPI project page:\n  " + "\n  ".join(relative)


# =============================================================================
# ANCHORS AND RELATIVE LINKS, EVERYWHERE
# =============================================================================


def test_every_in_page_anchor_points_at_a_real_heading():
    """`#known-limits` named no heading in 2.0.0's README and scrolled nowhere."""
    broken: list[str] = []
    for path in markdown_files():
        available = anchors_of(path)
        for target, line in links_in(path):
            if target.startswith("#") and target[1:] not in available:
                broken.append(f"{path.relative_to(REPO).as_posix()}:{line} -> {target}")
    assert not broken, "in-page anchors naming no heading:\n  " + "\n  ".join(broken)


def test_every_relative_link_resolves_on_disk():
    """
    Outside README a relative link is correct markdown -- but only if the file is there.
    Cross-file fragments (`../BENCHMARK_REGISTRY.md#exp-rerank`) are checked too: pointing
    at a real file and a heading that was since renamed is still a dead link.
    """
    broken: list[str] = []
    for path in markdown_files():
        for target, line in links_in(path):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            where = f"{path.relative_to(REPO).as_posix()}:{line} -> {target}"
            sub, _, fragment = target.partition("#")
            resolved = (path.parent / sub).resolve()
            if not resolved.exists():
                broken.append(f"{where}  (no such file)")
            elif fragment and resolved.is_file() and fragment not in anchors_of(resolved):
                broken.append(f"{where}  (file exists, anchor does not)")
    assert not broken, "relative links that go nowhere:\n  " + "\n  ".join(broken)


def test_every_github_url_into_this_repo_resolves_in_the_working_tree():
    """
    The 2.0.1 fix turned twelve relative README links into absolute GitHub URLs. That
    moved the failure mode rather than removing it: renaming `docs/API_REFERENCE.md` now
    leaves a URL that 404s, and no amount of local link checking notices unless the URL is
    resolved back to a path.

    Checked against the WORKING TREE, deliberately, not against a live 200: this fails on
    the branch that does the rename, before the push, rather than after a merge to main.
    """
    broken: list[str] = []
    for path in markdown_files():
        for target, line in links_in(path):
            match = SELF_URL.match(target)
            if not match:
                continue
            where = f"{path.relative_to(REPO).as_posix()}:{line} -> {target}"
            sub, _, fragment = match.group(1).partition("#")
            resolved = (REPO / sub) if sub else REPO
            if not resolved.exists():
                broken.append(f"{where}  (no such path in the repo)")
            elif fragment and resolved.is_file() and fragment not in anchors_of(resolved):
                broken.append(f"{where}  (path exists, anchor does not)")
    assert not broken, "GitHub URLs into this repo that name nothing:\n  " + "\n  ".join(broken)


def test_the_self_url_pattern_still_matches_something():
    """
    A pattern that matches nothing makes the test above vacuous. The README's entire
    Documentation section is written in this shape, so zero matches means either the
    README was rewritten or the pattern rotted.
    """
    matched = [
        target
        for path in markdown_files()
        for target, _line in links_in(path)
        if SELF_URL.match(target)
    ]
    assert len(matched) >= 8, f"only {len(matched)} self-referencing GitHub URLs found"


# =============================================================================
# LIVE URLS -- opt-in
# =============================================================================

NETWORK = os.environ.get("NEXUS_LINK_NETWORK") == "1"


@pytest.mark.skipif(
    not NETWORK,
    reason=(
        "set NEXUS_LINK_NETWORK=1 to check live URLs. Off by default for one concrete "
        "reason, not squeamishness: every self-URL names `main`, so a doc ADDED on a "
        "feature branch legitimately 404s until merge, and this test would fail the "
        "branch that wrote the doc. The offline resolution above covers the same links "
        "against the working tree, which is stricter on a branch and never flaky."
    ),
)
def test_every_url_in_the_readme_returns_200():
    """
    The live check, for the release path. Verified to have teeth: a bogus blob URL under
    this repository returns HTTP 404, so a 200 here is evidence and not a redirect that
    swallows everything.

    Network errors that are not HTTP status codes (DNS, timeouts, proxies) are reported
    as failures too -- an unreachable documentation URL is a broken documentation URL
    from the reader's point of view, and this test only runs when it was asked for.
    """
    import urllib.error
    import urllib.request

    urls = sorted(
        {
            target
            for target, _line in links_in(REPO / "README.md")
            if target.startswith(("http://", "https://"))
        }
    )
    assert urls, "no absolute URLs in README.md -- nothing to check"

    broken: list[str] = []
    for url in urls:
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "nexus-matcher-link-check"}
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                if response.status >= 400:
                    broken.append(f"{url} -> {response.status}")
        except urllib.error.HTTPError as exc:
            broken.append(f"{url} -> HTTP {exc.code}")
        except Exception as exc:
            broken.append(f"{url} -> {type(exc).__name__}: {exc}")
    assert not broken, "README URLs that do not return 200:\n  " + "\n  ".join(broken)
