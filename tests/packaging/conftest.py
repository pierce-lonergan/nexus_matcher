"""
tests/packaging/conftest.py | Env: ALL

A skipped packaging contract fails the run.

test_no_silent_skips.py reads the SOURCE and rejects unconditional skips written into it.
This is the same rule enforced at runtime, where it catches what no source scan can: a skip
raised from a fixture, from a plugin, or from library code several frames below a helper --
which is precisely the shape the three importorskips had, four call levels down inside
`_load_pyproject()`, reading like a helper rather than like a deletion of thirteen tests.

Failing the RUN rather than the individual test is deliberate. By the time a skip is
reported the test is over and its outcome cannot be rewritten without hookwrapper APIs that
differ across the pytest versions this project supports (>=7.4). `pytest_sessionfinish` is a
plain hook on every one of them, and a non-zero exit is what a CI step reads anyway.

ONE EXEMPTION, by name and by reason: the live-URL check in test_links.py is opt-in behind
NEXUS_LINK_NETWORK=1, because every self-URL in the docs names `main` and a doc added on a
branch legitimately 404s until merge. That is a test that must be REQUESTED, not one that
quietly vanishes. Adding to this list is a written decision, and the entry has to name both
the test and the switch that turns it on.
"""

from __future__ import annotations

# nodeid suffix -> the substring of the skip reason that proves it is the opt-in, not an
# absent dependency. Both must match, so a skip for any other reason still fails the run.
_OPT_IN_SKIPS = {
    "test_links.py::test_every_url_in_the_readme_returns_200": "NEXUS_LINK_NETWORK",
}

_PACKAGING = "tests/packaging/"


def is_allowed_skip(nodeid: str, reason: str) -> bool:
    """True only for a skip this directory has declared as an opt-in, with its switch."""
    for suffix, switch in _OPT_IN_SKIPS.items():
        if nodeid.endswith(suffix) and switch in reason:
            return True
    return False


def _reason(report) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def pytest_runtest_logreport(report) -> None:
    """Record every skipped packaging test, whatever raised it."""
    if not report.skipped:
        return
    nodeid = report.nodeid.replace("\\", "/")
    if not nodeid.startswith(_PACKAGING):
        return  # this hook fires for the whole session; only this directory is governed
    reason = _reason(report)
    if is_allowed_skip(nodeid, reason):
        return
    session_skips.append(f"{nodeid} -- {reason}")


# Module-level so the list survives whichever hook order pytest chooses, and so
# test_no_silent_skips.py can assert the mechanism exists rather than trusting it.
session_skips: list[str] = []


def pytest_sessionfinish(session, exitstatus) -> None:
    """A packaging contract that did not run is a failed run, not a green one with a note."""
    if not session_skips:
        return
    print(
        "\nERROR: packaging contracts were SKIPPED, so they checked nothing while the "
        "suite reported success:\n  " + "\n  ".join(session_skips) + "\n"
        "Make the missing thing a hard failure instead -- see _toml_parser() in "
        "test_extras_graph.py -- or, if the skip is a deliberate opt-in, declare it in "
        "_OPT_IN_SKIPS in tests/packaging/conftest.py with the switch that enables it."
    )
    if exitstatus == 0:
        session.exitstatus = 1
