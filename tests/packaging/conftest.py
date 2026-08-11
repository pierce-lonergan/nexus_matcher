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

TWO EXEMPTIONS, each by name and by reason.

The first is the live-URL check in test_links.py, opt-in behind NEXUS_LINK_NETWORK=1,
because every self-URL in the docs names `main` and a doc added on a branch legitimately
404s until merge. That is a test that must be REQUESTED, not one that quietly vanishes.

The second is the Java client drift gate in test_java_client_contract.py, which compares
the published `/openapi.json` against the hand-written DTOs under `clients/java/`. That
directory is an OPTIONAL part of this repository: it is not built by the Python packaging,
not shipped in the wheel, and a source distribution or sparse checkout that omits
`clients/` has nothing for the gate to compare against. The condition is the absence of the
directory itself rather than a switch, which is why the reason is matched on a phrase and
not an environment variable -- and the distinction that matters is preserved, because a
skip of those tests for ANY OTHER reason (an import failure, a missing fixture) still fails
the run. What is exempted is "the artifact under test is not in this checkout", not "the
test would not pass".

Adding to this list is a written decision, and the entry has to name both the test and the
condition that turns it off.
"""

from __future__ import annotations

# nodeid suffix -> the substring of the skip reason that proves it is the opt-in, not an
# absent dependency. Both must match, so a skip for any other reason still fails the run.
_JAVA_CLIENT_ABSENT = "clients/java/ is not in this checkout"

_OPT_IN_SKIPS = {
    "test_links.py::test_every_url_in_the_readme_returns_200": "NEXUS_LINK_NETWORK",
    # The drift gate, one entry per test, because the match is on the full nodeid and a
    # file-wide wildcard would exempt any test later added to the file without a decision.
    "test_java_client_contract.py::test_every_published_schema_property_has_a_java_record_component": _JAVA_CLIENT_ABSENT,
    "test_java_client_contract.py::test_every_published_schema_is_either_restated_or_deliberately_not": _JAVA_CLIENT_ABSENT,
    "test_java_client_contract.py::test_the_error_envelope_is_carried_by_the_exception_type": _JAVA_CLIENT_ABSENT,
    "test_java_client_contract.py::test_every_match_decision_value_exists_in_the_java_enum": _JAVA_CLIENT_ABSENT,
    "test_java_client_contract.py::test_every_published_status_code_has_a_mapped_exception": _JAVA_CLIENT_ABSENT,
    "test_java_client_contract.py::test_the_unconfigured_vocabulary_sentinel_matches_the_server": _JAVA_CLIENT_ABSENT,
    "test_java_client_contract.py::test_the_java_source_parsers_are_not_vacuous": _JAVA_CLIENT_ABSENT,
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
