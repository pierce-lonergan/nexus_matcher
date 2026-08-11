"""
tests/packaging/test_java_client_contract.py | Env: ALL

The Java client's DTOs are a SECOND DESCRIPTION of the wire contract, so they will drift.

`clients/java/` restates every published schema as a Java record: `MatchCandidateView`
becomes `MatchCandidate`, `GovernanceView` becomes `Governance`, the `MatchDecision` enum
becomes a Java enum, and each published failure status becomes an exception subclass. None
of that is generated -- it is hand-written -- so the moment `schemas.py` gains a member, the
Java side is silently one member short and nothing in either language notices.

That is not hypothetical here. While this client was being written, three wire changes
landed in the same working tree:

  * `MatchResponseView` gained a top-level `vocabulary` block,
  * `GovernanceView` gained a sixth member, `enhancement`,
  * `MatchRequest` moved from `extra="forbid"` to `extra="ignore"`.

The client was updated by hand for all three, and `docs/API_REFERENCE.md` was not updated
for any of them. A hand-written client that happens to be correct today is exactly the
thing this repository keeps a gate on, because the next change lands while nobody is
looking at both files at once.

## What this checks, and why it parses text

The published schema is read from a live `create_app().openapi()` -- the same source a
generated client would use, not a copy of it. The Java side is read as TEXT, deliberately:

  * no JVM is required at pytest time, so this gate runs in every environment the Python
    suite runs in rather than only where Maven and a JDK happen to be installed;
  * it reads what an adopter's Jackson actually binds, which is the `@JsonProperty("...")`
    annotation on each record component, not what a comment claims.

Textual parsing is the weaker tool and it is chosen knowingly. Its failure mode is a FALSE
GREEN -- a regex that stops matching reports nothing missing -- which is the failure mode
this directory exists to prevent, so `test_the_java_source_parsers_are_not_vacuous` below
asserts each parser still sees the things it is supposed to see, and would fail if the
annotations were renamed or the switch reshaped out from under it.

## The skip

If `clients/java/` is not in the checkout there is nothing to compare and the tests skip on
a real runtime condition. `tests/packaging/conftest.py` escalates any skip in this
directory into a failed run unless it is declared, so these five are declared there by name
-- see `_OPT_IN_SKIPS`. That declaration is the written decision the conftest asks for: the
Java client is an optional part of this repository, and a source distribution that omits
`clients/` should not fail the Python packaging suite. Absent that entry, this gate would
be the sixth way this directory has found to switch itself off quietly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_JAVA_PACKAGE = (
    _REPO
    / "clients"
    / "java"
    / "src"
    / "main"
    / "java"
    / "io"
    / "github"
    / "pierce_lonergan"
    / "nexusmatcher"
)

_CLIENT_ABSENT = not _JAVA_PACKAGE.is_dir()

# The substring tests/packaging/conftest.py matches to recognise this as the declared
# opt-in. Keep the two in step; the conftest names this file and these test functions.
_ABSENT_REASON = "clients/java/ is not in this checkout"


# =============================================================================
# WHAT RESTATES WHAT
# =============================================================================

# Published schema name -> the Java file that restates it, relative to the package root.
#
# A schema absent from this table is NOT checked, so the table itself is asserted against
# the two the task pins (`MatchCandidateView`, `GovernanceView`) plus the two that carry the
# changes which already drifted (`VocabularyView`, `MatchResponseView`). Adding a published
# schema without adding it here is caught by
# `test_every_published_schema_is_either_restated_or_deliberately_not`.
_SCHEMA_TO_JAVA = {
    "MatchCandidateView": "model/MatchCandidate.java",
    "GovernanceView": "model/Governance.java",
    "VocabularyView": "model/Vocabulary.java",
    "MatchResponseView": "model/MatchResponse.java",
    "ExplainView": "model/Explain.java",
    "FieldSpec": "model/FieldSpec.java",
    "MatchRequest": "model/MatchRequest.java",
    "FeedbackRequest": "model/Feedback.java",
    "FeedbackResponseView": "model/FeedbackReceipt.java",
    "HealthResponse": "model/HealthStatus.java",
    "ReadinessResponse": "model/Readiness.java",
}

# The error envelope is NOT a record on the Java side -- it is decoded straight into the
# exception hierarchy, which is the point of it being typed there. So each published member
# of `ErrorDetail` is pinned to the thing on `NexusMatcherException` that carries it.
# `message` goes to `Throwable`, so what is asserted is that it reaches `super(...)`.
_ERROR_DETAIL_CARRIERS = {
    "code": "errorCode()",
    "details": "details()",
    "message": "super(message",
}

# Published schemas with no Java counterpart, each with the reason. `ErrorDetail` and
# `ErrorResponse` are covered by `test_the_error_envelope_is_carried_by_the_exception_type`
# instead; `MatchDecision` is an enum and is covered by its own test.
_NOT_RESTATED_AS_A_RECORD = {
    "ErrorDetail": "carried by the NexusMatcherException hierarchy, not a record",
    "ErrorResponse": "carried by the NexusMatcherException hierarchy, not a record",
    "MatchDecision": "an enum; see test_every_match_decision_value_exists_in_the_java_enum",
}


# =============================================================================
# READING THE TWO SIDES
# =============================================================================


def _openapi() -> dict:
    """
    The published contract, from a live app rather than a checked-in copy.

    `environ={}` so no dictionary or vocabulary is loaded: the schema component of the app
    is registered by the router and does not depend on a matcher being ready, and starting
    without one keeps this gate off the encoder's several-second load.
    """
    from nexus_matcher.presentation.api.app import create_app

    return create_app(configure_logs=False, environ={}).openapi()


def _java(relative: str) -> str:
    """
    One Java source file.

    A missing file when `clients/java/` IS present is a hard failure, never a skip: it
    means the client was restructured and this gate is now comparing against nothing.
    """
    path = _JAVA_PACKAGE / relative
    assert path.is_file(), (
        f"{path} does not exist. The Java client was restructured and "
        f"tests/packaging/test_java_client_contract.py still points at the old layout, so "
        f"it is checking nothing."
    )
    return path.read_text(encoding="utf-8")


_JSON_PROPERTY = re.compile(r'@JsonProperty\(\s*"([^"]+)"\s*\)')


def _wire_names(source: str) -> set[str]:
    """Every wire name the record binds, as Jackson sees it."""
    return set(_JSON_PROPERTY.findall(source))


def _enum_constants(source: str, enum_name: str) -> set[str]:
    """The constants declared by `enum <enum_name>`, up to the first `;` or `}`."""
    match = re.search(rf"enum\s+{enum_name}\s*\{{(.*?)(?:;|\}})", source, re.DOTALL)
    if match is None:
        return set()
    body = re.sub(r"/\*.*?\*/", " ", match.group(1), flags=re.DOTALL)
    body = re.sub(r"//[^\n]*", " ", body)
    return set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", body))


def _mapped_statuses(source: str) -> set[int]:
    """
    Every status the `of(...)` switch maps to a SPECIFIC exception.

    The `default ->` arm is deliberately not counted. Every status maps to something, so
    counting it would make this check vacuous -- the question is which statuses get a class
    a caller can catch by name.
    """
    statuses: set[int] = set()
    for arm in re.findall(r"case\s+([0-9,\s]+?)\s*->", source):
        for number in re.findall(r"\d+", arm):
            statuses.add(int(number))
    return statuses


def _published_failure_statuses(spec: dict) -> set[int]:
    """Every >=400 status any published operation declares it can answer."""
    statuses: set[int] = set()
    for operations in spec["paths"].values():
        for operation in operations.values():
            for code in operation.get("responses", {}):
                if str(code).isdigit() and int(code) >= 400:
                    statuses.add(int(code))
    return statuses


# =============================================================================
# THE GATES
# =============================================================================


@pytest.mark.skipif(_CLIENT_ABSENT, reason=_ABSENT_REASON)
def test_every_published_schema_property_has_a_java_record_component():
    """
    The gate the `enhancement` and `vocabulary` additions would have tripped.

    Every property of every restated schema must be bound by name on the Java side. A
    member added to `schemas.py` and not to the record is a member a Java caller cannot
    read -- and for `GovernanceView` that is a governance instruction going missing.
    """
    schemas = _openapi()["components"]["schemas"]
    findings: list[str] = []
    for schema_name, java_file in sorted(_SCHEMA_TO_JAVA.items()):
        published = set(schemas.get(schema_name, {}).get("properties", {}))
        assert published, (
            f"{schema_name} publishes no properties, so this comparison proves nothing. "
            f"It was renamed or removed and _SCHEMA_TO_JAVA is stale."
        )
        bound = _wire_names(_java(java_file))
        missing = published - bound
        if missing:
            findings.append(
                f"{schema_name} -> {java_file}: no Java component binds "
                f"{sorted(missing)} (the record binds {sorted(bound)})"
            )

    assert not findings, (
        "the Java client's DTOs no longer cover the published contract, so a Java caller "
        "silently cannot read these members:\n  "
        + "\n  ".join(findings)
        + "\nAdd the component with its @JsonProperty to the record named above. If the "
        "member is deliberately not carried, say so in _SCHEMA_TO_JAVA's neighbourhood "
        "rather than deleting the entry."
    )


@pytest.mark.skipif(_CLIENT_ABSENT, reason=_ABSENT_REASON)
def test_every_published_schema_is_either_restated_or_deliberately_not():
    """
    Guards the table itself.

    The check above is only as complete as `_SCHEMA_TO_JAVA`, and a new published schema
    that nobody adds to it would be covered by nothing while the suite stayed green -- the
    vacuity this directory was built to catch, one level up.
    """
    published = set(_openapi()["components"]["schemas"])
    accounted = set(_SCHEMA_TO_JAVA) | set(_NOT_RESTATED_AS_A_RECORD)
    unaccounted = published - accounted
    assert not unaccounted, (
        f"these published schemas are in neither _SCHEMA_TO_JAVA nor "
        f"_NOT_RESTATED_AS_A_RECORD, so nothing checks whether the Java client covers "
        f"them: {sorted(unaccounted)}"
    )

    stale = accounted - published
    assert not stale, (
        f"these entries name schemas the service no longer publishes, so they are "
        f"checking a contract that no longer exists: {sorted(stale)}"
    )

    # The two the drift gate was asked for, pinned by name so a future tidy-up cannot
    # quietly drop them and leave the gate technically passing.
    for required in ("MatchCandidateView", "GovernanceView"):
        assert required in _SCHEMA_TO_JAVA, f"{required} must stay covered by this gate"


@pytest.mark.skipif(_CLIENT_ABSENT, reason=_ABSENT_REASON)
def test_the_error_envelope_is_carried_by_the_exception_type():
    """
    `{"error": {code, message, details}}` is the one failure shape, and the Java client
    decodes it into exceptions rather than a record. Each published member still has to
    land somewhere a caller can reach.
    """
    schemas = _openapi()["components"]["schemas"]

    assert set(schemas["ErrorResponse"]["properties"]) == {"error"}, (
        "the error envelope gained a member outside `error`, which the Java decoder does "
        "not look at"
    )

    published = set(schemas["ErrorDetail"]["properties"])
    assert published == set(_ERROR_DETAIL_CARRIERS), (
        f"ErrorDetail publishes {sorted(published)} but this gate knows how "
        f"{sorted(_ERROR_DETAIL_CARRIERS)} are carried. A new member is a member no Java "
        f"caller can read until NexusMatcherException carries it too."
    )

    source = _java("error/NexusMatcherException.java")
    missing = sorted(
        f"{member} (expected {carrier!r})"
        for member, carrier in _ERROR_DETAIL_CARRIERS.items()
        if carrier not in source
    )
    assert not missing, (
        "NexusMatcherException no longer carries these published members of the error "
        "envelope: " + ", ".join(missing)
    )


@pytest.mark.skipif(_CLIENT_ABSENT, reason=_ABSENT_REASON)
def test_every_match_decision_value_exists_in_the_java_enum():
    """
    `MatchDecision` is the ONE closed vocabulary in this client -- it is the library's own,
    not a caller's taxonomy -- so it is the one place a Java enum is correct, and the one
    place a new server value would throw on decode rather than degrade.
    """
    published = set(_openapi()["components"]["schemas"]["MatchDecision"]["enum"])
    assert published, "MatchDecision publishes no values; this comparison proves nothing"

    declared = _enum_constants(_java("model/MatchDecision.java"), "MatchDecision")
    missing = published - declared

    assert not missing, (
        f"the service publishes MatchDecision values the Java enum does not declare: "
        f"{sorted(missing)} (Java declares {sorted(declared)}). A response carrying one of "
        f"these fails to decode, so the whole match is lost rather than one field."
    )


@pytest.mark.skipif(_CLIENT_ABSENT, reason=_ABSENT_REASON)
def test_every_published_status_code_has_a_mapped_exception():
    """
    Every failure the API documents should arrive as a class a caller can catch by name.

    Falling through to the base `NexusMatcherException` is not counted as mapped: the whole
    reason the client has `PayloadTooLargeException` and `ServiceUnavailableException` is
    that "chunk it and retry" and "retry later" are different branches, and a caller who
    has to read `httpStatus()` to tell them apart has the string test back.
    """
    published = _published_failure_statuses(_openapi())
    assert published, "no failure statuses are published; this comparison proves nothing"

    mapped = _mapped_statuses(_java("error/NexusMatcherException.java"))
    missing = published - mapped

    assert not missing, (
        f"these statuses are published by the API and fall through to the generic "
        f"exception in NexusMatcherException.of: {sorted(missing)} (mapped: "
        f"{sorted(mapped)}). Add a case and a subclass, or the caller cannot branch on "
        f"the failure without reading the status by hand."
    )


@pytest.mark.skipif(_CLIENT_ABSENT, reason=_ABSENT_REASON)
def test_the_unconfigured_vocabulary_sentinel_matches_the_server():
    """
    `Vocabulary.UNCONFIGURED_OPEN_CLASSIFICATION` is a COPY of a server constant.

    It is the one string the client is allowed to know, because `isConfigured()` is
    implemented by comparing against it -- and that makes it a second description of
    `domain/governance.py:OPEN_CLASSIFICATION` with no compiler between them. If the server
    changes its sentinel, nothing fails: `isConfigured()` simply starts answering `true`
    for a deployment carrying no vocabulary at all, which is the wrong answer in the
    direction that matters, since the caller then trusts `openClassification` as a real
    tier name.

    Not a taxonomy: the whole point of the value is that it is deliberately not a word any
    real vocabulary uses.
    """
    from nexus_matcher.domain.governance import OPEN_CLASSIFICATION

    source = _java("model/Vocabulary.java")
    match = re.search(
        r'UNCONFIGURED_OPEN_CLASSIFICATION\s*=\s*"([^"]+)"',
        source,
    )
    assert match is not None, (
        "Vocabulary.java no longer declares UNCONFIGURED_OPEN_CLASSIFICATION as a string "
        "literal, so this comparison proves nothing"
    )

    assert match.group(1) == OPEN_CLASSIFICATION, (
        f"the Java client's unconfigured-vocabulary sentinel is {match.group(1)!r} but the "
        f"service's is {OPEN_CLASSIFICATION!r}. Vocabulary.isConfigured() is implemented by "
        f"comparing against this literal, so it now reports an UNCONFIGURED deployment as "
        f"configured and a caller reads the sentinel as a real tier."
    )


@pytest.mark.skipif(_CLIENT_ABSENT, reason=_ABSENT_REASON)
def test_the_java_source_parsers_are_not_vacuous():
    """
    The control, and the reason this file is allowed to parse Java with regexes.

    Every check above reports drift as a NON-EMPTY difference, so a parser that quietly
    stopped matching -- annotations renamed, the switch reshaped, the enum reformatted --
    would report nothing missing and go green forever. Each parser is therefore asserted to
    still see real content, and to still be able to SEE an omission when one exists.
    """
    # It sees the real files.
    governance = _wire_names(_java("model/Governance.java"))
    assert {"code", "classification", "enhancement"} <= governance, (
        f"the @JsonProperty scan no longer reads Governance.java; it found {sorted(governance)}"
    )

    decisions = _enum_constants(_java("model/MatchDecision.java"), "MatchDecision")
    assert {"AUTO_APPROVE", "REVIEW", "REJECT"} <= decisions, (
        f"the enum scan no longer reads MatchDecision.java; it found {sorted(decisions)}"
    )

    statuses = _mapped_statuses(_java("error/NexusMatcherException.java"))
    assert {413, 422, 500, 503, 504} <= statuses, (
        f"the switch scan no longer reads NexusMatcherException.java; it found {sorted(statuses)}"
    )

    # And it can see an omission. These are the shapes the checks above would have to
    # notice, run against synthetic sources so the control does not depend on the client
    # being broken.
    assert _wire_names('record X(@JsonProperty("kept") String a) {}') == {"kept"}
    assert "dropped" not in _wire_names('record X(@JsonProperty("kept") String a) {}')
    assert _enum_constants("enum MatchDecision { AUTO_APPROVE, REVIEW }", "MatchDecision") == {
        "AUTO_APPROVE",
        "REVIEW",
    }
    assert _mapped_statuses("case 400, 422 -> a; case 413 -> b; default -> c;") == {400, 422, 413}
    assert _mapped_statuses("default -> c;") == set(), (
        "the default arm must not count as a mapped status, or this gate passes for a "
        "client that maps nothing at all"
    )
