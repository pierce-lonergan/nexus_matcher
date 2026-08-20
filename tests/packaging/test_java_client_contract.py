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
    "ScoringContractView": "model/ScoringContract.java",
    "SourceMetadataView": "model/SourceMetadata.java",
    "ExplainView": "model/Explain.java",
    "FieldSpec": "model/FieldSpec.java",
    "MatchRequest": "model/MatchRequest.java",
    "FeedbackRequest": "model/Feedback.java",
    "FeedbackResponseView": "model/FeedbackReceipt.java",
    "HealthResponse": "model/HealthStatus.java",
    "ReadinessResponse": "model/Readiness.java",
    # The lookup plane.
    "LookupRequest": "model/LookupRequest.java",
    "LookupEntryView": "model/LookupEntry.java",
    "LookupResponseView": "model/LookupResponse.java",
    # GET /api/v1/status. The Java names drop the `View` suffix, which is a server-side
    # naming convention rather than part of the contract, and `StatusResponseView` becomes
    # `ServiceStatus` because `Status` alone collides with half of every Java codebase.
    "StatusResponseView": "model/ServiceStatus.java",
    "StatusWarningView": "model/StatusWarning.java",
    "DictionaryStatusView": "model/DictionaryStatus.java",
    "EncoderStatusView": "model/EncoderStatus.java",
    "ThresholdsView": "model/Thresholds.java",
    "ServiceLimitsView": "model/ServiceLimits.java",
    # POST /api/v1/diag/retrieval.
    "RetrievalDiagnosticRequest": "model/RetrievalDiagnosticRequest.java",
    "RetrievalDiagnosticView": "model/RetrievalDiagnostic.java",
    "RetrievalChannelView": "model/RetrievalChannel.java",
    "RetrievalCandidateView": "model/RetrievalCandidate.java",
    "ExpectedPlacementView": "model/ExpectedPlacement.java",
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
    "FieldDecision": "an enum; see test_every_match_decision_value_exists_in_the_java_enum",
}


# =============================================================================
# WHAT THE JAVA CLIENT DOES NOT CARRY YET -- a debt record, not an exemption
# =============================================================================

# The wire grows in this repository and the Java client is hand-written in another part of
# it, so a change can land on one side while the other is untouched. Silence about that is
# what this whole file exists to prevent; so is a table of excuses that nobody prunes.
#
# These two tables are the middle path. An entry RECORDS that a published member or schema
# is not on the Java side yet, names the file and the component that closes it, and is then
# policed in both directions by `test_the_java_debt_record_is_current`: an entry whose
# member stopped being published, or whose Java component has since been added, fails. The
# table can therefore only shrink by the client catching up, never rot into decoration.
#
# It is NOT a licence to ship an unreadable response. Every Java record is
# `@JsonIgnoreProperties(ignoreUnknown = true)` -- verified by
# `test_the_java_records_ignore_members_they_do_not_carry` below -- so an unbound member
# is invisible to a Java caller rather than fatal to one. That is the property that makes
# recording the debt honest instead of reckless, and it is asserted rather than assumed.

# (published schema, published member) -> why it is not bound yet, and what closes it.
_UNBOUND_ON_THE_JAVA_SIDE: dict[tuple[str, str], str] = {
    ("ThresholdsView", "absoluteScoreFloor"): (
        "NM-V2-01 AR-4 / NM-V2-03 SC-6: the active absolute-score floor is now on the "
        "status surface, where an operator can see which calibration is in force. Closes "
        'with `@JsonProperty("absoluteScoreFloor") Double absoluteScoreFloor` on '
        "model/Thresholds.java -- Double and not double, because null there means NO "
        "FLOOR IS CONFIGURED and a primitive would render that as 0.0, which is a "
        "configured floor at zero and a different deployment."
    ),
    ("ThresholdsView", "absoluteScoreMetric"): (
        "The metric that floor is compared against. Closes with "
        '`@JsonProperty("absoluteScoreMetric") String absoluteScoreMetric` on '
        "model/Thresholds.java. An open string, not an enum: `unknown` is a real value and "
        "a caller's own vector store may declare a metric this library has never seen."
    ),
    ("StatusResponseView", "calibration"): (
        "NM-V2-03 SC-7: the corpus the shipped defaults were fitted on, plus every setting "
        "this deployment has overridden. Closes with "
        '`@JsonProperty("calibration") Calibration calibration` on model/ServiceStatus.java '
        "and the two records named in _NOT_ON_THE_JAVA_SIDE_YET."
    ),
}

# Published schema -> why the Java client has no record for it yet. Separate from
# `_NOT_RESTATED_AS_A_RECORD`, which is for schemas that are carried by something OTHER
# than a record and are therefore finished; these are unfinished.
_NOT_ON_THE_JAVA_SIDE_YET: dict[str, str] = {
    "CalibrationView": (
        "reachable only through StatusResponseView.calibration, which is recorded in "
        "_UNBOUND_ON_THE_JAVA_SIDE. Closes as model/Calibration.java."
    ),
    "CalibrationCorpusView": (
        "reachable only through CalibrationView.corpus. Closes as model/CalibrationCorpus.java."
    ),
}


# =============================================================================
# THE PUBLISHED ENUMS
# =============================================================================

# Published enum schema -> (Java file, Java enum name). Every enum the service publishes has
# to appear here; `test_every_match_decision_value_exists_in_the_java_enum` fails on one that
# does not, because a closed Java enum bound to a vocabulary nobody is watching is the exact
# shape of the deserialisation break this file exists to prevent.
_SCHEMA_TO_JAVA_ENUM = {
    "MatchDecision": ("model/MatchDecision.java", "MatchDecision"),
    "FieldDecision": ("model/FieldDecision.java", "FieldDecision"),
}

# Constants the Java enum declares that the service does NOT publish, with the reason.
#
# There is exactly one, and it carries the client's answer to the question this whole seam
# exists for: what happens when a newer server sends a value an older client has never heard
# of. `MatchDecision` answers "refuse the response", which is right for a vocabulary the
# service has committed to freezing -- putting NO_MATCH on a NEW enum rather than widening
# that one IS that commitment. `FieldDecision` answers "degrade to UNKNOWN, keep the raw
# string, grant nothing", which is right for the vocabulary that was born by widening and is
# where the next verdict will land.
#
# The membership test below is the sharp half. If the service ever publishes a FieldDecision
# value literally spelled UNKNOWN, this gate goes red -- because at that moment the client's
# sentinel would silently start absorbing a real server verdict, and "this client could not
# read your answer" would become indistinguishable from the answer itself.
_CLIENT_SIDE_ENUM_SENTINELS = {
    "FieldDecision": {
        "UNKNOWN": (
            "the client-side sentinel for a verdict this build does not know. Never on the "
            "wire; FieldVerdict.wireValue() carries what the server actually sent."
        )
    },
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
    """
    The constants declared by `enum <enum_name>`, up to the first `;` or `}`.

    Comments are stripped BEFORE the body is located rather than after it, and the difference
    is not cosmetic. A javadoc on a constant routinely contains `{@link Foo#bar()}` or
    `{@code x}`, so a scan that hunts for the enum's closing brace in the raw source stops at
    the first brace inside a comment and reads only the constants above it.

    That under-read is loud in one direction -- a published value then looks undeclared and
    the gate goes red -- and SILENT in the other: the check for Java-side constants the
    service does not publish is a difference against `declared`, so an under-read makes it
    pass over every constant it never saw. `FieldDecision` has both long javadocs and a
    documented client-side sentinel, so it is the case that found this.
    """
    stripped = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    stripped = re.sub(r"//[^\n]*", " ", stripped)
    match = re.search(rf"enum\s+{enum_name}\s*\{{(.*?)(?:;|\}})", stripped, re.DOTALL)
    if match is None:
        return set()
    return set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", match.group(1)))


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


def _refs(node: object) -> set[str]:
    """Every `#/components/schemas/X` name reachable anywhere inside a JSON fragment."""
    found: set[str] = set()
    if isinstance(node, dict):
        target = node.get("$ref")
        if isinstance(target, str) and target.startswith("#/components/schemas/"):
            found.add(target.rsplit("/", 1)[-1])
        for value in node.values():
            found |= _refs(value)
    elif isinstance(node, list):
        for value in node:
            found |= _refs(value)
    return found


def _response_schema_names(spec: dict) -> set[str]:
    """
    Every schema a client DECODES, closed over nested references.

    Request bodies are deliberately excluded. A record the client only ever serialises is
    never handed a member it does not declare, so `ignoreUnknown` says nothing about it --
    and asserting it there would be a rule with no failure mode, which is the decoration
    this file is written to avoid.
    """
    reachable: set[str] = set()
    for operations in spec["paths"].values():
        for operation in operations.values():
            if isinstance(operation, dict):
                reachable |= _refs(operation.get("responses", {}))

    schemas = spec["components"]["schemas"]
    frontier = set(reachable)
    while frontier:
        nested = set()
        for name in frontier:
            nested |= _refs(schemas.get(name, {}))
        frontier = nested - reachable
        reachable |= nested
    return reachable


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
        recorded = {member for schema, member in _UNBOUND_ON_THE_JAVA_SIDE if schema == schema_name}
        missing = published - bound - recorded
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
        "rather than deleting the entry. If it is not carried YET -- the wire moved in "
        "one part of this repository and the hand-written client has not caught up -- "
        "record it in _UNBOUND_ON_THE_JAVA_SIDE with the component that closes it, which "
        "test_the_java_debt_record_is_current then holds to."
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
    accounted = (
        set(_SCHEMA_TO_JAVA) | set(_NOT_RESTATED_AS_A_RECORD) | set(_NOT_ON_THE_JAVA_SIDE_YET)
    )
    unaccounted = published - accounted
    assert not unaccounted, (
        f"these published schemas are in neither _SCHEMA_TO_JAVA, "
        f"_NOT_RESTATED_AS_A_RECORD nor _NOT_ON_THE_JAVA_SIDE_YET, so nothing checks "
        f"whether the Java client covers them: {sorted(unaccounted)}"
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

    overlap = sorted(set(_NOT_ON_THE_JAVA_SIDE_YET) & set(_SCHEMA_TO_JAVA))
    assert not overlap, (
        f"these schemas are recorded as not-on-the-Java-side-yet AND mapped to a Java "
        f"record, so the debt record is claiming a gap the mapping says is closed: "
        f"{overlap}"
    )


@pytest.mark.skipif(_CLIENT_ABSENT, reason=_ABSENT_REASON)
def test_the_java_debt_record_is_current():
    """
    Every recorded gap is still a gap, in both directions.

    A debt record that outlives its debt is worse than no record: it reads as a live
    exception, so the next reader believes a member is deliberately unbound when the Java
    side has carried it for months -- and the gate above is silently no longer checking it.
    That is the same rot `test_no_allowlist_entry_has_gone_stale` guards in
    `test_architecture.py`, and the same rot the `.ci-exceptions.yaml` check exists for.

    So an entry fails when the member stopped being published (the wire moved on and the
    note is describing nothing), and it fails when the Java component HAS been added (the
    debt is paid; delete the line). The only way an entry survives is by still being true.
    """
    schemas = _openapi()["components"]["schemas"]

    gone: list[str] = []
    paid: list[str] = []
    for (schema_name, member), reason in sorted(_UNBOUND_ON_THE_JAVA_SIDE.items()):
        assert reason.strip(), f"{schema_name}.{member} is recorded with no reason"
        if member not in schemas.get(schema_name, {}).get("properties", {}):
            gone.append(f"{schema_name}.{member}")
            continue
        java_file = _SCHEMA_TO_JAVA.get(schema_name)
        assert java_file is not None, (
            f"{schema_name} is recorded as having an unbound member but is not in "
            f"_SCHEMA_TO_JAVA, so nothing was ever checking it"
        )
        if member in _wire_names(_java(java_file)):
            paid.append(f"{schema_name}.{member} -> {java_file}")

    assert not gone, (
        "these members are recorded as unbound on the Java side but are no longer "
        "published at all, so the record describes nothing; delete the entries:\n  "
        + "\n  ".join(gone)
    )
    assert not paid, (
        "the Java client now binds these members, so the debt is paid and the record is "
        "suppressing a check that would otherwise be live; delete the entries:\n  "
        + "\n  ".join(paid)
    )

    restated = sorted(name for name in _NOT_ON_THE_JAVA_SIDE_YET if name in _SCHEMA_TO_JAVA)
    assert not restated, (
        f"these schemas are recorded as having no Java record while _SCHEMA_TO_JAVA names "
        f"one for them: {restated}"
    )


@pytest.mark.skipif(_CLIENT_ABSENT, reason=_ABSENT_REASON)
def test_the_java_records_ignore_members_they_do_not_carry():
    """
    The property that makes recording a gap honest rather than reckless.

    An unbound member is only survivable for a Java caller because Jackson is told to
    ignore what the record does not declare. If a record ever loses that annotation, an
    unbound member stops being invisible and becomes an exception on a response that was
    perfectly valid -- and `_UNBOUND_ON_THE_JAVA_SIDE` would then be a table of ways to
    break the client rather than a table of things it cannot yet read.

    Asserted over every RESPONSE record the contract table names, not only the ones with
    recorded gaps: the next gap can land on any of them. Request records are out of scope
    and `_response_schema_names` says why.
    """
    spec = _openapi()
    decoded = _response_schema_names(spec)
    assert decoded, "no response schemas were resolved -- this check would prove nothing"
    covered = sorted(name for name in _SCHEMA_TO_JAVA if name in decoded)
    assert len(covered) > 5, (
        f"only {len(covered)} decoded records are being checked; the reference walk is "
        f"broken and this assertion is close to vacuous"
    )

    naked = [
        _SCHEMA_TO_JAVA[name]
        for name in covered
        if "@JsonIgnoreProperties(ignoreUnknown = true)" not in _java(_SCHEMA_TO_JAVA[name])
    ]
    assert not naked, (
        "these Java records do not tell Jackson to ignore members they do not declare, so "
        "a member added to the wire would throw in the client instead of being invisible "
        "to it:\n  " + "\n  ".join(naked)
    )


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
    Every published enum, against the Java enum that binds it.

    The library's own vocabularies -- `MatchDecision` and `FieldDecision` -- are the only
    closed sets in this client, and they are closed because they are ITS words rather than a
    caller's taxonomy. Everything that comes out of somebody's controlled vocabulary
    (`GovernanceView.code`, `classification`, `EncoderStatusView.tier`,
    `ScoringContractView.absoluteScoreMetric`, `StatusWarningView.code`) is an open string on
    both sides and must stay one.

    A value the Java enum does not declare is the failure this file exists for: a Java enum
    refuses an unknown constant on decode, so ONE new value costs the whole response rather
    than one field.

    The name still says `match_decision` and now the body walks every published enum. Kept
    deliberately: `tests/packaging/conftest.py` declares this directory's opt-in skips by
    NODEID, so renaming the function here would silently un-declare its skip in a checkout
    without `clients/java/` -- a rename that switches a gate off is exactly the vacuity this
    file is written against. Renaming it is a two-file change and belongs with whoever owns
    the conftest.
    """
    schemas = _openapi()["components"]["schemas"]
    published_enums = {
        name: set(schema["enum"]) for name, schema in schemas.items() if "enum" in schema
    }
    assert published_enums, "no enums are published; this comparison proves nothing"

    unwatched = set(published_enums) - set(_SCHEMA_TO_JAVA_ENUM)
    assert not unwatched, (
        f"these published enums are bound by no entry in _SCHEMA_TO_JAVA_ENUM, so nothing "
        f"checks whether the Java client can decode their values: {sorted(unwatched)}"
    )

    findings: list[str] = []
    for schema_name, published in sorted(published_enums.items()):
        java_file, enum_name = _SCHEMA_TO_JAVA_ENUM[schema_name]
        declared = _enum_constants(_java(java_file), enum_name)

        missing = published - declared
        if missing:
            findings.append(
                f"{schema_name} -> {java_file}: the service publishes {sorted(missing)} and "
                f"the Java enum does not declare them (it declares {sorted(declared)}). A "
                f"response carrying one fails to decode."
            )

        # Extra constants are allowed only where this file has written down what they are
        # for. An undeclared extra is a client-side value somebody added without saying what
        # it means, which is how a sentinel quietly becomes a guess.
        sentinels = _CLIENT_SIDE_ENUM_SENTINELS.get(schema_name, {})
        undeclared_extras = declared - published - set(sentinels)
        if undeclared_extras:
            findings.append(
                f"{schema_name} -> {java_file}: the Java enum declares "
                f"{sorted(undeclared_extras)}, which the service does not publish and "
                f"_CLIENT_SIDE_ENUM_SENTINELS does not explain."
            )

        # And the sharp one: a sentinel that the server has started publishing is no longer
        # a sentinel, it is a real verdict wearing the client's "I could not read that" hat.
        collided = set(sentinels) & published
        if collided:
            findings.append(
                f"{schema_name}: {sorted(collided)} is declared as a CLIENT-SIDE sentinel in "
                f"{java_file} and the service now publishes it as a real value. The Java "
                f"client would read a genuine server verdict as 'this build did not "
                f"understand you'. Rename the sentinel."
            )

    assert not findings, "the published enums and the Java enums disagree:\n  " + "\n  ".join(
        findings
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

    # The second published enum, and the sentinel that is the client's answer to an unknown
    # value. Asserted here as well as in the gate above, because the gate's sentinel check is
    # a SUBSET test and a parser that stopped seeing UNKNOWN would satisfy it vacuously.
    field_decisions = _enum_constants(_java("model/FieldDecision.java"), "FieldDecision")
    assert {"AUTO_APPROVE", "REVIEW", "REJECT", "NO_MATCH", "UNKNOWN"} <= field_decisions, (
        f"the enum scan no longer reads FieldDecision.java; it found {sorted(field_decisions)}"
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
    assert _enum_constants("enum FieldDecision { AUTO_APPROVE }", "FieldDecision") == {
        "AUTO_APPROVE"
    }, "the enum scan must be able to see a FieldDecision that is missing NO_MATCH"
    assert _enum_constants(
        "enum E { /** {@link Foo#bar()} */ FIRST, /* {@code x} */ SECOND }", "E"
    ) == {"FIRST", "SECOND"}, (
        "a brace inside a javadoc must not end the enum body: reading only FIRST here would "
        "make the client-side-sentinel check pass over every constant it failed to see"
    )
    assert _mapped_statuses("case 400, 422 -> a; case 413 -> b; default -> c;") == {400, 422, 413}
    assert _mapped_statuses("default -> c;") == set(), (
        "the default arm must not count as a mapped status, or this gate passes for a "
        "client that maps nothing at all"
    )
