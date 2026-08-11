"""
tests/packaging/test_documented_behaviour.py | Env: ALL

`test_documented_routes.py` pinned the route TABLE. Six documents then rotted in one
session on statements about BEHAVIOUR, and every one of them walked straight past it.

The expensive one was `docs/GOVERNANCE.md` -- the document the Java caller is pointed at --
which said "a rejected match carries no class: `MatchResult` clears `governance` when the
decision is `REJECT`". True when it was written; false the moment the strip was narrowed to
rank 1. A caller reading it treats every `governance` on a `REJECT` candidate as absent,
which deletes exactly the rank-1-versus-rank-2 comparison the field exists to provide. No
route changed, so no route gate could see it. Four more statements went stale the same way:
CORS was `allow_origins=["*"]` and is now closed by default; the readiness `components` map
was four hardcoded `True`s and is now three of which one is a real check; and the OpenAPI
description offered an API-key header that no code implemented and that has since been
deleted from the description.

The shape of the defect is not "the docs are wrong". It is that a document made a claim only
the source could settle, and nothing compared the two. So each row below pairs a SENTENCE
with a WITNESS -- a predicate evaluated against a live `create_app()`, a live
`/openapi.json`, or a live `MatchResult`. The sentence is a defect only while the witness
disagrees with it, which means a deliberate behaviour change flips the gate to demanding the
opposite wording rather than freezing today's text forever.

## What this cannot do, said plainly

  * **It is a table of known sentences, not a reader.** A newly invented false statement
    about behaviour is invisible until somebody adds a row. Every row here was written from
    a statement that actually shipped.
  * **Silence passes.** Nothing forces README, DEPLOYMENT or ARCHITECTURE to describe CORS,
    authentication or the REJECT rule at all. The one exception is deliberate:
    `docs/GOVERNANCE.md` must state the REJECT rule in a form this file recognises, because
    that document is the one a caller is sent to and dropping the rule from it is the same
    defect as stating it wrongly.
  * **Only the readiness map is checked in both directions.** Its members are enumerable
    from the app, so adding a component and forgetting the document goes red. CORS, auth and
    the REJECT rule have no such enumeration; those rows catch a stale claim, not a missing
    one.
  * **Four behaviours out of everything six documents assert.** These four are the ones that
    rotted, and each is settleable by running the app rather than by reading it.

## A dated record is not a claim

`docs/DEPLOYMENT.md` §9 correctly writes down that hardening by the book "produced a server
whose CORS policy was still `allow_origins=["*"]`". That sentence is a retraction of a real
past state, and erasing it to satisfy a gate would be the opposite of what this file is for.
So a sentence carrying one of `_DATED` is skipped.

This is the reverse of the call `test_documented_routes.py` makes for denials, on purpose.
There, quoting "there is no HTTP matching endpoint" reads as a denial no matter how it is
framed, because a reader skimming for whether the endpoint exists takes the wrong answer
from the quote. Here the claims are configuration values, the retraction table is the
register this repository uses to record corrections, and the marker sits inside the same
sentence as the value. The `_DATED` list is kept short and literal for that reason: it is a
hole a determined writer can walk through, and a shorter list is a smaller hole.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import pytest

# The route gate owns the exemption list and the changelog scoping rule, and two copies of
# either is precisely the seam that let six documents rot unattended. Importing keeps one
# rule: `_scannable` restricts CHANGELOG.md to `## [Unreleased]` plus the newest version
# section -- the release being staged, which is a claim about today -- so a released section
# recording what 2.0.0 shipped is not dragged into the present tense.
from test_documented_routes import (
    _HISTORICAL,
    REPO,
    _scannable,
    tracked_markdown,
)

# =============================================================================
# WITNESSES -- what the application actually does
# =============================================================================
#
# Each is `@cache`d: a red run evaluates a witness once per matching sentence, and
# `readiness_can_refuse` costs four application startups. They observe a factory with no
# state between calls, so the only thing the cache changes is the clock.


def _live_app(environ: dict[str, str] | None = None):
    from nexus_matcher.presentation.api.app import create_app

    return create_app(configure_logs=False, environ=environ or {})


def _readiness(environ: dict[str, str]) -> tuple[int, dict[str, bool]]:
    """`/health/ready`'s status and its components map, wherever the body carries it."""
    from fastapi.testclient import TestClient

    with TestClient(_live_app(environ)) as client:
        response = client.get("/health/ready")
    body = response.json()
    components = body.get("components")
    if components is None:
        components = body.get("error", {}).get("details", {}).get("components", {})
    return response.status_code, components


# The configurations this service ships in. The union of their component maps is the set of
# names a document may enumerate; taking one configuration would miss `matcher`, which is
# absent only under the opt-out.
_SHIPPED_CONFIGURATIONS = (
    {},
    {"NEXUS_API_MATCHING_OPTIONAL": "true"},
    {"NEXUS_API_DICTIONARY": "does/not/exist.csv"},
    {"NEXUS_API_MATCHING_OPTIONAL": "true", "NEXUS_API_DICTIONARY": "does/not/exist.csv"},
)

# Spellings of a credential. `X-API-Key` and `NEXUS_API_KEY` are the two the deleted
# paragraph used; the rest are what a replacement would most likely say. Deliberately NOT
# the bare words "api key" -- the description's replacement text says the service implements
# "no API-key or OAuth check", and a gate that cannot tell a denial from an offer is a gate
# that gets muted.
_CREDENTIAL_SPELLINGS = ("x-api-key", "nexus_api_key", "authorization:", "bearer ")


@cache
def cors_reflects_any_origin() -> bool:
    """Whether a default deployment grants `*`. Today it mounts no CORS middleware at all."""
    from fastapi.middleware.cors import CORSMiddleware

    for middleware in _live_app().user_middleware:
        if middleware.cls is CORSMiddleware:
            return "*" in middleware.kwargs.get("allow_origins", [])
    return False


@cache
def readiness_can_refuse() -> bool:
    """Whether any component is a real check, i.e. whether readiness can answer 503."""
    return any(_readiness(env)[0] != 200 for env in _SHIPPED_CONFIGURATIONS)


@cache
def reported_components() -> frozenset[str]:
    """Every component name a shipped configuration can put in the map."""
    names: set[str] = set()
    for env in _SHIPPED_CONFIGURATIONS:
        names.update(_readiness(env)[1])
    return frozenset(names)


@cache
def openapi_description() -> str:
    return str(_live_app().openapi()["info"]["description"]).lower()


@cache
def openapi_offers_a_credential() -> bool:
    return any(spelling in openapi_description() for spelling in _CREDENTIAL_SPELLINGS)


@cache
def openapi_declares_a_security_scheme() -> bool:
    return "securitySchemes" in _live_app().openapi().get("components", {})


def _reject_at(rank: int) -> bool:
    """Whether a REJECTED candidate at `rank` keeps the class its entry confers."""
    from nexus_matcher.domain.governance import ProtectionClass
    from nexus_matcher.domain.models.entities import DictionaryEntry, MatchResult, SchemaField
    from nexus_matcher.shared.types.base import MatchDecision, PerformanceMetrics, ScoreBreakdown

    result = MatchResult(
        schema_field=SchemaField(name="legal_name", data_type="string"),
        dictionary_entry=DictionaryEntry(
            id="GBF-0001",
            business_name="Passenger Legal Name",
            logical_name="legal_name",
            definition="A ticketed passenger's full legal name.",
            data_type="string",
        ),
        rank=rank,
        final_confidence=0.4,
        score_breakdown=ScoreBreakdown(fused_retrieval_score=0.4),
        decision=MatchDecision.REJECT,
        performance=PerformanceMetrics(latency_ms=1.0),
        governance=ProtectionClass(
            code="MANIFEST_NAME",
            name="Passenger manifest identity",
            classification="SEALED_RESTRICTED",
            personal_information=True,
            direct_identifier=True,
        ),
    )
    return result.governance is not None


@cache
def rejected_runner_up_keeps_its_class() -> bool:
    return _reject_at(2)


@cache
def rejected_rank_one_keeps_its_class() -> bool:
    return _reject_at(1)


# =============================================================================
# THE CLAIMS
# =============================================================================


@dataclass(frozen=True)
class Claim:
    """A sentence a document can make, and the predicate that decides whether it is true."""

    pattern: str
    witness: Callable[[], bool]
    true_when: bool
    site: str

    def is_stale(self) -> bool:
        return self.witness() != self.true_when


_CLAIMS: tuple[Claim, ...] = (
    Claim(
        r"allow_origins ?= ?\[\"?\*\"?\]",
        cors_reflects_any_origin,
        True,
        "docs/API_REFERENCE.md middleware list, docs/PROJECT_STATE.md 'what does not exist'",
    ),
    Claim(
        r"treat readiness as .{0,3}the process started",
        readiness_can_refuse,
        False,
        "docs/API_REFERENCE.md readiness paragraph",
    ),
    Claim(
        r"/health/ready reports hardcoded",
        readiness_can_refuse,
        False,
        "docs/PROJECT_STATE.md 'what does not exist'",
    ),
    Claim(
        r"openapi description[^.;]{0,40}mentions",
        openapi_offers_a_credential,
        True,
        "docs/API_REFERENCE.md 'not implemented' table, docs/PROJECT_STATE.md",
    ),
    # Finding A, both halves. The first is the sentence that shipped; the second is its
    # replacement, so re-broadening the strip in `entities.py` reddens the document that
    # would then be describing a rule the model no longer has.
    Claim(
        r"(rejected match carries no class|clears governance when the decision is reject)",
        rejected_runner_up_keeps_its_class,
        False,
        "docs/GOVERNANCE.md 'what a match carries'",
    ),
    Claim(
        r"rejected runner-?up keeps",
        rejected_runner_up_keeps_its_class,
        True,
        "docs/GOVERNANCE.md 'what a match carries' (the replacement)",
    ),
)

# Sentence fragments that mark a value as a record of a past state. See the module docstring
# for why this file makes the opposite call to the route gate. Kept short on purpose.
_DATED = ("used to", "was still", "previously", "no longer", "earlier revision", "retracted")


def _oneline(text: str) -> str:
    """One line, backticks and bold markers gone, `*` KEPT.

    The route gate's `flatten` deletes every `*` as an emphasis marker, which would turn
    `allow_origins=["*"]` into `allow_origins=[""]` and quietly unmatch the one claim whose
    whole content is an asterisk. Only the doubled form is stripped here.
    """
    lines = [line.lstrip("> \t").strip() for line in text.splitlines()]
    joined = " ".join(lines).replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", joined).lower()


def stale_claims_in(doc: str, text: str) -> list[str]:
    """Every sentence in `text` that the live application contradicts."""
    found: list[str] = []
    for sentence in re.split(r"(?<=[.!?]) ", _oneline(_scannable(doc, text))):
        if any(marker in sentence for marker in _DATED):
            continue
        for claim in _CLAIMS:
            if re.search(claim.pattern, sentence) and claim.is_stale():
                found.append(
                    f"{doc}: {claim.pattern!r} matched, but "
                    f"{claim.witness.__name__}() is {claim.witness()} "
                    f"(the sentence is only true when it is {claim.true_when}). "
                    f"Originally at {claim.site}."
                )
    return found


# =============================================================================
# THE GATE
# =============================================================================


def test_no_document_states_a_behaviour_the_application_contradicts():
    """
    The general direction: a document asserts something the running service denies.

    All five statements finding D and finding A named were true when written. That is what
    makes them expensive -- they were written in this repository's honest-about-gaps
    register, so a caller believes them, and nothing re-checked them when the behaviour
    moved underneath.
    """
    found: list[str] = []
    for doc in tracked_markdown():
        if doc in _HISTORICAL:
            continue
        found.extend(stale_claims_in(doc, (REPO / doc).read_text(encoding="utf-8")))

    assert not found, (
        "documentation states behaviour this application contradicts:\n  "
        + "\n  ".join(found)
        + "\nCorrect the sentence, or -- if the behaviour change was the mistake -- "
        "revert the behaviour."
    )


def test_the_documented_readiness_components_are_the_ones_the_app_reports():
    """
    The one claim with a both-directions check, because its members are enumerable.

    `docs/API_REFERENCE.md` said the map held hardcoded `True` for `api`, `config`,
    `vector_store` and `cache`. `vector_store` and `cache` were deleted from the code -- each
    had been set `True` inside a `try:` block whose body was a comment -- and `matcher` was
    added as a real check. Nothing connected the two, so the document went on naming two
    components that no longer exist and omitting the only one that can fail.

    This asserts set equality against a live app, so adding a component and forgetting the
    document is as red as removing one and forgetting the document.
    """
    text = re.sub(r"\s+", " ", (REPO / "docs/API_REFERENCE.md").read_text(encoding="utf-8"))
    match = re.search(
        r"the `components` map reported by `/health/ready` carries ([^.]+)\.", text, re.I
    )
    assert match, (
        "docs/API_REFERENCE.md no longer carries the sentence that enumerates the readiness "
        "components, so nothing compares the documented map to the live one. Restore an "
        "enumeration of the form 'The `components` map reported by `/health/ready` carries "
        "`a`, `b` and `c`.' or rewrite this assertion deliberately."
    )
    documented = frozenset(re.findall(r"`([a-z_]+)`", match.group(1)))
    assert documented == reported_components(), (
        f"docs/API_REFERENCE.md documents readiness components {sorted(documented)}; a live "
        f"app reports {sorted(reported_components())} across its shipped configurations."
    )


def test_the_governance_document_states_the_reject_rule_in_a_checkable_form():
    """
    Non-vacuity for finding A, and the only place this file demands that a document speak.

    Deleting the rule from `docs/GOVERNANCE.md` costs a caller exactly what stating it
    wrongly cost them: `governance` is null for two unrelated reasons, `decision` does not
    separate them, and this is the document they are sent to. So the section must contain a
    sentence one of the rows above recognises -- which is also what stops somebody
    "simplifying" the fix away and leaving the gate green over silence.
    """
    text = (REPO / "docs/GOVERNANCE.md").read_text(encoding="utf-8")
    start = text.find("## What a match carries")
    assert start != -1, "docs/GOVERNANCE.md no longer has a 'What a match carries' section"
    end = text.find("\n## ", start + 1)
    section = _oneline(text[start:] if end == -1 else text[start:end])

    recognised = [claim for claim in _CLAIMS if re.search(claim.pattern, section)]
    assert recognised, (
        "docs/GOVERNANCE.md's 'What a match carries' section states nothing about the REJECT "
        "rule that this gate can check against `MatchResult`. That section is the contract a "
        "caller reads before deciding what a null `governance` means."
    )


def test_the_openapi_description_offers_no_credential_the_spec_does_not_declare():
    """
    Source against source, and the row that stops the deleted paragraph growing back.

    The description used to say "API key authentication can be enabled via the
    `NEXUS_API_KEY` environment variable. Pass the key in the `X-API-Key` header." Nothing
    read it: the variable appeared exactly once in `src/`, inside that string, and a request
    with no header returned 200 and a real protection class. Both documentation rows fixed
    alongside this file describe the description's current wording, so they go stale the
    moment it changes -- and a claim inside `/openapi.json` reaches a generated client
    directly, without a document in between.

    Written as an implication rather than a skip, because `conftest.py` in this directory
    fails the whole run on an undeclared skip: the day somebody implements the scheme, this
    must relax on its own, not take the build down with it.
    """
    named = [s for s in _CREDENTIAL_SPELLINGS if s in openapi_description()]
    assert not named or openapi_declares_a_security_scheme(), (
        f"the OpenAPI description names {named} while `components.securitySchemes` is "
        "absent, so a generated client is told to send a credential no route checks. "
        "Implement the scheme and declare it, or do not name it."
    )


# =============================================================================
# CONTROLS -- proof the gate can go red
# =============================================================================

# Findings A and D verbatim, each with the document it stood in. Without these, every
# pattern above is green over a tree that no longer contains the sentence it was written
# for, which is the vacuous shape this repository keeps rediscovering.
_STALE: tuple[tuple[str, str], ...] = (
    (
        "docs/GOVERNANCE.md",
        "A rejected match carries no class: `MatchResult` clears `governance` when the "
        "decision is\n`REJECT`, so a refused match cannot confer anything.",
    ),
    (
        "docs/API_REFERENCE.md",
        '- CORS, currently `allow_origins=["*"]` - narrow this before exposing the service.',
    ),
    (
        "docs/API_REFERENCE.md",
        "It does **not** currently probe a real vector store or cache connection - the code "
        'paths that\nwould do so are empty `try` blocks. Treat readiness as "the process '
        'started", not "the\ndependencies are reachable".',
    ),
    (
        "docs/API_REFERENCE.md",
        "| API key auth via `NEXUS_API_KEY` / `X-API-Key` | No authentication dependency is "
        "attached to any route. The OpenAPI description text mentions it; the code does not "
        "implement it. |",
    ),
    (
        "docs/PROJECT_STATE.md",
        "- **API authentication or rate limiting.** The OpenAPI description mentions API "
        'keys;\n  no route enforces one. CORS is currently `allow_origins=["*"]`.',
    ),
    (
        "docs/PROJECT_STATE.md",
        "- **Dependency health probing.** `/health/ready` reports hardcoded component status.",
    ),
)


@pytest.mark.parametrize(("source", "sentence"), _STALE, ids=[s for s, _ in _STALE])
def test_the_scan_catches_every_statement_that_went_stale(source, sentence):
    """Replay each stale statement through the detector, exactly as it stood."""
    assert stale_claims_in(source, sentence), (
        f"the detector no longer recognises the stale statement from {source}:\n{sentence}"
    )


def test_a_statement_is_stale_only_while_the_application_disagrees():
    """
    The gate reads the app, not a wish list.

    Every row is a pair, and the second element is the witness. If CORS were reopened, the
    retracted sentence would become true again and this file must go quiet rather than force
    a document to lie the other way -- which is the failure mode of a gate that pins text.
    """
    cors = next(c for c in _CLAIMS if "allow_origins" in c.pattern)
    sentence = 'CORS, currently `allow_origins=["*"]`.'

    assert stale_claims_in("docs/API_REFERENCE.md", sentence)
    reopened = Claim(cors.pattern, lambda: True, cors.true_when, cors.site)
    assert not reopened.is_stale()

    # ...and the same, from the other end: the replacement wording is what goes red if the
    # narrowed REJECT guard is ever broadened back.
    runner_up = next(c for c in _CLAIMS if "runner-?up" in c.pattern)
    broadened = Claim(runner_up.pattern, lambda: False, runner_up.true_when, runner_up.site)
    assert broadened.is_stale()


def test_the_replacement_wording_is_not_itself_flagged():
    """
    The other half of the control, and what keeps the patterns narrow.

    A pattern broad enough to match any sentence about CORS, readiness or REJECT would make
    the behaviour undocumentable -- green only while the documents stay silent, which is how
    the route defect started.
    """
    replacements = (
        (
            "docs/GOVERNANCE.md",
            "**A rejected runner-up keeps the class its entry confers.** The rank qualifier "
            "is the rule, not a detail of it.",
        ),
        (
            "docs/API_REFERENCE.md",
            "- CORS, only when `NEXUS_API_CORS_ORIGINS` names the origins that may use a "
            "browser. Empty is the default and mounts no `CORSMiddleware` at all.",
        ),
        (
            "docs/API_REFERENCE.md",
            "The `components` map reported by `/health/ready` carries `api`, `config` and "
            "`matcher`. `api` and `config` are still hardcoded `True`. `matcher` is a real "
            "check: it is `False` when no dictionary loaded.",
        ),
        (
            "docs/PROJECT_STATE.md",
            "- **API authentication or rate limiting.** No route enforces a key and "
            "`/openapi.json` declares no security scheme; the description says the service "
            "ships unauthenticated instead of offering a header nothing reads.",
        ),
    )
    for doc, sentence in replacements:
        assert stale_claims_in(doc, sentence) == [], sentence


def test_a_dated_retraction_is_not_read_as_a_present_claim():
    """
    `docs/DEPLOYMENT.md` §9 records the state its own advice used to produce. That is a true
    sentence in the register this repository corrects itself in, and a gate that demanded its
    deletion would be destroying the record to make the check pass.

    The second assertion is the price: strip the marker and the identical value is caught, so
    the exemption rests on one short phrase and not on the file it appears in.
    """
    dated = (
        "No code read any of them, so hardening by the book produced a server whose CORS "
        'policy was still `allow_origins=["*"]` with credentials enabled.'
    )
    undated = (
        "Hardening by the book produces a server whose CORS policy is "
        '`allow_origins=["*"]` with credentials enabled.'
    )
    assert stale_claims_in("docs/DEPLOYMENT.md", dated) == []
    assert stale_claims_in("docs/DEPLOYMENT.md", undated)


# =============================================================================
# THE WITNESSES ARE LOOKING AT A LIVE APPLICATION
# =============================================================================


def test_the_witnesses_are_reading_a_live_application():
    """
    Every assertion above is only as good as its witness. A witness that silently returned a
    constant -- a renamed middleware attribute, a readiness body whose components moved, an
    `openapi()` that stopped carrying a description -- would leave the whole file green.

    So each one is pinned to the value the shipped application produces today, measured:
    `create_app(environ={})` mounts no CORS middleware; `/health/ready` answers 503 with
    `matcher: false` and 200 under the opt-out; the description names no credential and the
    spec declares no scheme; a rank-1 REJECT is stripped and a rank-2 REJECT is not.
    """
    assert cors_reflects_any_origin() is False
    assert readiness_can_refuse() is True
    assert reported_components() == frozenset({"api", "config", "matcher"})
    assert _readiness({})[0] == 503
    assert _readiness({"NEXUS_API_MATCHING_OPTIONAL": "true"}) == (
        200,
        {"api": True, "config": True},
    )
    assert openapi_offers_a_credential() is False
    assert openapi_declares_a_security_scheme() is False
    assert rejected_rank_one_keeps_its_class() is False
    assert rejected_runner_up_keeps_its_class() is True


def test_the_scan_is_looking_at_the_documents_that_ship():
    """A `git ls-files` glob that stopped matching would leave the scan green over zero files."""
    tracked = set(tracked_markdown())
    assert {"docs/GOVERNANCE.md", "docs/API_REFERENCE.md", "docs/PROJECT_STATE.md"} <= tracked
    assert len(tracked) >= 40, sorted(tracked)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
