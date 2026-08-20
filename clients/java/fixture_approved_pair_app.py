"""
The fifth fixture server for the Java client: the example pack with a reviewer's verdict
attached, which is the only configuration on which `provenance: APPROVED_PAIR` exists.

    PYTHONPATH=clients/java .venv/Scripts/python.exe -m uvicorn \
        fixture_approved_pair_app:create_app --factory --host 127.0.0.1 --port 8004

`clients/java/serve-fixtures.sh` (and `.ps1`) starts it; nothing else in this repository
imports this module.

## Why a fixture server exists for this at all

`provenance` distinguishes a candidate the pipeline scored (`RETRIEVAL`) from one a
reviewer decided and matching skipped (`APPROVED_PAIR`). The second half is unreachable
on every server this package starts: `create_app()` builds no feedback consumer,
`NexusMatcher()` takes `feedback_consumer=None`, and that default is a measured decision
rather than an unfinished wire-up -- see `presentation/api/feedback.py` and
`application/feedback_loop.py`, which both say so at length.

So the Java client's handling of `APPROVED_PAIR` could only ever have been tested against
a response body somebody typed. That is the one thing `clients/java/capture-fixtures.sh`
exists to prevent: a hand-written fixture tests what its author believed the contract was,
which is the exact belief the fixture was there to check. This module is the smallest
honest way out -- it opts a THROWAWAY server into the documented seam, using the reference
consumer, so the capture is a real body from a real service.

Same reasoning, same shape, as `clients/java/fixture-absolute-floor.json`: a configuration
that exists so a reachable-but-not-default behaviour can be driven against something live.
It is a TEST FIXTURE and not a recommended default. Attaching the bypass makes precision
on a seen pair 100% BY CONSTRUCTION, which is a tautology and never an accuracy result.

## What it answers, and why that is the point

`booking.passenger.legal_name` is the field `fixture-approved-pairs.jsonl` carries an
APPROVED verdict for, and the id on that verdict is `GBF-0001` -- **the same entry
retrieval finds for it anyway**, which `src/test/resources/captured/match-response.json`
shows. That is deliberate. If the reviewer had chosen a different entry, the two answers
would be distinguishable by their `governanceId` and the fixture would prove nothing about
`provenance`. Here the id, the `decision` and the `confidence` all agree, and the ONLY
thing separating a human's answer from the pipeline's is the member this fixture exists
for.

The capture pairs it with `published.terminal_nm`, whose retrieval confidence against the
shipped pack is exactly **1.0** -- every one of the five signals is maximal, because the
column name is the glossary's own `logical_name` for that entry. One response therefore
carries two candidates that are identical on `(confidence 1.0, decision AUTO_APPROVE)`,
one scored and one not. That pair was once documented as identifying a bypass; the capture
is the evidence that it identifies nothing.

## The private attribute, and why it is not smuggling

`NexusMatcher.from_config()` does not take `feedback_consumer`, so the consumer is assigned
after construction and before indexing -- `_index_dictionary` calls `_rebind_feedback_consumer`,
which is what hands the consumer the glossary and resolves the pair. Both underscored names
are already load-bearing outside the class: `presentation/api/app.py` calls
`_index_dictionary` on this same path, for the reason its docstring gives.

Assigning a private is still a thing that can silently stop working, so it is CHECKED
rather than assumed: `_require_the_bypass_is_standing` fails the process at startup if the
verdict did not resolve. A fixture server that quietly answers by retrieval would make the
capture a lie about the one member it was captured for.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cost, not behaviour
    from fastapi import FastAPI

    from nexus_matcher.application.feedback_loop import ApprovedPairBypass

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

# Defaults, so the module can be started with no environment at all. Overridable by the
# same variables the other four fixtures use, so this server reads one pack and one
# vocabulary rather than a second copy of them.
_DEFAULT_DICTIONARY = _REPO / "examples" / "governance" / "glossary.csv"
_DEFAULT_VOCABULARY = _REPO / "examples" / "governance" / "protection_classes.json"
_DEFAULT_TRAIL = _HERE / "fixture-approved-pairs.jsonl"

# The variable that points at the trail. Named here rather than inline because the shell
# and PowerShell launchers both set it.
TRAIL_VARIABLE = "NEXUS_FIXTURE_APPROVED_PAIRS"


def _setting(variable: str, default: Path) -> str:
    value = (os.environ.get(variable) or "").strip()
    return value or str(default)


def _require_the_bypass_is_standing(bypass: ApprovedPairBypass, trail: str) -> None:
    """
    Refuse to serve unless every verdict in the trail actually resolved.

    A pair that did not bind is not an error the matcher reports -- it is held and retried,
    which is right for a production deployment whose glossary is momentarily wrong and
    exactly wrong for a fixture. The server would come up, answer by retrieval, and the
    capture taken from it would carry `provenance: RETRIEVAL` on the field the whole
    fixture exists to show as `APPROVED_PAIR`. Failing here makes that a startup error
    naming the file instead of a green build over a fixture that proves the opposite of
    what it claims.
    """
    report = bypass.bypass_report()
    if report.standing != report.verdicts or report.verdicts == 0:
        raise RuntimeError(
            f"{trail} yielded {report.verdicts} verdict(s) of which {report.standing} are "
            f"standing (unresolved={report.unresolved}, retired={report.retired}, "
            f"invalidated={report.invalidated}). Every verdict in a fixture trail must "
            f"resolve against the loaded glossary, or this server answers by retrieval and "
            f"the capture taken from it says the opposite of what it was taken to say."
        )


def create_app() -> FastAPI:
    """Build the pack-loaded service with the approved-pair bypass attached."""
    # Deferred exactly as `presentation/api/app.py` defers them: importing the matching
    # stack at module scope makes merely importing this file load the bundled encoder.
    from nexus_matcher.application.feedback_loop import ApprovedPairBypass
    from nexus_matcher.application.ingest import load_entries
    from nexus_matcher.application.use_cases.match_schema import NexusMatcher
    from nexus_matcher.presentation.api.app import create_app as create_service

    dictionary = _setting("NEXUS_API_DICTIONARY", _DEFAULT_DICTIONARY)
    vocabulary = _setting("NEXUS_API_GOVERNANCE", _DEFAULT_VOCABULARY)
    trail = _setting(TRAIL_VARIABLE, _DEFAULT_TRAIL)

    entries = load_entries(dictionary, governance=vocabulary)
    matcher = NexusMatcher.from_config(governance=vocabulary)
    bypass = ApprovedPairBypass.from_trail(trail)
    # Assigned BEFORE indexing: `_index_dictionary` is what binds the consumer to the
    # glossary, so a consumer attached afterwards would never resolve its pairs.
    matcher._feedback_consumer = bypass
    matcher._index_dictionary(entries)
    _require_the_bypass_is_standing(bypass, trail)

    # `matcher=` rather than NEXUS_API_DICTIONARY, so the service uses THIS object instead
    # of building a second one at startup. `GET /api/v1/status` then reports a null
    # dictionary source, which is the honest answer for an injected matcher and is why
    # nothing is captured from this port except the match body.
    return create_service(matcher=matcher)
