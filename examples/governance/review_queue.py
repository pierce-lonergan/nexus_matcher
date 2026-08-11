"""
examples.governance.review_queue | Layer: EXAMPLE (command 5 of 5)
The queue a governance reviewer actually works, with past decisions folded back in.

    python examples/governance/review_queue.py

Reads `out/results.json` and `feedback.jsonl`. A field is in the queue when its top match
was NOT auto-approved. Each entry says which glossary term is being proposed, what class
accepting it would confer, and whether a reviewer has already ruled on that exact pairing.

## Why the queue is built from the DECISION, not from a confidence cutoff

`MatchingSession.get_low_confidence_fields()` shipped with a default cutoff of 0.6, one
hundredth below the structural floor of 0.63, so it returned an empty list on every
schema ever matched -- and told a governance lead there was nothing to review on schemas
where nothing was trustworthy (DX-001, `tests/museum/NM-0027`). A queue built from the
decision cannot drift away from the configuration that way, because the decision was
already made using the calibrated threshold AND the margin rule. So an ambiguous
near-tie, which any numeric cutoff would wave through, is still queued.

An empty queue is therefore reported as a finding, not as success.

## Over-inheritance is called out by name

The costly error is inheriting MORE protection than a field warrants: nothing breaks,
nobody notices, and a lawful use of the data quietly stops being possible. So entries
proposing a DIRECT IDENTIFIER or the most closed tier are marked, and they are shown
first.
"""

from __future__ import annotations

import json

from _pack import (
    BANNER,
    FEEDBACK_FILE,
    RESULTS_FILE,
    describe,
    load_jsonl,
    load_vocabulary,
    rule,
    say,
)


def _build_queue(produced: dict, rulings: dict) -> list[dict]:
    """Every field whose top match was NOT auto-approved, riskiest first."""
    queue = []
    for key in sorted(produced):
        top = produced[key][0]
        if top["decision"] == "AUTO_APPROVE":
            continue
        governance = top["governance"]
        runner_up = produced[key][1] if len(produced[key]) > 1 else None
        queue.append(
            {
                "field": key,
                "candidate_id": top["governance_id"],
                "business_name": top["business_name"],
                "confidence": top["confidence"],
                "margin": (
                    round(top["confidence"] - runner_up["confidence"], 6) if runner_up else None
                ),
                "decision": top["decision"],
                "governance": governance,
                "risky": bool(governance and governance["direct_identifier"]),
                "ruling": rulings.get((key, top["governance_id"])),
            }
        )
    # Riskiest first, then by field name so two runs order identically.
    queue.sort(key=lambda item: (not item["risky"], item["field"]))
    return queue


def _print_item(item: dict, vocabulary) -> None:
    flag = (
        "  <- DIRECT IDENTIFIER, over-inheriting this is the costly error" if item["risky"] else ""
    )
    code = (item["governance"] or {}).get("code")
    margin = "n/a" if item["margin"] is None else f"{item['margin']:.4f}"
    say(f"  {item['field']}")
    say(f"      proposes  {item['candidate_id']}  {item['business_name']}")
    say(f"      would confer  {describe(vocabulary.get(code))}{flag}")
    say(
        f"      {item['decision']}  confidence {item['confidence']:.4f}  "
        f"margin over runner-up {margin}"
    )
    ruling = item["ruling"]
    if ruling:
        say(
            f"      RULED {ruling['action'].upper()} by {ruling['reviewer']} "
            f"on {ruling['reviewed_at']}"
        )
        say(f'      "{ruling["reason"]}"')


def _print_stray(events: list, queue: list, produced: dict) -> None:
    """Reviewer events matching nothing in the queue.

    An accept recorded against a field the matcher now auto-approves is not an error -- it
    is the normal outcome of a glossary improving. It is listed so that a stale feedback
    file, which is the other explanation, cannot hide inside the same silence.
    """
    queued_pairs = {(item["field"], item["candidate_id"]) for item in queue}
    stray = [e for e in events if (e["field"], e["candidate_id"]) not in queued_pairs]
    if not stray:
        say("  none")
    for event in sorted(stray, key=lambda e: e["event_id"]):
        top = produced.get(event["field"], [{}])[0]
        state = top.get("decision", "field not in the results file")
        say(
            f"  {event['event_id']}  {event['field']} / {event['candidate_id']}: "
            f"{event['action']}, field is now {state}"
        )


def main() -> int:
    say(BANNER)
    rule("the review queue")

    if not RESULTS_FILE.is_file():
        say(f"FAIL {RESULTS_FILE} does not exist. Run run_pack.py first.")
        return 1

    document = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    produced = document["fields"]
    vocabulary = load_vocabulary()

    events = load_jsonl(FEEDBACK_FILE)
    # Keyed by the exact pairing a reviewer ruled on. Keying by field alone would let a
    # ruling about one candidate silently resolve a different candidate for the same
    # field -- which is how a reviewer's "no, not that term" becomes "yes, any term".
    rulings = {(e["field"], e["candidate_id"]): e for e in events}
    say(f"reviewer events on file: {len(events)} ({FEEDBACK_FILE.name})")

    queue = _build_queue(produced, rulings)
    pending = [item for item in queue if item["ruling"] is None]
    resolved = [item for item in queue if item["ruling"] is not None]

    say(f"fields in the results file: {len(produced)}")
    say(f"not auto-approved, so queued: {len(queue)}")
    say(f"  already ruled on: {len(resolved)}")
    say(f"  awaiting a reviewer: {len(pending)}")

    if not queue:
        # Stated, not celebrated. See the module docstring.
        say("")
        say("The queue is EMPTY. That is a claim about the schema, not a success message:")
        say("it says every field was auto-approved. Check the auto-approvals before")
        say("believing it -- an empty queue is what DX-001 produced on schemas where")
        say("nothing was trustworthy.")

    for section, items in (("awaiting a reviewer", pending), ("already ruled on", resolved)):
        if not items:
            continue
        rule(section)
        for item in items:
            _print_item(item, vocabulary)

    rule("reviewer events that do not match anything in this queue")
    _print_stray(events, queue, produced)

    rule()
    say(f"REVIEW QUEUE: {len(pending)} awaiting a reviewer, {len(resolved)} already ruled on")
    say("Exit code is 0 either way: a queue is work to do, not a failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
