"""
examples.governance.check_expectations | Layer: EXAMPLE (command 3 of 5)
Hold the pack to the expectations written down in fields.json.

    python examples/governance/check_expectations.py

Reads `out/results.json` (write it with run_pack.py first) and checks every field
against its recorded `expected_id` and `expected_decision`. Exit 0 means the pack
behaved exactly as documented; exit 1 names every field that did not.

## The number that matters

`wrong auto-approvals` is the headline, not P@1. A wrong match that goes to a human
costs a few minutes. A wrong match that is AUTO-APPROVED silently confers another term's
protection class on a field -- and the expensive direction is OVER-inheritance, which
does not break anything and is invisible until an audit. So this check reports both, and
it fails on either. Reporting a retrieval number without the decision number is the
mistake `docs/HAZARDS.md` H-001 exists to stop: retrieval improving while the metric that
decides whether a class is applied without a human gets worse.

## What each assertion can catch

Every one of these has been observed going red against a mutation of fields.json or of
the results file:

  * a top match landing on the wrong entry     -> change an expected_id
  * a decision drifting                        -> change an expected_decision
  * a novel field starting to auto-approve     -> flip a null expected_id to an id
  * governance not matching the glossary       -> the inherited-class check below
  * a field disappearing from the results      -> delete a field from the results file
"""

from __future__ import annotations

import json

from _pack import (
    BANNER,
    RESULTS_FILE,
    describe,
    governed_entries,
    load_vocabulary,
    read_fields,
    rule,
    say,
)


def _check_one_field(row: dict, produced: dict, tally: dict) -> str:
    """Verdict for one field, updating the running tally. "ok" means it held."""
    key = row["flattenedName"]
    expected_id = row.get("expected_id")
    expected_decision = row.get("expected_decision")

    records = produced.get(key)
    if not records:
        return "MISSING from the results file"

    top = records[0]
    got_id = top["governance_id"]
    got_decision = top["decision"]
    verdict = "ok"

    if got_decision == "AUTO_APPROVE":
        tally["auto_approved"] += 1
    else:
        tally["review"].append(key)

    if expected_id is None:
        # No glossary term is a correct answer for this field. WHICH entry the matcher
        # ranked first is recorded in fields.json, not asserted -- the governance-
        # meaningful requirement is that it was not applied without a human. Asserting the
        # wrong entry by name would turn a documented trap into a pinned behaviour, and
        # would go red for the right reason (the matcher improving) as readily as the wrong.
        tally["novel"] += 1
        if got_decision == "AUTO_APPROVE":
            tally["wrong"].append(f"{key} auto-approved {got_id} with no correct answer")
            verdict = "FAIL auto-approved a field nothing governs"
    elif got_id != expected_id:
        verdict = f"FAIL expected {expected_id}"
        if got_decision == "AUTO_APPROVE":
            tally["wrong"].append(f"{key} auto-approved {got_id}, expected {expected_id}")

    if expected_decision is not None and got_decision != expected_decision:
        verdict = f"FAIL decision {got_decision}, expected {expected_decision}"

    say(f"{key:40} {got_decision:13} {got_id:10} {verdict}")
    return verdict


def _check_inherited_classes(produced: dict, entries: dict, vocabulary) -> list[str]:
    """The join is what silently goes wrong: a results file can carry the right id and the
    wrong class, and nothing about it looks broken."""
    failures: list[str] = []
    for key in sorted(produced):
        top = produced[key][0]
        entry = entries.get(top["governance_id"])
        wanted = vocabulary.get(getattr(entry, "governance_code", None))
        got = top["governance"]
        got_code = got["code"] if got else None
        wanted_code = wanted.code if wanted else None
        if got_code != wanted_code:
            failures.append(
                f"{key}: results say {got_code!r}, the glossary entry "
                f"{top['governance_id']} says {wanted_code!r}"
            )
    if failures:
        for line in failures:
            say(f"  FAIL {line}")
    else:
        say(f"  all {len(produced)} top matches carry the class their glossary entry declares")
    return failures


def _summary(
    document: dict, produced: dict, entries: dict, vocabulary, tally: dict, checked: int
) -> None:
    say(f"fields checked              : {checked}")
    say(f"auto-approved               : {tally['auto_approved']}")
    say(f"sent to review              : {len(tally['review'])}")
    say(f"fields nothing governs      : {tally['novel']}")
    say(f"WRONG AUTO-APPROVALS        : {len(tally['wrong'])}   <- the number that matters")
    for line in tally["wrong"]:
        say(f"    {line}")
    say(f"governance source(s)        : {', '.join(document['governance_source'])}")
    say(
        f"corpus                      : {document['corpus']['name']}, "
        f"{document['corpus']['glossary_entries']} glossary entries"
    )

    if tally["review"]:
        say("")
        say("sent to review, and what each would have inherited if accepted:")
        for key in tally["review"]:
            top = produced[key][0]
            protection_class = vocabulary.get(
                getattr(entries.get(top["governance_id"]), "governance_code", None)
            )
            say(f"  {key}")
            say(f"      {top['governance_id']}  {describe(protection_class)}")


def main() -> int:
    say(BANNER)
    rule("expectations recorded in fields.json vs what the pack just produced")

    if not RESULTS_FILE.is_file():
        say(f"FAIL {RESULTS_FILE} does not exist. Run run_pack.py first.")
        return 1

    document = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    produced = document["fields"]
    vocabulary = load_vocabulary()
    entries = governed_entries(vocabulary)
    expected_fields = read_fields()

    tally: dict = {"auto_approved": 0, "wrong": [], "review": [], "novel": 0}
    failures: list[str] = []

    say(f"{'field':40} {'decision':13} {'top-1':10} verdict")
    for row in expected_fields:
        verdict = _check_one_field(row, produced, tally)
        if verdict != "ok":
            failures.append(f"{row['flattenedName']}: {verdict}")

    rule("the class each field actually inherits, against the glossary")
    failures.extend(_check_inherited_classes(produced, entries, vocabulary))

    rule("summary")
    _summary(document, produced, entries, vocabulary, tally, len(expected_fields))

    rule()
    if failures:
        say(f"CHECK FAILED -- {len(failures)} expectation(s) not met")
        return 1
    say(f"CHECK OK -- {len(expected_fields)} fields, {len(tally['wrong'])} wrong auto-approvals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
