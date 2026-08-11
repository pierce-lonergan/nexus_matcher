"""
examples.governance.run_pack | Layer: EXAMPLE (command 2 of 5)
Match the schema fields against the glossary and write an auditable results file.

    python examples/governance/run_pack.py

Writes `examples/governance/out/results.json`, one record per candidate, carrying the
GOVERNANCE ID and the PROTECTION CLASS the field would inherit -- not just a business
name and a score. The whole reason a caller matches a field is to inherit the entry's
governance, so a results file that omits it forces every consumer to re-join against the
glossary and get that join wrong in their own way.

The document names its own inputs (vocabulary, glossary, schema, corpus size) because a
governance artifact that does not say what produced it cannot be checked later, and
because a quality number carried without its corpus identity is a number nobody can
compare against anything (docs/HAZARDS.md, H-002).
"""

from __future__ import annotations

import json
from typing import Any

from _pack import (
    BANNER,
    FIELDS_FILE,
    GLOSSARY_FILE,
    RESULTS_FILE,
    VOCABULARY_FILE,
    build_matcher,
    describe,
    governed_entries,
    load_vocabulary,
    resolve_governance,
    rule,
    say,
)

# Decimals kept for every emitted number, matching the CLI's JSON writer. Enough that a
# reader recomputing anything from this file cannot mistake rounding for an error.
PRECISION = 6


def _class_payload(protection_class: Any) -> dict[str, Any] | None:
    if protection_class is None:
        return None
    return {
        "code": protection_class.code,
        "name": protection_class.name,
        "classification": protection_class.classification,
        "personal_information": protection_class.personal_information,
        "direct_identifier": protection_class.direct_identifier,
        "enhancement": protection_class.enhancement,
    }


def main() -> int:
    say(BANNER)
    rule("matching the pack's schema fields against the pack's glossary")

    vocabulary = load_vocabulary()
    entries = governed_entries(vocabulary)
    matcher = build_matcher()
    results = matcher.match_schema(FIELDS_FILE)

    say(f"glossary entries indexed: {matcher.dictionary_size}")
    say(f"schema fields matched:    {len(results)}")

    sources: set[str] = set()
    payload: dict[str, Any] = {}
    rescued: list[str] = []
    for field_key in sorted(results):
        records = []
        for match in results[field_key]:
            governance_id, protection_class, source = resolve_governance(match, entries, vocabulary)
            sources.add(source)
            if source == "caller_side_join" and match.rank == 1:
                rescued.append(f"{field_key} -> {governance_id}")
            records.append(
                {
                    "rank": match.rank,
                    "governance_id": governance_id,
                    "business_name": match.dictionary_entry.business_name,
                    "confidence": round(float(match.final_confidence), PRECISION),
                    "decision": match.decision.value,
                    "governance": _class_payload(protection_class),
                }
            )
        payload[field_key] = records

    document = {
        "notice": BANNER,
        "vocabulary_file": VOCABULARY_FILE.name,
        "glossary_file": GLOSSARY_FILE.name,
        "schema_file": FIELDS_FILE.name,
        "corpus": {
            "name": "gravel-bay-ferry-authority (fictional)",
            "glossary_entries": matcher.dictionary_size,
            "schema_fields": len(results),
        },
        # Which path produced the governance on every record. "match_result" means the
        # matcher promoted it onto MatchResult, which is the contract. "caller_side_join"
        # means this example joined the id back to the glossary itself -- the same answer,
        # reached the long way, and stated rather than hidden.
        "governance_source": sorted(sources),
        "open_classification": vocabulary.classification_for(None),
        "fields": payload,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rule("what the top match would make each field inherit")
    for field_key in sorted(payload):
        top = payload[field_key][0]
        protection_class = vocabulary.get(
            (top["governance"] or {}).get("code") if top["governance"] else None
        )
        say(f"  {field_key}")
        say(
            f"      {top['decision']:12} {top['confidence']:.4f}  {top['governance_id']}  "
            f"{describe(protection_class)}"
        )

    rule()
    say(f"governance source: {', '.join(document['governance_source'])}")
    if rescued:
        # Printed as a block rather than a footnote. The failure it describes is silent
        # by construction: `governance=None` on a match reads exactly like "this entry
        # has no class", so nothing downstream can tell the two apart, and a governance
        # report full of blanks looks like a glossary that classifies nothing.
        say("")
        say("!! WIRING DEFECT -- the matcher returned NO CLASS for fields the glossary")
        say("!! classifies. matcher.load_dictionary() indexes through")
        say("!! BaseDictionaryLoader._convert_row, which never reads the protection-code")
        say("!! column, so every indexed entry carries governance_code=None. This pack")
        say("!! fell back to joining the id against ingest.load_entries() so the rest of")
        say("!! it still runs, and recorded 'caller_side_join' on every affected record.")
        say(f"!! affected top matches: {len(rescued)} of {len(results)}")
        for line in rescued[:5]:
            say(f"!!   {line}")
        if len(rescued) > 5:
            say(f"!!   ... and {len(rescued) - 5} more")
        say("!! See README.md, 'What this pack found'.")
    say(f"wrote {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
