"""
examples.governance.calibrate | Layer: EXAMPLE (command 4 of 5)
Sweep the auto-approve threshold over labelled query -> id pairs.

    python examples/governance/calibrate.py

`labels.jsonl` is a labelled set: each line is a schema field and the glossary id that is
the correct answer, or `null` when the correct answer is "nothing in this glossary". The
sweep replays the matcher's own decision rule at each candidate threshold and reports
what it would buy.

## Three rules this obeys, each of them learned the hard way here

**Both metrics, always.** P@1 and auto-approve precision are printed together.
`docs/HAZARDS.md` H-001 is a hazard because they move in OPPOSITE directions: better
retrieval lifts the whole score distribution, so more candidates clear a fixed bar,
including wrong ones. It has happened three times in this repository. A calibration
report that shows only the retrieval number is how it happens a fourth.

**No threshold below the floor.** `final_confidence` cannot fall below
`semantic_weight * fusion_alpha` for a rank-1 match, so any threshold at or under that
number selects everything and means nothing. A default of 0.6 -- one hundredth below the
floor -- is what made `get_low_confidence_fields()` answer "nothing to review" on every
schema ever matched (DX-001, `tests/museum/NM-0027`). Thresholds at or below the floor
are excluded here, by name, rather than quietly producing a 100%-coverage row.

**The corpus is named.** Every number below is 42 labels over a 30-entry glossary. It
calibrates THIS pack. H-002: a threshold calibrated on one corpus does not transfer, and
corpus size is a regime change rather than a scaling factor -- one option in this library
is worth +1.9 P@1 at 688 entries and -18.8 at 30,000. Do not carry these numbers
anywhere.
"""

from __future__ import annotations

from _pack import BANNER, LABELS_FILE, build_matcher, load_jsonl, rule, say

# The thresholds to try. Anything at or below the structural floor is dropped below, with
# its name printed, because a row reading "coverage 100%" at a threshold nothing can fall
# under is a row that reads as a spectacular result and means nothing at all.
CANDIDATE_THRESHOLDS = (0.60, 0.63, 0.70, 0.80, 0.85, 0.87, 0.90, 0.92)


def _top_matches(results) -> list[tuple[str, str, float, float]]:
    """(field, top-1 id, confidence, margin over the runner-up) per field."""
    tops: list[tuple[str, str, float, float]] = []
    for key, matches in results.items():
        if not matches:
            tops.append((key, "", 0.0, 0.0))
            continue
        top = float(matches[0].final_confidence)
        runner_up = float(matches[1].final_confidence) if len(matches) > 1 else 0.0
        tops.append((key, matches[0].dictionary_entry.id, top, top - runner_up))
    return tops


def _usable_thresholds(floor: float | None) -> list[float]:
    """Every candidate threshold above the structural floor, naming the ones dropped."""
    usable = []
    for threshold in CANDIDATE_THRESHOLDS:
        if floor is not None and threshold <= floor:
            say(
                f"  threshold {threshold:.2f} EXCLUDED: at or below the structural floor "
                f"{floor:.4f}, so every rank-1 match clears it"
            )
            continue
        usable.append(threshold)
    return usable


def _sweep(tops, gold, usable, min_gap: float) -> None:
    """Coverage and auto-approve precision at each threshold, then who was wrong."""
    say("")
    say(
        f"{'threshold':>10} {'coverage':>10} {'auto-approved':>14} "
        f"{'auto-approve precision':>24} {'wrong':>7}"
    )
    wrong_by_threshold: dict[float, list[str]] = {}
    for threshold in usable:
        approved = [
            (key, entry_id)
            for key, entry_id, confidence, margin in tops
            if confidence >= threshold and margin >= min_gap
        ]
        wrong = [key for key, entry_id in approved if gold.get(key) != entry_id]
        wrong_by_threshold[threshold] = wrong
        coverage = len(approved) / len(tops) if tops else 0.0
        precision = (len(approved) - len(wrong)) / len(approved) if approved else 0.0
        precision_text = "n/a" if not approved else f"{precision:.4f}"
        say(
            f"{threshold:>10.2f} {coverage:>10.4f} {len(approved):>14} "
            f"{precision_text:>24} {len(wrong):>7}"
        )

    for threshold in usable:
        wrong = wrong_by_threshold[threshold]
        if wrong:
            say("")
            say(f"  auto-approved and WRONG at {threshold:.2f}:")
            for key in sorted(wrong):
                answer = gold.get(key) or "none -- nothing governs it"
                say(f"    {key}  (correct answer: {answer})")


def main() -> int:
    from nexus_matcher.application.use_cases.match_schema import MatchingConfig

    say(BANNER)
    rule("calibration set")

    labels = load_jsonl(LABELS_FILE)
    gold = {row["flattenedName"]: row.get("expected_id") for row in labels}
    positives = sum(1 for value in gold.values() if value)
    say(f"file: {LABELS_FILE.name}")
    say(
        f"labels: {len(labels)}  with a correct answer: {positives}  "
        f"with none: {len(labels) - positives}"
    )
    say("the labels carry names and types only, no descriptions -- the harder case, and")
    say("the one most schemas actually present")

    matcher = build_matcher()
    results = matcher.match_schema(LABELS_FILE)
    config = MatchingConfig()
    floor = matcher.minimum_achievable_confidence

    say(f"glossary entries: {matcher.dictionary_size}")
    say(f"fields matched:   {len(results)}")

    rule("the decision rule being replayed")
    say("auto-approve when rank-1 confidence >= THRESHOLD")
    say(
        f"                and margin over the runner-up >= min_confidence_gap "
        f"({config.min_confidence_gap})"
    )
    say(f"shipped auto_approve_threshold: {config.auto_approve_threshold}")
    if floor is None:
        say("structural floor: unknown for this configuration (a reranker is wired)")
    else:
        say(
            f"structural floor: {floor:.4f} = semantic_weight {config.semantic_weight} "
            f"* fusion_alpha {config.fusion_alpha}"
        )

    tops = _top_matches(results)

    # P@1 is scored over the labels that HAVE a correct answer. Scoring the negatives here
    # too would fold two questions into one number: retrieval cannot return "nothing", so a
    # negative can only ever be counted wrong, and a set with more negatives in it would
    # report worse retrieval for no retrieval reason. The negatives are scored where they
    # belong -- against the DECISION, in the sweep.
    answerable = [(key, entry_id) for key, entry_id, _, _ in tops if gold.get(key)]
    correct_at_1 = sum(1 for key, entry_id in answerable if gold.get(key) == entry_id)
    p_at_1 = correct_at_1 / len(answerable) if answerable else 0.0
    missed = [key for key, entry_id in answerable if gold.get(key) != entry_id]

    rule("sweep")
    usable = _usable_thresholds(floor)
    if not usable:
        say("FAIL every candidate threshold sits at or below the floor; nothing to sweep.")
        return 1
    _sweep(tops, gold, usable, config.min_confidence_gap)

    rule("both metrics, together (H-001)")
    say(
        f"P@1 over the {len(answerable)} labels that have a correct answer: {p_at_1:.4f}   "
        f"({correct_at_1} correct at rank 1)"
    )
    for key in sorted(missed):
        say(f"    missed: {key} -> wanted {gold.get(key)}")
    say(f"the other {len(tops) - len(answerable)} labels have no correct answer and are")
    say("scored only by their decision, in the sweep above")
    say("P@1 does not depend on the threshold. Auto-approve precision does, and it is the")
    say("one that decides whether a protection class is applied without a human. A change")
    say("that lifts the first while lowering the second is a REGRESSION, however good the")
    say("headline looks.")

    rule("what these numbers are, and are not")
    say(
        f"corpus: gravel-bay-ferry-authority (fictional), {matcher.dictionary_size} glossary "
        f"entries, {len(labels)} labels"
    )
    say("These calibrate this pack and nothing else. Re-run against YOUR glossary, at ITS")
    say("size, and re-run it again after any change to the encoder, the fusion weights,")
    say("the query representation, or the glossary itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
