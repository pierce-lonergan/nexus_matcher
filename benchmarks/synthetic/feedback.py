"""
benchmarks.synthetic.feedback | Layer: BENCHMARK
Artifact 5 of 5: a reviewer-feedback trace, including the verdict class that costs the
most to lose.

Three verdicts
--------------
  APPROVED         the reviewer accepted the matcher's rank-1 candidate.
  REJECTED         the reviewer rejected it and chose nothing. Usually a field nothing in
                   the glossary governs.
  MANUAL_OVERRIDE  the reviewer chose a term THE MATCHER NEVER PROPOSED -- not rank 2, not
                   rank 5: absent from the candidate list entirely.

The third is the highest-signal record there is. An approval says the ranking was right; a
rejection says it was wrong; an override says what the right answer was, on a field where
retrieval did not surface it at all. It is the only record that can teach a system
something retrieval could not have found.

What this repository's wire format can and cannot carry
-------------------------------------------------------
Measured against `presentation/api/feedback.py:_RECORD_KEYS`, which stores

    ts, receivedAt, reviewer, field, doc, chosenGovernanceId, suggestedGovernanceId,
    wasCorrect

the loss is NARROWER than "a bool cannot express an override", and worth stating
precisely because the narrower version is the one that can be fixed:

  * APPROVED vs REJECTED survives -- `wasCorrect` carries it.
  * "the reviewer chose a DIFFERENT term" survives -- `chosenGovernanceId` differs from
    `suggestedGovernanceId`.
  * "the reviewer chose a term that was NEVER PROPOSED" does NOT survive.
    `suggestedGovernanceId` is rank 1 only, so a chosen id that differs from it may have
    been rank 2 or absent from a 50-candidate list, and after storage nothing can tell
    those apart. They are opposite facts: the first says the ranking was nearly right, the
    second says retrieval missed entirely.

So every record here carries `proposedIds` -- the whole candidate list the reviewer was
shown -- and `wire_projection()` returns what would survive a round trip through the
shipped endpoint. The difference between the two is the loss, in bytes, not in argument.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .glossary import Glossary
from .truth import TruthClass, TruthRow

VERDICT_APPROVED = "APPROVED"
VERDICT_REJECTED = "REJECTED"
VERDICT_MANUAL_OVERRIDE = "MANUAL_OVERRIDE"

_REVIEWERS: tuple[str, ...] = (
    "steward-a",
    "steward-b",
    "steward-c",
    "governance-analyst-1",
    "governance-analyst-2",
)

_REASONS: dict[str, tuple[str, ...]] = {
    VERDICT_APPROVED: (
        "Correct term for this column.",
        "Matches the governed element this feed carries.",
        "Agreed; the domain and the class word both line up.",
    ),
    VERDICT_REJECTED: (
        "Nothing in the glossary governs this column. Raise a term request.",
        "Shares tokens with the suggestion but means something else.",
        "Wrong domain, and no other candidate is right either.",
    ),
    VERDICT_MANUAL_OVERRIDE: (
        "The right term was not in the candidate list; supplied it by hand.",
        "None of the proposals were close. The governed element is the one named here.",
        "Retrieval missed this one entirely; chosen from the register directly.",
    ),
}


@dataclass(frozen=True)
class FeedbackEvent:
    """One reviewer verdict, in full."""

    event_id: str
    ts: str
    reviewer: str
    schema: str
    field: str
    proposed_ids: tuple[str, ...]
    verdict: str
    chosen_id: str
    reason: str

    @property
    def was_correct(self) -> bool:
        """What a boolean-only surface would record. APPROVED and nothing else."""
        return self.verdict == VERDICT_APPROVED

    @property
    def chose_an_unproposed_term(self) -> bool:
        return bool(self.chosen_id) and self.chosen_id not in self.proposed_ids

    def as_json(self) -> dict[str, object]:
        return {
            "eventId": self.event_id,
            "ts": self.ts,
            "reviewer": self.reviewer,
            "schema": self.schema,
            "field": self.field,
            "proposedIds": list(self.proposed_ids),
            "suggestedGovernanceId": self.proposed_ids[0] if self.proposed_ids else "",
            "verdict": self.verdict,
            "chosenGovernanceId": self.chosen_id,
            "wasCorrect": self.was_correct,
            "reason": self.reason,
        }

    def wire_projection(self) -> dict[str, object]:
        """Exactly the keys `_RECORD_KEYS` stores. Everything else is dropped."""
        return {
            "ts": self.ts,
            "reviewer": self.reviewer,
            "field": self.field,
            "doc": "",
            "chosenGovernanceId": self.chosen_id,
            "suggestedGovernanceId": self.proposed_ids[0] if self.proposed_ids else "",
            "wasCorrect": self.was_correct,
        }


def build_feedback(
    glossary: Glossary,
    truth_rows: tuple[TruthRow, ...],
    seed: int,
    count: int = 5_000,
    candidates_per_field: int = 5,
    override_share: float = 0.15,
) -> tuple[FeedbackEvent, ...]:
    """
    Synthesise a trace of `count` verdicts against the pack's own fields.

    The candidate lists are MANUFACTURED, not measured: this is a fixture for testing what
    a system does with reviewer history, not a record of what any matcher returned. That
    distinction matters -- replaying it proves the bypass mechanism works, and proves
    nothing about the accuracy of the matcher that would have produced the proposals.
    """
    rng = random.Random(seed ^ 0x5EED_0005)
    all_ids = [row.id for row in glossary.approved]
    if not all_ids:
        raise RuntimeError("no approved glossary rows; a feedback trace would have no terms")

    pool = list(truth_rows)
    rng.shuffle(pool)
    start = datetime(2026, 5, 4, 9, 0, 0, tzinfo=timezone.utc)

    events: list[FeedbackEvent] = []
    for i in range(count):
        row = pool[i % len(pool)]
        correct = row.correct_ids[0] if row.correct_ids else ""

        # The candidate list the reviewer was shown. Distractors are drawn from the
        # glossary at large, which is what a top-k list is.
        distractors = [rng.choice(all_ids) for _ in range(candidates_per_field)]

        if row.truth_class in (TruthClass.NO_MATCH, TruthClass.TRAP):
            verdict = VERDICT_REJECTED
            proposed = tuple(dict.fromkeys([row.trap_id or distractors[0], *distractors]))[
                :candidates_per_field
            ]
            chosen = ""
        elif correct and rng.random() < override_share:
            verdict = VERDICT_MANUAL_OVERRIDE
            # The defining property: the chosen term is NOT in the list. Enforced by
            # construction rather than by hoping the random draw misses it.
            proposed = tuple(d for d in dict.fromkeys(distractors) if d != correct)
            if not proposed:
                proposed = (rng.choice(all_ids),)
            chosen = correct
        elif correct:
            verdict = VERDICT_APPROVED
            proposed = tuple(dict.fromkeys([correct, *distractors]))[:candidates_per_field]
            chosen = correct
        else:
            verdict = VERDICT_REJECTED
            proposed = tuple(dict.fromkeys(distractors))[:candidates_per_field]
            chosen = ""

        events.append(
            FeedbackEvent(
                event_id=f"SYN-REV-{i + 1:06d}",
                ts=(start + timedelta(minutes=7 * i)).isoformat().replace("+00:00", "Z"),
                reviewer=rng.choice(_REVIEWERS),
                schema=row.schema,
                field=row.flattened_name,
                proposed_ids=proposed,
                verdict=verdict,
                chosen_id=chosen,
                reason=rng.choice(_REASONS[verdict]),
            )
        )
    return tuple(events)


def write_feedback_jsonl(path: Path, events: tuple[FeedbackEvent, ...]) -> None:
    """One record per line, pure ASCII, `\\n` endings -- the same promises the shipped
    recorder makes, so the file is a drop-in for anything that reads its output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        for event in events:
            fh.write(
                json.dumps(
                    event.as_json(), ensure_ascii=True, allow_nan=False, separators=(",", ":")
                )
                + "\n"
            )


def wire_loss(events: tuple[FeedbackEvent, ...]) -> dict[str, int]:
    """
    How much of the trace the shipped wire format cannot carry.

    `overrides_indistinguishable_from_reranks` is the number: records where the reviewer
    chose a term that was never proposed, but where the stored projection is
    byte-identical to what a "reviewer preferred rank 2" record would look like.
    """
    total_overrides = sum(1 for e in events if e.verdict == VERDICT_MANUAL_OVERRIDE)
    unproposed = sum(1 for e in events if e.chose_an_unproposed_term)
    return {
        "events": len(events),
        "approved": sum(1 for e in events if e.verdict == VERDICT_APPROVED),
        "rejected": sum(1 for e in events if e.verdict == VERDICT_REJECTED),
        "manual_override": total_overrides,
        "chose_an_unproposed_term": unproposed,
        "overrides_indistinguishable_from_reranks": unproposed,
    }
