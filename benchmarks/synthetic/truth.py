"""
benchmarks.synthetic.truth | Layer: BENCHMARK
Artifact 4 of 5: ground truth, free by construction.

The generator makes the term first and the column that should match it second, so the
answer is known without a steward ever looking at it. That is the compounding benefit of
a synthetic corpus and it is why feature-level accuracy is testable here at any scale for
nothing.

The four classes, and why the last two are the point
----------------------------------------------------
EXACT (60%) and AMBIGUOUS (20%) are what every public benchmark measures: the correct
answer exists, and the question is whether the matcher finds it.

NO_MATCH (15%) and TRAP (5%) are what none of them measure. Both are rows where the
correct behaviour is to DECLINE, and a matcher that cannot decline scores identically on
them whether it is working or broken. This library ships `absolute_score_floor=None` by
default and `final_confidence` has a structural floor of 0.63, so as shipped every one of
these rows comes back with a rank-1 candidate that is at least REVIEW -- which is what
makes the fixture worth having rather than a formality.

Both classes are constructed so that "no correct term exists" is a fact about the
generator, not an observation about the data:

  NO_MATCH  the column is built from the held-out orphan vocabulary (see pools.Pools),
            which the glossary generator is never given. No glossary row can describe it
            at 1,000 rows or at 100,000.
  TRAP      the column keeps a real term's qualifiers and class word and replaces its
            SUBJECT with an orphan. Lexical overlap with that term stays high, the
            meaning is unrelated, and again no correct term exists. `trap_id` records the
            term the matcher is expected to fall for, so "confidently wrong" is a number
            and not an anecdote.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class TruthClass(str, Enum):
    EXACT = "EXACT"
    AMBIGUOUS = "AMBIGUOUS"
    NO_MATCH = "NO_MATCH"
    TRAP = "TRAP"


# The specification's shares. A dial, because the interesting question about the no-match
# share is what happens when it is not 15%.
DEFAULT_SHARES: dict[TruthClass, float] = {
    TruthClass.EXACT: 0.60,
    TruthClass.AMBIGUOUS: 0.20,
    TruthClass.NO_MATCH: 0.15,
    TruthClass.TRAP: 0.05,
}


@dataclass(frozen=True)
class TruthRow:
    """One (schema, field) -> answer record."""

    schema: str
    flattened_name: str
    field_path: str
    data_type: str
    truth_class: TruthClass
    # Every defensible id, not just the best one. For AMBIGUOUS this is the whole cluster;
    # for NO_MATCH and TRAP it is empty, and empty is the answer rather than a gap.
    correct_ids: tuple[str, ...]
    trap_id: str = ""
    note: str = ""

    @property
    def key(self) -> str:
        """The handle a match result is looked up under: `field_result_key` returns the
        flattened name for a flattened schema, which is what every profile here emits."""
        return self.flattened_name


TRUTH_HEADER: tuple[str, ...] = (
    "schema",
    "flattened_name",
    "field_path",
    "data_type",
    "truth_class",
    "correct_ids",
    "trap_id",
    "note",
)

# The truth file's own multi-value separator. Deliberately NOT ';' or ',': those two are
# the glossary's, and the pack's whole delimiter trap depends on a reader being able to
# tell them apart. A third character here means a mistake in one file cannot be excused
# by a convention borrowed from the other.
TRUTH_ID_SEPARATOR = "|"


def write_truth_csv(path: Path, rows: tuple[TruthRow, ...]) -> None:
    """Write truth.csv. Newline handling is explicit so the file is byte-identical on
    every platform -- a corpus whose checksum depends on the OS is not reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(TRUTH_HEADER)
        for row in rows:
            writer.writerow(
                [
                    row.schema,
                    row.flattened_name,
                    row.field_path,
                    row.data_type,
                    row.truth_class.value,
                    TRUTH_ID_SEPARATOR.join(row.correct_ids),
                    row.trap_id,
                    row.note,
                ]
            )


def read_truth_csv(path: Path) -> tuple[TruthRow, ...]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return tuple(
            TruthRow(
                schema=r["schema"],
                flattened_name=r["flattened_name"],
                field_path=r["field_path"],
                data_type=r["data_type"],
                truth_class=TruthClass(r["truth_class"]),
                correct_ids=tuple(p for p in r["correct_ids"].split(TRUTH_ID_SEPARATOR) if p),
                trap_id=r["trap_id"],
                note=r["note"],
            )
            for r in csv.DictReader(fh)
        )


def class_counts(rows: tuple[TruthRow, ...]) -> dict[str, int]:
    counts: dict[str, int] = {c.value: 0 for c in TruthClass}
    for row in rows:
        counts[row.truth_class.value] += 1
    return counts
