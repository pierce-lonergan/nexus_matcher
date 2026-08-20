"""
benchmarks.synthetic_harness | Layer: BENCHMARK
The plumbing every synthetic-pack experiment shares: index a generated glossary, match a
generated schema, score against generated truth.

Kept out of `benchmarks/synthetic/` on purpose. That package is pure standard library and
imports nothing from `nexus_matcher`, so the generator runs anywhere -- on a machine with
no encoder, inside the enterprise that owns the real glossary, in a review that only wants
to read what the corpus looks like. This module is where the dependency starts.

Scoring, and what "correct" means per class
-------------------------------------------
    EXACT      rank 1 must be the one right term.
    AMBIGUOUS  rank 1 must be any of the recorded defensible terms. Scoring these against
               a single "best" id would measure a coin flip.
    NO_MATCH   the only correct behaviour is to DECLINE. There is no id to hit.
    TRAP       likewise -- and `trap_hit` additionally records how often the matcher
               returned the high-overlap term it was built to fall for, with what
               confidence. That number is the point of the class.

`answerable` is EXACT + AMBIGUOUS. P@1 quoted over anything else silently mixes "found the
right term" with "correctly found nothing", and those move in opposite directions when a
threshold changes.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic.glossary import ADMIT_APPROVED, COLUMN_MAPPING, VALUE_DELIMITERS
from synthetic.truth import TruthClass, TruthRow

from nexus_matcher.application.ingest import LoadReport, load_entries
from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.models.entities import FieldDecision, MatchingSession, MatchResult

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "benchmarks" / "results"


# =============================================================================
# WIRING
# =============================================================================


def build_matcher(config: MatchingConfig | None = None) -> NexusMatcher:
    """A fully wired matcher on the bundled offline encoder."""
    return NexusMatcher.from_config(config or MatchingConfig())


def index_glossary(
    matcher: NexusMatcher,
    rows: list[dict[str, str]],
    admit: bool = True,
    value_delimiters: dict[str, str] | None = None,
    delimiter_strict: bool = True,
) -> tuple[LoadReport, float]:
    """
    Read the generated rows through `load_entries` and index them.

    Through `load_entries` rather than `load_dictionary` because `admit` and
    `value_delimiters` -- the two features half this pack exists to exercise -- live there
    and nowhere else. `_index_dictionary` is reached directly for the same reason: the
    public path takes a file, and writing 100,000 rows to disk to vary one boolean is a
    minute of I/O per condition for no information.
    """
    report = LoadReport()
    entries = load_entries(
        rows,
        columns=dict(COLUMN_MAPPING),
        admit=dict(ADMIT_APPROVED) if admit else None,
        value_delimiters=dict(value_delimiters or VALUE_DELIMITERS),
        delimiter_strict=delimiter_strict,
        report=report,
    )
    t0 = time.perf_counter()
    matcher._index_dictionary(entries)
    return report, time.perf_counter() - t0


def parse_fields(flattened_rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    """The `SchemaField`s the matcher will see, without matching anything.

    Needed by the cache-key experiment, which has to compare the QUERY TEXT the pipeline
    builds for two columns before asking whether their answers may share a cache entry.
    """
    from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import (
        FlattenedAvroParser,
    )

    payload = {row["flattenedName"]: dict(row) for row in flattened_rows}
    result = FlattenedAvroParser().parse(payload)
    if result.is_failure:
        raise ValueError(result.error)
    return result.unwrap().fields


@dataclass
class Run:
    """One schema matched against one indexed glossary."""

    session: MatchingSession
    index_seconds: float
    match_seconds: float

    @property
    def results(self) -> dict[str, tuple[MatchResult, ...]]:
        return self.session.results

    @property
    def decisions(self) -> dict[str, FieldDecision]:
        return self.session.field_decisions()


def run_schema(
    matcher: NexusMatcher,
    flattened_rows: list[dict[str, Any]],
    index_seconds: float = 0.0,
    signals: Any = None,
) -> Run:
    """Match a pre-flattened field list -- the shape a production pipeline sends.

    Handed over KEYED BY FLATTENED NAME, which is one of the four shapes
    `FlattenedAvroParser` documents. The two obvious alternatives both fail in ways worth
    recording, because a caller with a list of columns in memory will reach for them:

      * a bare list raises `TypeError` from inside `pathlib` -- `_parse_schema` dispatches
        on `dict` or on a file suffix, and a list is neither;
      * `{"fields": [...]}`, the envelope the on-disk artifact uses, is refused by
        `can_parse`, which treats a `fields` key as the signature of raw Avro. That
        envelope works through `parse_file` (which never consults `can_parse`) and not in
        memory, so the same bytes parse from a file and do not parse from a variable.

    Neither is this harness's problem to fix -- src is another lane's -- but both are
    worth naming rather than working around silently.
    """
    payload = {row["flattenedName"]: dict(row) for row in flattened_rows}
    t0 = time.perf_counter()
    if signals is None:
        session = matcher.match_schema_session(payload)
    else:
        session = matcher.match_schema_session(payload, signals=signals)
    return Run(
        session=session,
        index_seconds=index_seconds,
        match_seconds=time.perf_counter() - t0,
    )


# =============================================================================
# SCORING
# =============================================================================


ANSWERABLE = (TruthClass.EXACT, TruthClass.AMBIGUOUS)
UNANSWERABLE = (TruthClass.NO_MATCH, TruthClass.TRAP)


@dataclass(frozen=True)
class FieldObs:
    """Everything one field's result says, before any threshold is applied.

    Separated from `Scored` so a threshold sweep does not need the matcher: the decision
    layer is `confidence >= auto_approve_threshold AND gap >= min_confidence_gap`, and
    recomputing it here means a 40-point sweep costs no re-indexing. Re-running the
    matcher per threshold would also re-embed the corpus 40 times, which is how a sweep
    quietly becomes the most expensive experiment in the suite.
    """

    key: str
    truth_class: TruthClass
    hit: int
    top_id: str
    conf1: float
    conf2: float
    absolute1: float | None
    declined: bool
    auto_approved: bool

    @property
    def gap(self) -> float:
        return self.conf1 - self.conf2

    @property
    def answerable(self) -> bool:
        return self.truth_class in ANSWERABLE


def observations(truth_rows: tuple[TruthRow, ...] | list[TruthRow], run: Run) -> list[FieldObs]:
    """One record per truth row that the run actually returned a key for."""
    decisions = run.decisions
    results = run.results
    out: list[FieldObs] = []
    for row in truth_rows:
        matches = results.get(row.key)
        if matches is None:
            continue
        top = matches[0] if matches else None
        correct = set(row.correct_ids)
        out.append(
            FieldObs(
                key=row.key,
                truth_class=row.truth_class,
                hit=1 if top is not None and top.dictionary_entry.id in correct else 0,
                top_id=top.dictionary_entry.id if top is not None else "",
                conf1=float(top.final_confidence) if top is not None else 0.0,
                conf2=float(matches[1].final_confidence) if len(matches) > 1 else 0.0,
                absolute1=(
                    None
                    if top is None or top.score_breakdown.absolute_cosine is None
                    else float(top.score_breakdown.absolute_cosine)
                ),
                declined=decisions.get(row.key) is FieldDecision.NO_MATCH or not matches,
                auto_approved=bool(top is not None and top.is_auto_approved),
            )
        )
    return out


def threshold_sweep(
    obs: list[FieldObs],
    min_gap: float,
    grid: tuple[float, ...],
) -> list[dict[str, float]]:
    """
    Coverage and auto-approve precision at each candidate threshold.

    Precision counts an auto-approval on an UNANSWERABLE row as wrong, which it is: there
    is no correct term, and an auto-approved match inherits its definition into whatever
    the consumer ships. A sweep that scored only the answerable rows would report a
    precision that improves as the no-match share rises.
    """
    rows: list[dict[str, float]] = []
    for t in grid:
        approved = [o for o in obs if o.conf1 >= t and o.gap >= min_gap]
        right = sum(o.hit for o in approved if o.answerable)
        rows.append(
            {
                "threshold": t,
                "coverage": len(approved) / len(obs) if obs else 0.0,
                "precision": right / len(approved) if approved else 0.0,
                "n_auto_approved": float(len(approved)),
            }
        )
    return rows


@dataclass
class Scored:
    """Metrics, plus the per-query vectors a paired test needs."""

    n: int = 0
    n_answerable: int = 0
    n_unanswerable: int = 0
    p_at_1: float = 0.0
    p_at_5: float = 0.0
    abstention_rate: float = 0.0
    false_abstention_rate: float = 0.0
    trap_hit_rate: float = 0.0
    auto_approve_coverage: float = 0.0
    auto_approve_precision: float = 0.0
    missing_keys: int = 0
    # key -> 1/0, over answerable rows only, in truth order. The input to McNemar.
    hits: dict[str, int] = field(default_factory=dict)
    confidences: dict[str, list[float]] = field(default_factory=dict)
    absolute_scores: dict[str, list[float]] = field(default_factory=dict)

    def render(self, title: str) -> str:
        return (
            f"\n  {title}\n"
            f"  {'-' * len(title)}\n"
            f"    fields scored        {self.n}  "
            f"({self.n_answerable} answerable, {self.n_unanswerable} not)\n"
            f"    P@1 / P@5            {self.p_at_1:.4f} / {self.p_at_5:.4f}\n"
            f"    abstained on the unanswerable   {self.abstention_rate:.4f}\n"
            f"    abstained on the answerable     {self.false_abstention_rate:.4f}\n"
            f"    returned the trap term at rank 1 {self.trap_hit_rate:.4f}\n"
            f"    auto-approve cov/prec {self.auto_approve_coverage:.4f} / "
            f"{self.auto_approve_precision:.4f}\n"
        )


def score(
    truth_rows: tuple[TruthRow, ...] | list[TruthRow],
    run: Run,
) -> Scored:
    """Score one run against the truth rows for the schema it matched."""
    out = Scored()
    decisions = run.decisions
    results = run.results

    hits: dict[str, int] = {}
    p1 = p5 = 0
    abstained_unanswerable = 0
    abstained_answerable = 0
    trap_hits = 0
    auto = 0
    auto_right = 0
    conf: dict[str, list[float]] = {c.value: [] for c in TruthClass}
    absolute: dict[str, list[float]] = {c.value: [] for c in TruthClass}

    for row in truth_rows:
        matches = results.get(row.key)
        if matches is None:
            out.missing_keys += 1
            continue
        out.n += 1
        declined = decisions.get(row.key) is FieldDecision.NO_MATCH or not matches
        top = matches[0] if matches else None

        if top is not None:
            conf[row.truth_class.value].append(float(top.final_confidence))
            raw = top.score_breakdown.absolute_cosine
            if raw is not None:
                absolute[row.truth_class.value].append(float(raw))

        if row.truth_class in ANSWERABLE:
            out.n_answerable += 1
            correct = set(row.correct_ids)
            hit1 = 1 if top is not None and top.dictionary_entry.id in correct else 0
            hits[row.key] = hit1
            p1 += hit1
            p5 += 1 if any(m.dictionary_entry.id in correct for m in matches[:5]) else 0
            if declined:
                abstained_answerable += 1
            if top is not None and top.is_auto_approved:
                auto += 1
                auto_right += hit1
        else:
            out.n_unanswerable += 1
            if declined:
                abstained_unanswerable += 1
            if row.trap_id and top is not None and top.dictionary_entry.id == row.trap_id:
                trap_hits += 1
            if top is not None and top.is_auto_approved:
                # An auto-approval on a row where nothing is correct is wrong by
                # definition, and it is the expensive kind of wrong: an auto-approved
                # match inherits its definition into whatever the consumer ships.
                auto += 1

    out.hits = hits
    out.confidences = conf
    out.absolute_scores = absolute
    if out.n_answerable:
        out.p_at_1 = p1 / out.n_answerable
        out.p_at_5 = p5 / out.n_answerable
        out.false_abstention_rate = abstained_answerable / out.n_answerable
    if out.n_unanswerable:
        out.abstention_rate = abstained_unanswerable / out.n_unanswerable
    traps = sum(1 for r in truth_rows if r.truth_class is TruthClass.TRAP)
    if traps:
        out.trap_hit_rate = trap_hits / traps
    if out.n:
        out.auto_approve_coverage = auto / out.n
    if auto:
        out.auto_approve_precision = auto_right / auto
    return out


def paired_vectors(a: Scored, b: Scored) -> tuple[list[float], list[float], list[str]]:
    """The two hit vectors over the keys BOTH runs scored, in a stable order.

    Restricting to the intersection is not a convenience. This repository has published a
    false regression from an unpaired comparison, and has a fixture that read a change as
    -1.33 points where the full corpus read the same change as +0.58. Two vectors of
    different lengths silently become an unpaired comparison.
    """
    keys = sorted(set(a.hits) & set(b.hits))
    return [float(a.hits[k]) for k in keys], [float(b.hits[k]) for k in keys], keys
