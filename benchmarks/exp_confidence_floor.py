"""
benchmarks.exp_confidence_floor | Layer: BENCHMARK
Is the documented confidence floor the EMPIRICAL floor, and how often does the matcher
decline when declining is the only right answer?

Two questions, one run, because they are the same measurement read twice.

E5 -- the floor
---------------
`MatchingConfig.minimum_achievable_confidence` computes `semantic_weight * fusion_alpha`
= 0.63 for the shipped configuration, and derives it from first principles: the fused
retrieval score is min-max normalised over the candidates retrieved for ONE field, so the
rank-1 candidate always lands at or above `fusion_alpha`, and it carries `semantic_weight`
of the final confidence. That is an argument. This script is the observation, over every
field in the pack, and it reports the histogram rather than a pass mark -- a floor that is
correct but never approached is a different fact from one that half the corpus sits on.

The floor was folklore in this repository before it was computed: `get_low_confidence_fields`
shipped with a default threshold of 0.6, below the floor, and therefore returned an empty
list on every schema ever matched -- telling a governance lead there was nothing to review
on a schema where nothing was trustworthy.

E4 -- abstention
----------------
20% of the pack's fields have no correct term BY CONSTRUCTION: 15% NO_MATCH built from a
held-out vocabulary the glossary generator never sees, and 5% TRAP -- high lexical
overlap, unrelated meaning. The only correct behaviour on those rows is to decline.

With `absolute_score_floor=None`, which is the shipped default and the only honest default
for a library that has not seen the caller's corpus, no field can ever be declined: rank 1
cannot fall below 0.63 and `review_threshold` is 0.50, so every field comes back at least
REVIEW however irrelevant its best candidate is. The measured abstention rate is expected
to be exactly 0.0000, and printing it is worth more than arguing it.

The script then sweeps `absolute_score_floor` across the observed range and prints what a
caller would actually get at each value: how many unanswerable rows it declines against
how many answerable ones it throws away. That table is the measurement
`docs/guides/absolute_score_floor.md` tells a caller to make on their own corpus, made
here on a corpus that has one.

Usage
-----
    python benchmarks/exp_confidence_floor.py --rows 10000 --save
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimization_ledger import provenance
from synthetic.pack import DEFAULT_SEED, PackSpec, SyntheticPack
from synthetic.truth import TruthClass
from synthetic_harness import (
    RESULTS,
    FieldObs,
    build_matcher,
    index_glossary,
    observations,
    run_schema,
)

from nexus_matcher.application.use_cases.match_schema import MatchingConfig

BAR_WIDTH = 46


def _histogram(
    values: list[float], lo: float, hi: float, bins: int = 20
) -> list[tuple[float, float, int]]:
    if not values:
        return []
    width = (hi - lo) / bins if hi > lo else 1.0
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / width) if width else 0
        counts[min(max(idx, 0), bins - 1)] += 1
    return [(lo + i * width, lo + (i + 1) * width, c) for i, c in enumerate(counts)]


def _render_histogram(rows: list[tuple[float, float, int]]) -> str:
    if not rows:
        return "    (no values)\n"
    peak = max(c for _a, _b, c in rows) or 1
    out = []
    for a, b, c in rows:
        bar = "#" * round(BAR_WIDTH * c / peak)
        out.append(f"    {a:.4f}-{b:.4f} {c:>7} |{bar}")
    return "\n".join(out) + "\n"


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": 0.0, "p05": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    s = sorted(values)
    return {
        "n": len(s),
        "min": s[0],
        "p05": s[max(0, int(0.05 * len(s)) - 1)],
        "median": s[len(s) // 2],
        "mean": sum(s) / len(s),
        "max": s[-1],
    }


def _floor_sweep(obs: list[FieldObs], steps: int = 24) -> list[dict[str, float]]:
    """
    What an `absolute_score_floor` at each value would actually do.

    `declined` follows `derive_field_decision` exactly, including its rule that a rank-1
    candidate with NO absolute score is declined when a floor is configured: it reached
    the shortlist through the lexical arm alone, and clearing a similarity floor on
    evidence that does not exist is the expensive direction of the mistake.
    """
    scores = [o.absolute1 for o in obs if o.absolute1 is not None]
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    answerable = [o for o in obs if o.answerable]
    unanswerable = [o for o in obs if not o.answerable]
    rows = []
    for i in range(steps + 1):
        floor = lo + (hi - lo) * i / steps
        declined_bad = sum(1 for o in unanswerable if o.absolute1 is None or o.absolute1 < floor)
        declined_good = sum(1 for o in answerable if o.absolute1 is None or o.absolute1 < floor)
        # Of the answerable fields still answered, how many are right. A floor that
        # discards the fields it was going to get wrong buys precision honestly; one that
        # discards the ones it was going to get right does not.
        kept = [o for o in answerable if o.absolute1 is not None and o.absolute1 >= floor]
        rows.append(
            {
                "floor": floor,
                "abstains_on_unanswerable": (
                    declined_bad / len(unanswerable) if unanswerable else 0.0
                ),
                "loses_answerable": declined_good / len(answerable) if answerable else 0.0,
                "p_at_1_on_kept": (sum(o.hit for o in kept) / len(kept)) if kept else 0.0,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=10_000, help="glossary rows")
    ap.add_argument("--schema-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    config = MatchingConfig()
    spec = PackSpec(
        rows=args.rows, seed=args.seed, schema_scale=args.schema_scale, feedback_events=100
    )
    print(f"\nGenerating pack: {spec.rows} glossary rows, seed {spec.seed}")
    pack = SyntheticPack.generate(spec)

    matcher = build_matcher(config)
    report, index_s = index_glossary(matcher, pack.glossary_dicts())
    print(
        f"Indexed {report.entries} approved entries in {index_s:.1f}s "
        f"({report.refused} rows refused by the status filter)"
    )

    obs: list[FieldObs] = []
    session_floors: dict[str, float | None] = {}
    for schema in pack.schemas:
        run = run_schema(matcher, schema.flattened, index_s)
        session_floors[schema.name] = run.session.minimum_achievable_confidence
        obs.extend(observations(schema.truth, run))
        print(
            f"  matched {schema.name:<18} {len(schema.flattened):>5} fields "
            f"in {run.match_seconds:.1f}s"
        )

    documented = config.minimum_achievable_confidence
    matcher_floor = matcher.minimum_achievable_confidence
    conf = [o.conf1 for o in obs]
    overall = _stats(conf)
    below = [o for o in obs if o.conf1 < documented]

    print("\n" + "=" * 74)
    print("E5  THE CONFIDENCE FLOOR")
    print("=" * 74)
    print(
        f"  documented (semantic_weight {config.semantic_weight} * fusion_alpha "
        f"{config.fusion_alpha})   {documented:.4f}"
    )
    print(f"  reported by the matcher                                {matcher_floor}")
    print(
        f"  reported on every session                              "
        f"{sorted(set(session_floors.values()))}"
    )
    print(f"  EMPIRICAL minimum over {overall['n']} rank-1 results          {overall['min']:.4f}")
    print(f"  fields below the documented floor                      {len(below)}")
    print(
        f"  median / mean / max                                    "
        f"{overall['median']:.4f} / {overall['mean']:.4f} / {overall['max']:.4f}"
    )
    slack = overall["min"] - documented
    if below:
        print(f"\n  THE FLOOR IS WRONG: {len(below)} field(s) fall below it")
    else:
        print(f"\n  The documented floor HOLDS: nothing fell below {documented:.4f}.")
        print("  It is a valid bound and it is not TIGHT -- the lowest confidence any")
        print(f"  field actually reached was {overall['min']:.4f}, {slack:.4f} above it.")
        print("  That slack is the residual of the same defect the floor was computed to")
        print("  kill. `get_low_confidence_fields(threshold=...)` refuses a threshold at or")
        print(f"  below {documented:.4f} because it could only return []; a caller who takes")
        print(
            f"  that refusal at its word and passes {documented + 0.005:.4f} still gets [] on this"
        )
        print("  corpus, and gets it with no refusal and no explanation. The number that")
        print("  makes the API honest is the observed minimum, not the derived bound.")
    print("\n  Rank-1 confidence over the whole corpus")
    print(_render_histogram(_histogram(conf, min(conf), max(conf))))

    print("  By truth class")
    print("  " + "-" * 12)
    by_class: dict[str, dict[str, float]] = {}
    for cls in TruthClass:
        vals = [o.conf1 for o in obs if o.truth_class is cls]
        st = _stats(vals)
        by_class[cls.value] = st
        if st["n"]:
            print(
                f"    {cls.value:<10} n={st['n']:<6} min {st['min']:.4f}  "
                f"median {st['median']:.4f}  max {st['max']:.4f}"
            )
    answerable = [o for o in obs if o.answerable]
    unanswerable = [o for o in obs if not o.answerable]
    if answerable and unanswerable:
        gap = sum(o.conf1 for o in answerable) / len(answerable) - sum(
            o.conf1 for o in unanswerable
        ) / len(unanswerable)
        print(f"\n    mean confidence, answerable minus unanswerable: {gap:+.4f}")
        print("    (confidence is rank-relative, so it is not expected to separate them;")
        print("     that is the reason absolute_score_floor exists at all)")

    print("\n" + "=" * 74)
    print("E4  ABSTENTION")
    print("=" * 74)
    declined = sum(1 for o in obs if o.declined)
    declined_unanswerable = sum(1 for o in unanswerable if o.declined)
    auto_on_unanswerable = sum(1 for o in unanswerable if o.auto_approved)
    print(
        f"  fields with no correct term by construction            "
        f"{len(unanswerable)} of {len(obs)} ({len(unanswerable) / len(obs):.1%})"
    )
    print(f"  absolute_score_floor                                   {config.absolute_score_floor}")
    print(f"  declined (FieldDecision.NO_MATCH), all fields          {declined}")
    print(
        f"  declined, of the ones that SHOULD be declined          "
        f"{declined_unanswerable} ({declined_unanswerable / len(unanswerable):.4f})"
    )
    print(f"  AUTO_APPROVED although nothing is correct              {auto_on_unanswerable}")
    traps = [o for o in obs if o.truth_class is TruthClass.TRAP]
    if traps:
        print(
            f"  trap rows, median rank-1 confidence                    "
            f"{_stats([o.conf1 for o in traps])['median']:.4f}"
        )

    sweep = _floor_sweep(obs)
    if sweep:
        print("\n  What an absolute_score_floor would buy on this corpus")
        print("  " + "-" * 52)
        print("    floor    abstains on   loses        P@1 on")
        print("             unanswerable  answerable   what is kept")
        for row in sweep:
            print(
                f"    {row['floor']:.4f}   {row['abstains_on_unanswerable']:>10.4f}   "
                f"{row['loses_answerable']:>9.4f}   {row['p_at_1_on_kept']:>10.4f}"
            )

    artifact = {
        "experiment": "exp_confidence_floor (E5 + E4)",
        "provenance": provenance(spec.seed, config.__dict__.copy()),
        "pack": pack.manifest(),
        "floor": {
            "documented": documented,
            "matcher_reported": matcher_floor,
            "session_reported": sorted({v for v in session_floors.values() if v is not None}),
            "empirical": overall,
            "fields_below_documented_floor": len(below),
            "holds": not below,
            "slack_empirical_minus_documented": overall["min"] - documented,
        },
        "confidence_by_truth_class": by_class,
        "abstention": {
            "absolute_score_floor": config.absolute_score_floor,
            "fields": len(obs),
            "unanswerable": len(unanswerable),
            "declined": declined,
            "declined_unanswerable": declined_unanswerable,
            "abstention_rate_on_unanswerable": (
                declined_unanswerable / len(unanswerable) if unanswerable else 0.0
            ),
            "auto_approved_on_unanswerable": auto_on_unanswerable,
        },
        "absolute_score_floor_sweep": sweep,
    }
    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"exp_confidence_floor_synthetic_{spec.rows}.json"
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nSaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
