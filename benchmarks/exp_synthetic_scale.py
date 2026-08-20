"""
benchmarks.exp_synthetic_scale | Layer: BENCHMARK
E10 -- where a threshold fitted on a small corpus stops transferring to a large one.

This is the pack's central empirical claim, and the one this repository most needs to know
about its own defaults. `auto_approve_threshold` ships at 0.87, calibrated on a 688-pair
public benchmark whose dictionary holds a few hundred entries. An enterprise glossary holds
five orders of magnitude more. If the threshold that buys 95% auto-approve precision at
1,000 entries buys materially less at 100,000, then the shipped number is not a
conservative default -- it is a number measured somewhere else.

Design
------
The QUERIES never change and the GOLD entries are present at every size; only the number of
competing entries grows. That is the same construction `benchmarks/exp_scale.py` uses on
the public corpus, and it is what makes the sizes comparable: a P@1 drop is attributable to
the larger candidate pool rather than to a different task.

    entries(N) = every gold entry, plus rows from the generated glossary in order until N

Three things are read at each size:

  1. P@1 and P@5 on the answerable fields.
  2. The confidence distribution, including its empirical minimum -- because the
     structural floor `semantic_weight * fusion_alpha` is a bound on the RANK-1 fused
     score, and how far above it the corpus actually sits is a property of how many
     candidates were retrieved.
  3. The full (threshold -> coverage, precision) curve, so the threshold that hits a
     precision target can be READ OFF at one size and APPLIED at another.

The transfer table at the end is the result. It fits the threshold at the smallest size,
which is what a calibration on a small benchmark amounts to, and reports what that same
number delivers at every larger one.

Usage
-----
    python benchmarks/exp_synthetic_scale.py --sizes 1000 10000 100000 --save
    python benchmarks/exp_synthetic_scale.py --sizes 1000 5000 --schemas flat-english
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimization_ledger import paired_precision_ci, provenance
from synthetic.pack import DEFAULT_SEED, PackSpec, SyntheticPack
from synthetic_harness import (
    RESULTS,
    FieldObs,
    build_matcher,
    index_glossary,
    observations,
    run_schema,
    threshold_sweep,
)

from nexus_matcher.application.use_cases.match_schema import MatchingConfig

DEFAULT_SCHEMAS = ("flat-english", "mixed-production")
GRID = tuple(round(0.60 + 0.01 * i, 2) for i in range(41))  # 0.60 .. 1.00


def _subset(pack: SyntheticPack, gold_ids: set[str], size: int) -> list[dict[str, str]]:
    """The glossary trimmed to `size` rows, gold entries always present.

    Order is preserved from the generated glossary so a smaller corpus is a genuine
    subset of a larger one: two sizes that shared no rows would differ in their
    vocabulary as well as their size, and the comparison would confound the two.
    """
    rows = pack.glossary_dicts()
    gold = [r for r in rows if r["id"] in gold_ids]
    if len(gold) > size:
        raise SystemExit(
            f"{len(gold)} gold entries do not fit in a corpus of {size}. Raise the "
            f"smallest size, or lower --schema-scale so fewer distinct terms are asked "
            f"for."
        )
    filler = [r for r in rows if r["id"] not in gold_ids]
    return gold + filler[: size - len(gold)]


def _trim_queries(truth: list, smallest: int, budget: float) -> list:
    """
    Drop queries until their gold entries fit comfortably inside the SMALLEST corpus.

    Two reasons, and the second is the one that matters.

    The first is arithmetic: the gold entries have to be present at every size, and a
    query set whose answers outnumber the smallest corpus cannot be run at all. An
    AMBIGUOUS row records its whole near-duplicate cluster as defensible, and the pack's
    wide clusters hold dozens of members each, so the gold set grows faster than the
    query count.

    The second is that the experiment is about the DISTRACTOR ratio. If the gold entries
    were most of the 1,000-row corpus, the small condition would not be a small corpus --
    it would be a corpus that is mostly answers, and the comparison against 100,000 would
    be measuring two different tasks. `budget` caps the gold at a fraction of the smallest
    size so that every condition is a genuine needle-in-haystack, differing only in how
    much hay.

    THE ORDER IS SHUFFLED FIRST, and that is not tidiness. Taking a prefix of the truth
    rows takes the schemas in the order they were generated, so the first schema's queries
    fill the entire gold budget and the rest of the pack contributes almost nothing. The
    first run of this script did exactly that: the readable `flat-english` profile ate the
    budget and the reported P@1 was a measurement of the easiest profile in the pack
    wearing the label of the whole query set.

    THE UNANSWERABLE ROWS ARE THEN SUBSAMPLED to the ratio they hold in the pack, and that
    correction is the second thing this function got wrong. Keeping all of them was the
    obvious move -- they cost no gold budget, and they are the abstention fixture. It made
    them 69% of the query set against 20% in the pack, and auto-approve precision counts an
    approval on a row with no correct answer as wrong. So the precision curve collapsed,
    no threshold reached the target at any size, and the transfer table had nothing to
    report -- a result that was entirely an artefact of the sampling and looked like a
    finding about the matcher.

    Even corrected the sample is not uniform, and the printed class mixture says so: a row
    with many defensible terms costs more of the gold budget than a row with one, so
    AMBIGUOUS rows stay under-represented. The comparison ACROSS sizes is unaffected -- it
    is the same query set every time -- but the absolute P@1 here is not comparable to a
    run that scored every field.
    """
    cap = int(smallest * budget)
    order = list(truth)
    random.Random(0x5EED_5CA1).shuffle(order)

    answerable: list = []
    unanswerable: list = []
    gold: set[str] = set()
    for row in order:
        if not row.correct_ids:
            unanswerable.append(row)
            continue
        if len(gold | set(row.correct_ids)) > cap:
            continue
        gold |= set(row.correct_ids)
        answerable.append(row)

    n_answerable_all = sum(1 for r in truth if r.correct_ids)
    ratio = (len(truth) - n_answerable_all) / n_answerable_all if n_answerable_all else 0.0
    return answerable + unanswerable[: round(len(answerable) * ratio)]


def _fit(sweep: list[dict[str, float]], target: float) -> dict[str, float] | None:
    """The lowest threshold whose auto-approve precision reaches `target`.

    Lowest rather than highest: a calibration picks the cheapest threshold that clears the
    precision bar, because every step above it costs coverage. Picking the highest would
    describe a different and easier decision.
    """
    for row in sweep:
        if row["n_auto_approved"] >= 20 and row["precision"] >= target:
            return row
    return None


def _at(sweep: list[dict[str, float]], threshold: float) -> dict[str, float]:
    return min(sweep, key=lambda r: abs(r["threshold"] - threshold))


def _precision_vectors(
    obs_by_key: dict, keys: list[str], threshold: float, min_gap: float
) -> tuple[list[float], list[float]]:
    """Per-query (auto-approved, auto-approved AND right) over `keys`, in `keys` order.

    An auto-approval on a row where nothing is correct counts as auto-approved and not
    right, which is what it is: there is no term to have found, and the field inherits a
    definition anyway.
    """
    flag: list[float] = []
    correct: list[float] = []
    for key in keys:
        o = obs_by_key[key]
        approved = o.conf1 >= threshold and o.gap >= min_gap
        flag.append(1.0 if approved else 0.0)
        correct.append(1.0 if approved and o.answerable and o.hit else 0.0)
    return flag, correct


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    ap.add_argument("--schemas", nargs="+", default=list(DEFAULT_SCHEMAS))
    ap.add_argument("--schema-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument(
        "--gold-budget",
        type=float,
        default=0.25,
        help="cap the gold entries at this fraction of the SMALLEST corpus, so every "
        "size is a genuine needle-in-haystack and only the amount of hay differs",
    )
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    sizes = sorted(args.sizes)
    config = MatchingConfig()
    spec = PackSpec(
        rows=max(sizes), seed=args.seed, schema_scale=args.schema_scale, feedback_events=100
    )
    print(
        f"\nGenerating pack at the largest size: {spec.rows} rows, "
        f"{spec.subjects} subject words, seed {spec.seed}"
    )
    pack = SyntheticPack.generate(spec)

    schemas = [pack.schema(name) for name in args.schemas]
    truth_all = [t for s in schemas for t in s.truth]
    truth = _trim_queries(truth_all, sizes[0], args.gold_budget)
    by_schema = {s.name: [t for t in truth if t.schema == s.name] for s in schemas}
    gold_ids = {i for t in truth for i in t.correct_ids}
    mixture: dict[str, int] = {}
    for row in truth:
        mixture[row.truth_class.value] = mixture.get(row.truth_class.value, 0) + 1
    per_schema = {s.name: len(by_schema[s.name]) for s in schemas}
    print(
        f"Query set: {len(truth)} of {len(truth_all)} fields, "
        f"{len(gold_ids)} distinct gold entries held present at every size "
        f"({len(gold_ids) / sizes[0]:.0%} of the smallest corpus)"
    )
    print(f"           by schema {per_schema}")
    print(f"           by class  {mixture}")
    print("           (a sample, not the whole pack: a row with several defensible terms")
    print("            costs more of the gold budget, so AMBIGUOUS is under-represented.")
    print("            The comparison across sizes is unaffected; the absolute P@1 is not")
    print("            comparable to a run that scored every field.)")

    per_size: dict[int, dict] = {}
    for size in sizes:
        rows = _subset(pack, gold_ids, size)
        matcher = build_matcher(config)
        report, index_s = index_glossary(matcher, rows, admit=True)
        print(f"\n[{size:>7}] indexed {report.entries} approved entries in {index_s:.1f}s")

        obs: list[FieldObs] = []
        for schema in schemas:
            run = run_schema(matcher, schema.flattened, index_s)
            obs.extend(observations(by_schema[schema.name], run))
            print(
                f"          {schema.name:<18} {len(schema.flattened):>5} fields "
                f"in {run.match_seconds:.1f}s"
            )

        answerable = [o for o in obs if o.answerable]
        conf = sorted(o.conf1 for o in obs)
        sweep = threshold_sweep(obs, config.min_confidence_gap, GRID)
        per_size[size] = {
            "requested_rows": size,
            "indexed_entries": report.entries,
            "rows_read": report.rows_read,
            "fields": len(obs),
            "answerable": len(answerable),
            "p_at_1": sum(o.hit for o in answerable) / len(answerable) if answerable else 0.0,
            "confidence_min": conf[0] if conf else 0.0,
            "confidence_median": conf[len(conf) // 2] if conf else 0.0,
            "confidence_max": conf[-1] if conf else 0.0,
            "sweep": sweep,
            "at_shipped_default": _at(sweep, config.auto_approve_threshold),
            "fitted": _fit(sweep, args.target_precision),
            # Kept for the paired precision comparison below, which needs the same
            # queries at two sizes rather than two summary ratios.
            "obs_by_key": {o.key: o for o in obs},
        }
        s = per_size[size]
        print(
            f"          P@1 {s['p_at_1']:.4f}   confidence min/median/max "
            f"{s['confidence_min']:.4f}/{s['confidence_median']:.4f}/"
            f"{s['confidence_max']:.4f}"
        )

    print("\n" + "=" * 74)
    print("E10  SCALE")
    print("=" * 74)
    print("  entries    P@1     conf min   conf median   at the shipped 0.87")
    print("                                              coverage  precision")
    for size in sizes:
        s = per_size[size]
        d = s["at_shipped_default"]
        print(
            f"  {s['indexed_entries']:>7}   {s['p_at_1']:.4f}   {s['confidence_min']:.4f}     "
            f"{s['confidence_median']:.4f}       {d['coverage']:.4f}    {d['precision']:.4f}"
        )

    smallest = sizes[0]
    fitted = per_size[smallest]["fitted"]
    print()
    if fitted is None:
        print(
            f"  No threshold reaches {args.target_precision:.0%} auto-approve precision at "
            f"{smallest} entries,"
        )
        print("  so there is nothing to transfer. That is itself the answer: a precision")
        print("  target this corpus cannot meet at its EASIEST size will not be met by")
        print("  raising the bar at a larger one.")
    else:
        t = fitted["threshold"]
        print(f"  THRESHOLD TRANSFER. Fitted at {smallest} entries, the lowest threshold")
        print(
            f"  reaching {args.target_precision:.0%} precision is {t:.2f} "
            f"(coverage {fitted['coverage']:.4f})."
        )
        print("  Applying that same number at every size:")
        print()
        print("  entries    coverage   precision   precision vs the target")
        for size in sizes:
            row = _at(per_size[size]["sweep"], t)
            delta = row["precision"] - args.target_precision
            print(
                f"  {per_size[size]['indexed_entries']:>7}    {row['coverage']:.4f}     "
                f"{row['precision']:.4f}      {delta:+.4f}"
            )
        largest = _at(per_size[sizes[-1]]["sweep"], t)
        drop = fitted["precision"] - largest["precision"]
        print()
        if drop > 0.02:
            print(f"  The threshold does NOT transfer: {drop:.4f} of precision is lost")
            print(f"  between {smallest} and {sizes[-1]} entries at a fixed number. A")
            print("  threshold is a statement about a score distribution, and the")
            print("  distribution moves with the size of the candidate pool -- which is")
            print("  the argument for calibrating on the corpus a deployment will actually")
            print("  hold, and against inheriting a number measured on a few hundred rows.")
        else:
            print(f"  The threshold transfers on this corpus: precision moves {drop:+.4f}")
            print(f"  between {smallest} and {sizes[-1]} entries. That is a real result and")
            print("  it is corpus-specific -- it does not license inheriting a threshold")
            print("  measured on a different corpus, only this one.")

    # -- the paired reading, which does not depend on a target being reachable ---------
    #
    # `precision` above is a ratio over however many fields cleared the bar, and at a
    # small corpus that can be a couple of dozen -- an estimate thin enough to move by
    # several points on one field. So the transfer question is also asked the way this
    # repository asks every other before/after question: paired over the SAME queries,
    # with a bootstrap interval that resamples queries and recomputes the whole ratio, so
    # "precision rose because coverage fell" cannot masquerade as an improvement.
    smallest_obs = per_size[smallest]["obs_by_key"]
    threshold = fitted["threshold"] if fitted else config.auto_approve_threshold
    print()
    print(f"  PAIRED, at threshold {threshold:.2f}, every larger size against {smallest} entries")
    print("  entries   auto-approved  coverage  precision   change vs the smallest")
    paired_rows: list[dict[str, object]] = []
    for size in sizes:
        obs_by_key = per_size[size]["obs_by_key"]
        keys = sorted(set(smallest_obs) & set(obs_by_key))
        fb, cb = _precision_vectors(smallest_obs, keys, threshold, config.min_confidence_gap)
        fa, ca = _precision_vectors(obs_by_key, keys, threshold, config.min_confidence_gap)
        point, lo, hi = paired_precision_ci(fb, cb, fa, ca)
        n_flagged = int(sum(fa))
        precision = (sum(ca) / n_flagged) if n_flagged else float("nan")
        coverage = n_flagged / len(keys) if keys else 0.0
        row = {
            "entries": per_size[size]["indexed_entries"],
            "n_auto_approved": n_flagged,
            "coverage": coverage,
            "precision": precision,
            "precision_change_vs_smallest": point,
            "ci_low": lo,
            "ci_high": hi,
        }
        paired_rows.append(row)
        change = (
            "  -- (this is the baseline)"
            if size == smallest
            else (f"  {point:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        )
        print(f"  {row['entries']:>7}   {n_flagged:>13}  {coverage:.4f}    {precision:.4f}{change}")
    print()
    print("  Coverage is the reading to trust when the approval counts are small: it is")
    print("  estimated over every query, not over the handful that cleared the bar, and a")
    print("  fixed threshold admitting a steadily smaller share of fields as the corpus")
    print("  grows is the same statement as 'the threshold does not transfer', made from")
    print("  the side where the sample size is not the problem.")

    print()
    print("  Confidence minimum by size is worth reading on its own. The structural floor")
    print(
        f"  is {config.minimum_achievable_confidence:.4f} at every size, because it is derived from the weights"
    )
    print("  and not from the data. What the corpus actually reaches is not, and a")
    print("  'low confidence' threshold chosen just above the derived floor selects")
    print("  nothing at any of these sizes.")

    artifact = {
        "experiment": "exp_synthetic_scale (E10)",
        "provenance": provenance(spec.seed, config.__dict__.copy()),
        "pack": pack.manifest(),
        "schemas": args.schemas,
        "query_fields": len(truth),
        "query_fields_before_trim": len(truth_all),
        "gold_budget": args.gold_budget,
        "query_class_mixture": mixture,
        "query_fields_per_schema": per_schema,
        "gold_entries": len(gold_ids),
        "target_precision": args.target_precision,
        "sizes": {
            str(k): {kk: vv for kk, vv in v.items() if kk != "obs_by_key"}
            for k, v in per_size.items()
        },
        "paired_at_threshold": {"threshold": threshold, "rows": paired_rows},
        "fitted_at_smallest": fitted,
        "structural_floor": config.minimum_achievable_confidence,
    }
    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"exp_synthetic_scale_{'_'.join(str(s) for s in sizes)}.json"
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nSaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
