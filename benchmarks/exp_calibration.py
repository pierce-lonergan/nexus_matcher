"""
benchmarks.exp_calibration | Layer: BENCHMARK
Calibrate the auto-approve threshold against a target precision.

The shipped `auto_approve_threshold = 0.75` is an arbitrary constant. It is not tied to
any measured precision, so its behaviour drifts whenever retrieval quality changes --
improving the retriever pushed MORE candidates over the fixed bar and DROPPED
auto-approve precision from 0.909 to 0.825, which is exactly backwards from what a
person would expect after an upgrade.

This script produces the precision/coverage curve so the threshold can be chosen to hit
an explicit operating point ("auto-approve at >=95% precision") instead of a magic
number, and reports the coverage you buy at each precision target.

Definitions
-----------
coverage  : fraction of fields auto-approved (rank-1 confidence >= threshold and the
            margin over the runner-up >= min_confidence_gap)
precision : of those auto-approved, the fraction whose top match is the gold entry
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import RESULTS, Dataset
from eval_pipeline import build_matcher, to_data_type

from nexus_matcher.domain.models.entities import SchemaField


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="combined")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--min-gap", type=float, default=0.10)
    ap.add_argument("--targets", nargs="*", type=float, default=[0.99, 0.95, 0.90, 0.85, 0.80])
    ap.add_argument("--aliases", type=int, default=0)
    args = ap.parse_args()

    prefix = "Represent this sentence for searching relevant passages: "
    ds = Dataset.load(args.benchmark)
    matcher, _ = build_matcher(ds, True, prefix, args.model, args.aliases)

    fields = [
        SchemaField(
            name=q.field_name,
            data_type=to_data_type(q.data_type),
            full_path=q.field_path,
            parent_path=q.parent_path,
        )
        for q in ds.queries
    ]
    results = matcher._match_fields(fields)

    # (confidence, margin over runner-up, is_correct) for each field's top match.
    rows: list[tuple[float, float, bool]] = []
    for q, f in zip(ds.queries, fields, strict=False):
        matches = results.get(f.full_path, ())
        if not matches:
            continue
        top = matches[0]
        margin = (
            top.final_confidence - matches[1].final_confidence if len(matches) > 1 else float("inf")
        )
        rows.append((top.final_confidence, margin, top.dictionary_entry.id == q.gold_id))

    n = len(rows)
    print(f"\nCalibration on '{ds.name}' ({n} fields, min_gap={args.min_gap})\n")
    print(f"  {'threshold':>9s} {'coverage':>9s} {'auto-P':>8s} {'n_auto':>7s} {'n_correct':>10s}")
    print(f"  {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 7} {'-' * 10}")

    curve = []
    for t in [i / 100 for i in range(50, 100)]:
        auto = [r for r in rows if r[0] >= t and r[1] >= args.min_gap]
        if not auto:
            continue
        correct = sum(1 for r in auto if r[2])
        prec = correct / len(auto)
        cov = len(auto) / n
        curve.append({"threshold": t, "coverage": cov, "precision": prec, "n_auto": len(auto)})
        if round(t * 100) % 5 == 0:
            print(f"  {t:9.2f} {cov:9.3f} {prec:8.3f} {len(auto):7d} {correct:10d}")

    print("\n  Lowest threshold meeting each precision target (max coverage):")
    recommended = {}
    for target in sorted(args.targets, reverse=True):
        ok = [c for c in curve if c["precision"] >= target]
        if ok:
            best = max(ok, key=lambda c: c["coverage"])
            recommended[str(target)] = best
            print(
                f"    P>={target:.2f} -> threshold {best['threshold']:.2f}  "
                f"coverage {best['coverage']:.1%}  actual precision {best['precision']:.3f}  "
                f"({best['n_auto']} fields auto-approved)"
            )
        else:
            print(f"    P>={target:.2f} -> NOT ACHIEVABLE at any threshold on this benchmark")

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"exp_calibration_{ds.name}.json"
    p.write_text(
        json.dumps(
            {
                "benchmark": ds.name,
                "min_gap": args.min_gap,
                "n": n,
                "curve": curve,
                "recommended": recommended,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
