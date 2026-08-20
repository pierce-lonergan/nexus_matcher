"""
benchmarks.exp_row_admission | Layer: BENCHMARK
E8 -- what indexing a glossary's drafts and retired terms costs.

The question
-----------
A glossary export carries every term, not just the approved ones. `load_entries(admit=...)`
is the filter that keeps the rest out of the index. Without it a draft competes as a real
term, because it IS a real term -- same shape, same class word, same domain, often the
same words as the approved term it was drafted to replace. No threshold repairs that: a
draft that outranks its approved twin is not a low-confidence match, it is a confident
match on the wrong row.

The pack ships 12% non-approved rows for exactly this. The two conditions are the same
corpus, the same queries and the same seed, differing only in whether `admit` is passed --
which is what makes the comparison legitimate. The paired test is exact McNemar over the
answerable fields both runs scored.

The second half: the delimiter trap
-----------------------------------
The same loader call carries the pack's other deliberate trap. `sample_values` in the
glossary is comma-separated and `enum_values` in the same file is semicolon-separated, and
reading either with the other's separator produces one value per row containing every
element -- which indexes, and matches, and is silently wrong.

`delimiter_strict=True` refuses that, and this script exercises the refusal in both
directions: the declared-correctly load, the swapped load under strict (must raise), and
the swapped load with the check disabled (must succeed, and produce the giant values that
show why the check exists). A gate is not live until it has been seen to fail.

Usage
-----
    python benchmarks/exp_row_admission.py --rows 10000 --save
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimization_ledger import paired_compare_metric, provenance
from synthetic.pack import DEFAULT_SEED, PackSpec, SyntheticPack
from synthetic_harness import (
    RESULTS,
    build_matcher,
    index_glossary,
    observations,
    run_schema,
)

from nexus_matcher.application.ingest import LoadReport, load_entries
from nexus_matcher.application.use_cases.match_schema import MatchingConfig

# The schemas the comparison runs over. Every profile in the pack, so the result is not a
# property of one naming style: the point of admission is that it applies to the INDEX,
# and an index defect shows up on whatever is asked of it.
SCHEMAS = ("flat-english", "flat-contracted", "nested-deep", "no-doc", "mixed-production")


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _delimiter_trap(rows: list[dict[str, str]]) -> dict[str, object]:
    """Load the glossary three ways and report what each one does."""
    from synthetic.glossary import ADMIT_APPROVED, COLUMN_MAPPING

    correct = {"sample_values": ",", "enum_values": ";"}
    swapped = {"sample_values": ";", "enum_values": ","}
    out: dict[str, object] = {}

    report = LoadReport()
    entries = load_entries(
        rows,
        columns=dict(COLUMN_MAPPING),
        admit=dict(ADMIT_APPROVED),
        value_delimiters=correct,
        delimiter_strict=True,
        report=report,
    )
    multi = [e for e in entries if len(e.sample_values) > 1]
    out["declared_correctly"] = {
        "loaded": len(entries),
        "entries_with_several_sample_values": len(multi),
        "longest_single_value_chars": max(
            (len(v) for e in entries for v in e.sample_values), default=0
        ),
    }

    try:
        load_entries(
            rows,
            columns=dict(COLUMN_MAPPING),
            admit=dict(ADMIT_APPROVED),
            value_delimiters=swapped,
            delimiter_strict=True,
        )
    except ValueError as exc:
        out["swapped_under_strict"] = {"refused": True, "message": _first_line(str(exc))}
    else:
        out["swapped_under_strict"] = {
            "refused": False,
            "message": "THE GATE DID NOT FIRE -- a swapped separator loaded cleanly",
        }

    entries = load_entries(
        rows,
        columns=dict(COLUMN_MAPPING),
        admit=dict(ADMIT_APPROVED),
        value_delimiters=swapped,
        delimiter_strict=False,
    )
    out["swapped_with_the_check_off"] = {
        "loaded": len(entries),
        "entries_with_several_sample_values": sum(1 for e in entries if len(e.sample_values) > 1),
        "longest_single_value_chars": max(
            (len(v) for e in entries for v in e.sample_values), default=0
        ),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=10_000)
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
    rows = pack.glossary_dicts()
    non_approved = {r.id for r in pack.glossary.rows if not r.is_approved}

    conditions: dict[str, dict] = {}
    hits: dict[str, dict[str, int]] = {}
    for label, admit in (("filtered", True), ("unfiltered", False)):
        matcher = build_matcher(config)
        report, index_s = index_glossary(matcher, rows, admit=admit)
        print(
            f"\n[{label}] indexed {report.entries} entries in {index_s:.1f}s "
            f"(read {report.rows_read}, refused {report.refused})"
        )
        obs_all = []
        for name in SCHEMAS:
            schema = pack.schema(name)
            run = run_schema(matcher, schema.flattened, index_s)
            obs_all.extend(observations(schema.truth, run))
            print(f"   {name:<18} {len(schema.flattened):>5} fields in {run.match_seconds:.1f}s")
        answerable = [o for o in obs_all if o.answerable]
        polluted = sum(1 for o in obs_all if o.top_id in non_approved)
        auto = [o for o in obs_all if o.auto_approved]
        conditions[label] = {
            "indexed_entries": report.entries,
            "rows_read": report.rows_read,
            "refused": report.refused,
            "refused_by_column": dict(report.refused_by_column),
            "fields": len(obs_all),
            "p_at_1": sum(o.hit for o in answerable) / len(answerable) if answerable else 0.0,
            "rank1_is_a_non_approved_term": polluted,
            "rank1_non_approved_share": polluted / len(obs_all) if obs_all else 0.0,
            "auto_approve_coverage": len(auto) / len(obs_all) if obs_all else 0.0,
            "auto_approve_precision": (
                sum(o.hit for o in auto if o.answerable) / len(auto) if auto else 0.0
            ),
            "auto_approved_a_non_approved_term": sum(1 for o in auto if o.top_id in non_approved),
        }
        hits[label] = {o.key: o.hit for o in answerable}

    shared = sorted(set(hits["filtered"]) & set(hits["unfiltered"]))
    before = [float(hits["unfiltered"][k]) for k in shared]
    after = [float(hits["filtered"][k]) for k in shared]
    stat = paired_compare_metric("p_at_1", before, after) if shared else None

    f, u = conditions["filtered"], conditions["unfiltered"]
    print("\n" + "=" * 74)
    print("E8  ROW ADMISSION (the status filter)")
    print("=" * 74)
    print(f"  glossary rows                                {u['rows_read']}")
    print(f"  indexed, admit={{'status': {{'Approved'}}}}        {f['indexed_entries']}")
    print(f"  indexed, no admit                            {u['indexed_entries']}")
    print(
        f"  index pollution                              "
        f"{u['indexed_entries'] - f['indexed_entries']} non-approved terms"
    )
    print()
    print(f"  P@1 filtered                                 {f['p_at_1']:.4f}")
    print(f"  P@1 unfiltered                               {u['p_at_1']:.4f}")
    print(f"  delta (filtered minus unfiltered)            {f['p_at_1'] - u['p_at_1']:+.4f}")
    if stat is not None:
        print(f"  paired test                                  {stat.test}")
        print(
            f"  95% CI                                       "
            f"[{stat.ci_low:+.4f}, {stat.ci_high:+.4f}]  p = {stat.p_value:.4g}"
        )
    print()
    print(
        f"  rank 1 was a DRAFT or RETIRED term, unfiltered "
        f"{u['rank1_is_a_non_approved_term']} of {u['fields']} "
        f"({u['rank1_non_approved_share']:.2%})"
    )
    print(
        f"  ... and AUTO-APPROVED anyway                   {u['auto_approved_a_non_approved_term']}"
    )
    print(
        f"  auto-approve precision  filtered / unfiltered  "
        f"{f['auto_approve_precision']:.4f} / {u['auto_approve_precision']:.4f}"
    )
    print()
    print("  A term the glossary marks Draft is not a low-confidence match. It is a")
    print("  confident match on a row the business has not agreed to, and the field")
    print("  inherits its definition into whatever the consumer ships.")
    print()
    print("  READ THE TWO NUMBERS SEPARATELY, and do not expect them to agree.")
    print("  P@1 is not where admission pays, and it can move either way: removing rows")
    print("  changes the candidate set, so it changes the per-field min-max normalisation")
    print("  and the lexical arm's document statistics, and survivors reorder. A P@1 delta")
    print("  near zero -- or negative, with a confidence interval spanning it -- is")
    print("  therefore NOT evidence that admission is worthless, and 'fixing' it by")
    print("  dropping the filter would trade a governance property for retrieval noise.")
    print("  The number admission is for is the one above it: how often rank 1 was a term")
    print("  nobody approved. There is no threshold that makes that share acceptable,")
    print("  because the row is not wrong -- it is unratified, and only its status says so.")

    trap = _delimiter_trap(rows)
    print("\n" + "=" * 74)
    print("THE DELIMITER TRAP (same loader call)")
    print("=" * 74)
    ok = trap["declared_correctly"]
    sw = trap["swapped_under_strict"]
    off = trap["swapped_with_the_check_off"]
    print(
        f"  declared correctly   {ok['loaded']} entries, "
        f"{ok['entries_with_several_sample_values']} with several sample values, "
        f"longest value {ok['longest_single_value_chars']} chars"
    )
    print(f"  swapped, strict      refused={sw['refused']}")
    print(f"                       {sw['message']}")
    print(
        f"  swapped, check off   {off['loaded']} entries, "
        f"{off['entries_with_several_sample_values']} with several sample values, "
        f"longest value {off['longest_single_value_chars']} chars"
    )
    print()
    print("  The third row is the failure the check exists to stop: it loads, it indexes,")
    print("  it matches, and every multi-valued cell became one value containing all of")
    print("  them. Nothing anywhere reports a problem.")

    artifact = {
        "experiment": "exp_row_admission (E8) + the delimiter trap",
        "provenance": provenance(spec.seed, config.__dict__.copy()),
        "pack": pack.manifest(),
        "schemas": list(SCHEMAS),
        "conditions": conditions,
        "paired": None
        if stat is None
        else {
            "test": stat.test,
            "delta": stat.delta,
            "ci_low": stat.ci_low,
            "ci_high": stat.ci_high,
            "p_value": stat.p_value,
            "n_pairs": stat.n_pairs,
            "n_discordant": stat.n_discordant,
        },
        "delimiter_trap": trap,
    }
    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"exp_row_admission_synthetic_{spec.rows}.json"
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nSaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
