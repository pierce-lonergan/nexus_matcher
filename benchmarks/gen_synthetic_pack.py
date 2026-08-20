"""
benchmarks.gen_synthetic_pack | Layer: BENCHMARK
Generate the synthetic mock-data pack, and hold the generator to its own claims.

The pack is written under `data/`, which this repository gitignores, for the same reason
`data/benchmarks/` is: it is reproducible from source, it is large, and a generated corpus
committed to git becomes a second answer to "what does the generator produce" that nobody
regenerates.

Usage
-----
    python benchmarks/gen_synthetic_pack.py --rows 10000 --verify
    python benchmarks/gen_synthetic_pack.py --rows 100000 --out data/synthetic/large
    python benchmarks/gen_synthetic_pack.py --rows 5000 --difficulty 2.0 --verify --no-write

Exit codes
----------
    0  written, and every verification passed
    1  a verification failed (the pack is still written; the findings say what is wrong)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic.pack import DEFAULT_SEED, PackSpec, SyntheticPack
from synthetic.verify import verify

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "synthetic" / "pack"


def _render_manifest(manifest: dict) -> str:
    g = manifest["glossary"]
    a = manifest["abbreviations"]
    t = manifest["truth"]
    f = manifest["feedback"]
    lines = [
        "",
        "  Glossary",
        "  --------",
        f"    rows                        {g['rows']}",
        f"    approved                    {g['approved']}  "
        f"(non-approved {g['non_approved_share']:.1%})",
        f"    in a near-duplicate cluster {g['in_near_duplicate_cluster_share']:.1%} "
        f"across {g['clusters']} clusters, widest {g['widest_cluster']}",
        f"    definition echoes name      {g['definition_echoes_name_share']:.1%}",
        f"    name tokens                 median {g['name_tokens_median']}, "
        f"max {g['name_tokens_max']}",
        f"    domains / class words       {g['distinct_domains']} / {g['distinct_class_words']}",
        "",
        "  Abbreviation catalog",
        "  --------------------",
        f"    entries                     {a['entries']}",
        f"    ambiguous short forms       {a['ambiguous_shorts']}",
        f"    multi-word rules            {a['multi_word_rules']}",
        f"    stopword collisions         {a['stopword_collisions']}",
        f"    never expand                {a['never_expand']}",
        f"    delta (v{a['delta_version']})               {a['delta_entries']} changed mappings",
        "",
        "  Schemas",
        "  -------",
    ]
    for name, s in manifest["schemas"].items():
        lines.append(
            f"    {name:<18} {s['fields']:>6} fields  "
            f"{s['with_doc']:>6} with doc  "
            f"{s['distinct_leaf_names']:>6} distinct leaves  "
            f"max repetition {s['max_leaf_repetition']}"
        )
    lines += [
        "",
        "  Ground truth",
        "  ------------",
        f"    EXACT {t['EXACT']}   AMBIGUOUS {t['AMBIGUOUS']}   "
        f"NO_MATCH {t['NO_MATCH']}   TRAP {t['TRAP']}",
        "",
        "  Feedback trace",
        "  --------------",
        f"    events {f['events']}   approved {f['approved']}   rejected {f['rejected']}   "
        f"override {f['manual_override']}",
        f"    overrides the shipped wire format cannot distinguish from a rerank: "
        f"{f['overrides_indistinguishable_from_reranks']}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=10_000, help="glossary rows")
    ap.add_argument(
        "--difficulty",
        type=float,
        default=1.0,
        help="1.0 is the specified shape; above it, near-duplicates, non-approved rows "
        "and tautological definitions all rise together",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--schema-scale",
        type=float,
        default=1.0,
        help="multiplier on every schema profile's leaf count",
    )
    ap.add_argument("--feedback-events", type=int, default=5_000)
    ap.add_argument(
        "--paraphrase-strength",
        type=float,
        default=0.6,
        help="how far a generated column drifts from its term. 0 makes the column a copy "
        "of the term, which measures string identity",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-write", action="store_true", help="generate and verify only")
    ap.add_argument("--verify", action="store_true", help="check the advertised properties")
    ap.add_argument(
        "--skip-determinism",
        action="store_true",
        help="skip the two-run checksum comparison (it regenerates a small pack twice)",
    )
    args = ap.parse_args()

    spec = PackSpec(
        rows=args.rows,
        difficulty=args.difficulty,
        seed=args.seed,
        schema_scale=args.schema_scale,
        feedback_events=args.feedback_events,
        paraphrase_strength=args.paraphrase_strength,
    )
    print(
        f"\nGenerating synthetic pack: rows={spec.rows} difficulty={spec.difficulty} "
        f"seed={spec.seed} vocabulary={spec.subjects} subjects"
    )
    pack = SyntheticPack.generate(spec)

    if args.no_write:
        manifest = pack.manifest()
    else:
        manifest = pack.write(args.out)
        print(f"Wrote -> {args.out}")

    print(_render_manifest(manifest))

    if not args.verify:
        return 0

    findings, report = verify(pack, check_determinism=not args.skip_determinism)
    print("  Verification")
    print("  ------------")
    o = report["overlap"]
    print(
        f"    query/gold token overlap    mean {o['mean']:.3f}  median {o['median']:.3f}  "
        f"identical {o['identical_share']:.1%} of {o['n']}"
    )
    e = report["english_collisions"]
    if e.get("available"):
        print(
            f"    manufactured subjects that spell an English word: "
            f"{e['collide_with_english']}/{e['subjects']} ({e['share']:.1%})"
        )
    else:
        print("    English-collision check: encoder vocabulary not found, not run")
    d = report["delimiter_trap"]
    print(
        f"    delimiter trap              {d['multi_sample_values']} rows with several "
        f"comma-separated sample values, {d['multi_enum_values']} with several "
        f"semicolon-separated enum values"
    )
    print(f"    widest contested leaf       {report['widest_contested_leaf']} distinct answers")
    if "determinism" in report:
        print(f"    determinism                 {report['determinism']}")

    if findings:
        print(f"\n  {len(findings)} finding(s):")
        for finding in findings:
            print(finding.render())
        if not args.no_write:
            (args.out / "verify.json").write_text(
                json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                encoding="ascii",
                newline="\n",
            )
        return 1

    print("\n  Every advertised property holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
