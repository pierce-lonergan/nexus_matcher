"""
benchmarks.exp_repeated_leaf | Layer: BENCHMARK
E7 -- what a cache keyed on the leaf column name costs, measured rather than argued.

The fixture
-----------
`nested-repeated` is built from the glossary's deliberately WIDE near-duplicate clusters:
one term name governed separately in N domains. The schema puts that one leaf name under N
different parents, and the correct answer for each occurrence is the cluster member owned
by that parent's domain. So the leaf name carries no information and the parent carries all
of it -- which is the real shape of a flattened production schema, where one column name
appears dozens of times under different records and means a different governed element each
time.

The two conditions
------------------
    KEYED ON THE FIELD   what this library does today: every column is matched, and the
                         result dict is keyed by the caller's own flattened name.
    KEYED ON THE LEAF    the plausible optimisation: match each DISTINCT leaf name once and
                         reuse the answer for every occurrence. Nothing in this library
                         does that; the control is computed here from the real run's own
                         answers, so the comparison is exact and paired.

What makes it worth measuring rather than reasoning about is the failure MODE. Collapsing
does not raise, does not drop a field, and does not lower a confidence. The conservation
law still holds -- every column sent comes back exactly once. The wrong answers arrive at
the same confidence as the right ones, because they ARE the right one, for a different
column. The only symptom is a count nobody has reason to check.

Why the query text is also reported
-----------------------------------
The answer to "may these two columns share a cache entry?" is decided by what the pipeline
turns them into, not by what they are called. `ContextEnricher` injects the parent path
into the query -- worth +19.3 points of P@1 on the public benchmark and the single largest
factor in this pipeline -- so two occurrences of one leaf under different parents produce
DIFFERENT query text. An embedding cache keyed on that text is safe. One keyed on the leaf
name is not, and the count of distinct query texts per leaf name is the evidence.

Usage
-----
    python benchmarks/exp_repeated_leaf.py --rows 10000 --save
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimization_ledger import paired_compare_metric, provenance
from synthetic.pack import DEFAULT_SEED, PackSpec, SyntheticPack
from synthetic_harness import RESULTS, build_matcher, index_glossary, parse_fields, run_schema

from nexus_matcher.application.use_cases.match_schema import MatchingConfig
from nexus_matcher.domain.services.context_enricher import ContextEnricher


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
    schema = pack.schema("nested-repeated")

    matcher = build_matcher(config)
    report, index_s = index_glossary(matcher, pack.glossary_dicts())
    print(f"Indexed {report.entries} approved entries in {index_s:.1f}s")

    run = run_schema(matcher, schema.flattened, index_s)
    results = run.results
    print(f"Matched {len(schema.flattened)} fields in {run.match_seconds:.1f}s")

    leaf_of = {r["flattenedName"]: r["leafName"] for r in schema.flattened}
    truth_by_key = {t.flattened_name: t for t in schema.truth}

    # -- the conservation law, first, because everything else assumes it ------
    keys = list(results)
    conservation_ok = len(keys) == len(schema.flattened) and len(set(keys)) == len(keys)

    # -- condition A: keyed on the field ------------------------------------
    per_field: dict[str, int] = {}
    top_id: dict[str, str] = {}
    conf: dict[str, float] = {}
    for key, matches in results.items():
        truth = truth_by_key.get(key)
        if truth is None or not matches:
            continue
        top_id[key] = matches[0].dictionary_entry.id
        conf[key] = float(matches[0].final_confidence)
        per_field[key] = 1 if matches[0].dictionary_entry.id in set(truth.correct_ids) else 0

    # -- condition B: keyed on the leaf name --------------------------------
    # The first occurrence of each leaf name is the one that would have populated the
    # cache; every later occurrence reads its answer instead of computing its own.
    first_for_leaf: dict[str, str] = {}
    for key in results:
        leaf = leaf_of.get(key, key)
        first_for_leaf.setdefault(leaf, key)

    per_leaf: dict[str, int] = {}
    collapsed_answer: dict[str, str] = {}
    for key in per_field:
        donor = first_for_leaf[leaf_of.get(key, key)]
        answer = top_id.get(donor, "")
        collapsed_answer[key] = answer
        per_leaf[key] = 1 if answer in set(truth_by_key[key].correct_ids) else 0

    occurrences: dict[str, int] = {}
    for key in results:
        occurrences[leaf_of.get(key, key)] = occurrences.get(leaf_of.get(key, key), 0) + 1
    max_repetition = max(occurrences.values(), default=0)

    shared = sorted(set(per_field) & set(per_leaf))
    field_hits = [float(per_field[k]) for k in shared]
    leaf_hits = [float(per_leaf[k]) for k in shared]
    p1_field = sum(field_hits) / len(shared) if shared else 0.0
    p1_leaf = sum(leaf_hits) / len(shared) if shared else 0.0

    # The subset the collapse can actually touch. A leaf that appears once is answered
    # identically either way, and including those dilutes the effect by however many
    # unrepeated columns the schema happens to carry -- which is a property of the fixture,
    # not of the defect. Both numbers are reported: the contested one is the size of the
    # error, the whole-schema one is what a run-level metric would show.
    contested = [k for k in shared if occurrences.get(leaf_of.get(k, k), 1) > 1]
    c_field = [float(per_field[k]) for k in contested]
    c_leaf = [float(per_leaf[k]) for k in contested]
    p1_field_contested = sum(c_field) / len(contested) if contested else 0.0
    p1_leaf_contested = sum(c_leaf) / len(contested) if contested else 0.0
    stat_contested = (
        paired_compare_metric("p_at_1_contested", c_field, c_leaf) if contested else None
    )

    changed = [k for k in shared if collapsed_answer[k] != top_id.get(k, "")]
    broke = [k for k in changed if per_field[k] == 1 and per_leaf[k] == 0]
    # Every one of these carries the DONOR's confidence, which is a real confidence for a
    # real match -- of another column. That is what "confidently wrong" means here.
    donor_conf = [conf[first_for_leaf[leaf_of.get(k, k)]] for k in broke]
    auto_threshold = config.auto_approve_threshold

    stat = paired_compare_metric("p_at_1", field_hits, leaf_hits) if shared else None

    # -- the query text ------------------------------------------------------
    enricher = ContextEnricher()
    fields = parse_fields(schema.flattened)
    texts_per_leaf: dict[str, set[str]] = {}
    for f in fields:
        leaf = f.source_metadata.get("leafName") or f.name
        texts_per_leaf.setdefault(leaf, set()).add(enricher.enrich(f))
    counts_per_leaf = {leaf: len(v) for leaf, v in texts_per_leaf.items()}
    widest_leaf = max(counts_per_leaf, key=lambda k: counts_per_leaf[k], default="")

    print("\n" + "=" * 74)
    print("E7  CACHE-KEY COMPOSITION ON A REPEATED LEAF NAME")
    print("=" * 74)
    print(f"  fields                                       {len(schema.flattened)}")
    print(f"  distinct leaf names                          {len(occurrences)}")
    print(f"  most repeated leaf name appears              {max_repetition} times")
    print(
        f"  conservation law (one result per field sent) {'holds' if conservation_ok else 'BROKEN'}"
    )
    print(f"  distinct query texts for that leaf name      {counts_per_leaf.get(widest_leaf, 0)}")
    print()
    print(
        f"  ON THE {len(contested)} FIELDS WHOSE LEAF NAME IS REPEATED "
        f"(the ones a collapse touches)"
    )
    print(f"    P@1 keyed on the FIELD                     {p1_field_contested:.4f}")
    print(f"    P@1 keyed on the LEAF NAME                 {p1_leaf_contested:.4f}")
    print(
        f"    delta                                      "
        f"{p1_leaf_contested - p1_field_contested:+.4f}"
    )
    if stat_contested is not None:
        print(f"    paired test                                {stat_contested.test}")
        print(
            f"    95% CI                                     "
            f"[{stat_contested.ci_low:+.4f}, {stat_contested.ci_high:+.4f}]  "
            f"p = {stat_contested.p_value:.4g}"
        )
    print()
    print(f"  OVER THE WHOLE SCHEMA ({len(shared)} fields), which is what a run-level")
    print("  metric would report -- diluted by every leaf that appears only once")
    print(f"    P@1 keyed on the FIELD                     {p1_field:.4f}")
    print(f"    P@1 keyed on the LEAF NAME                 {p1_leaf:.4f}")
    print(f"    delta                                      {p1_leaf - p1_field:+.4f}")
    if stat is not None:
        print(f"    paired test                                {stat.test}")
        print(
            f"    95% CI                                     "
            f"[{stat.ci_low:+.4f}, {stat.ci_high:+.4f}]  p = {stat.p_value:.4g}"
        )
    print()
    print(f"  answers CHANGED by collapsing                {len(changed)} of {len(shared)}")
    print(f"  answers BROKEN by collapsing                 {len(broke)}")
    if donor_conf:
        donor_conf_sorted = sorted(donor_conf)
        above = sum(1 for c in donor_conf if c >= auto_threshold)
        print(
            f"  those wrong answers arrive at confidence      "
            f"min {donor_conf_sorted[0]:.4f}  "
            f"median {donor_conf_sorted[len(donor_conf_sorted) // 2]:.4f}  "
            f"max {donor_conf_sorted[-1]:.4f}"
        )
        print(
            f"  of them, at or above auto_approve_threshold  {above} "
            f"({above / len(donor_conf):.1%})"
        )
    print()
    print("  Nothing above is an error, a miss or a dropped field. The collapsed run")
    print("  returns exactly as many results, with confidences drawn from real matches,")
    print("  and it is wrong on the fields it changed.")

    # How much of this fixture is decidable at all from the query side. Within a wide
    # cluster every member has the SAME name and differs only by the domain that governs
    # it, so a matcher with no view of the domain is choosing at random among the
    # members. The gap between that and the field-keyed P@1 is what the parent path
    # recovers on its own; the gap between the field-keyed P@1 and 1.0 is the size of
    # what a request-level domain prior would be worth, and is not measurable here.
    widths = [len(c) for c in pack.glossary.wide_clusters()]
    chance = (sum(1 / w for w in widths) / len(widths)) if widths else 0.0
    print()
    print(f"  For scale: choosing at random inside a wide cluster scores {chance:.4f}.")
    print(f"  The parent path alone gets {p1_field:.4f}. The distance from there to 1.0 is")
    print("  the disambiguation only the CALLER holds -- which schema namespace this")
    print("  column came from -- and no experiment in this pack can close it from the")
    print("  query side. That is E3, and it needs a request-level domain prior to run.")

    artifact = {
        "experiment": "exp_repeated_leaf (E7)",
        "provenance": provenance(spec.seed, config.__dict__.copy()),
        "pack": pack.manifest(),
        "schema": schema.name,
        "fields": len(schema.flattened),
        "distinct_leaf_names": len(occurrences),
        "max_leaf_repetition": max_repetition,
        "conservation_law_holds": conservation_ok,
        "distinct_query_texts_for_widest_leaf": counts_per_leaf.get(widest_leaf, 0),
        "p_at_1_keyed_on_field": p1_field,
        "p_at_1_keyed_on_leaf_name": p1_leaf,
        "delta": p1_leaf - p1_field,
        "contested_fields": len(contested),
        "p_at_1_keyed_on_field_contested": p1_field_contested,
        "p_at_1_keyed_on_leaf_name_contested": p1_leaf_contested,
        "delta_contested": p1_leaf_contested - p1_field_contested,
        "paired_contested": None
        if stat_contested is None
        else {
            "test": stat_contested.test,
            "delta": stat_contested.delta,
            "ci_low": stat_contested.ci_low,
            "ci_high": stat_contested.ci_high,
            "p_value": stat_contested.p_value,
            "n_pairs": stat_contested.n_pairs,
            "n_discordant": stat_contested.n_discordant,
        },
        "answers_changed": len(changed),
        "answers_broken": len(broke),
        "broken_confidence": {
            "n": len(donor_conf),
            "min": min(donor_conf) if donor_conf else 0.0,
            "max": max(donor_conf) if donor_conf else 0.0,
            "at_or_above_auto_approve_threshold": sum(1 for c in donor_conf if c >= auto_threshold),
        },
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
    }
    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"exp_repeated_leaf_synthetic_{spec.rows}.json"
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nSaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
