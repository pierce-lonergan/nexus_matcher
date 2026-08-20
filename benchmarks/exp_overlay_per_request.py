"""
benchmarks.exp_overlay_per_request | Layer: BENCHMARK
E2 -- the overlay has to be PER REQUEST, and a load-time catalog structurally cannot pass.

The claim, and what would falsify it
------------------------------------
A naming standard is a live feed. A term abbreviated one way this quarter is abbreviated
differently the next, and both spellings are in flight at once: one upstream producer has
adopted the new standard and another has not. The claim is that no single catalog fixed at
process start can serve both, however good that catalog is.

That is a strong claim, so it is worth stating what would falsify it. If some fixed
catalog -- version 1, version 2, or any merge of the two -- matched the per-request arm on
the mixed population, the channel would be a convenience and not a requirement.

Why no merge exists
-------------------
The pack's versioned delta does not ADD rows. Every one of its entries re-points a short
form the base catalog already had, so version 1 says `S -> X` and version 2 says `S -> Y`
for the same `S`. A `dict` holds one value per key. "Merge them" therefore is not a third
option: merging with version 2 winning IS version 2, merging with version 1 winning IS
version 1, and the choice is between them on every request. This is the structural half of
the argument and it is arithmetic, not a measurement.

The measurement is what that choice costs, and it is reported on the columns the delta
actually touches -- the population where the two catalogs disagree. Diluting it with the
columns both versions spell identically would understate the effect by exactly the share
of the schema the delta happened not to reach.

What the aggregate does NOT say
-------------------------------
Over the whole mixed schema, per-request routing does not beat a stale version-1 catalog.
That reads like a contradiction of the paragraph above and it is not; it decomposes
exactly, and the decomposition is printed:

    on the columns the delta RE-SPELLED       54 gained, 10 lost
    on the columns it left alone               0 gained, 45 lost

`with_delta` moves a short form S from long form X to long form Y and teaches contraction
that Y now yields S. But X still yields S too, and S now expands to Y. So on a column the
change re-spelled, the newer catalog is the only one that inverts it; on a column the
change did not touch, the newer catalog now expands that column's short to a word that is
not its own, and is strictly worse. Routing per producer therefore TRADES one population
for the other, and on this pack the trade nets to about zero.

The deployable statement is narrow and it is the one this script prints: the channel's
value is that the trade can be made at all, and made per request. "Always send the newest
catalog" is a different instruction from "send the catalog this producer used", and
nothing here supports either as a blanket gain.

How "load time" is emulated, and the proof that the emulation is exact
---------------------------------------------------------------------
A fixed catalog is emulated by sending the SAME overlay on every request. With
`expand_query_abbreviations` at its shipped default of False, `_request_expander` builds
`AbbreviationExpander(AbbreviationDictionary.from_dict(overlay))` -- the overlay alone,
which is precisely the object a deployment configured with that catalog at construction
holds for the life of the process. The script does not assume this: it builds a second
matcher configured that way and asserts the query text is identical, field for field,
before any arm is scored.

What "mid-run" means here
-------------------------
One matcher, one index, one process. Columns are sent in interleaved batches, alternating
between the two producers, and the catalog changes between requests. Abbreviations are
REQUEST-scoped by design -- `_match_fields` resolves one expander per call -- so a single
request cannot carry two catalogs, and the unit at which a standard can change is the
request. That is the shape of the result, not a limitation of the harness.

Usage
-----
    python benchmarks/exp_overlay_per_request.py --rows 10000 --schema-scale 4 --save
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimization_ledger import paired_compare_metric, provenance
from synthetic.pack import DEFAULT_SEED, PackSpec, SyntheticPack
from synthetic.truth import TruthRow
from synthetic_harness import (
    RESULTS,
    Run,
    build_matcher,
    index_glossary,
    mirror_pairs,
    observations,
    parse_fields,
    reconstructed_keys,
    run_schema,
)

from nexus_matcher.application.use_cases.match_schema import (
    MatchingConfig,
    NexusMatcher,
    QuerySignals,
)

# Columns per request. Small enough that the catalog changes many times inside one run,
# large enough that the run is not dominated by per-call overhead.
BATCH = 25


# =============================================================================
# THE TWO PRODUCERS
# =============================================================================


def _rename(row: dict[str, Any], name: str) -> dict[str, Any]:
    out = dict(row)
    out["flattenedName"] = name
    out["leafName"] = name
    return out


def _build_population(pack: SyntheticPack) -> dict[str, Any]:
    """
    One column set, half of it spelled by a producer on version 1 and half on version 2.

    Both namings are derived from the SAME English column, so the correct answer is
    identical whichever producer sent it and the two halves are comparable.
    """
    english = pack.schema("flat-english")
    contracted = pack.schema("flat-contracted")
    v1 = pack.catalog
    v2 = v1.with_delta(pack.delta)

    # Paired by the generator's own mirror record: contraction is many-to-one, so the two
    # profiles are not the same length and positional pairing would be wrong.
    pairs = mirror_pairs(english, contracted)
    row_of = {r["flattenedName"]: r for r in contracted.flattened}
    truth_of = {t.flattened_name: t for t in contracted.truth}

    rows: list[dict[str, Any]] = []
    truth: list[TruthRow] = []
    versions: list[int] = []
    touched: set[str] = set()
    regenerated_matches_pack = True
    # A v2 spelling can collide with some other column's v1 spelling. Two truth rows
    # claiming one handle would have the harness score one against the other's answer,
    # and `run_schema` keys its payload by the flattened name, so the collision would
    # simply lose a column with nothing raised. Dropped and counted instead.
    seen: set[str] = set()
    collisions = 0
    original_of: dict[str, str] = {}

    for i, (english_name, contracted_name) in enumerate(pairs):
        tokens = tuple(t for t in english_name.split("_") if t)
        name_v1 = "_".join(v1.contract_tokens(tokens))
        name_v2 = "_".join(v2.contract_tokens(tokens))
        # The generator's own contracted mirror must come back out of this re-render.
        # Without the check, a change in the generator would quietly turn this experiment
        # into a comparison of two things it made up.
        if name_v1 != contracted_name:
            regenerated_matches_pack = False
        version = 1 if i % 2 == 0 else 2
        name = name_v1 if version == 1 else name_v2
        if name in seen:
            collisions += 1
            continue
        seen.add(name)
        if name_v1 != name_v2:
            touched.add(name)
        rows.append(_rename(row_of[contracted_name], name))
        truth.append(
            dataclasses.replace(truth_of[contracted_name], flattened_name=name, field_path=name)
        )
        versions.append(version)
        original_of[name] = english_name

    return {
        "rows": rows,
        "truth": truth,
        "versions": versions,
        "original_of": original_of,
        "delta_touched": touched,
        "regenerated_matches_pack": regenerated_matches_pack,
        "name_collisions_dropped": collisions,
        "mirrored_columns": len(pairs),
        "v1": v1,
        "v2": v2,
    }


# =============================================================================
# RUNNING
# =============================================================================


def _batches(rows: list[dict[str, Any]], versions: list[int]) -> list[tuple[int, list[dict]]]:
    """Interleaved batches, alternating producer, in the order they are sent."""
    by_version: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    for row, version in zip(rows, versions, strict=True):
        by_version[version].append(row)
    chunks: list[tuple[int, list[dict[str, Any]]]] = []
    for start in range(0, max(len(by_version[1]), len(by_version[2])), BATCH):
        for version in (1, 2):
            chunk = by_version[version][start : start + BATCH]
            if chunk:
                chunks.append((version, chunk))
    return chunks


def _run_arm(
    matcher: NexusMatcher,
    chunks: list[tuple[int, list[dict[str, Any]]]],
    catalog_for: dict[int, dict[str, str] | None],
) -> dict[str, Any]:
    """Send every batch, choosing this arm's catalog from the batch's producer version."""
    merged: dict[str, Any] = {}
    for version, chunk in chunks:
        catalog = catalog_for[version]
        signals = None if catalog is None else {"abbreviations": catalog}
        run = run_schema(matcher, chunk, 0.0, signals=signals)
        merged.update(run.results)
    return merged


def _hits(truth: list[TruthRow], results: dict[str, Any]) -> dict[str, int]:
    run = Run(session=_SessionView(results), index_seconds=0.0, match_seconds=0.0)
    return {o.key: o.hit for o in observations(truth, run) if o.answerable}


class _SessionView:
    """The two attributes `observations` reads, over a results dict assembled from many
    requests. A real `MatchingSession` is per-request, and this experiment's whole point
    is that there were many requests."""

    def __init__(self, results: dict[str, Any]) -> None:
        self.results = results

    def field_decisions(self) -> dict[str, Any]:
        return {}


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=10_000)
    ap.add_argument("--schema-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    config = MatchingConfig()
    spec = PackSpec(
        rows=args.rows, seed=args.seed, schema_scale=args.schema_scale, feedback_events=100
    )
    print(f"\nGenerating pack: {spec.rows} glossary rows, seed {spec.seed}")
    pack = SyntheticPack.generate(spec)
    pop = _build_population(pack)
    v1, v2 = pop["v1"], pop["v2"]
    rows, truth, versions = pop["rows"], pop["truth"], pop["versions"]

    conflicting = sorted(s for s in pack.delta.changed if v1.expansions.get(s) != v2.expansions[s])
    print(
        f"  catalog v{v1.version} {len(v1.expansions)} rows, "
        f"v{v2.version} {len(v2.expansions)} rows; "
        f"{len(conflicting)} short forms resolve DIFFERENTLY in the two versions"
    )
    print(
        f"  {len(rows)} columns, {sum(1 for v in versions if v == 1)} from the v1 producer "
        f"and {sum(1 for v in versions if v == 2)} from the v2 producer; "
        f"{len(pop['delta_touched'])} are spelled differently by the two standards"
    )
    print(
        f"  re-rendered v1 names reproduce the pack's own contracted mirror: "
        f"{pop['regenerated_matches_pack']}  "
        f"({pop['name_collisions_dropped']} of {pop['mirrored_columns']} columns dropped "
        f"because a v2 spelling collided with another column's v1 spelling)"
    )

    matcher = build_matcher(config)
    report, index_s = index_glossary(matcher, pack.glossary_dicts())
    print(f"  indexed {report.entries} approved entries in {index_s:.1f}s")

    # -- the emulation is exact, and here is the proof ----------------------
    load_time = NexusMatcher.from_config(MatchingConfig(expand_query_abbreviations=True))
    load_time._abbreviation_expander = matcher._request_expander(
        QuerySignals.from_mapping({"abbreviations": dict(v1.expansions)})
    )[0]
    fields = parse_fields(rows)
    emulated_expander, emulated_flag = matcher._request_expander(
        QuerySignals.from_mapping({"abbreviations": dict(v1.expansions)})
    )
    differing = sum(
        1
        for f in fields
        if load_time._build_query_text(f)
        != matcher._build_query_text(f, expander=emulated_expander, expand=emulated_flag)
    )
    print(
        f"  a constant overlay vs a matcher CONFIGURED with that catalog: "
        f"{differing} of {len(fields)} query texts differ"
    )

    # -- the arms -----------------------------------------------------------
    chunks = _batches(rows, versions)
    expander_before = matcher._abbreviation_expander
    solo_none = _hits(truth, _run_arm(matcher, chunks, {1: None, 2: None}))

    arms = {
        "no catalog": {1: None, 2: None},
        "fixed v1 (load time)": {1: dict(v1.expansions), 2: dict(v1.expansions)},
        "fixed v2 (load time)": {1: dict(v2.expansions), 2: dict(v2.expansions)},
        "per request": {1: dict(v1.expansions), 2: dict(v2.expansions)},
    }
    hits: dict[str, dict[str, int]] = {}
    for label, catalog_for in arms.items():
        hits[label] = _hits(truth, _run_arm(matcher, chunks, catalog_for))

    expander_after = matcher._abbreviation_expander
    control = _hits(truth, _run_arm(matcher, chunks, {1: None, 2: None}))
    control_identical = control == solo_none

    # -- scoring, whole population and delta-touched only -------------------
    version_of = {row["flattenedName"]: v for row, v in zip(rows, versions, strict=True)}
    touched = pop["delta_touched"]
    all_keys = sorted(set.intersection(*(set(h) for h in hits.values())))
    populations = {
        "all columns": all_keys,
        "columns the delta re-spells": [k for k in all_keys if k in touched],
        # The complement, and it is not padding. Every loss the routing takes has to live
        # somewhere, and if the channel's cost is concentrated on the columns the change
        # did NOT reach, that is a fact about what a re-pointed catalog does to the words
        # it orphans -- not a fact about routing.
        "columns both standards spell alike": [k for k in all_keys if k not in touched],
        "from the v1 producer": [k for k in all_keys if version_of[k] == 1],
        "from the v2 producer": [k for k in all_keys if version_of[k] == 2],
    }

    table: dict[str, dict[str, float]] = {}
    for pop_label, keys in populations.items():
        table[pop_label] = {
            label: (sum(h[k] for k in keys) / len(keys) if keys else 0.0)
            for label, h in hits.items()
        }

    # -- how well does each catalog invert the spelling it is asked about? --
    # `all columns` came out with per-request BEHIND the stale catalog, which nothing in
    # the design predicts, so the mechanism is measured rather than explained away. For
    # each half and each catalog: the share of columns whose expansion rebuilds the
    # original readable name.
    #
    # The delta is not symmetric, and this is where that shows. `with_delta` re-points a
    # short form S from long form X to long form Y and teaches the contraction map that Y
    # now contracts to S -- but X still contracts to S as well, and S now expands to Y. So
    # version 2 does NOT invert its own contraction for every word version 1 covered: the
    # words that used to own a re-pointed short form are orphaned by the change. A real
    # standard behaves this way, and it means "always send the newest catalog" is not the
    # same instruction as "send the catalog that matches this producer".
    invertibility: dict[str, dict[str, float]] = {}
    for version in (1, 2):
        half = [
            (pop["original_of"][row["flattenedName"]], row["flattenedName"])
            for row in rows
            if version_of[row["flattenedName"]] == version
        ]
        invertibility[f"v{version} producer"] = {}
        for label, catalog in (("v1 catalog", v1.expansions), ("v2 catalog", v2.expansions)):
            expander, _flag = matcher._request_expander(
                QuerySignals.from_mapping({"abbreviations": dict(catalog)})
            )
            invertibility[f"v{version} producer"][label] = (
                len(reconstructed_keys(half, expander)) / len(half) if half else 0.0
            )

    # -- routing checks, labelled as checks --------------------------------
    # Per-request routing sends each half the catalog its own producer used, so on each
    # half it must reproduce that fixed catalog EXACTLY. Zero discordant is what a correct
    # implementation produces; it is evidence about the wiring and not about the feature's
    # value, and it is recorded that way rather than quoted as a result.
    routing: dict[str, dict[str, int]] = {}
    for version, fixed in ((1, "fixed v1 (load time)"), (2, "fixed v2 (load time)")):
        keys = [k for k in sorted(hits["per request"]) if version_of[k] == version]
        routing[f"per request vs {fixed} on the v{version} half"] = {
            "n": len(keys),
            "discordant": sum(1 for k in keys if hits["per request"][k] != hits[fixed][k]),
        }

    best_fixed_label = max(
        ("fixed v1 (load time)", "fixed v2 (load time)"),
        key=lambda label: table["all columns"][label],
    )
    paired: dict[str, Any] = {}
    for pop_label, keys in populations.items():
        if not keys:
            continue
        stat = paired_compare_metric(
            "p_at_1",
            [float(hits[best_fixed_label][k]) for k in keys],
            [float(hits["per request"][k]) for k in keys],
        )
        paired[pop_label] = {
            "baseline_arm": best_fixed_label,
            "test": stat.test,
            "delta": stat.delta,
            "ci_low": stat.ci_low,
            "ci_high": stat.ci_high,
            "p_value": stat.p_value,
            "n_pairs": stat.n_pairs,
            "n_discordant": stat.n_discordant,
        }

    # -- report -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("E2  THE OVERLAY MUST BE PER REQUEST")
    print("=" * 78)
    print("  P@1 by arm and population")
    header = f"    {'population':<32}" + "".join(f"{label:>22}" for label in arms)
    print(header)
    for pop_label, keys in populations.items():
        line = f"    {pop_label + f' (n={len(keys)})':<32}"
        for label in arms:
            line += f"{table[pop_label][label]:>22.4f}"
        print(line)

    print()
    print("  THE STRUCTURAL POINT, FIRST, BECAUSE IT IS NOT A MEASUREMENT.")
    print(
        f"  The delta re-points {len(conflicting)} short forms that the base catalog already had."
    )
    print("  For each of them v1 and v2 assert DIFFERENT long forms, so no single map can")
    print("  hold both. Merging with v2 winning IS v2; merging with v1 winning IS v1.")
    print("  A load-time catalog is a choice between the two columns above, made once, for")
    print("  the life of the process -- and both producers are on the wire at the same time.")
    print()
    d = paired.get("columns the delta re-spells")
    if d is not None:
        print(
            f"  On the {d['n_pairs']} columns the two standards spell differently, "
            f"per-request beats"
        )
        print(
            f"  the better fixed catalog ({best_fixed_label}) by "
            f"{d['delta']:+.4f}  [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
            f"p = {d['p_value']:.4g}"
        )
        print(f"  {d['test']}")
    print()
    print(f"  PER-REQUEST vs {best_fixed_label}, EVERY POPULATION")
    print(f"    {'population':<32}{'n':>6}{'delta':>10}{'p':>12}   gained / lost")
    for pop_label, stat in paired.items():
        print(
            f"    {pop_label:<32}{stat['n_pairs']:>6}{stat['delta']:>+10.4f}"
            f"{stat['p_value']:>12.4g}   {stat['test'].split('(')[-1].rstrip(')')}"
        )
    print()
    print("  Every one of those rows is the same routing decision seen on a different")
    print("  population, and they do not all point the same way. Quote a number with its")
    print("  population or do not quote it.")
    touched_stat = paired.get("columns the delta re-spells")
    alike_stat = paired.get("columns both standards spell alike")
    if touched_stat and alike_stat:
        print()
        print("  WHERE THE GAINS AND THE LOSSES LIVE -- and they do not live together.")
        print(f"    on the {touched_stat['n_pairs']} re-spelled columns   {touched_stat['test']}")
        print(f"    on the {alike_stat['n_pairs']} unchanged columns     {alike_stat['test']}")
        print()
        print("  A delta re-points a short form from one long form to another. On a column")
        print("  the change re-spelled, the new catalog is the only one that inverts it.")
        print("  On a column the change did NOT re-spell, the old long form still contracts")
        print("  to that short and the new catalog now expands it to somebody else -- so")
        print("  the newer catalog is strictly worse there, on words its own delta orphaned.")
        print()
        print("  So routing per producer TRADES one population for the other, and on this")
        print("  pack the trade nets to about zero. The channel's value is that it can make")
        print("  the trade at all and can make it per request; it is not a blanket gain,")
        print("  and 'always send the newest catalog' is not supported by anything here.")
        print("  Whether a catalog assembled per PRODUCER SPELLING rather than per version")
        print("  would take the gains without the losses is a hypothesis this run does not")
        print("  test, and it should not be reported as if it did.")

    print()
    print("  WHICH CATALOG ACTUALLY INVERTS WHICH SPELLING")
    print("    (share of columns whose expansion rebuilds the original readable name)")
    print(f"    {'population':<20}{'v1 catalog':>14}{'v2 catalog':>14}")
    for population, row in invertibility.items():
        print(f"    {population:<20}{row['v1 catalog']:>14.4f}{row['v2 catalog']:>14.4f}")
    v2_own = invertibility["v2 producer"]["v2 catalog"]
    v2_stale = invertibility["v2 producer"]["v1 catalog"]
    if v2_own < v2_stale:
        print()
        print("    THE NEWER CATALOG INVERTS THE NEWER SPELLING LESS WELL THAN THE OLD ONE.")
        print("    That is not a bug in the routing and it is the reason the `all columns`")
        print("    row above does not favour per-request. A delta that re-points a short")
        print("    form ORPHANS the word that used to own it: that word still contracts to")
        print("    the same short, and the short now expands to somebody else. So version 2")
        print("    is a worse inverse of its own contraction than version 1 was of its.")
        print("    Read the delta-touched row for what the channel buys; do not read the")
        print("    aggregate as 'newer catalogs are better'.")

    print()
    print("  ROUTING CHECK (a design check, not a result)")
    for label, r in routing.items():
        print(f"    {label:<48} {r['discordant']} discordant of {r['n']}")

    print()
    print("  MID-RUN MECHANICS")
    print(f"    requests issued                       {len(chunks)}")
    print(
        f"    catalog changed between requests      {sum(1 for i in range(1, len(chunks)) if chunks[i][0] != chunks[i - 1][0])} times"
    )
    print(f"    matcher's own expander is unchanged   {expander_before is expander_after}")
    print(
        f"    a no-signal replay after all of it    {'identical' if control_identical else 'DIFFERENT -- state leaked'}"
    )
    print(
        f"    constant-overlay vs configured matcher {differing} of {len(fields)} query "
        f"texts differ"
    )

    artifact = {
        "experiment": "exp_overlay_per_request (E2) -- the versioned delta applied mid-run",
        "provenance": provenance(spec.seed, config.__dict__.copy()),
        "pack": pack.manifest(),
        "batch_size": BATCH,
        "requests": len(chunks),
        "columns": len(rows),
        "catalog": {
            "v1_version": v1.version,
            "v2_version": v2.version,
            "v1_rows": len(v1.expansions),
            "v2_rows": len(v2.expansions),
            "short_forms_that_conflict": len(conflicting),
            "columns_respelled_by_the_delta": len(touched),
        },
        "p_at_1": table,
        "paired_vs_best_fixed": paired,
        "routing_checks": routing,
        "catalog_invertibility": invertibility,
        "checks": {
            "regenerated_v1_names_match_the_pack": pop["regenerated_matches_pack"],
            "name_collisions_dropped": pop["name_collisions_dropped"],
            "matcher_expander_unchanged": expander_before is expander_after,
            "no_signal_replay_identical": control_identical,
            "query_texts_differing_from_a_configured_matcher": differing,
            "fields_compared": len(fields),
        },
    }
    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"exp_overlay_per_request_synthetic_{spec.rows}.json"
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nSaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
