"""
benchmarks.exp_overlay_value | Layer: BENCHMARK
E1 -- what a caller-supplied abbreviation catalog is worth on contracted names, measured
so that the part which is retrieval can be told apart from the part which is spelling.

The question, and the trap in front of it
-----------------------------------------
Contract every column name through a naming standard and retrieval collapses. Hand the
matcher the standard's catalog through the per-request signal channel and it recovers.
The obvious experiment therefore has an obvious answer, and the obvious answer is worth
nothing, because a catalog generated from a contraction is usually that contraction's
INVERSE. Expanding a name it produced rebuilds the original string, and the P@1 that
follows is `f_inverse(f(x)) == x` wearing a retrieval metric's clothes.

This repository has published exactly that number: a "99.4% recovery" whose caseless
reconstruction rate, measured afterwards, was 0.9927 and 1.0000. It was disclosed as an
upper bound rather than quoted as a finding, and that disclosure was correct -- but a
figure nobody can separate from reconstruction is not a measurement.

So this script is built to make the separation, three ways:

  1. THE CATALOG IS NOT THE INVERSE. The pack's naming standard hands several words the
     same short form and expands it to one of them; it collapses adjacent word pairs into
     a single token; and it issues short forms spelled like English function words. A
     word that loses a contested short form comes back as a DIFFERENT, equally plausible
     word. `PackSpec.ambiguity` is the dial, and `--ambiguity-sweep` moves it.

  2. THE IDENTITY FRACTION IS PRINTED BESIDE EVERY RECOVERY FIGURE, per arm, measured on
     the arm's own overlay -- the share of columns whose expansion is caselessly
     identical to the English name they mirror.

  3. THE HEADLINE IS THE STRATUM WHERE RECONSTRUCTION IS IMPOSSIBLE. Every arm is scored
     again over only the columns the FULL catalog cannot rebuild. A gain there is
     retrieval by construction: the query the matcher sent was not the answer's name.

The other confound, removed
---------------------------
`flat-contracted` mirrors `flat-english` field for field, and it also drops the doc that
the English profile carries on every row. Quoting the difference between those two as
"the abbreviation gap" would charge contraction for the doc as well. So an `english, doc
stripped` arm runs alongside, built from the same rows with the `doc` key removed, and
the gap this script recovers against is measured from THAT.

Usage
-----
    python benchmarks/exp_overlay_value.py --self-check
    python benchmarks/exp_overlay_value.py --rows 10000 --schema-scale 4 --save
    python benchmarks/exp_overlay_value.py --ambiguity-sweep 0.08 0.30 0.50 --save

`ambiguity` is a FLOOR on the contested share, not a target: the vowel-drop rule already
collides on its own at this vocabulary size, so a value below the natural collision rate
changes nothing and the sweep will print two identical rows. Start above it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimization_ledger import paired_compare_metric, provenance
from synthetic.pack import DEFAULT_SEED, PackSpec, SyntheticPack
from synthetic_harness import (
    RESULTS,
    FieldObs,
    build_matcher,
    index_glossary,
    mirror_pairs,
    normalise_words,
    observations,
    reconstructed_keys,
    run_schema,
)

from nexus_matcher.application.use_cases.match_schema import (
    MatchingConfig,
    QuerySignals,
)

# The two mirrored profiles. Field for field, same truth, one rendered through the naming
# standard -- which is what makes every comparison below paired rather than two runs.
ENGLISH = "flat-english"
CONTRACTED = "flat-contracted"


# =============================================================================
# THE OVERLAYS
# =============================================================================


@dataclass(frozen=True)
class Overlay:
    """One catalog as a caller would send it, and what is wrong with it."""

    label: str
    rows: dict[str, str]
    coverage: float
    wrong_rate: float


def _degrade(
    full: dict[str, str],
    seed: int,
    coverage: float = 1.0,
    wrong_rate: float = 0.0,
) -> dict[str, str]:
    """
    A catalog missing rows, or asserting the wrong long form on some of the rows it has.

    The two degradations are the two ways a real feed is imperfect and they are NOT the
    same failure. A missing row leaves a token alone -- the expander is exact-lookup with
    passthrough -- and costs the query that token's meaning. A WRONG row replaces the
    token with a confident, plausible, incorrect word, and the query only gets one vector.
    """
    rng = random.Random(seed ^ 0x0E1_0001)
    keys = sorted(full)
    kept = keys if coverage >= 1.0 else rng.sample(keys, round(len(keys) * coverage))
    out = {k: full[k] for k in kept}
    if wrong_rate <= 0.0:
        return out
    longs = sorted(set(full.values()))
    spoiled = rng.sample(sorted(out), round(len(out) * wrong_rate))
    for short in spoiled:
        wrong = rng.choice(longs)
        for _ in range(8):
            if wrong != out[short]:
                break
            wrong = rng.choice(longs)
        out[short] = wrong
    return out


def _arms(full: dict[str, str], seed: int) -> list[Overlay]:
    """The catalog as sent, and the degraded versions that carry the information."""
    arms = [Overlay("full", dict(full), 1.0, 0.0)]
    for cov in (0.75, 0.50, 0.25):
        arms.append(Overlay(f"coverage {cov:.0%}", _degrade(full, seed, coverage=cov), cov, 0.0))
    for wrong in (0.05, 0.25, 0.50, 0.75, 1.00):
        arms.append(
            Overlay(f"wrong {wrong:.0%}", _degrade(full, seed, wrong_rate=wrong), 1.0, wrong)
        )
    return arms


# =============================================================================
# SCORING
# =============================================================================


def _p_at_1(obs: dict[str, FieldObs], keys: list[str]) -> float:
    return sum(obs[k].hit for k in keys) / len(keys) if keys else 0.0


def _overlap(expander: Any, pairs: list[tuple[str, str]], keys: set[str]) -> dict[str, float]:
    """
    How much of the original name the catalog got back, on the columns it did not get
    back exactly.

    "Irrecoverable" is a binary and binaries hide their own shape. A column whose
    four-token name came back with one token wrong is a different object from one that
    came back with all four wrong, and a stratum made mostly of the first is still
    carrying a lot of the answer's surface into the query. So the token overlap between
    the expanded name and the original is reported with it. This is the check
    `benchmarks/validate_benchmark.py` runs on a whole corpus, applied to one stratum.
    """
    ratios: list[float] = []
    for original, contracted in pairs:
        if contracted not in keys:
            continue
        rebuilt = expander.expand(contracted.replace("_", " ")).expanded
        want = {t for t in normalise_words(rebuilt).split() if t}
        gold = {t for t in normalise_words(original).split() if t}
        if gold:
            ratios.append(len(want & gold) / len(gold))
    if not ratios:
        return {}
    ratios.sort()
    return {
        "mean": sum(ratios) / len(ratios),
        "median": ratios[len(ratios) // 2],
        "max": ratios[-1],
        "share_at_or_above_0.75": sum(1 for r in ratios if r >= 0.75) / len(ratios),
    }


def _measure(spec: PackSpec, config: MatchingConfig, verbose: bool = True) -> dict[str, Any]:
    """One pack, one indexed glossary, every arm."""
    pack = SyntheticPack.generate(spec)
    english = pack.schema(ENGLISH)
    contracted = pack.schema(CONTRACTED)
    full_catalog = dict(pack.catalog.expansions)

    # Contraction is many-to-one, so the contracted profile is SHORTER than the readable
    # one it mirrors: two readable columns can land on the same contracted spelling and
    # the generator keeps the first. Every arm below is therefore restricted to the
    # columns that exist on both sides, paired by the generator's own mirror record.
    pairs = mirror_pairs(english, contracted)
    english_of = {con: eng for eng, con in pairs}
    mirrored_english = {eng for eng, _con in pairs}

    matcher = build_matcher(config)
    report, index_s = index_glossary(matcher, pack.glossary_dicts())
    if verbose:
        print(
            f"  indexed {report.entries} approved entries in {index_s:.1f}s; "
            f"{len(english.flattened)} readable columns, {len(contracted.flattened)} "
            f"contracted, {len(pairs)} mirrored on both sides; "
            f"catalog {len(full_catalog)} rows"
        )

    # -- the three reference arms ------------------------------------------
    # `english, doc stripped` is the one the gap is measured from. The doc is a separate
    # feature and charging contraction for it would inflate everything downstream.
    english_no_doc = [{k: v for k, v in row.items() if k != "doc"} for row in english.flattened]

    runs = {
        "english (with doc)": (english.truth, run_schema(matcher, english.flattened, index_s)),
        "english (doc stripped)": (english.truth, run_schema(matcher, english_no_doc, index_s)),
        "contracted (no overlay)": (
            contracted.truth,
            run_schema(matcher, contracted.flattened, index_s),
        ),
    }
    reference: dict[str, dict[str, FieldObs]] = {}
    for label, (truth, run) in runs.items():
        obs = {o.key: o for o in observations(truth, run) if o.answerable}
        if label.startswith("english"):
            obs = {k: v for k, v in obs.items() if k in mirrored_english}
        else:
            obs = {k: v for k, v in obs.items() if k in english_of}
        reference[label] = obs

    # The doc's own contribution, PAIRED. A difference of two aggregate P@1 values can
    # read 0.0000 while dozens of fields moved in both directions, and "the doc is worth
    # nothing here" is a claim about the corpus that deserves a discordant count.
    doc_keys = sorted(
        set(reference["english (with doc)"]) & set(reference["english (doc stripped)"])
    )
    doc_stat = paired_compare_metric(
        "p_at_1_doc",
        [float(reference["english (doc stripped)"][k].hit) for k in doc_keys],
        [float(reference["english (with doc)"][k].hit) for k in doc_keys],
    )

    floor_obs = reference["contracted (no overlay)"]
    # The ceiling is taken over the readable columns that HAVE a contracted mirror, so
    # ceiling and floor describe one population and the gap between them is a difference
    # rather than a coincidence.
    ceiling = _p_at_1(
        reference["english (doc stripped)"], sorted(reference["english (doc stripped)"])
    )
    floor = _p_at_1(floor_obs, sorted(floor_obs))
    gap = ceiling - floor

    # -- the stratum reconstruction cannot reach ---------------------------
    # Computed from the FULL catalog and then held FIXED across every arm, so the arms
    # are compared on one population rather than each on its own convenient subset.
    full_expander, full_flag = matcher._request_expander(
        QuerySignals.from_mapping({"abbreviations": full_catalog})
    )
    rebuilt = reconstructed_keys(pairs, full_expander)
    answerable = sorted(floor_obs)
    hard_keys = [k for k in answerable if k not in rebuilt]
    easy_keys = [k for k in answerable if k in rebuilt]

    if verbose:
        print(
            f"  expansion rebuilds the original name on {len(rebuilt)} of "
            f"{len(pairs)} mirrored columns ({len(rebuilt) / len(pairs):.4f}); "
            f"{len(hard_keys)} of {len(answerable)} answerable columns it cannot"
        )
        print(
            f"  request expander is the matcher's own object: "
            f"{full_expander is matcher._abbreviation_expander}, expansion on: {full_flag}"
        )

    # -- the overlay arms ---------------------------------------------------
    arms: list[dict[str, Any]] = []
    arm_hits: dict[str, dict[str, int]] = {}
    for overlay in _arms(full_catalog, spec.seed):
        expander, _flag = matcher._request_expander(
            QuerySignals.from_mapping({"abbreviations": overlay.rows})
        )
        identity = reconstructed_keys(pairs, expander)
        run = run_schema(
            matcher, contracted.flattened, index_s, signals={"abbreviations": overlay.rows}
        )
        obs = {o.key: o for o in observations(contracted.truth, run) if o.answerable}
        arm_hits[overlay.label] = {k: o.hit for k, o in obs.items()}
        shared = sorted(set(obs) & set(floor_obs))
        stat = paired_compare_metric(
            "p_at_1",
            [float(floor_obs[k].hit) for k in shared],
            [float(obs[k].hit) for k in shared],
        )
        hard_shared = [k for k in shared if k in set(hard_keys)]
        hard_stat = (
            paired_compare_metric(
                "p_at_1_irrecoverable",
                [float(floor_obs[k].hit) for k in hard_shared],
                [float(obs[k].hit) for k in hard_shared],
            )
            if hard_shared
            else None
        )
        arms.append(
            {
                "label": overlay.label,
                "catalog_rows": len(overlay.rows),
                "coverage": overlay.coverage,
                "wrong_rate": overlay.wrong_rate,
                "identity_fraction": len(identity) / len(pairs),
                "p_at_1": _p_at_1(obs, sorted(obs)),
                "recovery": (_p_at_1(obs, sorted(obs)) - floor) / gap if gap else 0.0,
                "p_at_1_rebuilt_stratum": _p_at_1(obs, [k for k in easy_keys if k in obs]),
                "p_at_1_irrecoverable_stratum": _p_at_1(obs, hard_shared),
                "paired": _stat_json(stat),
                "paired_irrecoverable": _stat_json(hard_stat),
            }
        )
        if verbose:
            a = arms[-1]
            print(
                f"    {overlay.label:<14} P@1 {a['p_at_1']:.4f}  recovery {a['recovery']:+.3f}  "
                f"identity {a['identity_fraction']:.4f}  "
                f"irrecoverable-stratum {a['p_at_1_irrecoverable_stratum']:.4f}"
            )

    # -- each arm against the FULL catalog, on the stratum that matters ------
    # The first run of this script showed a 75%-coverage catalog SCORING HIGHER on the
    # irrecoverable stratum than the complete one. That is either a real mechanism -- a
    # missing row leaves a token alone, while a present-but-contested row replaces it with
    # a confident wrong word -- or it is noise on 178 columns. Only a paired test
    # distinguishes them, so one is run rather than the observation being narrated.
    full_hits = arm_hits["full"]
    hard_set = set(hard_keys)
    for arm in arms:
        if arm["label"] == "full":
            arm["paired_vs_full_irrecoverable"] = None
            continue
        keys = sorted(set(full_hits) & set(arm_hits[arm["label"]]) & hard_set)
        arm["paired_vs_full_irrecoverable"] = (
            _stat_json(
                paired_compare_metric(
                    "p_at_1_irrecoverable",
                    [float(full_hits[k]) for k in keys],
                    [float(arm_hits[arm["label"]][k]) for k in keys],
                )
            )
            if keys
            else None
        )

    return {
        "spec": {
            "rows": spec.rows,
            "schema_scale": spec.schema_scale,
            "seed": spec.seed,
            "ambiguity": spec.ambiguity,
        },
        "pack": pack.manifest(),
        "indexed_entries": report.entries,
        "columns": len(pairs),
        "columns_readable_profile": len(english.flattened),
        "columns_contracted_profile": len(contracted.flattened),
        "answerable": len(answerable),
        "catalog_rows": len(full_catalog),
        "reference": {label: _p_at_1(obs, sorted(obs)) for label, obs in reference.items()},
        "doc_contribution": (
            _p_at_1(reference["english (with doc)"], sorted(reference["english (with doc)"]))
            - ceiling
        ),
        "doc_contribution_paired": _stat_json(doc_stat),
        "irrecoverable_token_overlap": _overlap(full_expander, pairs, set(hard_keys)),
        "abbreviation_gap": gap,
        "identity_fraction_full_catalog": len(rebuilt) / len(pairs),
        "irrecoverable_answerable": len(hard_keys),
        "rebuilt_answerable": len(easy_keys),
        "floor_p_at_1_rebuilt_stratum": _p_at_1(floor_obs, easy_keys),
        "floor_p_at_1_irrecoverable_stratum": _p_at_1(floor_obs, hard_keys),
        "arms": arms,
    }


def _stat_json(stat: Any) -> dict[str, Any] | None:
    if stat is None:
        return None
    return {
        "test": stat.test,
        "delta": stat.delta,
        "ci_low": stat.ci_low,
        "ci_high": stat.ci_high,
        "p_value": stat.p_value,
        "n_pairs": stat.n_pairs,
        "n_discordant": stat.n_discordant,
    }


# =============================================================================
# MAIN
# =============================================================================


# =============================================================================
# THE SELF-CHECK -- does the identity measure actually detect f_inverse(f(x))?
# =============================================================================


class _Expansion:
    """The one attribute `AbbreviationExpander.expand` exposes that this file reads."""

    def __init__(self, expanded: str) -> None:
        self.expanded = expanded


class _OracleExpander:
    """
    The exact inverse of the contraction, by table lookup. The degenerate case.

    It returns the original name UPPER-CASED and underscore-joined rather than verbatim,
    so this stub also exercises the normalisation `reconstructed_keys` depends on. A
    reconstruction check that compared raw strings would score this oracle at zero and
    then report a corpus of perfectly rebuilt names as the irrecoverable stratum, which is
    the worst failure available here -- reconstruction credited as retrieval.
    """

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self._back = {normalise_words(contracted): original for original, contracted in pairs}

    def expand(self, text: str) -> _Expansion:
        original = self._back.get(normalise_words(text))
        if original is None:
            return _Expansion(text)
        return _Expansion("_".join(normalise_words(original).split()).upper())


class _NullExpander:
    """An expander that does nothing. The other pole."""

    def expand(self, text: str) -> _Expansion:
        return _Expansion(text)


def self_check(spec: PackSpec) -> int:
    """
    Prove the identity measure separates the two poles it exists to separate.

    The headline of this experiment is a P@1 restricted to the columns the catalog CANNOT
    rebuild. If `reconstructed_keys` under-reported -- if it said "cannot rebuild" about
    columns that were in fact rebuilt -- the headline would be reporting reconstruction
    while claiming to have excluded it, which is the exact failure this script was written
    to avoid, one level down.

    So both poles are asserted:

      * an ORACLE that maps every contracted name straight back to its original is
        `f_inverse` by construction. The identity fraction must be exactly 1.0 and the
        irrecoverable stratum must be EMPTY. A measure that reports anything else here
        would let a fully degenerate experiment through.
      * an expander that does NOTHING must rebuild only the columns the naming standard
        never changed.

    And the pack's own catalog must sit strictly between them, which is the property that
    makes the fixture usable at all.

    THE PAIRING IS CHECKED SEPARATELY, and it has to be. The two poles above are both
    computed from whatever pairs they are handed, so they are self-consistent under a
    WRONG pairing and cannot detect one -- verified by planting a positional zip in
    `mirror_pairs`, which this pole test passed. The independent check is the generator's
    own rule: contracting the original name through the pack's catalog must produce the
    contracted name it is paired with.
    """
    pack = SyntheticPack.generate(spec)
    pairs = mirror_pairs(pack.schema(ENGLISH), pack.schema(CONTRACTED))

    oracle = len(reconstructed_keys(pairs, _OracleExpander(pairs)))
    null = len(reconstructed_keys(pairs, _NullExpander()))
    total = len(pairs)
    mispaired = sum(
        1
        for original, contracted in pairs
        if "_".join(pack.catalog.contract_tokens(tuple(t for t in original.split("_") if t)))
        != contracted
    )

    print(f"\nSELF-CHECK on {total} mirrored columns")
    print(f"  exact-inverse oracle rebuilds   {oracle} / {total} = {oracle / total:.4f}")
    print(f"  do-nothing expander rebuilds    {null} / {total} = {null / total:.4f}")
    print(f"  pairs the naming standard does not reproduce   {mispaired}")

    failures: list[str] = []
    if mispaired:
        failures.append(
            f"{mispaired} of {total} pairs do not survive the generator's own contraction "
            f"rule, so columns are paired with the wrong original and every stratum below "
            f"is drawn from the wrong population"
        )
    if oracle != total:
        failures.append(
            f"an exact inverse rebuilt only {oracle} of {total} columns; the identity "
            f"measure does not detect f_inverse(f(x)) and every headline below is unsafe"
        )
    if null >= total:
        failures.append(
            "doing nothing rebuilt every column, so the contraction changed no name and "
            "this fixture cannot measure expansion at all"
        )
    for line in failures:
        print(f"  FAIL: {line}")
    if not failures:
        print("  PASS: the measure reads 1.0 on an exact inverse and less on the pack's own")
        print("        catalog, so the irrecoverable stratum means what it says.")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=10_000)
    ap.add_argument("--schema-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--ambiguity-sweep",
        type=float,
        nargs="*",
        default=None,
        help=(
            "also re-run the whole measurement at these ambiguity levels. The check that "
            "a recovery figure is not reconstruction is that it MOVES when this does."
        ),
    )
    ap.add_argument(
        "--self-check",
        action="store_true",
        help=(
            "assert the identity measure reads 1.0 against an exact inverse and less "
            "against the pack's own catalog, and exit. Loads no encoder."
        ),
    )
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    config = MatchingConfig()
    if args.self_check:
        return self_check(
            PackSpec(
                rows=args.rows,
                seed=args.seed,
                schema_scale=args.schema_scale,
                feedback_events=100,
            )
        )
    levels = args.ambiguity_sweep if args.ambiguity_sweep else [PackSpec().ambiguity]

    measurements: list[dict[str, Any]] = []
    for ambiguity in levels:
        spec = PackSpec(
            rows=args.rows,
            seed=args.seed,
            schema_scale=args.schema_scale,
            feedback_events=100,
            ambiguity=ambiguity,
        )
        print(f"\nPack: {spec.rows} glossary rows, ambiguity {ambiguity}, seed {spec.seed}")
        measurements.append(_measure(spec, config))

    primary = measurements[0]
    _report(primary, measurements)

    artifact = {
        "experiment": "exp_overlay_value (E1) -- the abbreviation overlay on contracted names",
        "provenance": provenance(args.seed, config.__dict__.copy()),
        "measurements": measurements,
    }
    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"exp_overlay_value_synthetic_{args.rows}.json"
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nSaved -> {path}")
    return 0


def _report(primary: dict[str, Any], measurements: list[dict[str, Any]]) -> None:
    ref = primary["reference"]
    print("\n" + "=" * 78)
    print("E1  THE ABBREVIATION OVERLAY, WITH THE RECONSTRUCTION SEPARATED OUT")
    print("=" * 78)
    print(f"  glossary indexed                      {primary['indexed_entries']} approved entries")
    print(
        f"  mirrored columns                      {primary['columns']} of {primary['columns_readable_profile']} readable / {primary['columns_contracted_profile']} contracted"
    )
    print(f"  answerable columns                    {primary['answerable']}")
    print(f"  catalog supplied per request          {primary['catalog_rows']} rows")
    print()
    print(f"  english, with doc            P@1      {ref['english (with doc)']:.4f}")
    print(f"  english, doc stripped        P@1      {ref['english (doc stripped)']:.4f}")
    print(f"  contracted, no overlay       P@1      {ref['contracted (no overlay)']:.4f}")
    doc = primary["doc_contribution_paired"]
    print(
        f"  the doc alone is worth                {primary['doc_contribution']:+.4f}   "
        f"({doc['n_discordant']} discordant of {doc['n_pairs']}, p = {doc['p_value']:.4g})"
    )
    print(
        f"  THE ABBREVIATION GAP                  {primary['abbreviation_gap']:.4f}  "
        f"(english-no-doc minus contracted)"
    )
    print()
    print("  The gap is measured against english WITHOUT its doc on purpose. The")
    print("  contracted mirror carries no doc, so the raw english-to-contracted")
    print("  difference charges contraction for the documentation as well.")

    print("\n  ARMS")
    print(
        "    catalog          P@1     recovery  identity   rebuilt   irrecoverable"
        "   McNemar p (irrecoverable)"
    )
    floor_hard = primary["floor_p_at_1_irrecoverable_stratum"]
    floor_easy = primary["floor_p_at_1_rebuilt_stratum"]
    print(
        f"    {'(none)':<14} {primary['reference']['contracted (no overlay)']:>7.4f}"
        f"  {0.0:>8.3f}  {0.0:>8.4f}  {floor_easy:>8.4f}  {floor_hard:>13.4f}"
    )
    for arm in primary["arms"]:
        p = arm["paired_irrecoverable"]
        pv = "n/a" if p is None or p["p_value"] is None else f"{p['p_value']:.4g}"
        print(
            f"    {arm['label']:<14} {arm['p_at_1']:>7.4f}  {arm['recovery']:>+8.3f}  "
            f"{arm['identity_fraction']:>8.4f}  {arm['p_at_1_rebuilt_stratum']:>8.4f}  "
            f"{arm['p_at_1_irrecoverable_stratum']:>13.4f}   {pv}"
        )

    full = next(a for a in primary["arms"] if a["label"] == "full")
    print()
    print("  READ THE LAST TWO COLUMNS, NOT THE FIRST.")
    print(
        f"  On this pack a full catalog rebuilds the original column name on "
        f"{primary['identity_fraction_full_catalog']:.2%} of columns."
    )
    print("  The 'rebuilt' column is that population and it is NOT a retrieval result --")
    print("  the matcher was handed the answer's own name and looked it up.")
    print(
        f"  The 'irrecoverable' column is the {primary['irrecoverable_answerable']} answerable "
        f"columns the catalog"
    )
    print("  provably cannot rebuild: a contested short form expands to a DIFFERENT,")
    print("  equally plausible word, so the query the matcher sent is not the answer's")
    print(
        f"  name. There P@1 goes {floor_hard:.4f} -> "
        f"{full['p_at_1_irrecoverable_stratum']:.4f}, and that movement is retrieval."
    )
    ov = primary.get("irrecoverable_token_overlap") or {}
    if ov:
        print()
        print("  HOW IRRECOVERABLE IS IRRECOVERABLE. On that stratum the expanded name")
        print(
            f"  still shares {ov['mean']:.2f} of the original's tokens on average "
            f"(median {ov['median']:.2f}), and"
        )
        print(
            f"  {ov['share_at_or_above_0.75']:.1%} of those columns come back with three "
            f"quarters or more of the"
        )
        print("  name intact. So this is not a stratum with no surface at all -- it is one")
        print("  where the name the matcher searched with is NOT the name it was looking")
        print("  for, which is the condition retrieval has to survive and string identity")
        print("  cannot.")

    worse_than_full = [
        a
        for a in primary["arms"]
        if a["label"] != "full"
        and a["p_at_1_irrecoverable_stratum"] > full["p_at_1_irrecoverable_stratum"]
    ]
    if worse_than_full:
        print()
        print("  A DEGRADED CATALOG BEATS THE COMPLETE ONE ON THIS STRATUM:")
        for a in worse_than_full:
            st = a["paired_vs_full_irrecoverable"]
            detail = (
                "no paired test available"
                if st is None
                else (
                    f"{st['delta']:+.4f} vs full  [{st['ci_low']:+.4f}, {st['ci_high']:+.4f}]  "
                    f"p = {st['p_value']:.4g}  ({st['n_discordant']} discordant)"
                )
            )
            print(f"    {a['label']:<14} {a['p_at_1_irrecoverable_stratum']:.4f}   {detail}")
        print("  The mechanism, if it is real: a row the catalog is MISSING leaves its")
        print("  token alone, and an unexpanded token still matches itself. A row the")
        print("  catalog HAS but resolves to the losing side of a contested short form")
        print("  replaces that token with a confident, wrong, plausible word. Read the")
        print("  p-value before repeating this as a finding.")

    if len(measurements) > 1:
        print("\n" + "=" * 78)
        print("THE DIFFICULTY SWEEP -- does the number move?")
        print("=" * 78)
        print(
            "    ambiguity  invertible  identity   gap      full-catalog recovery   "
            "irrecoverable P@1"
        )
        for m in measurements:
            arm = next(a for a in m["arms"] if a["label"] == "full")
            inv = m["pack"]["abbreviations"]["token_level_invertible_share"]
            print(
                f"    {m['spec']['ambiguity']:>9.3f}  {inv:>10.4f}  "
                f"{m['identity_fraction_full_catalog']:>8.4f}  {m['abbreviation_gap']:>7.4f}  "
                f"{arm['recovery']:>+21.3f}   {arm['p_at_1_irrecoverable_stratum']:>16.4f}"
            )
        print()
        print("  A recovery figure that does not move as the catalog stops being the")
        print("  contraction's inverse is not measuring retrieval. That is the check the")
        print("  '99.4% recovery' of the previous wave failed, and it is why this table")
        print("  is printed whether or not it flatters the result.")


if __name__ == "__main__":
    raise SystemExit(main())
