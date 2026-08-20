"""
benchmarks.exp_mode_selection | Layer: BENCHMARK
E6 -- should this library learn to decline? Measured BEFORE anything is built.

The proposal
------------
A field with no doc, no caller-supplied parent record, and a path that merely echoes its
own name is downgraded to sparse-only retrieval, on the reasoning that a dense query built
from nothing still returns a nearest neighbour and still reports a confidence that looks
fine. The claim made for it is not speed. It is that the value of the policy is the
confident garbage it declines to emit.

This library has no equivalent and always runs the full pipeline. This script asks three
questions, in the order that decides whether the feature is worth writing.

  1. IS THE TRIGGER DETECTABLE AT ALL? "A path that merely echoes its name" is not a
     definition. Three candidate definitions are proposed below and each is counted on the
     synthetic profiles AND on the committed real corpora. A policy whose trigger never
     fires on real input is dead code with a docstring.

  2. WHAT DOES EACH MODE ACTUALLY DO on the profile the policy is written for? P@1 on the
     rows that have an answer, and -- the number the proposal rests on -- how often each
     mode emits a confident answer on the rows where no correct answer exists.

  3. IS THERE A CHEAPER KNOB THAT ALREADY DOES THIS? `absolute_score_floor` ships, is off
     by default, and is compared against the raw dense score, which has no per-field
     normalisation and therefore no floor. If that score separates "correct" from "nothing
     is correct", the existing knob is the answer and mode selection is not needed.

THE CONFOUND, STATED UP FRONT
-----------------------------
"Sparse-only" here is `fusion_alpha = 0.0`. That does not merely change retrieval: it
moves the structural confidence floor, which is `semantic_weight * fusion_alpha` -- 0.63
for the shipped configuration and 0.00 at alpha 0. So SOME of any drop in confident
answers is arithmetic, and both floors are printed beside both empirical minima so a
reader can see how much.

That is not a reason to discard the comparison, and it is worth being precise about why.
The 0.63 floor exists because min-max normalisation maps the best dense candidate to
exactly 1.0 no matter how bad it is. The floor is therefore not evidence about the match;
it is a manufactured number, and it is manufactured by the arm this policy proposes to
switch off. The confound and the mechanism are the same fact seen from two sides.

Usage
-----
    python benchmarks/exp_mode_selection.py --rows 10000 --save
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimization_ledger import paired_compare_metric, provenance
from synthetic.pack import DEFAULT_SEED, PackSpec, SyntheticPack
from synthetic.schemas import ARRAY_BOUNDARY
from synthetic_harness import (
    RESULTS,
    FieldObs,
    build_matcher,
    index_glossary,
    observations,
    parse_fields,
    run_schema,
)

from nexus_matcher.application.use_cases.match_schema import MatchingConfig

# The profile the proposal is written for, and a realistic mixture where the trigger has
# to pick fields out rather than claiming all of them.
NO_DOC = "no-doc"
MIXED = "mixed-production"

# The bars a "confident answer" is counted at. 0.63 is the shipped structural floor and
# 0.87 the shipped auto-approve threshold; 0.50 is `review_threshold`, below which a
# rank-1 result would not even be REVIEW.
BARS = (0.50, 0.63, 0.87)


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^0-9A-Za-z]+", (text or "").lower()) if t}


# =============================================================================
# QUESTION 1 -- THREE DEFINITIONS OF "A PATH THAT MERELY ECHOES ITS NAME"
# =============================================================================


def _split(flattened: str) -> tuple[str, str]:
    """`(parent path, leaf)` for a flattened name from this pack."""
    if ARRAY_BOUNDARY in flattened:
        parent, _, leaf = flattened.rpartition(ARRAY_BOUNDARY)
        return parent, leaf
    return "", flattened


def structural_trigger(parent: str, leaf: str, doc: str, entity: str) -> bool:
    """
    D1, the proposal's own wording read literally.

    No doc, no caller-supplied parent record, and every token of the path already present
    in the leaf -- which includes the case of no path at all, where the subset holds
    trivially and the leaf really is everything the query has.
    """
    if doc or entity:
        return False
    return _tokens(parent) <= _tokens(leaf)


def lexical_trigger(query_tokens: set[str], vocabulary: set[str], doc: str, entity: str) -> bool:
    """
    D2: no doc, no entity, and not one token of the query occurs anywhere in the indexed
    corpus.

    This is the reading that matches the STATED REASON rather than the stated wording --
    "a dense query built from nothing" -- and unlike D1 it is a statement about the query
    and the corpus together rather than about the field's shape. It is computable from the
    query text and the index's vocabulary alone, before any retrieval runs, so a mode
    selector could act on it and genuinely skip the dense call.
    """
    if doc or entity:
        return False
    return not (query_tokens & vocabulary)


def sparse_empty_trigger(usable_sparse_candidates: int, doc: str, entity: str) -> bool:
    """
    D3: the lexical arm came back with nothing usable.

    The most honest predictor of the three and the least useful as a MODE selector,
    because knowing it requires running retrieval. Kept because it bounds the other two:
    it is what they are trying to predict.
    """
    if doc or entity:
        return False
    return usable_sparse_candidates < 2


# =============================================================================
# MEASUREMENT
# =============================================================================


def _confidence_profile(obs: list[FieldObs]) -> dict[str, Any]:
    confidences = sorted(o.conf1 for o in obs)
    if not confidences:
        return {}
    return {
        "n": len(confidences),
        "min": confidences[0],
        "p25": confidences[len(confidences) // 4],
        "median": confidences[len(confidences) // 2],
        "max": confidences[-1],
        **{f"share_at_or_above_{bar}": _share(confidences, bar) for bar in BARS},
    }


def _share(sorted_values: list[float], bar: float) -> float:
    return sum(1 for v in sorted_values if v >= bar) / len(sorted_values) if sorted_values else 0.0


def _auc(positive: list[float], negative: list[float]) -> float:
    """P(a correct rank-1 outscores a rank-1 on a row with no correct answer)."""
    if not positive or not negative:
        return 0.0
    wins = sum(1 for p in positive for n in negative if p > n)
    ties = sum(1 for p in positive for n in negative if p == n)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def _best_floor(positive: list[float], negative: list[float]) -> dict[str, Any]:
    """
    The best any `absolute_score_floor` can do at separating the two populations.

    Reported as the maximum of (kept share of correct rank-1s) minus (kept share of
    rank-1s on rows where nothing is correct). At 0 the score carries no information the
    floor can use and the knob cannot help however it is set.
    """
    if not positive or not negative:
        return {}
    best = {"floor": 0.0, "youden_j": -1.0, "kept_correct": 1.0, "kept_no_answer": 1.0}
    for floor in sorted({round(v, 4) for v in positive + negative}):
        kept_pos = sum(1 for v in positive if v >= floor) / len(positive)
        kept_neg = sum(1 for v in negative if v >= floor) / len(negative)
        j = kept_pos - kept_neg
        if j > best["youden_j"]:
            best = {
                "floor": floor,
                "youden_j": j,
                "kept_correct": kept_pos,
                "kept_no_answer": kept_neg,
            }
    return best


def _observe(matcher: Any, pack: SyntheticPack, schema_name: str) -> dict[str, FieldObs]:
    """One indexed matcher, one schema, one observation per field."""
    schema = pack.schema(schema_name)
    run = run_schema(matcher, schema.flattened, 0.0)
    return {o.key: o for o in observations(schema.truth, run)}


def _summarise(obs: dict[str, FieldObs], keys: list[str] | None = None) -> dict[str, Any]:
    rows = [obs[k] for k in (keys if keys is not None else sorted(obs))]
    answerable = [o for o in rows if o.answerable]
    unanswerable = [o for o in rows if not o.answerable]
    return {
        "fields": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "p_at_1": sum(o.hit for o in answerable) / len(answerable) if answerable else 0.0,
        "confidence_answerable": _confidence_profile(answerable),
        "confidence_unanswerable": _confidence_profile(unanswerable),
        "auto_approved_on_unanswerable": sum(1 for o in unanswerable if o.auto_approved),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=10_000)
    ap.add_argument("--schema-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    full_config = MatchingConfig()
    sparse_config = MatchingConfig(fusion_alpha=0.0)
    dense_config = MatchingConfig(fusion_alpha=1.0)
    spec = PackSpec(
        rows=args.rows, seed=args.seed, schema_scale=args.schema_scale, feedback_events=100
    )
    print(f"\nGenerating pack: {spec.rows} glossary rows, seed {spec.seed}")
    pack = SyntheticPack.generate(spec)

    # -- QUESTION 1: is the trigger detectable? -----------------------------
    print("\nIndexing once to read the corpus vocabulary the lexical trigger needs...")
    probe = build_matcher(full_config)
    report, index_s = index_glossary(probe, pack.glossary_dicts())
    vocabulary = {
        t
        for entry in probe._dictionary_entries.values()
        for t in _tokens(entry.to_searchable_text())
    }
    print(
        f"  indexed {report.entries} entries in {index_s:.1f}s; {len(vocabulary)} distinct tokens"
    )

    triggers: dict[str, dict[str, Any]] = {}
    for schema in pack.schemas:
        fields = {f.name: f for f in parse_fields(schema.flattened)}
        counts = {"structural": 0, "lexical": 0, "sparse_empty": 0, "no_doc": 0}
        for row in schema.flattened:
            name = row["flattenedName"]
            parent, leaf = _split(name)
            doc = row.get("doc", "")
            if not doc:
                counts["no_doc"] += 1
            field = fields.get(name)
            query = probe._build_query_text(field) if field is not None else name
            usable = 0
            search = probe._sparse_retriever.search(query, top_k=100)
            if search.is_success:
                hits = search.unwrap()
                usable = len({h.score for h in hits}) if hits else 0
            if structural_trigger(parent, leaf, doc, ""):
                counts["structural"] += 1
            if lexical_trigger(_tokens(query), vocabulary, doc, ""):
                counts["lexical"] += 1
            if sparse_empty_trigger(usable, doc, ""):
                counts["sparse_empty"] += 1
        triggers[schema.name] = {"fields": len(schema.flattened), **counts}

    real = _real_corpora()

    print("\n" + "=" * 78)
    print("QUESTION 1  IS THE TRIGGER EVEN DETECTABLE?")
    print("=" * 78)
    print("    corpus                fields    no doc   D1 path   D2 no lex   D3 sparse")
    print("                                              echoes    surface      empty")
    for name, t in triggers.items():
        print(
            f"    {name:<20} {t['fields']:>6}  {t['no_doc']:>8}  {t['structural']:>8}  "
            f"{t['lexical']:>10}  {t['sparse_empty']:>10}"
        )
    for name, t in real.items():
        print(
            f"    {name + ' (real)':<20} {t['fields']:>6}  {t['no_doc']:>8}  "
            f"{t['structural']:>8}  {'-':>10}  {'-':>10}"
        )
    print()
    print("  D1 is the proposal's wording read literally: no doc, no entity, and no token")
    print("  in the path that is not already in the leaf. D2 is its stated REASON -- no")
    print("  token of the query occurs anywhere in the indexed corpus, so the dense query")
    print("  is built from nothing -- and it is computable before retrieval runs. D3 is")
    print("  what D2 is trying to predict and needs retrieval to know.")

    # -- QUESTION 2: what does each mode do? --------------------------------
    print("\nIndexing the two remaining retrieval settings...")
    matchers = {"full pipeline": probe}
    for label, cfg in (
        ("sparse-only (alpha=0)", sparse_config),
        ("dense-only (alpha=1)", dense_config),
    ):
        candidate = build_matcher(cfg)
        _rep, secs = index_glossary(candidate, pack.glossary_dicts())
        matchers[label] = candidate
        print(f"  {label} indexed in {secs:.1f}s")

    arms = {label: {"obs": _observe(m, pack, NO_DOC)} for label, m in matchers.items()}
    summary = {label: _summarise(a["obs"]) for label, a in arms.items()}
    shared = sorted(
        {k for k, o in arms["full pipeline"]["obs"].items() if o.answerable}
        & {k for k, o in arms["sparse-only (alpha=0)"]["obs"].items() if o.answerable}
    )
    p1_stat = paired_compare_metric(
        "p_at_1",
        [float(arms["full pipeline"]["obs"][k].hit) for k in shared],
        [float(arms["sparse-only (alpha=0)"]["obs"][k].hit) for k in shared],
    )

    print("\n" + "=" * 78)
    print(f"QUESTION 2  THE TWO MODES ON '{NO_DOC}'")
    print("=" * 78)
    for label, s in summary.items():
        cfg = full_config if label.startswith("full") else sparse_config
        cu = s["confidence_unanswerable"]
        print(f"\n  {label}")
        print(f"    structural confidence floor       {cfg.minimum_achievable_confidence:.4f}")
        print(f"    P@1 on the {s['answerable']} answerable rows   {s['p_at_1']:.4f}")
        print(
            f"    rank-1 confidence, no-answer rows  min {cu['min']:.4f}  "
            f"median {cu['median']:.4f}  max {cu['max']:.4f}"
        )
        for bar in BARS:
            print(
                f"    ... at or above {bar:.2f}                 "
                f"{cu[f'share_at_or_above_{bar}']:.4f}  "
                f"({round(cu[f'share_at_or_above_{bar}'] * cu['n'])} of {cu['n']})"
            )
    print(
        f"\n  paired P@1, full -> sparse-only: {p1_stat.delta:+.4f}  "
        f"[{p1_stat.ci_low:+.4f}, {p1_stat.ci_high:+.4f}]  p = {p1_stat.p_value:.4g}"
    )
    print(f"  {p1_stat.test}")

    # -- QUESTION 3: does a floor already do this? --------------------------
    full_obs = arms["full pipeline"]["obs"]
    positive = [
        o.absolute1 for o in full_obs.values() if o.answerable and o.hit and o.absolute1 is not None
    ]
    negative = [
        o.absolute1 for o in full_obs.values() if not o.answerable and o.absolute1 is not None
    ]
    auc = _auc(positive, negative)
    floor = _best_floor(positive, negative)

    print("\n" + "=" * 78)
    print("QUESTION 3  WOULD `absolute_score_floor` ALREADY DO THIS?")
    print("=" * 78)
    print(f"    correct rank-1s                      {len(positive)}")
    print(f"    rank-1s where nothing is correct     {len(negative)}")
    print(f"    AUC of the raw dense score           {auc:.4f}")
    if floor:
        print(
            f"    best floor                           {floor['floor']:.4f}  "
            f"keeps {floor['kept_correct']:.2%} of the correct answers and "
            f"{floor['kept_no_answer']:.2%} of the garbage"
        )
        print(f"    Youden J at that floor               {floor['youden_j']:+.4f}")

    # -- THE DEPLOYMENT QUESTION: the policy applied where it fires ---------
    print(f"\nApplying the policy field by field on '{MIXED}'...")
    mixed_arms = {label: {"obs": _observe(m, pack, MIXED)} for label, m in matchers.items()}
    mixed_schema = pack.schema(MIXED)
    mixed_fields = {f.name: f for f in parse_fields(mixed_schema.flattened)}
    fired: list[str] = []
    for row in mixed_schema.flattened:
        name = row["flattenedName"]
        field = mixed_fields.get(name)
        query = probe._build_query_text(field) if field is not None else name
        if lexical_trigger(_tokens(query), vocabulary, row.get("doc", ""), ""):
            fired.append(name)
    fired_set = set(fired)

    policy_obs: dict[str, FieldObs] = {}
    for key, o in mixed_arms["full pipeline"]["obs"].items():
        source = "sparse-only (alpha=0)" if key in fired_set else "full pipeline"
        policy_obs[key] = mixed_arms[source]["obs"].get(key, o)

    mixed_summary = {
        "full pipeline": _summarise(mixed_arms["full pipeline"]["obs"]),
        "sparse-only everywhere": _summarise(mixed_arms["sparse-only (alpha=0)"]["obs"]),
        "policy (D2 fields sparse-only)": _summarise(policy_obs),
    }
    mixed_shared = sorted(
        {k for k, o in mixed_arms["full pipeline"]["obs"].items() if o.answerable}
        & {k for k, o in policy_obs.items() if o.answerable}
    )
    policy_stat = paired_compare_metric(
        "p_at_1",
        [float(mixed_arms["full pipeline"]["obs"][k].hit) for k in mixed_shared],
        [float(policy_obs[k].hit) for k in mixed_shared],
    )

    print("\n" + "=" * 78)
    print(f"THE DEPLOYMENT QUESTION  the policy applied only where D2 fires, on '{MIXED}'")
    print("=" * 78)
    print(
        f"    D2 fires on {len(fired)} of {len(mixed_schema.flattened)} columns "
        f"({len(fired) / len(mixed_schema.flattened):.2%})"
    )
    print("    condition                          P@1     conf>=0.63 on no-answer rows")
    for label, s in mixed_summary.items():
        cu = s["confidence_unanswerable"]
        print(
            f"    {label:<32} {s['p_at_1']:.4f}   "
            f"{cu['share_at_or_above_0.63']:.4f}  "
            f"({round(cu['share_at_or_above_0.63'] * cu['n'])} of {cu['n']})"
        )
    print(
        f"\n    paired P@1, full -> policy: {policy_stat.delta:+.4f}  "
        f"[{policy_stat.ci_low:+.4f}, {policy_stat.ci_high:+.4f}]  "
        f"p = {policy_stat.p_value:.4g}"
    )

    # -- what the policy costs and what it buys, field by field -------------
    # The two aggregate columns above are a benefit and a cost added together. Separated
    # here, because the decision to build this is a decision about the exchange rate: how
    # many confident answers on unanswerable rows does the policy withdraw, and how many
    # CORRECT auto-approvable answers does it withdraw at the same time.
    full_mixed = mixed_arms["full pipeline"]["obs"]
    withdrawn_garbage = sum(
        1
        for k, o in full_mixed.items()
        if not o.answerable and o.conf1 >= 0.63 and policy_obs[k].conf1 < 0.63
    )
    withdrawn_correct = sum(
        1
        for k, o in full_mixed.items()
        if o.answerable and o.hit and o.conf1 >= 0.63 and policy_obs[k].conf1 < 0.63
    )
    lost_correct = sum(
        1 for k, o in full_mixed.items() if o.answerable and o.hit and not policy_obs[k].hit
    )
    gained_correct = sum(
        1 for k, o in full_mixed.items() if o.answerable and not o.hit and policy_obs[k].hit
    )
    unanswerable_fired = sum(
        1 for k, o in full_mixed.items() if not o.answerable and k in fired_set
    )
    unanswerable_total = sum(1 for o in full_mixed.values() if not o.answerable)
    exchange = {
        "trigger_fires_on_unanswerable_rows": unanswerable_fired,
        "unanswerable_rows": unanswerable_total,
        "confident_answers_withdrawn_where_nothing_is_correct": withdrawn_garbage,
        "confident_answers_withdrawn_where_rank1_was_right": withdrawn_correct,
        "correct_rank1_lost": lost_correct,
        "correct_rank1_gained": gained_correct,
    }

    print()
    print("    THE EXCHANGE RATE, which is what the build decision is actually about")
    print(
        f"      the trigger fires on {unanswerable_fired} of the {unanswerable_total} rows "
        f"where nothing is correct"
    )
    print(f"      confident answers withdrawn where NOTHING is correct   {withdrawn_garbage}")
    print(f"      confident answers withdrawn where rank 1 was RIGHT     {withdrawn_correct}")
    print(
        f"      correct rank-1s lost / gained outright                 {lost_correct} / {gained_correct}"
    )
    if withdrawn_correct or lost_correct:
        ratio = withdrawn_garbage / max(withdrawn_correct + lost_correct, 1)
        print(f"      garbage withdrawn per correct answer damaged           {ratio:.1f}")

    # -- is this corpus entitled to answer this question at all? ------------
    entitlement: dict[str, dict[str, float]] = {}
    for schema_name in (NO_DOC, MIXED):
        source = arms if schema_name == NO_DOC else mixed_arms
        entitlement[schema_name] = {
            label: _summarise(a["obs"])["p_at_1"] for label, a in source.items()
        }
    dense_ahead = {
        name: row["dense-only (alpha=1)"] - row["sparse-only (alpha=0)"]
        for name, row in entitlement.items()
    }

    print("\n" + "=" * 78)
    print("IS THIS CORPUS ENTITLED TO ANSWER THE QUESTION?")
    print("=" * 78)
    print("    schema              dense-only   shipped (0.9)   sparse-only   dense - sparse")
    for name, row in entitlement.items():
        print(
            f"    {name:<18} {row['dense-only (alpha=1)']:>10.4f}   "
            f"{row['full pipeline']:>13.4f}   {row['sparse-only (alpha=0)']:>11.4f}   "
            f"{dense_ahead[name]:>+14.4f}"
        )
    print()
    print("  `benchmarks/validate_benchmark.py` states this repository's own rule: on a")
    print("  real semantic task the dense encoder beats lexical retrieval, and when it")
    print("  does not, the corpus is rewarding string overlap rather than meaning.")
    failing = [name for name, d in dense_ahead.items() if d < 0]
    passing = [name for name, d in dense_ahead.items() if d >= 0]
    print()
    if passing:
        print(f"  PASSES on: {', '.join(passing)}. There the dense arm is genuinely ahead")
        print("  of the lexical one, so switching it off is a real trade and the P@1 cost")
        print("  measured above is a cost.")
    if failing:
        print(f"  FAILS on: {', '.join(failing)}. There the LEXICAL arm is ahead, and any")
        print("  P@1 the policy appears to buy on that profile is the fixture, not the")
        print("  policy. The pack's vocabulary is manufactured by a syllable grammar, so")
        print("  its words carry no meaning a pretrained encoder has ever seen: the lexical")
        print("  arm has real tokens to match and the dense arm has an embedding of noise.")
        print("  Do not quote a P@1 gain from a profile in this list.")
    print()
    print("  WHAT SURVIVES EITHER WAY is the confidence result, because it does not depend")
    print("  on the encoder being any good. Min-max normalisation maps the best dense")
    print("  candidate to exactly 1.0 whatever its raw score was, so the full pipeline")
    print("  reports a rank-1 above the structural floor on EVERY row -- including every")
    print("  row where no correct answer exists. That is arithmetic, and it is the")
    print("  mechanism the proposal is actually about.")

    artifact = {
        "experiment": "exp_mode_selection (E6) -- should the pipeline learn to decline?",
        "corpus_entitlement": {
            "p_at_1_by_retrieval_arm": entitlement,
            "dense_minus_sparse": dense_ahead,
        },
        "provenance": provenance(spec.seed, full_config.__dict__.copy()),
        "pack": pack.manifest(),
        "indexed_entries": report.entries,
        "corpus_vocabulary_tokens": len(vocabulary),
        "triggers_synthetic": triggers,
        "triggers_real": real,
        "no_doc_profile": {
            "schema": NO_DOC,
            "arms": summary,
            "floors": {
                "full pipeline": full_config.minimum_achievable_confidence,
                "sparse-only (alpha=0)": sparse_config.minimum_achievable_confidence,
            },
            "paired_p_at_1": {
                "test": p1_stat.test,
                "delta": p1_stat.delta,
                "ci_low": p1_stat.ci_low,
                "ci_high": p1_stat.ci_high,
                "p_value": p1_stat.p_value,
                "n_pairs": p1_stat.n_pairs,
                "n_discordant": p1_stat.n_discordant,
            },
        },
        "absolute_score_floor": {
            "n_correct": len(positive),
            "n_no_answer": len(negative),
            "auc": auc,
            "best": floor,
        },
        "policy_on_mixed": {
            "schema": MIXED,
            "columns": len(mixed_schema.flattened),
            "trigger_fired": len(fired),
            "conditions": mixed_summary,
            "exchange_rate": exchange,
            "paired_p_at_1": {
                "test": policy_stat.test,
                "delta": policy_stat.delta,
                "ci_low": policy_stat.ci_low,
                "ci_high": policy_stat.ci_high,
                "p_value": policy_stat.p_value,
                "n_pairs": policy_stat.n_pairs,
                "n_discordant": policy_stat.n_discordant,
            },
        },
    }
    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"exp_mode_selection_synthetic_{spec.rows}.json"
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nSaved -> {path}")
    return 0


def _real_corpora() -> dict[str, dict[str, Any]]:
    """
    D1 counted on the committed benchmarks, which is where the honest answer lives.

    D2 and D3 are not computed here: both are statements about a query against an INDEX,
    and building an index per corpus for a count is a different experiment. D1 is the
    proposal's own wording and needs nothing but the query records.
    """
    from eval_harness import Dataset

    out: dict[str, dict[str, Any]] = {}
    for name in ("fhir", "combined", "bird", "omop"):
        try:
            ds = Dataset.load(name)
        except SystemExit:
            continue
        no_doc = [q for q in ds.queries if not q.doc]
        fires = [q for q in no_doc if _tokens(q.parent_path) <= _tokens(q.field_name)]
        out[name] = {
            "fields": len(ds.queries),
            "no_doc": len(no_doc),
            "structural": len(fires),
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
