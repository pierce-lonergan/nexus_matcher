"""
benchmarks.exp_instance_signal | Layer: BENCHMARK
Instance values as a SEPARATE scoring signal, not as query text.

Why this exists
---------------
exp_instance.py tested the obvious thing -- append sample values to the query string --
and every variant LOST accuracy (bird P@1 0.5651 -> 0.5346 at k=10, and -0.0637 for the
full profile). The mechanism is dilution, the same failure as appending type words: the
dictionary side is a SHORT business name ("County Name") while the enriched query became
long and semantically dominated by its own contents ("Los Angeles, Alameda, Orange..."),
so cosine similarity drifted toward the values and away from the label.

That result condemns the representation, not the signal. This script keeps the primary
query embedding clean and scores values SEPARATELY:

    sim_name   = cos(embed(context + field name),  embed(business name + description))
    sim_values = cos(embed(sample values),         embed(business name + description))
    score      = (1 - w) * sim_name + w * sim_values

Sweeping w from 0 (current behaviour) upward answers the real question: does the value
signal carry ANY independent information about the right dictionary entry? If the best w
is 0, instance data genuinely does not help this task at this scale and we stop.

A second variant tests a targeted rule rather than a global weight: use values only where
the NAME is uninformative (very short or unsplittable), which is exactly where the
abbreviation ceiling bites.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import RESULTS, Dataset, score_rankings
from exp_instance import fmt_values, load_instances
from exp_query_repr import VARIANTS, split_identifier

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def rank_from_sims(sims: np.ndarray, ids: list[str], top_k: int) -> list[list[str]]:
    k = min(top_k, len(ids))
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    out = []
    for row, cand in zip(sims, part, strict=False):
        order = cand[np.argsort(-row[cand])]
        out.append([ids[i] for i in order])
    return out


def name_is_opaque(q) -> bool:
    """
    Heuristic for 'the name tells you nothing'.

    True when the identifier splits into few, short tokens -- "sname", "cds", "enroll12"
    -- which is precisely the population where the abbreviation ceiling lives.
    """
    toks = split_identifier(q.field_name)
    if not toks:
        return True
    longest = max(len(t) for t in toks)
    return len(toks) <= 2 and longest <= 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="bird")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--index-fields", default="business+desc")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--value-k", type=int, default=10)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    ds = Dataset.load(args.benchmark)
    instances = load_instances(args.benchmark)
    if not instances:
        raise SystemExit("No instance data; run benchmarks/datasets/extract_instances.py")

    model = SentenceTransformer(args.model, device="cpu")
    ids = [e.id for e in ds.entries]
    golds = [q.gold_id for q in ds.queries]

    doc_emb = model.encode(
        [e.searchable_text(args.index_fields) for e in ds.entries],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    name_texts = [BGE_PREFIX + VARIANTS["context"](q) for q in ds.queries]
    name_emb = model.encode(
        name_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)

    # Value-only text. Empty when there is nothing useful (numeric or missing), and those
    # rows fall back to the name similarity so they are never penalised.
    value_texts, has_values = [], []
    for q in ds.queries:
        inst = instances.get(q.gold_id)
        v = ""
        if inst and not inst.get("is_numeric") and inst.get("sample_values"):
            v = fmt_values(inst["sample_values"], args.value_k)
        has_values.append(bool(v))
        value_texts.append(BGE_PREFIX + (v or "empty"))
    value_emb = model.encode(
        value_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)

    sim_name = name_emb @ doc_emb.T
    sim_values = value_emb @ doc_emb.T
    mask = np.array(has_values, dtype=bool)

    n_vals = int(mask.sum())
    print(
        f"\nBenchmark {ds.name}: {len(ds.queries)} queries, "
        f"{n_vals} with usable text values ({n_vals / len(ds.queries):.0%})"
    )

    base = score_rankings(rank_from_sims(sim_name, ids, args.top_k), golds, ks=(1, 5, 10))
    print(f"\n  baseline (name only)  P@1 {base['p_at_1']:.4f}  MRR@10 {base['mrr_at_10']:.4f}")

    # Is the value signal informative ON ITS OWN? If P@1 here is near zero the signal is
    # simply absent and no weighting can rescue it.
    only = score_rankings(rank_from_sims(sim_values, ids, args.top_k), golds, ks=(1, 5, 10))
    print(f"  values only           P@1 {only['p_at_1']:.4f}  MRR@10 {only['mrr_at_10']:.4f}")

    print(f"\n  {'weight w':>9s} {'P@1':>7s} {'P@5':>7s} {'MRR@10':>7s}  {'delta':>8s}")
    print(f"  {'-' * 9} {'-' * 7} {'-' * 7} {'-' * 7}  {'-' * 8}")

    out: dict[str, dict] = {
        "baseline": {k: v for k, v in base.items() if k != "recall"},
        "values_only": {k: v for k, v in only.items() if k != "recall"},
    }
    best = (0.0, base["p_at_1"])

    for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        blend = sim_name.copy()
        blend[mask] = (1 - w) * sim_name[mask] + w * sim_values[mask]
        s = score_rankings(rank_from_sims(blend, ids, args.top_k), golds, ks=(1, 5, 10))
        d = s["p_at_1"] - base["p_at_1"]
        print(f"  {w:9.2f} {s['p_at_1']:7.4f} {s['p_at_5']:7.4f} {s['mrr_at_10']:7.4f}  {d:+8.4f}")
        out[f"w={w}"] = {k: v for k, v in s.items() if k != "recall"}
        if s["p_at_1"] > best[1]:
            best = (w, s["p_at_1"])

    # Targeted variant: apply the value signal ONLY where the name is opaque.
    opaque = np.array([name_is_opaque(q) for q in ds.queries], dtype=bool) & mask
    print(f"\n  Targeted: values only for OPAQUE names ({int(opaque.sum())} queries)")
    print(f"  {'weight w':>9s} {'P@1':>7s} {'MRR@10':>7s}  {'delta':>8s}")
    for w in (0.15, 0.30, 0.50, 0.70):
        blend = sim_name.copy()
        blend[opaque] = (1 - w) * sim_name[opaque] + w * sim_values[opaque]
        s = score_rankings(rank_from_sims(blend, ids, args.top_k), golds, ks=(1, 5, 10))
        d = s["p_at_1"] - base["p_at_1"]
        print(f"  {w:9.2f} {s['p_at_1']:7.4f} {s['mrr_at_10']:7.4f}  {d:+8.4f}")
        out[f"opaque_w={w}"] = {k: v for k, v in s.items() if k != "recall"}
        if s["p_at_1"] > best[1]:
            best = (f"opaque_w={w}", s["p_at_1"])

    print(f"\n  BEST: {best[0]}  P@1 {best[1]:.4f}  ({best[1] - base['p_at_1']:+.4f} vs baseline)")
    if best[1] <= base["p_at_1"] + 1e-9:
        print("  => The value signal adds NOTHING at any weight. Instance data does not")
        print("     help this asymmetric name-to-glossary task. Stop here.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"exp_instance_signal_{ds.name}.json"
    p.write_text(
        json.dumps({"benchmark": ds.name, "variants": out, "best": str(best[0])}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
