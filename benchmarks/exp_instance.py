"""
benchmarks.exp_instance | Layer: BENCHMARK
Does using the column's DATA VALUES break the abbreviation ceiling?

The gap that motivates this: bird P@1 0.598 vs omop 0.831. BIRD column names are opaque
("sname", "NumTstTakr"), OMOP's are descriptive. But BIRD ships the real databases, and
"sname" holds "Middle College High", "John F. Kennedy High" -- which makes the column's
meaning obvious to a reader in a way the name never does.

This is an ASYMMETRIC signal and that is the whole design point: the SOURCE schema has
data, the business glossary does not. So values enrich the QUERY side only; the dictionary
is still indexed on business_name + description alone. No leakage: the values are the
column's contents, not its label.

Expectation worth stating BEFORE measuring, so the result can falsify it: values should
help TEXT/categorical columns a lot and numeric columns barely at all, because "0, 1, 2, 3"
says nothing about whether a column counts test takers or enrolled students.

Variants
--------
baseline        context + field name (current production representation)
values_k        + the k most frequent values
text_only_k     + values, but ONLY for non-numeric columns
stats           + numeric min/max/mean instead of raw numeric values
profile         + cardinality / null-rate / pattern signature
best            the winning combination
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import RESULTS, Dataset, score_rankings
from exp_query_repr import VARIANTS

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def load_instances(benchmark: str) -> dict[str, dict]:
    p = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / benchmark / "instances.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def fmt_values(vals: list[str], k: int) -> str:
    """Comma-joined sample. Order is frequency-descending from the profiler."""
    picked = [v.strip() for v in vals[:k] if v and v.strip()]
    return ", ".join(picked)


def build_query(q, inst: dict | None, mode: str, k: int) -> str:
    """Compose the query text for a given variant."""
    base = VARIANTS["context"](q)
    if inst is None or mode == "baseline":
        return base

    is_numeric = inst.get("is_numeric", False)
    vals = inst.get("sample_values") or []

    if mode == "values":
        v = fmt_values(vals, k)
        return f"{base}. Example values: {v}" if v else base

    if mode == "text_only":
        if is_numeric:
            return base
        v = fmt_values(vals, k)
        return f"{base}. Example values: {v}" if v else base

    if mode == "stats":
        # Values for text columns, numeric summary for numeric ones.
        if is_numeric:
            ns = inst.get("numeric_stats")
            if not ns:
                return base
            return (
                f"{base}. Numeric column ranging from {ns['min']:g} to {ns['max']:g}, "
                f"averaging {ns['mean']:.1f}"
            )
        v = fmt_values(vals, k)
        return f"{base}. Example values: {v}" if v else base

    if mode == "profile":
        parts = [base]
        if not is_numeric:
            v = fmt_values(vals, k)
            if v:
                parts.append(f"Example values: {v}")
        else:
            ns = inst.get("numeric_stats")
            if ns:
                parts.append(f"Numeric from {ns['min']:g} to {ns['max']:g}")
        parts.append(f"{inst.get('n_distinct', 0)} distinct values")
        return ". ".join(parts)

    raise ValueError(f"unknown mode {mode}")


def evaluate(ds: Dataset, instances, model, doc_emb, ids, mode: str, k: int, top_k: int):
    texts = []
    for q in ds.queries:
        # Instance data is keyed by the dictionary id, which for these benchmarks is the
        # same column the query refers to -- it is the SOURCE column's own data.
        texts.append(build_query(q, instances.get(q.gold_id), mode, k))

    q_emb = model.encode(
        [BGE_PREFIX + t for t in texts],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    sims = q_emb @ doc_emb.T
    kk = min(top_k, len(ids))
    part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
    rankings = []
    for row, cand in zip(sims, part, strict=False):
        order = cand[np.argsort(-row[cand])]
        rankings.append([ids[i] for i in order])

    golds = [q.gold_id for q in ds.queries]
    return score_rankings(rankings, golds, ks=(1, 5, 10)), texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="bird")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--index-fields", default="business+desc")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--ks", nargs="*", type=int, default=[3, 5, 10, 25])
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    ds = Dataset.load(args.benchmark)
    instances = load_instances(args.benchmark)
    if not instances:
        raise SystemExit(
            f"No instance data for '{args.benchmark}'. "
            f"Run: python benchmarks/datasets/extract_instances.py"
        )

    covered = sum(1 for q in ds.queries if q.gold_id in instances)
    print(
        f"\nBenchmark {ds.name}: {len(ds.queries)} queries, "
        f"{covered} ({covered / len(ds.queries):.0%}) with instance data"
    )

    n_num = sum(1 for q in ds.queries if instances.get(q.gold_id, {}).get("is_numeric"))
    print(f"  numeric columns: {n_num}   text columns: {len(ds.queries) - n_num}\n")

    model = SentenceTransformer(args.model, device="cpu")
    corpus = [e.searchable_text(args.index_fields) for e in ds.entries]
    ids = [e.id for e in ds.entries]
    doc_emb = model.encode(
        corpus, batch_size=128, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)

    print(
        f"  {'variant':18s} {'P@1':>7s} {'P@5':>7s} {'MRR@10':>7s} {'R@10':>7s}  {'delta P@1':>10s}"
    )
    print(f"  {'-' * 18} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}  {'-' * 10}")

    out: dict[str, dict] = {}
    base_s, _ = evaluate(ds, instances, model, doc_emb, ids, "baseline", 0, args.top_k)
    base_p1 = base_s["p_at_1"]
    print(
        f"  {'baseline':18s} {base_p1:7.4f} {base_s['p_at_5']:7.4f} "
        f"{base_s['mrr_at_10']:7.4f} {base_s['recall'][10]:7.4f}  {'--':>10s}"
    )
    out["baseline"] = {k: v for k, v in base_s.items() if k != "recall"}

    runs = [("values", k) for k in args.ks]
    runs += [("text_only", k) for k in args.ks]
    runs += [("stats", 10), ("profile", 10)]

    best = ("baseline", base_p1)
    for mode, k in runs:
        s, texts = evaluate(ds, instances, model, doc_emb, ids, mode, k, args.top_k)
        label = f"{mode}_k{k}" if mode in ("values", "text_only") else mode
        delta = s["p_at_1"] - base_p1
        print(
            f"  {label:18s} {s['p_at_1']:7.4f} {s['p_at_5']:7.4f} "
            f"{s['mrr_at_10']:7.4f} {s['recall'][10]:7.4f}  {delta:+10.4f}"
        )
        out[label] = {k2: v for k2, v in s.items() if k2 != "recall"}
        if s["p_at_1"] > best[1]:
            best = (label, s["p_at_1"])

    print(f"\n  BEST: {best[0]} at P@1 {best[1]:.4f} ({best[1] - base_p1:+.4f} vs baseline)")

    # Split the winner by column kind -- the hypothesis is that the gain is concentrated
    # in text columns. If it is not, the mechanism is not what we think it is.
    mode, k = ("values", 10)
    _s_all, texts = evaluate(ds, instances, model, doc_emb, ids, mode, k, args.top_k)
    [q.gold_id for q in ds.queries]
    idx_num = [
        i for i, q in enumerate(ds.queries) if instances.get(q.gold_id, {}).get("is_numeric")
    ]
    idx_txt = [i for i in range(len(ds.queries)) if i not in set(idx_num)]

    def subset_p1(indices, mode_, k_):
        sub_ds = Dataset(name=ds.name, entries=ds.entries, queries=[ds.queries[i] for i in indices])
        s, _ = evaluate(sub_ds, instances, model, doc_emb, ids, mode_, k_, args.top_k)
        return s["p_at_1"]

    print("\n  Gain split by column kind (values_k10):")
    for label, idxs in (("text columns", idx_txt), ("numeric columns", idx_num)):
        if not idxs:
            continue
        b = subset_p1(idxs, "baseline", 0)
        v = subset_p1(idxs, "values", 10)
        print(
            f"    {label:16s} n={len(idxs):4d}  baseline {b:.4f} -> values {v:.4f}  ({v - b:+.4f})"
        )

    print("\n  Example enriched queries:")
    for i in (0, 1, 2):
        if i < len(texts):
            print(f"    {texts[i][:150]}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"exp_instance_{ds.name}.json"
    p.write_text(
        json.dumps({"benchmark": ds.name, "variants": out, "best": best[0]}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
