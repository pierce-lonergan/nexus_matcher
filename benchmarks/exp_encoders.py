"""
benchmarks.exp_encoders | Layer: BENCHMARK
Does a different encoder break the ceiling? A head-to-head on OUR task.

Motivation
----------
Three techniques that work elsewhere have now failed here: instance values (-3.1),
cross-encoder reranking (-1.3), query-side expansion (-2.0). The common thread is that
all of them are trained or designed for natural language, while our query is
"satscores sname". That suggests the encoder itself may be the binding constraint.

MTEB rank is a poor guide for this: it is dominated by long natural-language retrieval.
A model that wins MTEB may lose on three-token opaque identifiers, and there is no way to
know without measuring. So this script measures.

Each model is scored on the SAME corrected benchmark with the same query representation
(parent-path context) and the model's own recommended query prefix, since BGE-style models
lose ~5 points without one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import RESULTS, Dataset, score_rankings
from exp_query_repr import VARIANTS

# Each model's documented query-side instruction. Getting this wrong is worth several
# points, so it is part of the model definition rather than a global constant.
MODELS: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
    "intfloat/e5-small-v2": "query: ",
    "intfloat/e5-base-v2": "query: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "thenlper/gte-small": "",
    "mixedbread-ai/mxbai-embed-large-v1": "Represent this sentence for searching relevant passages: ",
}

# Document-side prefix, where the model uses an asymmetric scheme.
DOC_PREFIX = {
    "intfloat/e5-small-v2": "passage: ",
    "intfloat/e5-base-v2": "passage: ",
}


def evaluate(model_name: str, ds: Dataset, index_mode: str, top_k: int) -> dict | None:
    from sentence_transformers import SentenceTransformer

    q_prefix = MODELS.get(model_name, "")
    d_prefix = DOC_PREFIX.get(model_name, "")

    try:
        t0 = time.perf_counter()
        model = SentenceTransformer(model_name, device="cpu")
        load_s = time.perf_counter() - t0
    except Exception as exc:
        print(f"  {model_name:45s} LOAD FAILED: {type(exc).__name__}: {str(exc)[:70]}")
        return None

    ids = [e.id for e in ds.entries]
    golds = [q.gold_id for q in ds.queries]

    t0 = time.perf_counter()
    doc_emb = model.encode(
        [d_prefix + e.searchable_text(index_mode) for e in ds.entries],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    index_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_emb = model.encode(
        [q_prefix + VARIANTS["context"](q) for q in ds.queries],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    sims = q_emb @ doc_emb.T
    k = min(top_k, len(ids))
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rankings = []
    for row, cand in zip(sims, part, strict=False):
        order = cand[np.argsort(-row[cand])]
        rankings.append([ids[i] for i in order])
    query_s = time.perf_counter() - t0

    s = score_rankings(rankings, golds, ks=(1, 5, 10))
    return {
        "p_at_1": s["p_at_1"],
        "p_at_5": s["p_at_5"],
        "mrr_at_10": s["mrr_at_10"],
        "recall_at_10": s["recall"][10],
        "dim": int(doc_emb.shape[1]),
        "queries_per_sec": len(ds.queries) / query_s if query_s else 0.0,
        "index_seconds": index_s,
        "load_seconds": load_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="combined")
    ap.add_argument("--index-fields", default="business+desc")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()

    ds = Dataset.load(args.benchmark)
    print(
        f"\nEncoder sweep on '{ds.name}' ({len(ds.entries)} entries, {len(ds.queries)} queries)\n"
    )
    print(
        f"  {'model':45s} {'dim':>5s} {'P@1':>7s} {'P@5':>7s} {'MRR@10':>7s} "
        f"{'R@10':>7s} {'q/sec':>8s}"
    )
    print(f"  {'-' * 45} {'-' * 5} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8}")

    out: dict[str, dict] = {}
    for name in args.models:
        r = evaluate(name, ds, args.index_fields, args.top_k)
        if r is None:
            continue
        out[name] = r
        print(
            f"  {name:45s} {r['dim']:5d} {r['p_at_1']:7.4f} {r['p_at_5']:7.4f} "
            f"{r['mrr_at_10']:7.4f} {r['recall_at_10']:7.4f} {r['queries_per_sec']:8.1f}"
        )

    if out:
        best = max(out.items(), key=lambda kv: kv[1]["p_at_1"])
        base = out.get("BAAI/bge-small-en-v1.5")
        print(f"\n  BEST: {best[0]}  P@1 {best[1]['p_at_1']:.4f}")
        if base:
            print(f"  vs current default (bge-small): {best[1]['p_at_1'] - base['p_at_1']:+.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"exp_encoders_{ds.name}.json"
    p.write_text(json.dumps({"benchmark": ds.name, "models": out}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
