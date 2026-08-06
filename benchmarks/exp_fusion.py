"""
benchmarks.exp_fusion | Layer: BENCHMARK
Choose the hybrid fusion method by measurement rather than by reputation.

RRF (Cormack et al., SIGIR 2009) is the conventional default and is what the repo's
docstrings recommend, but RRF discards score MAGNITUDE and keeps only rank. When the
dense retriever is confident and correct at rank 1 while BM25 is confidently wrong, RRF
happily averages the two ranks and demotes the right answer. Whether that matters is an
empirical question about this corpus, not a matter of citation.

This script sweeps every fusion method the codebase implements, plus the RRF k constant,
on the labelled benchmark so the default can be chosen from data.
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

from nexus_matcher.core.fusion import (
    FusionConfig,
    FusionMethod,
    HybridFuser,
)

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def retrieve(ds: Dataset, index_mode: str, model_name: str, top_k: int):
    """Return per-query (dense, sparse) ranked (id, score) lists."""
    import bm25s
    from sentence_transformers import SentenceTransformer

    ids = [e.id for e in ds.entries]
    corpus = [e.searchable_text(index_mode) for e in ds.entries]
    texts = [VARIANTS["context"](q) for q in ds.queries]

    model = SentenceTransformer(model_name, device="cpu")
    doc_emb = model.encode(
        corpus, batch_size=128, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    q_emb = model.encode(
        [BGE_PREFIX + t for t in texts],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    sims = q_emb @ doc_emb.T
    k = min(top_k, len(ids))
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    dense = []
    for row, cand in zip(sims, part, strict=False):
        order = cand[np.argsort(-row[cand])]
        dense.append([(ids[i], float(row[i])) for i in order])

    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)
    docs, scores = retriever.retrieve(
        bm25s.tokenize(texts, show_progress=False), k=k, show_progress=False
    )
    sparse = [
        [(ids[j], float(sc)) for j, sc in zip(drow, srow, strict=False)]
        for drow, srow in zip(docs, scores, strict=False)
    ]
    return dense, sparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="combined")
    ap.add_argument("--index-fields", default="business+desc")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    ds = Dataset.load(args.benchmark)
    golds = [q.gold_id for q in ds.queries]
    dense, sparse = retrieve(ds, args.index_fields, args.model, args.top_k)

    print(f"\nFusion sweep on '{ds.name}' ({len(ds.queries)} queries)\n")
    print(f"  {'method':22s} {'P@1':>7s} {'P@5':>7s} {'MRR@10':>7s} {'R@10':>7s}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")

    out: dict[str, dict] = {}

    def record(label: str, rankings):
        s = score_rankings(rankings, golds, ks=(1, 5, 10))
        print(
            f"  {label:22s} {s['p_at_1']:7.4f} {s['p_at_5']:7.4f} "
            f"{s['mrr_at_10']:7.4f} {s['recall'][10]:7.4f}"
        )
        out[label] = {
            "p_at_1": s["p_at_1"],
            "p_at_5": s["p_at_5"],
            "mrr_at_10": s["mrr_at_10"],
            "recall_at_10": s["recall"][10],
        }

    # Single-arm baselines for reference.
    record("dense only", [[i for i, _ in d] for d in dense])
    record("sparse only", [[i for i, _ in s] for s in sparse])

    # Every implemented fusion method.
    for method in FusionMethod:
        fuser = HybridFuser(config=FusionConfig(method=method))
        rankings = [
            [x.id for x in fuser.fuse(d, s, top_k=args.top_k)]
            for d, s in zip(dense, sparse, strict=False)
        ]
        record(f"fusion:{method.value}", rankings)

    # RRF is famously sensitive to k on short candidate lists; sweep it.
    for k in (10, 20, 60, 200):
        fuser = HybridFuser(config=FusionConfig(method=FusionMethod.RRF, rrf_k=k))
        rankings = [
            [x.id for x in fuser.fuse(d, s, top_k=args.top_k)]
            for d, s in zip(dense, sparse, strict=False)
        ]
        record(f"rrf k={k}", rankings)

    # Weighted linear over min-max normalized scores, sweeping the dense weight.
    for w in (0.5, 0.7, 0.8, 0.9):
        fuser = HybridFuser(
            config=FusionConfig(method=FusionMethod.LINEAR, semantic_weight=w, lexical_weight=1 - w)
        )
        rankings = [
            [x.id for x in fuser.fuse(d, s, top_k=args.top_k)]
            for d, s in zip(dense, sparse, strict=False)
        ]
        record(f"linear dense={w}", rankings)

    best = max(out.items(), key=lambda kv: kv[1]["p_at_1"])
    print(f"\n  BEST by P@1: {best[0]}  ({best[1]['p_at_1']:.4f})")

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"exp_fusion_{ds.name}.json"
    p.write_text(
        json.dumps({"benchmark": ds.name, "results": out, "best": best[0]}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
