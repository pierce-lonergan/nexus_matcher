"""
benchmarks.exp_rerank | Layer: BENCHMARK
Quantify the accuracy headroom from reranking a first-stage shortlist.

The first-stage evaluation showed Recall@50 = 0.90 against P@1 = 0.49 on the combined
benchmark -- a 41-point gap. That gap is exactly what a reranker can recover. This script
measures how much of it each candidate cross-encoder actually recovers, and at what cost.
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

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

# Query representation used for both stages. `context` (parent table + split identifier)
# measured best in exp_query_repr: P@1 0.691 vs 0.491 for the bare field name.
QUERY_REPR = "context"


def _query_text(q) -> str:
    return VARIANTS[QUERY_REPR](q)


def first_stage(ds: Dataset, index_mode: str, top_k: int, model_name: str):
    """Dense retrieval with the BGE query prefix -- the strongest first stage we measured."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    corpus = [e.searchable_text(index_mode) for e in ds.entries]
    ids = [e.id for e in ds.entries]

    doc_emb = model.encode(
        corpus, batch_size=128, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    q_emb = model.encode(
        [BGE_PREFIX + _query_text(q) for q in ds.queries],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    sims = q_emb @ doc_emb.T
    k = min(top_k, len(ids))
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    shortlists = []
    for row, cand in zip(sims, part, strict=False):
        order = cand[np.argsort(-row[cand])]
        shortlists.append([(ids[i], float(row[i]), corpus[i]) for i in order])
    return shortlists


def rerank(ds: Dataset, shortlists, model_name: str, depth: int, batch_size: int = 256):
    """Cross-encoder rerank of the top `depth` candidates per query."""
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model_name, device="cpu", max_length=256)

    pairs: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for q, sl in zip(ds.queries, shortlists, strict=False):
        qt = _query_text(q)
        start = len(pairs)
        for _id, _s, text in sl[:depth]:
            pairs.append((qt, text))
        spans.append((start, len(pairs)))

    t0 = time.perf_counter()
    scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    elapsed = time.perf_counter() - t0

    rankings = []
    for (start, end), sl in zip(spans, shortlists, strict=False):
        seg = scores[start:end]
        order = np.argsort(-np.asarray(seg))
        head = [sl[i][0] for i in order]
        tail = [c[0] for c in sl[depth:]]
        rankings.append(head + tail)
    return rankings, elapsed, len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="combined")
    ap.add_argument("--index-fields", default="business+desc")
    ap.add_argument("--first-stage-k", type=int, default=50)
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument(
        "--rerankers",
        nargs="*",
        default=[
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "BAAI/bge-reranker-base",
        ],
    )
    args = ap.parse_args()

    ds = Dataset.load(args.benchmark, limit=args.limit)
    golds = [q.gold_id for q in ds.queries]
    print(f"\nBenchmark {ds.name}: {len(ds.entries)} entries, {len(ds.queries)} queries")

    t0 = time.perf_counter()
    shortlists = first_stage(ds, args.index_fields, args.first_stage_k, args.embed_model)
    fs_time = time.perf_counter() - t0

    base_rank = [[c[0] for c in sl] for sl in shortlists]
    base = score_rankings(base_rank, golds)
    print(
        f"\n  first stage (dense+prefix, k={args.first_stage_k}, {fs_time:.1f}s)\n"
        f"    P@1={base['p_at_1']:.4f}  P@5={base['p_at_5']:.4f}  "
        f"MRR@10={base['mrr_at_10']:.4f}  R@50={base['recall'][50]:.4f}"
    )
    ceiling = base["recall"][min(args.depth, 50)]
    print(f"    reranker ceiling at depth {args.depth}: P@1 <= {ceiling:.4f}")

    out = {
        "benchmark": ds.name,
        "index_fields": args.index_fields,
        "first_stage": {k: v for k, v in base.items() if k != "recall"},
        "ceiling": ceiling,
        "rerankers": {},
    }

    for name in args.rerankers:
        try:
            ranks, elapsed, n_pairs = rerank(ds, shortlists, name, args.depth)
        except Exception as exc:
            print(f"\n  {name}: FAILED -> {type(exc).__name__}: {str(exc)[:200]}")
            continue
        s = score_rankings(ranks, golds)
        gain = s["p_at_1"] - base["p_at_1"]
        recovered = gain / (ceiling - base["p_at_1"]) if ceiling > base["p_at_1"] else 0.0
        qps = len(ds.queries) / elapsed
        print(
            f"\n  {name}\n"
            f"    P@1={s['p_at_1']:.4f} ({gain:+.4f})  P@5={s['p_at_5']:.4f}  "
            f"MRR@10={s['mrr_at_10']:.4f}\n"
            f"    recovered {recovered:.1%} of the headroom\n"
            f"    {elapsed:.1f}s for {n_pairs} pairs -> {qps:.1f} queries/sec "
            f"({n_pairs / elapsed:.0f} pairs/sec)"
        )
        out["rerankers"][name] = {
            "p_at_1": s["p_at_1"],
            "p_at_5": s["p_at_5"],
            "mrr_at_10": s["mrr_at_10"],
            "gain_p_at_1": gain,
            "headroom_recovered": recovered,
            "queries_per_sec": qps,
            "seconds": elapsed,
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"exp_rerank_{ds.name}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
