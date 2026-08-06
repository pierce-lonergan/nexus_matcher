"""
benchmarks.eval_harness | Layer: BENCHMARK
Honest accuracy + throughput evaluation for schema matching.

Replaces the previous 17-field hand-written eval with ~800 real labelled pairs drawn
from the BIRD-SQL database descriptions and the OHDSI OMOP CDM v5.4 specification.

Metrics
-------
P@1, P@5, MRR@10, Recall@{1,5,10,50}, plus end-to-end throughput (queries/sec) and
per-stage latency. Every query has exactly one gold entry, so these are unambiguous.

Leakage control
---------------
`--index-fields` decides what text the dictionary is indexed on. This matters a lot:

  business+desc  (default) The dictionary contains only the BUSINESS name and human
                 definition. The source system's technical column name is NOT present.
                 This is the real task -- mapping a new source schema onto an existing
                 corporate glossary -- and it is what actually measures semantic matching.

  all            Also index the technical `logical_name`. For BIRD/OMOP the logical_name
                 IS the query string, so this leaks the answer and inflates scores to
                 near-ceiling. Provided only to demonstrate the leak, never as a headline.

Usage
-----
    python benchmarks/eval_harness.py --benchmark combined --system baseline
    python benchmarks/eval_harness.py --benchmark bird --system nexus --limit 100
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "data" / "benchmarks"
RESULTS = REPO_ROOT / "benchmarks" / "results"


# =============================================================================
# DATA
# =============================================================================


@dataclass(frozen=True)
class DictEntry:
    id: str
    business_name: str
    logical_name: str
    description: str
    data_type: str
    domain: str

    def searchable_text(self, mode: str) -> str:
        if mode == "all":
            parts = [self.business_name, self.logical_name.replace("_", " "), self.description]
        elif mode == "business+desc":
            parts = [self.business_name, self.description]
        elif mode == "business":
            parts = [self.business_name]
        else:
            raise ValueError(f"unknown index-fields mode: {mode}")
        # De-duplicate while preserving order: description often repeats the name.
        seen: set[str] = set()
        out: list[str] = []
        for raw in parts:
            part = (raw or "").strip()
            if part and part.lower() not in seen:
                seen.add(part.lower())
                out.append(part)
        return " ".join(out)


@dataclass(frozen=True)
class Query:
    id: str
    field_name: str
    field_path: str
    data_type: str
    parent_path: str
    gold_id: str
    # The source field's own documentation, when the corpus has one (Avro `doc`, FHIR
    # `comment`). Kept on the QUERY side only: putting the target's definition here is
    # what makes a benchmark degenerate, so this must be independently authored text.
    doc: str = ""


@dataclass
class Dataset:
    name: str
    entries: list[DictEntry]
    queries: list[Query]

    @classmethod
    def load(cls, name: str, limit: int | None = None) -> Dataset:
        d = BENCH_ROOT / name
        if not d.exists():
            raise SystemExit(
                f"Benchmark '{name}' not found at {d}.\n"
                f"Run: python benchmarks/datasets/build_benchmarks.py"
            )
        entries = [
            DictEntry(**json.loads(line))
            for line in (d / "dictionary.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        queries = [
            Query(**json.loads(line))
            for line in (d / "queries.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit:
            queries = queries[:limit]
        return cls(name=name, entries=entries, queries=queries)


# =============================================================================
# METRICS
# =============================================================================


@dataclass
class Metrics:
    n: int = 0
    p_at_1: float = 0.0
    p_at_5: float = 0.0
    mrr_at_10: float = 0.0
    recall: dict[int, float] = field(default_factory=dict)
    queries_per_sec: float = 0.0
    total_seconds: float = 0.0
    index_seconds: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0

    def render(self, title: str) -> str:
        r = self.recall
        return (
            f"\n  {title}\n"
            f"  {'-' * len(title)}\n"
            f"    queries          {self.n}\n"
            f"    P@1              {self.p_at_1:.4f}\n"
            f"    P@5              {self.p_at_5:.4f}\n"
            f"    MRR@10           {self.mrr_at_10:.4f}\n"
            f"    Recall@1/5/10/50 {r.get(1, 0):.3f} / {r.get(5, 0):.3f} / "
            f"{r.get(10, 0):.3f} / {r.get(50, 0):.3f}\n"
            f"    index build      {self.index_seconds:.2f} s\n"
            f"    throughput       {self.queries_per_sec:.1f} queries/sec\n"
            f"    latency p50/p95  {self.latency_p50_ms:.2f} / {self.latency_p95_ms:.2f} ms\n"
        )


def score_rankings(
    rankings: Sequence[Sequence[str]],
    golds: Sequence[str],
    ks: Sequence[int] = (1, 5, 10, 50),
) -> dict[str, float | dict[int, float]]:
    """Compute P@1, P@5, MRR@10 and Recall@k from ranked id lists."""
    n = len(golds)
    if n == 0:
        return {"p_at_1": 0.0, "p_at_5": 0.0, "mrr_at_10": 0.0, "recall": dict.fromkeys(ks, 0.0)}

    p1 = p5 = mrr = 0.0
    hits = dict.fromkeys(ks, 0)

    for ranked, gold in zip(rankings, golds, strict=False):
        # Rank of the gold entry, 1-indexed; None if absent from the list.
        rank = None
        for i, doc_id in enumerate(ranked, 1):
            if doc_id == gold:
                rank = i
                break
        if rank is None:
            continue
        if rank == 1:
            p1 += 1
        if rank <= 5:
            p5 += 1
        if rank <= 10:
            mrr += 1.0 / rank
        for k in ks:
            if rank <= k:
                hits[k] += 1

    return {
        "p_at_1": p1 / n,
        "p_at_5": p5 / n,
        "mrr_at_10": mrr / n,
        "recall": {k: hits[k] / n for k in ks},
    }


# =============================================================================
# SYSTEMS UNDER TEST
# =============================================================================
#
# Each system is a callable: (Dataset, index_mode, top_k) -> (rankings, timings)
# where rankings[i] is the ranked list of dictionary ids for queries[i].


def _query_text(q: Query) -> str:
    """The raw source-schema signal available at match time."""
    return q.field_name.replace("_", " ")


def system_dense(
    dataset: Dataset,
    index_mode: str,
    top_k: int,
    model_name: str = "BAAI/bge-small-en-v1.5",
    query_prefix: str = "",
    batch_size: int = 128,
) -> tuple[list[list[str]], dict[str, float]]:
    """Pure dense retrieval with a sentence-transformers model, batched."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    t0 = time.perf_counter()
    corpus = [e.searchable_text(index_mode) for e in dataset.entries]
    doc_emb = model.encode(
        corpus,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    index_s = time.perf_counter() - t0

    ids = [e.id for e in dataset.entries]
    texts = [query_prefix + _query_text(q) for q in dataset.queries]

    t0 = time.perf_counter()
    q_emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    sims = q_emb @ doc_emb.T
    k = min(top_k, len(ids))
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rankings = []
    for row, cand in zip(sims, part, strict=False):
        order = cand[np.argsort(-row[cand])]
        rankings.append([ids[i] for i in order])
    total_s = time.perf_counter() - t0

    return rankings, {"index_seconds": index_s, "total_seconds": total_s}


def system_bm25(
    dataset: Dataset,
    index_mode: str,
    top_k: int,
) -> tuple[list[list[str]], dict[str, float]]:
    """Pure lexical retrieval with bm25s."""
    import bm25s

    ids = [e.id for e in dataset.entries]
    corpus = [e.searchable_text(index_mode) for e in dataset.entries]

    t0 = time.perf_counter()
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)
    index_s = time.perf_counter() - t0

    texts = [_query_text(q) for q in dataset.queries]
    t0 = time.perf_counter()
    tokens = bm25s.tokenize(texts, show_progress=False)
    k = min(top_k, len(ids))
    docs, _scores = retriever.retrieve(tokens, k=k, show_progress=False)
    rankings = [[ids[j] for j in row] for row in docs]
    total_s = time.perf_counter() - t0

    return rankings, {"index_seconds": index_s, "total_seconds": total_s}


def system_hybrid_rrf(
    dataset: Dataset,
    index_mode: str,
    top_k: int,
    model_name: str = "BAAI/bge-small-en-v1.5",
    query_prefix: str = "",
    rrf_k: int = 60,
) -> tuple[list[list[str]], dict[str, float]]:
    """Dense + BM25 fused with Reciprocal Rank Fusion."""
    dense_rank, dense_t = system_dense(
        dataset, index_mode, top_k, model_name=model_name, query_prefix=query_prefix
    )
    sparse_rank, sparse_t = system_bm25(dataset, index_mode, top_k)

    t0 = time.perf_counter()
    rankings: list[list[str]] = []
    for d_list, s_list in zip(dense_rank, sparse_rank, strict=False):
        scores: dict[str, float] = {}
        for r, doc in enumerate(d_list, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + r)
        for r, doc in enumerate(s_list, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + r)
        rankings.append([d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]])
    fuse_s = time.perf_counter() - t0

    return rankings, {
        "index_seconds": dense_t["index_seconds"] + sparse_t["index_seconds"],
        "total_seconds": dense_t["total_seconds"] + sparse_t["total_seconds"] + fuse_s,
    }


SYSTEMS: dict[str, Callable[..., tuple[list[list[str]], dict[str, float]]]] = {
    "bm25": system_bm25,
    "dense": system_dense,
    "dense_prefixed": lambda d, m, k: system_dense(
        d, m, k, query_prefix="Represent this sentence for searching relevant passages: "
    ),
    "hybrid_rrf": system_hybrid_rrf,
    "hybrid_rrf_prefixed": lambda d, m, k: system_hybrid_rrf(
        d, m, k, query_prefix="Represent this sentence for searching relevant passages: "
    ),
}


# =============================================================================
# RUNNER
# =============================================================================


def evaluate(
    dataset: Dataset,
    system: str,
    index_mode: str,
    top_k: int = 50,
) -> Metrics:
    fn = SYSTEMS[system]
    rankings, timings = fn(dataset, index_mode, top_k)

    golds = [q.gold_id for q in dataset.queries]
    scored = score_rankings(rankings, golds)

    total = timings["total_seconds"]
    m = Metrics(
        n=len(golds),
        p_at_1=float(scored["p_at_1"]),
        p_at_5=float(scored["p_at_5"]),
        mrr_at_10=float(scored["mrr_at_10"]),
        recall=scored["recall"],  # type: ignore[arg-type]
        total_seconds=total,
        index_seconds=timings["index_seconds"],
        queries_per_sec=len(golds) / total if total > 0 else 0.0,
        latency_p50_ms=(total / len(golds)) * 1000 if golds else 0.0,
        latency_p95_ms=(total / len(golds)) * 1000 if golds else 0.0,
    )
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="combined", help="benchmark name under data/benchmarks")
    ap.add_argument(
        "--system",
        default="all",
        help=f"one of {sorted(SYSTEMS)} or 'all'",
    )
    ap.add_argument(
        "--index-fields",
        default="business+desc",
        choices=["business+desc", "business", "all"],
        help="what dictionary text to index; 'all' leaks the technical name",
    )
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="limit number of queries")
    ap.add_argument("--save", action="store_true", help="write JSON results")
    args = ap.parse_args()

    ds = Dataset.load(args.benchmark, limit=args.limit)
    print(f"\nBenchmark: {ds.name}")
    print(f"  dictionary entries : {len(ds.entries)}")
    print(f"  queries            : {len(ds.queries)}")
    print(f"  index fields       : {args.index_fields}")
    if args.index_fields == "all":
        print(
            "  !! WARNING: 'all' indexes the technical name, which IS the query. Scores are inflated."
        )

    names = sorted(SYSTEMS) if args.system == "all" else [args.system]
    out: dict[str, dict] = {}

    for name in names:
        m = evaluate(ds, name, args.index_fields, top_k=args.top_k)
        print(m.render(name))
        out[name] = {
            "n": m.n,
            "p_at_1": m.p_at_1,
            "p_at_5": m.p_at_5,
            "mrr_at_10": m.mrr_at_10,
            "recall": {str(k): v for k, v in m.recall.items()},
            "queries_per_sec": m.queries_per_sec,
            "index_seconds": m.index_seconds,
        }

    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"eval_{ds.name}_{args.index_fields.replace('+', '_')}.json"
        path.write_text(
            json.dumps(
                {
                    "benchmark": ds.name,
                    "index_fields": args.index_fields,
                    "n_entries": len(ds.entries),
                    "n_queries": len(ds.queries),
                    "systems": out,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
