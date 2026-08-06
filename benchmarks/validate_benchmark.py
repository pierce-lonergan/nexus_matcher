"""
benchmarks.validate_benchmark | Layer: BENCHMARK
Does this benchmark measure semantic matching, or string copying?

Run this on ANY benchmark before trusting a number from it. This repo has shipped a
degenerate benchmark twice:

  * OMOP derived the entry's business name from the field name, so query and gold label
    were token-identical (measured overlap 1.000). Half the corpus scored string identity.
    Two techniques INVERTED when it was fixed -- cross-encoder reranking went +5.5 to -1.3.
  * A proposed FHIR design put the element `definition` on BOTH the query and target side.
    BM25 scored 0.81 on it and BEAT the embedding model.

Both were caught by the same two checks, which is why they live in one script.

The tests
---------
1. TOKEN OVERLAP between the query text and its gold target text. High mean overlap means
   the answer is being copied into the question. Reported with the zero-overlap and
   exact-match rates, because a mean can hide a bimodal corpus.

2. BM25 vs EMBEDDING. The diagnostic that actually matters. On a real semantic task a
   dense encoder beats lexical BM25. If BM25 wins, the optimal strategy is string overlap
   and the benchmark is rewarding copying, not meaning. This inversion is the signature,
   and it is what a raw overlap number alone will miss.

A benchmark PASSES when overlap is modest AND the embedding model is ahead of BM25 AND the
scores are not saturated (a corpus everything solves measures nothing).
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import Dataset, score_rankings

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^0-9A-Za-z]+", (text or "").lower()) if t}


def query_text(q, use_doc: bool) -> str:
    """The query as the matcher would see it: path context + leaf + optional doc."""
    parts = [q.parent_path, q.field_name.replace("__", " ").replace("_", " ")]
    if use_doc and q.doc:
        parts.append(q.doc)
    return " ".join(p for p in parts if p).strip()


def overlap_report(ds: Dataset, use_doc: bool) -> dict:
    entries = {e.id: e for e in ds.entries}
    ratios, exact, zero = [], 0, 0
    for q in ds.queries:
        gold = entries.get(q.gold_id)
        if gold is None:
            continue
        qt = tokens(query_text(q, use_doc))
        gt = tokens(gold.searchable_text("business+desc"))
        if not qt or not gt:
            continue
        inter = len(qt & gt)
        ratio = inter / len(gt)
        ratios.append(ratio)
        if gt <= qt:
            exact += 1
        if inter == 0:
            zero += 1
    n = len(ratios) or 1
    return {
        "mean": statistics.mean(ratios) if ratios else 0.0,
        "median": statistics.median(ratios) if ratios else 0.0,
        "exact_rate": exact / n,
        "zero_rate": zero / n,
    }


def retrieval_report(ds: Dataset, use_doc: bool, model_name: str) -> dict:
    import bm25s
    from sentence_transformers import SentenceTransformer

    ids = [e.id for e in ds.entries]
    corpus = [e.searchable_text("business+desc") for e in ds.entries]
    queries = [query_text(q, use_doc) for q in ds.queries]
    golds = [q.gold_id for q in ds.queries]

    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)
    k = min(50, len(ids))
    docs, _ = retriever.retrieve(
        bm25s.tokenize(queries, show_progress=False), k=k, show_progress=False
    )
    bm25_score = score_rankings([[ids[j] for j in row] for row in docs], golds, ks=(1, 5, 10))

    model = SentenceTransformer(model_name, device="cpu")
    doc_emb = model.encode(
        corpus, batch_size=128, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    q_emb = model.encode(
        [BGE_PREFIX + t for t in queries],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    sims = q_emb @ doc_emb.T
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rankings = [[ids[i] for i in c[np.argsort(-r[c])]] for r, c in zip(sims, part, strict=True)]
    embed_score = score_rankings(rankings, golds, ks=(1, 5, 10))

    return {"bm25": bm25_score, "embed": embed_score}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="fhir")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument(
        "--compare-doc",
        action="store_true",
        help="also evaluate WITHOUT the query doc, to isolate its contribution",
    )
    args = ap.parse_args()

    ds = Dataset.load(args.benchmark)
    print(f"\nValidating '{ds.name}': {len(ds.entries)} entries, {len(ds.queries)} queries")
    has_doc = sum(1 for q in ds.queries if q.doc)
    print(f"  queries carrying a doc: {has_doc} ({has_doc / max(len(ds.queries), 1):.0%})\n")

    variants = [("with doc", True)] + ([("no doc", False)] if args.compare_doc else [])

    verdicts = []
    for label, use_doc in variants:
        ov = overlap_report(ds, use_doc)
        rr = retrieval_report(ds, use_doc, args.model)
        bm25_p1 = rr["bm25"]["p_at_1"]
        embed_p1 = rr["embed"]["p_at_1"]
        margin = embed_p1 - bm25_p1

        print(f"  --- {label} ---")
        print(f"    token overlap with gold   mean {ov['mean']:.3f}  median {ov['median']:.3f}")
        print(
            f"    gold fully inside query   {ov['exact_rate']:.1%}   zero overlap {ov['zero_rate']:.1%}"
        )
        print(f"    BM25       P@1 {bm25_p1:.4f}  P@5 {rr['bm25']['p_at_5']:.4f}")
        print(
            f"    embedding  P@1 {embed_p1:.4f}  P@5 {rr['embed']['p_at_5']:.4f}  "
            f"R@10 {rr['embed']['recall'][10]:.4f}"
        )
        print(f"    embedding - BM25          {margin:+.4f}")

        problems = []
        if ov["mean"] > 0.5:
            problems.append(
                f"token overlap {ov['mean']:.2f} is high - gold text may be in the query"
            )
        if ov["exact_rate"] > 0.2:
            problems.append(f"{ov['exact_rate']:.0%} of gold texts sit entirely inside their query")
        if margin < 0:
            problems.append(
                f"BM25 BEATS the embedding model by {-margin:.3f} - the corpus rewards "
                f"string overlap, not meaning"
            )
        if embed_p1 > 0.95:
            problems.append(f"P@1 {embed_p1:.3f} is saturated - no headroom to measure")

        if problems:
            print("    VERDICT: DEGENERATE")
            for p in problems:
                print(f"      - {p}")
        else:
            print("    VERDICT: sound - semantics beat lexical overlap, with headroom")
        print()
        verdicts.append((label, not problems))

    if all(ok for _, ok in verdicts):
        print("  All variants passed.")
    else:
        failed = [label for label, ok in verdicts if not ok]
        print(f"  FAILED: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
