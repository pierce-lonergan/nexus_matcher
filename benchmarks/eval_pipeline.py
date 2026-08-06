"""
benchmarks.eval_pipeline | Layer: BENCHMARK
End-to-end accuracy + throughput evaluation of the REAL NexusMatcher pipeline.

Unlike eval_harness.py (which measures retrieval strategies in isolation), this drives
the actual `NexusMatcher` orchestrator -- context enrichment, abbreviation expansion,
dense retrieval, BM25, fusion, multi-signal scoring and the decision policy -- against
the labelled BIRD + OMOP benchmark. It is the number that matters, because it is the
number a user of the library actually gets.

Leakage control: dictionary entries are indexed with `logical_name` blanked, so the
source system's technical column name is NOT in the corpus. Retrieval must work from
the business name and human definition alone.

Usage:
    python benchmarks/eval_pipeline.py --benchmark combined
    python benchmarks/eval_pipeline.py --benchmark bird --no-sparse
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

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import DataType

_TYPE_MAP = {
    "integer": DataType.INTEGER,
    "float": DataType.FLOAT,
    "boolean": DataType.BOOLEAN,
    "date": DataType.DATE,
    "datetime": DataType.TIMESTAMP,
    "time": DataType.STRING,
    "string": DataType.STRING,
    "unknown": DataType.UNKNOWN,
}


def to_data_type(name: str) -> DataType:
    return _TYPE_MAP.get(name, DataType.UNKNOWN)


class PrefixedProvider:
    """
    Wraps a SentenceTransformer with asymmetric query/document prefixes.

    BGE retrieval models are trained with an instruction on the QUERY side only.
    Omitting it costs ~5 points of P@1 on this benchmark, so the provider -- not the
    caller -- owns applying it, and documents are deliberately left unprefixed.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", query_prefix: str = "") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self._query_prefix = query_prefix
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "prefixed"

    def _encode(self, texts, prefix: str) -> np.ndarray:
        return self._model.encode(
            [prefix + t for t in texts],
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    # -- EmbeddingProvider surface used by NexusMatcher -----------------------

    def embed(self, texts):
        from nexus_matcher.shared.types.base import Result

        arr = self._encode(list(texts), self._query_prefix)

        class _Batch:
            embeddings = arr

        return Result.success(_Batch())

    def embed_single(self, text):
        from nexus_matcher.shared.types.base import Result

        return Result.success(self._encode([text], self._query_prefix)[0])

    def embed_documents(self, texts) -> np.ndarray:
        return self._encode(list(texts), "")


def build_matcher(
    ds: Dataset,
    use_sparse: bool,
    query_prefix: str,
    model_name: str,
    # Mirrors the SHIPPED default. Aliasing is off because its small-corpus gain
    # inverts at scale (-13.7 P@1 at 10k entries); see MatchingConfig.
    alias_count: int = 0,
):
    provider = PrefixedProvider(model_name, query_prefix)

    store = InMemoryVectorStore(
        VectorStoreConfig(collection_name="dictionary", dimension=provider.dimension)
    )
    matcher = NexusMatcher(
        embedding_provider=provider,
        vector_store=store,
        sparse_retriever=BM25Retriever() if use_sparse else None,
        config=MatchingConfig(results_per_field=10, dictionary_alias_count=alias_count),
    )

    # Blank logical_name so the technical column name is not indexed (no leakage).
    entries = [
        DictionaryEntry(
            id=e.id,
            business_name=e.business_name,
            logical_name="",
            definition=e.description,
            data_type=to_data_type(e.data_type),
            domain=e.domain,
        )
        for e in ds.entries
    ]

    t0 = time.perf_counter()
    # Index documents WITHOUT the query prefix by temporarily swapping the provider's
    # prefix -- documents and queries must be encoded asymmetrically.
    original_prefix = provider._query_prefix
    provider._query_prefix = ""
    matcher._index_dictionary(entries)
    provider._query_prefix = original_prefix
    index_s = time.perf_counter() - t0

    return matcher, index_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="combined")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--no-prefix", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save", action="store_true")
    ap.add_argument(
        "--aliases",
        type=int,
        default=0,
        help="fabricated technical spellings indexed per dictionary entry",
    )
    args = ap.parse_args()

    prefix = "" if args.no_prefix else "Represent this sentence for searching relevant passages: "

    ds = Dataset.load(args.benchmark, limit=args.limit)
    matcher, index_s = build_matcher(ds, not args.no_sparse, prefix, args.model, args.aliases)

    fields = [
        SchemaField(
            name=q.field_name,
            data_type=to_data_type(q.data_type),
            full_path=q.field_path,
            parent_path=q.parent_path,
        )
        for q in ds.queries
    ]

    t0 = time.perf_counter()
    results = matcher._match_fields(fields)
    elapsed = time.perf_counter() - t0

    rankings = []
    decisions = {"AUTO_APPROVE": 0, "REVIEW": 0, "REJECT": 0}
    auto_correct = 0
    auto_total = 0
    for q, f in zip(ds.queries, fields, strict=False):
        matches = results.get(f.full_path, ())
        rankings.append([m.dictionary_entry.id for m in matches])
        if matches:
            d = matches[0].decision.name
            decisions[d] = decisions.get(d, 0) + 1
            if d == "AUTO_APPROVE":
                auto_total += 1
                if matches[0].dictionary_entry.id == q.gold_id:
                    auto_correct += 1

    golds = [q.gold_id for q in ds.queries]
    s = score_rankings(rankings, golds, ks=(1, 5, 10))

    print(f"\nEnd-to-end NexusMatcher on '{ds.name}'")
    print(f"  dictionary   {len(ds.entries)}   queries {len(ds.queries)}")
    print(f"  model        {args.model}")
    print(
        f"  sparse       {'off' if args.no_sparse else 'on'}    "
        f"query prefix {'off' if args.no_prefix else 'on'}"
    )
    print(f"\n  P@1          {s['p_at_1']:.4f}")
    print(f"  P@5          {s['p_at_5']:.4f}")
    print(f"  MRR@10       {s['mrr_at_10']:.4f}")
    print(f"  Recall@10    {s['recall'][10]:.4f}")
    print(f"\n  index build  {index_s:.2f} s")
    print(f"  match time   {elapsed:.2f} s  ->  {len(ds.queries) / elapsed:.1f} fields/sec")
    print(f"  decisions    {decisions}")
    if auto_total:
        print(
            f"  auto-approve precision {auto_correct}/{auto_total} = {auto_correct / auto_total:.3f}"
        )

    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        p = RESULTS / f"eval_pipeline_{ds.name}.json"
        p.write_text(
            json.dumps(
                {
                    "benchmark": ds.name,
                    "model": args.model,
                    "sparse": not args.no_sparse,
                    "query_prefix": not args.no_prefix,
                    "p_at_1": s["p_at_1"],
                    "p_at_5": s["p_at_5"],
                    "mrr_at_10": s["mrr_at_10"],
                    "recall": {str(k): v for k, v in s["recall"].items()},
                    "fields_per_sec": len(ds.queries) / elapsed,
                    "index_seconds": index_s,
                    "decisions": decisions,
                    "auto_approve_precision": (auto_correct / auto_total) if auto_total else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
