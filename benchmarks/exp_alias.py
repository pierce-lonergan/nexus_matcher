"""
benchmarks.exp_alias | Layer: BENCHMARK
Invert the abbreviation problem: abbreviate the DICTIONARY, not the query.

The failures that motivate this
-------------------------------
Two attempts to attack the abbreviation ceiling from the QUERY side both lost accuracy:

  * hand-written expansion of "sname" -> "school name"    -2.0 points
  * appending sample data values to the query             -3.1 points (dilution)

Both fail for the same structural reason: the query is ONE vector, so any wrong guess
corrupts it. Expanding "st" to "state" inside a street column is strictly worse than
leaving "st" alone, and there is no way to recover.

Dictionary-side alias generation does not have that failure mode. For the entry
"Number of Test Takers" we generate plausible technical spellings -- "number of test
takers", "num test takers", "numtsttakr", "num_tst_takr", "ntt" -- index EACH as its own
vector, and take the MAX similarity over the entry's aliases. A wrong alias simply never
wins the max; it cannot corrupt the entry's other surface forms. Generation is
rule-based, deterministic, needs no labels and no LLM.

This is the inverse of doc2query: instead of predicting queries that a document answers,
we predict the abbreviations a business term would be written as in a real schema.

Cost: index size grows by the number of aliases, and index-time encoding grows with it.
Query cost is unchanged apart from a larger candidate matrix.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import RESULTS, Dataset, score_rankings
from exp_query_repr import VARIANTS

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

# Words that carry no discriminating content in a column name and are usually dropped
# when a business term becomes an identifier.
STOP = {"of", "the", "a", "an", "in", "on", "at", "for", "to", "and", "or", "by", "with"}

# Conventional data-engineering contractions, applied to the DICTIONARY side. Being wrong
# here is cheap: a bad alias just never matches anything.
CONTRACT = {
    "number": "num",
    "identifier": "id",
    "description": "desc",
    "amount": "amt",
    "date": "dt",
    "datetime": "dttm",
    "timestamp": "ts",
    "code": "cd",
    "name": "nm",
    "address": "addr",
    "account": "acct",
    "customer": "cust",
    "quantity": "qty",
    "percent": "pct",
    "average": "avg",
    "maximum": "max",
    "minimum": "min",
    "count": "cnt",
    "total": "tot",
    "value": "val",
    "reference": "ref",
    "source": "src",
    "category": "cat",
    "department": "dept",
    "organization": "org",
    "person": "prsn",
    "product": "prod",
    "transaction": "txn",
    "balance": "bal",
    "payment": "pmt",
    "telephone": "tel",
    "phone": "phn",
    "email": "eml",
    "school": "sch",
    "district": "dist",
    "county": "cnty",
    "state": "st",
    "country": "cntry",
    "city": "cty",
    "score": "scr",
    "reading": "read",
    "writing": "writ",
    "mathematics": "math",
    "enrollment": "enroll",
    "grade": "gr",
    "student": "stu",
    "teacher": "tchr",
    "test": "tst",
    "taker": "takr",
    "year": "yr",
    "month": "mo",
    "day": "dy",
    "start": "strt",
    "end": "end",
    "concept": "cncpt",
    "occurrence": "occ",
    "observation": "obs",
    "measurement": "meas",
    "condition": "cond",
    "procedure": "proc",
    "provider": "prov",
    "visit": "vis",
    "specimen": "spec",
    "vocabulary": "vocab",
    "relationship": "rel",
    "domain": "dmn",
}


def words(text: str) -> list[str]:
    return [w for w in re.split(r"[^0-9A-Za-z]+", text.lower()) if w]


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOP]


def generate_aliases(business_name: str, max_aliases: int = 8, min_chars: int = 1) -> list[str]:
    """
    Generate plausible technical spellings of a business term.

    Deliberately rule-based: deterministic, label-free, and instant. The max-pool at
    query time means precision of any individual alias does not matter much; coverage does.
    """
    w = content_words(business_name)
    if not w:
        return []

    out: list[str] = []

    def add(s: str) -> None:
        s = s.strip()
        if s and s not in out:
            out.append(s)

    # 1. Content words only, stopwords dropped: "number of test takers" -> "number test takers"
    add(" ".join(w))

    # 2. Conventional contractions: "number test takers" -> "num tst takr"
    contracted = [CONTRACT.get(x, x) for x in w]
    add(" ".join(contracted))

    # 3. Truncation to 4 characters, the most common ad-hoc abbreviation style.
    add(" ".join(x[:4] for x in w))

    # 4. Initialism: "number test takers" -> "ntt". Only useful for multiword terms.
    if len(w) >= 2:
        add("".join(x[0] for x in w))

    # 5. First word in full plus initials of the rest: "school nm" style compromise.
    if len(w) >= 2:
        add(w[0] + " " + " ".join(CONTRACT.get(x, x[:3]) for x in w[1:]))

    # 6. Concatenated contraction, mimicking camelCase/no-separator identifiers.
    add("".join(contracted))

    # 7. Drop the head noun -- schemas often omit the obvious ("Customer Account Balance"
    #    inside a customer table becomes "account balance").
    if len(w) >= 3:
        add(" ".join(w[1:]))

    # Drop very short aliases. A 2-3 character string like "cc" or "ntt" carries almost
    # no lexical content, so its embedding sits in an arbitrary region and it wins
    # max-pool matches against unrelated queries -- the same "confident garbage" failure
    # mode that sank query-side expansion. Longer aliases are self-limiting: if they do
    # not resemble the query they simply lose the max.
    out = [a for a in out if len(a.replace(" ", "")) >= min_chars]
    return out[:max_aliases]


def discriminative_names(entries, max_share: int = 3) -> set[str]:
    """
    Entry ids whose business name is specific enough to be worth aliasing.

    A business name shared by many entries cannot identify any one of them. In the OMOP
    benchmark the business name is the TABLE ("person"), shared by ~30 fields, so its
    aliases ("prsn", "pers", "p") are short, meaningless vectors that still win max-pool
    matches against unrelated queries. Measured: aliasing everything cost OMOP 11.9
    points of P@1 while gaining BIRD 4.2, because BIRD names ("Free or Reduced Price Meal
    Count") are specific and OMOP's are not.

    So alias only where the name is (a) not shared with more than `max_share` entries and
    (b) long enough to carry content.
    """
    counts: dict[str, int] = {}
    for e in entries:
        key = e.business_name.strip().lower()
        counts[key] = counts.get(key, 0) + 1

    keep = set()
    for e in entries:
        key = e.business_name.strip().lower()
        if counts[key] <= max_share and len(content_words(e.business_name)) >= 2:
            keep.add(e.id)
    return keep


def build_index(
    model, entries, index_mode: str, n_aliases: int, selective: bool = False, min_chars: int = 1
):
    """Return (matrix, owner_ids) where several rows may map to the same entry id."""
    eligible = discriminative_names(entries) if selective else {e.id for e in entries}

    texts: list[str] = []
    owners: list[str] = []
    for e in entries:
        base = e.searchable_text(index_mode)
        texts.append(base)
        owners.append(e.id)
        if n_aliases > 0 and e.id in eligible:
            for alias in generate_aliases(e.business_name, n_aliases, min_chars):
                texts.append(alias)
                owners.append(e.id)

    emb = model.encode(
        texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    return emb, owners


def rank_maxpool(sims: np.ndarray, owners: list[str], top_k: int) -> list[list[str]]:
    """Max-pool similarities per owning entry, then rank entries."""
    owners_arr = np.asarray(owners)
    uniq, inverse = np.unique(owners_arr, return_inverse=True)

    rankings: list[list[str]] = []
    n_entries = len(uniq)
    for row in sims:
        # Max over all rows belonging to each entry.
        pooled = np.full(n_entries, -np.inf, dtype=np.float32)
        np.maximum.at(pooled, inverse, row)
        k = min(top_k, n_entries)
        cand = np.argpartition(-pooled, k - 1)[:k]
        order = cand[np.argsort(-pooled[cand])]
        rankings.append([uniq[i] for i in order])
    return rankings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="bird")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--index-fields", default="business+desc")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--alias-counts", nargs="*", type=int, default=[0, 2, 4, 6, 8])
    ap.add_argument(
        "--min-chars",
        type=int,
        default=1,
        help="drop generated aliases shorter than this many characters",
    )
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    ds = Dataset.load(args.benchmark)
    golds = [q.gold_id for q in ds.queries]
    model = SentenceTransformer(args.model, device="cpu")

    q_emb = model.encode(
        [BGE_PREFIX + VARIANTS["context"](q) for q in ds.queries],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    print(f"\nBenchmark {ds.name}: {len(ds.entries)} entries, {len(ds.queries)} queries")
    sample = ds.entries[0]
    print(f"\n  Example aliases for {sample.business_name!r}:")
    for a in generate_aliases(sample.business_name, 8):
        print(f"    {a!r}")

    print(
        f"\n  {'aliases':>8s} {'index rows':>11s} {'P@1':>7s} {'P@5':>7s} "
        f"{'MRR@10':>7s} {'R@10':>7s}  {'delta':>8s}"
    )
    print(f"  {'-' * 8} {'-' * 11} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}  {'-' * 8}")

    eligible = discriminative_names(ds.entries)
    print(
        f"  ({len(eligible)}/{len(ds.entries)} = {len(eligible) / len(ds.entries):.0%} "
        f"of entries have a discriminative business name; '*' = selective)"
    )

    out: dict[str, dict] = {}
    base_p1 = None
    for n, selective in [(n, False) for n in args.alias_counts] + [
        (n, True) for n in args.alias_counts if n > 0
    ]:
        emb, owners = build_index(
            model, ds.entries, args.index_fields, n, selective, args.min_chars
        )
        sims = q_emb @ emb.T
        rankings = rank_maxpool(sims, owners, args.top_k)
        s = score_rankings(rankings, golds, ks=(1, 5, 10))
        if base_p1 is None:
            base_p1 = s["p_at_1"]
        d = s["p_at_1"] - base_p1
        label = f"{n}{'*' if selective else ''}"
        print(
            f"  {label:>8s} {len(owners):11d} {s['p_at_1']:7.4f} {s['p_at_5']:7.4f} "
            f"{s['mrr_at_10']:7.4f} {s['recall'][10]:7.4f}  {d:+8.4f}"
        )
        out[f"aliases={n}{'_selective' if selective else ''}"] = {
            k: v for k, v in s.items() if k != "recall"
        }

    best = max(out.items(), key=lambda kv: kv[1]["p_at_1"])
    print(
        f"\n  BEST: {best[0]}  P@1 {best[1]['p_at_1']:.4f}  "
        f"({best[1]['p_at_1'] - base_p1:+.4f} vs no aliases)"
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"exp_alias_{ds.name}.json"
    p.write_text(
        json.dumps({"benchmark": ds.name, "variants": out, "best": best[0]}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
