"""
benchmarks.exp_query_repr | Layer: BENCHMARK
Measure how much the QUERY-SIDE representation matters.

Motivation
----------
First-stage retrieval gets P@1 = 0.49 while Recall@50 = 0.90, and a strong off-the-shelf
cross-encoder recovers only 11% of that gap (a stronger one made it worse). That pattern
says the bottleneck is not the ranker -- it is that the query "sname" carries almost no
signal for a model trained on natural language.

This script isolates the query representation and holds everything else fixed, so the
contribution of each enrichment step is measured independently rather than assumed.

Variants
--------
raw           "sname"                       -- the bare technical token
underscores   "sname"                       -- underscores to spaces (current code)
split         "s name"                      -- camelCase + underscore splitting
abbrev        "school name"                 -- abbreviation expansion
context       "schools . sname"             -- parent table as context
type          "sname (string)"              -- declared data type
full          "schools school name (string)" -- everything combined
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import RESULTS, Dataset, Query, score_rankings

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


# =============================================================================
# ABBREVIATION DICTIONARY
# =============================================================================
# Data-engineering abbreviations. Deliberately conservative: only unambiguous
# expansions, because a wrong expansion is worse than none.

ABBREV = {
    "acct": "account",
    "addr": "address",
    "amt": "amount",
    "auth": "authorization",
    "avg": "average",
    "bal": "balance",
    "cd": "code",
    "cde": "code",
    "chg": "charge",
    "cnt": "count",
    "cntry": "country",
    "cty": "city",
    "cust": "customer",
    "desc": "description",
    "dept": "department",
    "dob": "date of birth",
    "dt": "date",
    "dttm": "datetime",
    "eff": "effective",
    "eml": "email",
    "emp": "employee",
    "exp": "expiration",
    "fk": "foreign key",
    "flg": "flag",
    "fname": "first name",
    "freq": "frequency",
    "gr": "grade",
    "grp": "group",
    "id": "identifier",
    "idx": "index",
    "ind": "indicator",
    "info": "information",
    "ins": "insurance",
    "inv": "invoice",
    "lat": "latitude",
    "lname": "last name",
    "lng": "longitude",
    "loc": "location",
    "lgl": "legal",
    "max": "maximum",
    "mgr": "manager",
    "min": "minimum",
    "mo": "month",
    "msg": "message",
    "nbr": "number",
    "nm": "name",
    "num": "number",
    "org": "organization",
    "pct": "percent",
    "phn": "phone",
    "pk": "primary key",
    "pmt": "payment",
    "pos": "position",
    "prc": "price",
    "prin": "principal",
    "prod": "product",
    "qty": "quantity",
    "qtr": "quarter",
    "rcv": "received",
    "ref": "reference",
    "reg": "registration",
    "req": "required",
    "rev": "revenue",
    "rpt": "report",
    "scr": "score",
    "seq": "sequence",
    "sname": "school name",
    "src": "source",
    "st": "state",
    "stat": "status",
    "std": "standard",
    "svc": "service",
    "tel": "telephone",
    "temp": "temperature",
    "tot": "total",
    "txn": "transaction",
    "typ": "type",
    "usr": "user",
    "val": "value",
    "ver": "version",
    "yr": "year",
    "zip": "postal code",
    "cnty": "county",
    "dist": "district",
    "enroll": "enrollment",
    "tst": "test",
    "tsttakr": "test takers",
    "takr": "takers",
    "concept": "concept",
    "occurrence": "occurrence",
    "visit": "visit",
    "obs": "observation",
    "qual": "qualification",
    "cond": "condition",
    "meas": "measurement",
    "spec": "specimen",
    "prov": "provider",
    "care": "care",
    "drug": "drug",
    "proc": "procedure",
    "dev": "device",
    "vocab": "vocabulary",
    "rel": "relationship",
}


def split_identifier(name: str) -> list[str]:
    """person_idFoo -> ['person', 'id', 'Foo']; handles snake, camel, digits."""
    s = re.sub(r"[^0-9A-Za-z]+", " ", name)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    s = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", s)
    return [t for t in s.split() if t]


def expand_abbreviations(tokens: list[str]) -> list[str]:
    """Expand known abbreviations; keep the original token alongside the expansion."""
    out: list[str] = []
    for t in tokens:
        low = t.lower()
        exp = ABBREV.get(low)
        if exp:
            # Keep both: the literal token still helps exact lexical overlap.
            out.extend(exp.split())
        else:
            out.append(t)
    return out


# =============================================================================
# VARIANTS
# =============================================================================


def v_raw(q: Query) -> str:
    return q.field_name


def v_underscores(q: Query) -> str:
    return q.field_name.replace("_", " ")


def v_split(q: Query) -> str:
    return " ".join(split_identifier(q.field_name))


def v_abbrev(q: Query) -> str:
    return " ".join(expand_abbreviations(split_identifier(q.field_name)))


def v_context(q: Query) -> str:
    parent = " ".join(split_identifier(q.parent_path))
    return f"{parent} {' '.join(split_identifier(q.field_name))}".strip()


def v_type(q: Query) -> str:
    return f"{' '.join(split_identifier(q.field_name))} ({q.data_type})"


def v_full(q: Query) -> str:
    parent = " ".join(expand_abbreviations(split_identifier(q.parent_path)))
    fname = " ".join(expand_abbreviations(split_identifier(q.field_name)))
    return f"{parent} {fname} ({q.data_type})".strip()


def v_full_no_type(q: Query) -> str:
    parent = " ".join(expand_abbreviations(split_identifier(q.parent_path)))
    fname = " ".join(expand_abbreviations(split_identifier(q.field_name)))
    return f"{parent} {fname}".strip()


VARIANTS = {
    "raw": v_raw,
    "underscores": v_underscores,
    "split": v_split,
    "abbrev": v_abbrev,
    "context": v_context,
    "type": v_type,
    "full_no_type": v_full_no_type,
    "full": v_full,
}


# =============================================================================
# RUN
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="combined")
    ap.add_argument("--index-fields", default="business+desc")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--no-prefix", action="store_true")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    ds = Dataset.load(args.benchmark)
    golds = [q.gold_id for q in ds.queries]
    ids = [e.id for e in ds.entries]
    prefix = "" if args.no_prefix else BGE_PREFIX

    model = SentenceTransformer(args.model, device="cpu")
    corpus = [e.searchable_text(args.index_fields) for e in ds.entries]
    doc_emb = model.encode(
        corpus, batch_size=128, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)

    print(f"\nBenchmark {ds.name}: {len(ds.entries)} entries, {len(ds.queries)} queries")
    print(f"Model {args.model}   prefix={'yes' if prefix else 'no'}\n")
    print(f"  {'variant':14s} {'P@1':>7s} {'P@5':>7s} {'MRR@10':>7s} {'R@10':>7s}  example")
    print(f"  {'-' * 14} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}  {'-' * 40}")

    out: dict[str, dict] = {}
    example_q = next((q for q in ds.queries if q.field_name.lower() == "sname"), ds.queries[0])

    for name, fn in VARIANTS.items():
        texts = [prefix + fn(q) for q in ds.queries]
        q_emb = model.encode(
            texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        sims = q_emb @ doc_emb.T
        k = min(args.top_k, len(ids))
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rankings = []
        for row, cand in zip(sims, part, strict=False):
            order = cand[np.argsort(-row[cand])]
            rankings.append([ids[i] for i in order])
        s = score_rankings(rankings, golds)
        print(
            f"  {name:14s} {s['p_at_1']:7.4f} {s['p_at_5']:7.4f} "
            f"{s['mrr_at_10']:7.4f} {s['recall'][10]:7.4f}  {fn(example_q)!r}"
        )
        out[name] = {
            "p_at_1": s["p_at_1"],
            "p_at_5": s["p_at_5"],
            "mrr_at_10": s["mrr_at_10"],
            "recall": {str(kk): vv for kk, vv in s["recall"].items()},
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"exp_query_repr_{ds.name}.json"
    p.write_text(json.dumps({"benchmark": ds.name, "variants": out}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
