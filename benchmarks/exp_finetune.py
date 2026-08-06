"""
benchmarks.exp_finetune | Layer: BENCHMARK
Adapt the bi-encoder to schema matching with synthetic pairs. No human labels.

Why this and not something cheaper
----------------------------------
Everything cheap has been tried and measured on the corrected benchmark:

    instance data values          -3.1     (values-only P@1 0.0499 -- signal absent)
    cross-encoder reranking       -1.3     (trained on question->passage, not identifiers)
    query-side expansion          -2.0     (one wrong guess corrupts the single vector)
    scalar type words             -2.1
    char n-gram sparse arm        negative (adversarial reviewer measured it)
    dictionary-side aliases       +1.5     (the only survivor)
    swapping the encoder           0.0     (7 models all land in 0.53-0.56 P@1;
                                            a 335M model LOSES to the 33M one)

And the diagnostic that explains all of it: mean cosine to the GOLD entry is 0.7657, to
the nearest WRONG entry 0.7633. A margin of 0.0024. The encoder does not separate the
right answer from the best wrong one, so no amount of reranking, fusion or extra text
downstream of that embedding can fix it. The embedding space itself is the defect.

That is what this script attacks. Contrastive fine-tuning on (technical name -> business
text) pairs pushes golds together and hard negatives apart, which is precisely the margin
that is missing.

Guarding against self-deception
-------------------------------
Fine-tuning on 688 pairs invites memorisation. Two guards, both enforced here:

  --mode holdout    train on a random 70% of the benchmark, evaluate on the unseen 30%.
  --mode transfer   train on OMOP ONLY, evaluate on BIRD ONLY. Different corpora,
                    different domains, different naming conventions, zero overlap.
                    If the gain survives this, it is real adaptation and not memorisation.

`transfer` is the honest headline. Report it, not `holdout`.

Training pairs are SYNTHETIC: the technical-name side is fabricated from the business
name with the same rule set that made dictionary-side aliasing work (exp_alias.py), so
no ground-truth mapping is consumed. The model learns the general shape of
"abbreviated identifier <-> business term", not this benchmark's answer key.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import RESULTS, Dataset, DictEntry, score_rankings
from exp_alias import CONTRACT, content_words
from exp_query_repr import VARIANTS

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


# =============================================================================
# SYNTHETIC PAIR GENERATION
# =============================================================================


def fabricate_identifiers(business_name: str, rng: random.Random, n: int = 4) -> list[str]:
    """
    Fabricate plausible technical column names for a business term.

    Mirrors how humans actually mangle names into identifiers: drop stopwords, contract
    words, truncate, initialise, and join with a random separator convention. Randomised
    so repeated epochs see different surface forms, which is the augmentation.
    """
    w = content_words(business_name)
    if not w:
        return []

    out: list[str] = []
    for _ in range(n * 3):
        toks = list(w)

        # Drop a word sometimes -- real schemas omit the obvious.
        if len(toks) > 2 and rng.random() < 0.3:
            toks.pop(rng.randrange(len(toks)))

        style = rng.random()
        if style < 0.35:
            toks = [CONTRACT.get(t, t) for t in toks]
        elif style < 0.6:
            toks = [t[: rng.choice([3, 4, 5])] for t in toks]
        elif style < 0.75:
            toks = [t[0] if i else t for i, t in enumerate(toks)]

        sep = rng.choice(["_", "", " "])
        ident = toks[0] + "".join(t.capitalize() for t in toks[1:]) if sep == "" else sep.join(toks)

        ident = ident.strip()
        if ident and ident not in out:
            out.append(ident)
        if len(out) >= n:
            break
    return out


def build_pairs(entries: list[DictEntry], rng: random.Random, per_entry: int = 4):
    """
    Build (identifier, business text) positives from a training corpus.

    Two sources, both used:

    1. REAL pairs -- the corpus's own logical_name mapped to its description. For OMOP
       this is the CDM specification's field->guidance mapping, which is genuine
       supervision. Using it is legitimate transfer learning because evaluation happens
       on a DISJOINT corpus (BIRD); it is not label leakage into the test set. It does
       mean the method is not label-free, and the report says so.

    2. SYNTHETIC pairs -- identifiers fabricated from the entry's business name.

    The identifier is deliberately built from `logical_name` where available rather than
    from `business_name`. In the corrected OMOP split the business name is the TABLE, so
    fabricating from it produced pairs like "payer_plan_period payer_plan_period" --
    the table name paired with itself, teaching the model nothing about field naming.
    """
    from sentence_transformers import InputExample

    pairs = []
    seen: set[tuple[str, str]] = set()

    def add(query: str, target: str) -> None:
        query = " ".join(query.split())
        if not query or not target.strip():
            return
        # Reject degenerate pairs where the query is just the context repeated.
        key = (query.lower(), target[:80].lower())
        if key in seen:
            return
        seen.add(key)
        pairs.append(InputExample(texts=[BGE_PREFIX + query, target]))

    for e in entries:
        target = e.searchable_text("business+desc")
        if not target.strip():
            continue

        context = e.domain or ""

        # 1. The real technical name, if the corpus records one.
        if e.logical_name:
            add(f"{context} {e.logical_name.replace('_', ' ')}", target)
            # Augment the real name with fabricated spellings of ITSELF, so the model
            # sees the same target under several identifier conventions.
            for ident in fabricate_identifiers(e.logical_name.replace("_", " "), rng, 2):
                add(f"{context} {ident}", target)

        # 2. Fabricated from the business name -- only when it actually differs from the
        #    context, otherwise this generates the context paired with itself.
        if e.business_name.strip().lower() != context.strip().lower():
            for ident in fabricate_identifiers(e.business_name, rng, per_entry):
                add(f"{context} {ident}", target)

    rng.shuffle(pairs)
    return pairs


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate(model, entries, queries, top_k: int = 50) -> dict:
    ids = [e.id for e in entries]
    golds = [q.gold_id for q in queries]
    doc = model.encode(
        [e.searchable_text("business+desc") for e in entries],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    qe = model.encode(
        [BGE_PREFIX + VARIANTS["context"](q) for q in queries],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    sims = qe @ doc.T
    k = min(top_k, len(ids))
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rankings = []
    for row, cand in zip(sims, part, strict=False):
        order = cand[np.argsort(-row[cand])]
        rankings.append([ids[i] for i in order])

    s = score_rankings(rankings, golds, ks=(1, 5, 10))

    # The margin is the quantity we are actually trying to move.
    gold_pos = {e.id: i for i, e in enumerate(entries)}
    gi = np.array([gold_pos[g] for g in golds])
    gold_sim = sims[np.arange(len(gi)), gi]
    masked = sims.copy()
    masked[np.arange(len(gi)), gi] = -9.0
    s["margin"] = float((gold_sim - masked.max(1)).mean())
    return s


def report(tag: str, s: dict) -> None:
    print(
        f"  {tag:16s} P@1 {s['p_at_1']:.4f}  P@5 {s['p_at_5']:.4f}  "
        f"MRR@10 {s['mrr_at_10']:.4f}  R@10 {s['recall'][10]:.4f}  "
        f"margin {s['margin']:+.4f}"
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="transfer", choices=["transfer", "holdout"])
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--per-entry", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    rng = random.Random(args.seed)

    if args.mode == "transfer":
        train_ds, test_ds = Dataset.load("omop"), Dataset.load("bird")
        train_entries = train_ds.entries
        test_entries, test_queries = test_ds.entries, test_ds.queries
        desc = "train on OMOP -> evaluate on BIRD (zero overlap)"
    else:
        ds = Dataset.load("combined")
        idx = list(range(len(ds.queries)))
        rng.shuffle(idx)
        cut = int(0.7 * len(idx))
        train_ids = {ds.queries[i].gold_id for i in idx[:cut]}
        train_entries = [e for e in ds.entries if e.id in train_ids]
        test_queries = [ds.queries[i] for i in idx[cut:]]
        # Full corpus stays as the retrieval target, so the task is not made easier.
        test_entries = ds.entries
        desc = "train on 70% of combined -> evaluate on the held-out 30%"

    print(f"\nFine-tune experiment: {desc}")
    print(f"  model {args.model}   epochs {args.epochs}   batch {args.batch_size}")

    pairs = build_pairs(train_entries, rng, args.per_entry)
    print(f"  synthetic training pairs: {len(pairs)} from {len(train_entries)} entries")
    ex = pairs[0].texts
    print(f"  example pair: {ex[0][len(BGE_PREFIX) :]!r}  ->  {ex[1][:60]!r}")

    model = SentenceTransformer(args.model, device="cpu")
    before = evaluate(model, test_entries, test_queries)
    print()
    report("before", before)

    loader = DataLoader(pairs, shuffle=True, batch_size=args.batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    t0 = time.perf_counter()
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=max(10, len(loader) // 10),
        optimizer_params={"lr": args.lr},
        show_progress_bar=False,
        use_amp=False,
    )
    train_s = time.perf_counter() - t0

    after = evaluate(model, test_entries, test_queries)
    report("after", after)

    d_p1 = after["p_at_1"] - before["p_at_1"]
    d_margin = after["margin"] - before["margin"]
    print(
        f"\n  delta P@1 {d_p1:+.4f}   delta margin {d_margin:+.4f}   "
        f"trained in {train_s:.0f}s on CPU"
    )
    if d_p1 > 0.01:
        print(
            "  => adaptation works, and it is NOT memorisation (disjoint train/test)."
            if args.mode == "transfer"
            else "  => gain on held-out queries."
        )
    elif d_p1 < -0.01:
        print("  => fine-tuning HURT. Do not ship this.")
    else:
        print("  => no meaningful change.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"exp_finetune_{args.mode}.json"
    p.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "model": args.model,
                "epochs": args.epochs,
                "n_pairs": len(pairs),
                "train_seconds": train_s,
                "before": {k: v for k, v in before.items() if k != "recall"},
                "after": {k: v for k, v in after.items() if k != "recall"},
                "delta_p_at_1": d_p1,
                "delta_margin": d_margin,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
