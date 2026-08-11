"""
benchmarks.exp_governed_abbrev | Layer: BENCHMARK
Does a GOVERNED (exact-by-construction) expander recover the accuracy that a naming
standard's abbreviations destroy?

Why this experiment exists
--------------------------
Two earlier attempts at abbreviation handling in this repo lost accuracy:

  * query-side expansion with a GUESSING expander        -2.0 points P@1
  * dictionary-side alias generation                     +1.9 @688, -18.8 @30k

`acronymkit.expand_identifier` against a `GovernedDictionary` is a different mechanism
from either. It never guesses: a token in the caller's catalog expands with
`is_known=True, confidence=1.0, source=governed`; a token that is not passes through as
its own raw text with `confidence=0.0`. So the -2.0 result, measured against a guesser,
does not automatically transfer -- and that is a hypothesis to test, not a conclusion.

The public benchmarks cannot show the win directly, because their field names are full
words or ad-hoc truncations and NO approved-abbreviation catalog exists for them, so
`expand_identifier` would pass everything through and change nothing. This script
therefore builds the missing half of the world: it generates a naming standard, rewrites
the benchmark's field names through it, and measures three conditions on the same
queries and the same corpus.

  (i)   ORIGINAL   original field names, no expansion        -- the ceiling
  (ii)  ABBREV     standard-abbreviated names, no expansion  -- the damage
  (iii) GOVERNED   abbreviated names + expand_identifier     -- the recovery

The dictionary side never changes. Only the query text does. That is the point: this
isolates the query representation, exactly as exp_query_repr.py does, and never touches
the index, so it cannot reproduce the alias max-pool failure by accident.

Catalog degradation
-------------------
A real catalog is incomplete and partly stale, so a result that only holds for a perfect
catalog is a toy. Conditions (iii) are re-measured with 75/50/25% of the catalog present
and with a fraction of entries WRONG -- mapped to a plausible but incorrect long form
borrowed from elsewhere in the same catalog. A wrong entry is the -2.0 failure mode
returning by another route, and it is measured, not argued about.

Statistics
----------
Every comparison is PAIRED over the same queries and tested with an exact two-sided
McNemar (binomial on the discordant pairs). H-007 in this repo records a published false
regression from an unpaired comparison, and a fixture that read a change as -1.33 points
where the full corpus read the same change as +0.58. Full corpus only, no fixtures.

Vocabulary
----------
The generated standard is derived mechanically from the PUBLIC benchmark field names
(BIRD-SQL / OMOP / FHIR). No real organisation's approved abbreviations appear here and
none may ever be added.

Usage
-----
    python benchmarks/exp_governed_abbrev.py --benchmark combined --save
    python benchmarks/exp_governed_abbrev.py --benchmark fhir --retrieval dense --save
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import RESULTS, Dataset, Query

from nexus_matcher.domain.services.abbreviation import (
    AbbreviationDictionary,
    AbbreviationExpander,
)

BGE_PREFIX = "Represent this sentence for searching relevant passages: "
VOWELS = set("aeiou")


# =============================================================================
# 1. THE GENERATED NAMING STANDARD
# =============================================================================


def split_identifier(name: str) -> list[str]:
    """person_idFoo -> ['person', 'id', 'Foo']; handles snake, camel, digits."""
    s = re.sub(r"[^0-9A-Za-z]+", " ", name)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    s = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", s)
    return [t for t in s.split() if t]


def _candidate_forms(word: str) -> list[str]:
    """
    The forms a naming standard actually reaches for, in the order it reaches for them.

    Vowel-drop first ("customer" -> "cstmr" -> "CSTM"), then plain truncation
    ("customer" -> "CUST"), then progressively longer truncations as a tie-break. This
    is a mechanical rule, not a curated list, so it cannot encode anyone's real standard.
    """
    w = word.lower()
    devowelled = w[0] + "".join(c for c in w[1:] if c not in VOWELS)
    forms = []
    for n in (4, 5):
        if len(devowelled) >= 3:
            forms.append(devowelled[:n])
    for n in (4, 5, 6, 7):
        forms.append(w[:n])
    forms.append(w)
    out: list[str] = []
    for f in forms:
        if len(f) >= 2 and f not in out:
            out.append(f)
    return out


def build_standard(words: list[str], min_len: int = 4) -> dict[str, str]:
    """
    ABBREV -> Canonical Long Form, injective by construction.

    Words shorter than `min_len` are left alone and get NO catalog row, so they pass
    through `expand_identifier` untouched -- which is what a real standard does with
    "id", "sex", "age".

    Injectivity is the property that makes the catalog governed: one short form, one
    long form. Assignment walks a sorted vocabulary so it does not depend on set order.
    """
    standard: dict[str, str] = {}
    taken: set[str] = set()
    for word in sorted(set(words)):
        if len(word) < min_len or not word.isalpha():
            continue
        for form in _candidate_forms(word):
            if form not in taken:
                taken.add(form)
                standard[form.upper()] = word.capitalize()
                break
    return standard


def collect_vocabulary(queries: list[Query]) -> list[str]:
    """Every alphabetic word appearing in a field name or its parent path."""
    words: list[str] = []
    for q in queries:
        for source in (q.field_name, q.parent_path):
            words.extend(t.lower() for t in split_identifier(source))
    return words


def abbreviate(name: str, long_to_short: dict[str, str]) -> str:
    """Rewrite a name through the standard: 'service period' -> 'SRVC_PRD'."""
    toks = split_identifier(name)
    return "_".join(long_to_short.get(t.lower(), t).upper() for t in toks)


# =============================================================================
# 2. THE THREE QUERY REPRESENTATIONS
# =============================================================================
#
# All three use the SAME shape -- parent context then field name -- because
# EXP-QUERY-REPR measured parent context as worth +20.1 points and dropping it here
# would make every condition worse for a reason that has nothing to do with
# abbreviation. Only the NAMES differ between conditions.


def text_original(q: Query) -> str:
    parent = " ".join(split_identifier(q.parent_path))
    field = " ".join(split_identifier(q.field_name))
    return f"{parent} {field}".strip()


def text_abbrev(q: Query, l2s: dict[str, str]) -> str:
    parent = " ".join(split_identifier(abbreviate(q.parent_path, l2s)))
    field = " ".join(split_identifier(abbreviate(q.field_name, l2s)))
    return f"{parent} {field}".strip()


def text_governed(q: Query, l2s: dict[str, str], catalog: AbbreviationExpander) -> str:
    parent = catalog.expand(abbreviate(q.parent_path, l2s)).expanded
    field = catalog.expand(abbreviate(q.field_name, l2s)).expanded
    return " ".join(split_identifier(f"{parent} {field}".strip()))


def text_keep_both(q: Query, l2s: dict[str, str], catalog: AbbreviationExpander) -> str:
    """
    Expansion APPENDED to the raw abbreviated name rather than replacing it.

    This is the hedge worth measuring before shipping any hook. If the catalog row is
    wrong, replacement destroys the only true signal the query had; appending leaves the
    original token in the string, so a stale row costs dilution instead of corruption.
    The cost is a longer, noisier query, which EXP-QUERY-REPR already showed this model
    dislikes -- appending scalar type words cost 2.1 points. So this is not free and the
    question is which effect is larger.
    """
    ab_parent = abbreviate(q.parent_path, l2s)
    ab_field = abbreviate(q.field_name, l2s)
    raw = f"{ab_parent.replace('_', ' ')} {ab_field.replace('_', ' ')}".strip()
    exp = " ".join(
        split_identifier(
            f"{catalog.expand(ab_parent).expanded} {catalog.expand(ab_field).expanded}".strip()
        )
    )
    return f"{raw} {exp}".strip()


def text_guessed(q: Query, l2s: dict[str, str], expander) -> str:
    """
    The SAME damaged names, expanded by this repo's own guessing expander.

    This is the arm that makes the comparison non-circular. `expand_identifier` with a
    complete catalog reconstructs the original words almost exactly -- it is exact by
    construction, so of course it scores like the ceiling, and that number on its own
    proves only that the plumbing is lossless. What is NOT settled by construction is
    whether a hand-written expansion table, which is what cost -2.0 points here before,
    can do the same job. Same input, same pipeline, different expander.
    """
    parent = expander.expand(abbreviate(q.parent_path, l2s).replace("_", " ")).expanded
    field = expander.expand(abbreviate(q.field_name, l2s).replace("_", " ")).expanded
    return f"{parent} {field}".strip()


# =============================================================================
# 3. CATALOG DEGRADATION
# =============================================================================


def degrade(
    standard: dict[str, str],
    coverage: float,
    wrong_rate: float,
    seed: int,
) -> dict[str, str]:
    """
    A realistically imperfect copy of the standard.

    `coverage` < 1 drops rows -- the catalog has not caught up with the schema, and the
    missing tokens pass through as raw abbreviation text.

    `wrong_rate` > 0 repoints rows at a long form belonging to a DIFFERENT token in the
    same catalog. That is what stale looks like: not gibberish, a plausible business word
    that is wrong here. This is the condition under which a governed expander degenerates
    into the guesser that cost -2.0 points, so it is the one that decides the question.
    """
    rng = random.Random(seed)
    keys = sorted(standard)
    keep_n = round(len(keys) * coverage)
    kept = sorted(rng.sample(keys, keep_n))
    out = {k: standard[k] for k in kept}

    if wrong_rate > 0 and len(kept) > 1:
        n_wrong = round(len(kept) * wrong_rate)
        victims = rng.sample(kept, n_wrong)
        pool = [standard[k] for k in keys]
        for v in victims:
            replacement = standard[v]
            for _ in range(10):
                cand = rng.choice(pool)
                if cand != standard[v]:
                    replacement = cand
                    break
            out[v] = replacement
    return out


def make_catalog(mapping: dict[str, str]) -> AbbreviationExpander:
    """The caller's approved-abbreviation catalog, loaded into the SHIPPED expander.

    This experiment was first written against a third-party governed-naming library. It
    was then measured against `AbbreviationExpander` fed the same catalog and the two were
    indistinguishable -- identical query text on 688/688 and 1556/1556 queries, and 0
    discordant pairs at every point of the coverage and wrong-rate sweeps. So the
    dependency was dropped and the experiment rewritten onto the expander that ships,
    which is the one a reader can actually run.

    `lookup` is an exact dict hit and an unknown token passes through unchanged, so a
    catalog miss costs nothing and a catalog row is the only thing that can assert an
    expansion. That property is what the wrong-rate sweep below is measuring the limits of.
    """
    return AbbreviationExpander(AbbreviationDictionary.from_dict(mapping))


# =============================================================================
# 4. RETRIEVAL + METRICS
# =============================================================================


@dataclass
class Run:
    label: str
    hits1: np.ndarray  # bool per query: gold ranked first
    p_at_1: float
    r_at_5: float
    mrr_at_10: float
    r_at_10: float


def score(rankings: list[list[str]], golds: list[str], label: str) -> Run:
    n = len(golds)
    hits1 = np.zeros(n, dtype=bool)
    p5 = mrr = r10 = 0.0
    for i, (ranked, gold) in enumerate(zip(rankings, golds, strict=False)):
        rank = None
        for j, doc_id in enumerate(ranked, 1):
            if doc_id == gold:
                rank = j
                break
        if rank is None:
            continue
        if rank == 1:
            hits1[i] = True
        if rank <= 5:
            p5 += 1
        if rank <= 10:
            mrr += 1.0 / rank
            r10 += 1
    return Run(
        label=label,
        hits1=hits1,
        p_at_1=float(hits1.sum()) / n,
        r_at_5=p5 / n,
        mrr_at_10=mrr / n,
        r_at_10=r10 / n,
    )


class Retriever:
    """Dense BGE, optionally fused with BM25 exactly as the shipped matcher fuses."""

    def __init__(self, ds: Dataset, model_name: str, mode: str, top_k: int, alpha: float):
        from sentence_transformers import SentenceTransformer

        self.ids = [e.id for e in ds.entries]
        self.top_k = min(top_k, len(self.ids))
        self.mode = mode
        self.alpha = alpha
        self.model = SentenceTransformer(model_name, device="cpu")

        corpus = [e.searchable_text("business+desc") for e in ds.entries]
        self.doc_emb = self.model.encode(
            corpus, batch_size=128, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        if mode == "hybrid":
            import bm25s

            self.bm25 = bm25s.BM25()
            self.bm25.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)

    def rank(self, texts: list[str]) -> list[list[str]]:
        q_emb = self.model.encode(
            [BGE_PREFIX + t for t in texts],
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        sims = q_emb @ self.doc_emb.T
        k = self.top_k
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        dense: list[list[tuple[str, float]]] = []
        for row, cand in zip(sims, part, strict=False):
            order = cand[np.argsort(-row[cand])]
            dense.append([(self.ids[i], float(row[i])) for i in order])

        if self.mode == "dense":
            return [[d for d, _ in row] for row in dense]

        import bm25s

        from nexus_matcher.core.fusion import fuse_linear_ids

        tokens = bm25s.tokenize(texts, show_progress=False)
        docs, scores = self.bm25.retrieve(tokens, k=k, show_progress=False)
        out: list[list[str]] = []
        for drow, srow, dense_row in zip(docs, scores, dense, strict=False):
            sparse = [(self.ids[j], float(s)) for j, s in zip(drow, srow, strict=False)]
            fused = fuse_linear_ids(
                dense_row,
                sparse,
                semantic_weight=self.alpha,
                lexical_weight=1.0 - self.alpha,
                normalize_scores=True,
            )
            out.append([d for d, _ in fused])
        return out


def mcnemar(a: Run, b: Run) -> dict:
    """
    Exact two-sided McNemar on hit@1. `a` is the new condition, `b` the reference.

    b_gain = queries `a` gets right that `b` gets wrong; b_loss = the reverse. The test
    is on those discordant pairs alone -- the concordant ones carry no information about
    the difference, which is the whole reason an unpaired comparison on this data lies.
    """
    from scipy.stats import binomtest

    gain = int(np.sum(a.hits1 & ~b.hits1))
    loss = int(np.sum(~a.hits1 & b.hits1))
    n_disc = gain + loss
    if n_disc == 0:
        p = 1.0
    else:
        p = float(binomtest(min(gain, loss), n_disc, 0.5, alternative="two-sided").pvalue)
    return {
        "delta_p_at_1": a.p_at_1 - b.p_at_1,
        "gained_at_1": gain,
        "lost_at_1": loss,
        "discordant": n_disc,
        "mcnemar_p": p,
    }


# =============================================================================
# 5. MAIN
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="combined")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--retrieval", default="both", choices=["dense", "hybrid", "both"])
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    ds = Dataset.load(args.benchmark)
    golds = [q.gold_id for q in ds.queries]

    # ---- build the standard --------------------------------------------------------
    vocab = collect_vocabulary(ds.queries)
    standard = build_standard(vocab)  # ABBREV -> Long
    long_to_short = {v.lower(): k for k, v in standard.items()}

    n_tok = len(vocab)
    n_abbreviated = sum(1 for w in vocab if w in long_to_short)
    changed = sum(
        1 for q in ds.queries if text_abbrev(q, long_to_short).lower() != text_original(q).lower()
    )

    print(f"\nBenchmark {ds.name}: {len(ds.entries)} entries, {len(ds.queries)} queries")
    print(f"Generated standard: {len(standard)} approved abbreviations")
    print(f"  query name tokens           : {n_tok}")
    print(f"  tokens the standard rewrites: {n_abbreviated} ({n_abbreviated / n_tok:.1%})")
    print(f"  queries whose name changes  : {changed} ({changed / len(ds.queries):.1%})")
    sample = sorted(standard.items())[:12]
    print(f"  sample rows: {', '.join(f'{k}->{v}' for k, v in sample)}")

    full_catalog = make_catalog(standard)
    from nexus_matcher.domain.services.abbreviation import AbbreviationExpander

    guess_expander = AbbreviationExpander.default()
    ex = ds.queries[0]
    print("\n  example query")
    print(f"    (i)   original : {text_original(ex)!r}")
    print(f"    (ii)  abbrev   : {text_abbrev(ex, long_to_short)!r}")
    print(f"    (iii) governed : {text_governed(ex, long_to_short, full_catalog)!r}")
    print(f"    (iv)  guessing : {text_guessed(ex, long_to_short, guess_expander)!r}")

    modes = ["dense", "hybrid"] if args.retrieval == "both" else [args.retrieval]
    artifact: dict = {
        "benchmark": ds.name,
        "n_entries": len(ds.entries),
        "n_queries": len(ds.queries),
        "model": args.model,
        "index_fields": "business+desc",
        "standard": {
            "n_rows": len(standard),
            "vocab_tokens": n_tok,
            "tokens_rewritten": n_abbreviated,
            "queries_changed": changed,
            "sample": dict(sample),
        },
        "retrieval": {},
    }

    for mode in modes:
        print(
            f"\n{'=' * 78}\nRETRIEVAL: {mode}"
            + (f" (alpha={args.alpha})" if mode == "hybrid" else "")
        )
        print("=" * 78)
        t0 = time.perf_counter()
        r = Retriever(ds, args.model, mode, args.top_k, args.alpha)
        print(f"  index built in {time.perf_counter() - t0:.1f}s")

        runs: dict[str, Run] = {}

        def measure(
            label: str,
            texts: list[str],
            *,
            _r: Retriever = r,
            _runs: dict[str, Run] = runs,
        ) -> Run:
            # `_r` binds the retriever for THIS corpus at definition time. Closing over the
            # loop variable would make every call use the last corpus's index once the loop
            # advanced -- inert as written, since measure() is only called inside its own
            # iteration, but it is one moved line away from silently scoring FHIR against
            # the combined index.
            run = score(_r.rank(texts), golds, label)
            _runs[label] = run
            return run

        original = measure("original", [text_original(q) for q in ds.queries])
        abbrev = measure("abbrev", [text_abbrev(q, long_to_short) for q in ds.queries])
        governed = measure(
            "governed_full",
            [text_governed(q, long_to_short, full_catalog) for q in ds.queries],
        )
        guessed = measure(
            "guessing_expander",
            [text_guessed(q, long_to_short, guess_expander) for q in ds.queries],
        )
        # The -2.0 finding, re-measured on today's corpus with a paired test: this repo's
        # guessing expander applied to the ORIGINAL, un-abbreviated names. That is what
        # `expand_query_abbreviations = True` actually does in the shipped code, and it
        # is the number the "one wrong expansion corrupts the query vector" comment is
        # about. It has never been checked with McNemar.
        keep_both = measure(
            "keep_both_full",
            [text_keep_both(q, long_to_short, full_catalog) for q in ds.queries],
        )
        guessed_orig = measure(
            "guessing_on_original",
            [
                f"{guess_expander.expand(' '.join(split_identifier(q.parent_path))).expanded} "
                f"{guess_expander.expand(' '.join(split_identifier(q.field_name))).expanded}".strip()
                for q in ds.queries
            ],
        )

        gap = original.p_at_1 - abbrev.p_at_1
        rec = (governed.p_at_1 - abbrev.p_at_1) / gap if gap else float("nan")

        print(f"\n  {'condition':26s} {'P@1':>7s} {'R@5':>7s} {'MRR@10':>7s} {'R@10':>7s}")
        print(f"  {'-' * 26} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")
        for run in (original, abbrev, governed, keep_both, guessed, guessed_orig):
            print(
                f"  {run.label:26s} {run.p_at_1:7.4f} {run.r_at_5:7.4f} "
                f"{run.mrr_at_10:7.4f} {run.r_at_10:7.4f}"
            )

        vs_abbrev = mcnemar(governed, abbrev)
        vs_original = mcnemar(governed, original)
        damage = mcnemar(abbrev, original)
        guess_vs_abbrev = mcnemar(guessed, abbrev)
        governed_vs_guess = mcnemar(governed, guessed)
        guess_orig_vs_orig = mcnemar(guessed_orig, original)

        print("\n  paired exact McNemar on hit@1")
        for name, t in (
            ("(ii) abbrev vs (i) original", damage),
            ("(iii) governed vs (ii) abbrev", vs_abbrev),
            ("(iii) governed vs (i) original", vs_original),
            ("(iv) guessing vs (ii) abbrev", guess_vs_abbrev),
            ("(iii) governed vs (iv) guessing", governed_vs_guess),
            ("(v) guess-on-original vs (i)", guess_orig_vs_orig),
            ("keep-both vs (iii) governed", mcnemar(keep_both, governed)),
        ):
            print(
                f"    {name:33s} d={t['delta_p_at_1']:+.4f}  "
                f"gain={t['gained_at_1']:4d} loss={t['lost_at_1']:4d} "
                f"disc={t['discordant']:4d}  p={t['mcnemar_p']:.4g}"
            )
        print(f"\n  abbreviation gap (i)-(ii) = {gap:+.4f} P@1")
        print(f"  fraction of gap recovered by a FULL catalog = {rec:.1%}")
        if gap:
            print(
                "  fraction of gap recovered by the GUESSING expander = "
                f"{(guessed.p_at_1 - abbrev.p_at_1) / gap:.1%}"
            )

        # ---- degradation ------------------------------------------------------------
        # Each seed is a separate paired test against the SAME reference. The per-seed
        # results are NOT pooled: the seeds share their queries, so concatenating them
        # would multiply n without adding information and would shrink every p-value by
        # a factor that means nothing. Range across seeds is reported instead.
        print(
            f"\n  {'degraded catalog':22s} {'P@1':>7s} {'sd':>6s} {'recovery':>9s} "
            f"{'p vs abbrev (max)':>18s} {'p vs orig (min)':>16s} {'keep-both':>10s}"
        )
        print(f"  {'-' * 22} {'-' * 7} {'-' * 6} {'-' * 9} {'-' * 18} {'-' * 16} {'-' * 10}")
        degraded: dict[str, dict] = {}
        conditions = [
            ("coverage_75", 0.75, 0.0),
            ("coverage_50", 0.50, 0.0),
            ("coverage_25", 0.25, 0.0),
            ("wrong_05", 1.00, 0.05),
            ("wrong_10", 1.00, 0.10),
            ("wrong_25", 1.00, 0.25),
            # Past 25% wrong the question stops being "how much does it recover" and
            # becomes "is expansion now worse than leaving the abbreviation alone".
            # A catalog is only worth wiring in if that crossover is far from where a
            # real catalog sits.
            ("wrong_50", 1.00, 0.50),
            ("wrong_75", 1.00, 0.75),
            ("wrong_100", 1.00, 1.00),
            ("coverage_50_wrong_05", 0.50, 0.05),
        ]
        for label, cov, wrong in conditions:
            per_seed: list[Run] = []
            both_seed: list[Run] = []
            tests_ab: list[dict] = []
            tests_or: list[dict] = []
            for seed in range(args.seeds):
                mapping = degrade(standard, cov, wrong, seed)
                cat = make_catalog(mapping)
                run = score(
                    r.rank([text_governed(q, long_to_short, cat) for q in ds.queries]),
                    golds,
                    f"{label}_s{seed}",
                )
                per_seed.append(run)
                tests_ab.append(mcnemar(run, abbrev))
                tests_or.append(mcnemar(run, original))
                both_seed.append(
                    score(
                        r.rank([text_keep_both(q, long_to_short, cat) for q in ds.queries]),
                        golds,
                        f"{label}_both_s{seed}",
                    )
                )

            p1s = np.array([x.p_at_1 for x in per_seed])
            both_p1s = np.array([x.p_at_1 for x in both_seed])
            mean_p1 = float(p1s.mean())
            sd_p1 = float(p1s.std(ddof=1)) if len(p1s) > 1 else 0.0
            recovery = (mean_p1 - abbrev.p_at_1) / gap if gap else float("nan")
            p_ab = [t["mcnemar_p"] for t in tests_ab]
            p_or = [t["mcnemar_p"] for t in tests_or]
            print(
                f"  {label:22s} {mean_p1:7.4f} {sd_p1:6.4f} {recovery:8.1%} "
                f"{max(p_ab):18.4g} {min(p_or):16.4g} {both_p1s.mean():10.4f}"
            )
            degraded[label] = {
                "keep_both_p_at_1_mean": float(both_p1s.mean()),
                "keep_both_p_at_1_per_seed": [float(x) for x in both_p1s],
                "coverage": cov,
                "wrong_rate": wrong,
                "seeds": args.seeds,
                "p_at_1_mean": mean_p1,
                "p_at_1_sd": sd_p1,
                "p_at_1_per_seed": [float(x) for x in p1s],
                "r_at_5_mean": float(np.mean([x.r_at_5 for x in per_seed])),
                "mrr_at_10_mean": float(np.mean([x.mrr_at_10 for x in per_seed])),
                "recovery_fraction": recovery,
                "vs_abbrev_per_seed": tests_ab,
                "vs_original_per_seed": tests_or,
                "vs_abbrev_p_range": [min(p_ab), max(p_ab)],
                "vs_original_p_range": [min(p_or), max(p_or)],
            }

        # ---- how abbreviated does the schema have to be? -----------------------------
        # The gap above is measured against a standard that rewrites EVERY qualifying
        # token. No real schema is 100% governed, so the absolute point figure above is
        # an upper bound and is useless for estimating anyone's corpus. This sweep gives
        # the transfer function: subsample the standard so only a fraction of the
        # vocabulary is abbreviated at all, then measure the gap and the recovery with a
        # catalog that is COMPLETE for whatever standard was used.
        intensity: dict[str, dict] = {}
        print(
            f"\n  {'schema abbreviated':22s} {'P@1 (ii)':>9s} {'P@1 (iii)':>10s} "
            f"{'gap':>8s} {'recovery':>9s} {'p (iii vs ii)':>14s}"
        )
        print(f"  {'-' * 22} {'-' * 9} {'-' * 10} {'-' * 8} {'-' * 9} {'-' * 14}")
        for frac in (0.25, 0.50, 0.75):
            gaps, recs, p1_ab, p1_gv, ps = [], [], [], [], []
            for seed in range(args.seeds):
                sub = degrade(standard, frac, 0.0, 1000 + seed)
                l2s_sub = {v.lower(): k for k, v in sub.items()}
                sub_cat = make_catalog(sub)
                a = score(r.rank([text_abbrev(q, l2s_sub) for q in ds.queries]), golds, "a")
                g = score(
                    r.rank([text_governed(q, l2s_sub, sub_cat) for q in ds.queries]),
                    golds,
                    "g",
                )
                gp = original.p_at_1 - a.p_at_1
                gaps.append(gp)
                recs.append((g.p_at_1 - a.p_at_1) / gp if gp else float("nan"))
                p1_ab.append(a.p_at_1)
                p1_gv.append(g.p_at_1)
                ps.append(mcnemar(g, a)["mcnemar_p"])
            print(
                f"  {f'{frac:.0%} of vocabulary':22s} {np.mean(p1_ab):9.4f} "
                f"{np.mean(p1_gv):10.4f} {np.mean(gaps):8.4f} {np.mean(recs):8.1%} "
                f"{max(ps):14.4g}"
            )
            intensity[f"intensity_{int(frac * 100)}"] = {
                "fraction_of_vocabulary_abbreviated": frac,
                "p_at_1_abbrev_mean": float(np.mean(p1_ab)),
                "p_at_1_governed_mean": float(np.mean(p1_gv)),
                "gap_mean": float(np.mean(gaps)),
                "gap_sd": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0,
                "recovery_mean": float(np.mean(recs)),
                "mcnemar_p_range": [float(min(ps)), float(max(ps))],
            }
        print(
            f"  {'100% of vocabulary':22s} {abbrev.p_at_1:9.4f} {governed.p_at_1:10.4f} "
            f"{gap:8.4f} {rec:8.1%} {vs_abbrev['mcnemar_p']:14.4g}"
        )

        artifact["retrieval"][mode] = {
            "intensity_sweep": intensity,
            "alpha": args.alpha if mode == "hybrid" else None,
            "original": {
                "p_at_1": original.p_at_1,
                "r_at_5": original.r_at_5,
                "mrr_at_10": original.mrr_at_10,
                "r_at_10": original.r_at_10,
            },
            "abbrev": {
                "p_at_1": abbrev.p_at_1,
                "r_at_5": abbrev.r_at_5,
                "mrr_at_10": abbrev.mrr_at_10,
                "r_at_10": abbrev.r_at_10,
            },
            "governed_full": {
                "p_at_1": governed.p_at_1,
                "r_at_5": governed.r_at_5,
                "mrr_at_10": governed.mrr_at_10,
                "r_at_10": governed.r_at_10,
            },
            "guessing_expander": {
                "p_at_1": guessed.p_at_1,
                "r_at_5": guessed.r_at_5,
                "mrr_at_10": guessed.mrr_at_10,
                "r_at_10": guessed.r_at_10,
            },
            "keep_both_full": {
                "p_at_1": keep_both.p_at_1,
                "r_at_5": keep_both.r_at_5,
                "mrr_at_10": keep_both.mrr_at_10,
                "r_at_10": keep_both.r_at_10,
                "vs_governed": mcnemar(keep_both, governed),
            },
            "guessing_on_original": {
                "p_at_1": guessed_orig.p_at_1,
                "r_at_5": guessed_orig.r_at_5,
                "mrr_at_10": guessed_orig.mrr_at_10,
                "r_at_10": guessed_orig.r_at_10,
            },
            "abbreviation_gap_p_at_1": gap,
            "recovery_fraction_full_catalog": rec,
            "recovery_fraction_guessing": ((guessed.p_at_1 - abbrev.p_at_1) / gap if gap else None),
            "mcnemar": {
                "abbrev_vs_original": damage,
                "governed_vs_abbrev": vs_abbrev,
                "governed_vs_original": vs_original,
                "guessing_vs_abbrev": guess_vs_abbrev,
                "governed_vs_guessing": governed_vs_guess,
                "guessing_on_original_vs_original": guess_orig_vs_orig,
            },
            "degraded": degraded,
        }

    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        p = RESULTS / f"exp_governed_abbrev_{ds.name}.json"
        p.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
