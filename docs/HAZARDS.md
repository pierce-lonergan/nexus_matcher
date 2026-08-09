# Standing hazards

A **defect** is a wrong line. You fix it once and add a regression test.

A **hazard** is a property of this problem domain that will keep producing wrong lines
forever. Fixing one instance teaches you nothing durable, because the next instance looks
completely different.

This file exists because a known hazard recurred for the **third** time and was caught
only because an agent happened to read a notes file outside the repository. That is luck,
and luck does not scale.

Each entry carries a **status**. Five of seven have an executable check in
`tests/hazards/`; H-002 has none and H-003 is enforced only by hand. Those two are
stated as unguarded rather than quietly listed alongside the rest, because a ledger
that overstates its own coverage is worse than no ledger -- it is the same failure as
a green CI over a broken release.

**Rules**

- An entry is never deleted. It is marked *guarded* once a check exists, and the check is
  named. A hazard that stopped mattering is a hazard nobody has triggered lately.
- Every change reports which hazards it was evaluated against.
- When something escapes, the first question is not "which test was missing" but **"was
  this a known hazard with no check?"** If it was, that is a failure of this file, and it
  is a worse failure than a missing regression test — because the knowledge was already
  in the building.

---

## H-001 — Better retrieval lowers auto-approve precision at a fixed threshold

**Status:** guarded — `tests/hazards/test_h001_retrieval_vs_decision.py`

Improving retrieval shifts the whole score distribution upward. The auto-approve threshold
does not move with it, so more candidates cross a fixed bar — including wrong ones. Recall
goes up, and the metric that decides whether a PII label is applied *without a human*
goes down.

**Occurrences — three, and counting**

| # | Change | Effect |
|---|---|---|
| 1 | threshold re-derivation during tuning | noted in `MatchingConfig`: "this happened twice during tuning" |
| 2 | dictionary aliasing | caps achievable auto-approve precision at ~91% via max-pool |
| 3 | ContextEnricher level-wise dedup | P@1 **+0.0154** (p=0.018, genuine) while auto-approve precision fell **0.8049 → 0.7391**. Auto-approvals 41 → 46; **5 of the 7 new ones were wrong**; wrong-and-unreviewed went 8 → 12 |

Occurrence 3 was *claimed* as +0.030 precision. The sign was inverted. It was caught in
adjudication, not by any gate.

**Why it keeps happening:** the headline metric and the decision metric are measured on the
same run but reported separately, and a change that improves the headline reads as a win.

**The check:** a quality report that emits P@1 without auto-approve precision is rejected,
and a divergent-sign combination — retrieval up, decision down — fails the gate *even when
P@1 improves*.

---

## H-002 — A threshold calibrated on one corpus does not transfer

**Status:** UNGUARDED — no executable check yet. Tracked in `docs/DEFENSIBILITY.md`.

`auto_approve_threshold = 0.87` was calibrated on 688 labelled fields. `MatchingConfig`
already warns these numbers "move with the retriever AND with the benchmark."

Corpus size is not a scaling factor here, it is a *regime change*. Dictionary aliasing is
worth **+1.9** P@1 at 688 entries and **−18.8** at 30,000 — the sign inverts. Anything
validated at one corpus size is unvalidated at another.

**The check:** any recorded quality result carries its corpus identity and size, and a
comparison across different corpora is refused rather than silently performed.

---

## H-003 — Optimizations "fix" artifacts of their own measurement environment

**Status:** PARTIALLY GUARDED — the rule is written and was applied by hand to reject
the small-corpus fallback, but nothing enforces it automatically. Tracked in
`docs/DEFENSIBILITY.md`.

A small-corpus fallback was added to `search_batch` because the GEMM path lost at 793
entries. The hypothesis was cache behaviour. It was wrong:

```
 1 thread:   loop 14.7 ms   GEMM   7.4 ms   -> GEMM 2.00x FASTER
 4 threads:  loop 17.3 ms   GEMM   6.9 ms   -> GEMM 2.51x FASTER
24 threads:  loop 16.5 ms   GEMM 176.1 ms   -> GEMM 0.09x
```

Identical FLOPs. The loss only exists when OpenBLAS dispatches 24 threads across a small
GEMM **on a saturated box** — which is exactly what 22 concurrent agents create. The
optimization was fixing its own measurement conditions.

**The check:** any accepted optimization whose mechanism touches threading, BLAS, or batch
scheduling must demonstrate its win at 1 thread *and* at N threads.

---

## H-004 — Differential tests are invariant to errors both sides share

**Status:** guarded — `tests/hazards/test_h004_oracle_absoluteness.py`

A GEMM orientation flip passed **19 of 19** tests. Those tests compared the batched path
against the per-query loop, and the transpose was applied consistently, so both sides moved
together and agreement held perfectly while both were wrong.

An implementation-vs-implementation oracle can only detect *divergence*. It is structurally
blind to any error the two implementations share — which includes every error in the shared
assumption they were built from.

**The check:** every oracle test also pins **absolute expected values** on a small
hand-verified fixture, and matrix-shaped code paths are exercised with **non-square,
non-symmetric** inputs so a transpose cannot pass unnoticed.

---

## H-005 — The 0.0024 margin means ties are everywhere

**Status:** guarded — `tests/hazards/test_h005_total_order.py`

Measured cosine to the gold entry: **0.7657**. To the nearest wrong entry: **0.7633**. A
margin of **0.0024**.

At that separation, any iteration order that leaks into ranking changes the answer. Ranking
did in fact depend on `PYTHONHASHSEED` until the deterministic tie-break landed. It also
means near-ties are the normal case, not an edge case, so silently picking one is
frequently silently picking wrong.

**The check:** ranking is total and seed-independent — identical across `PYTHONHASHSEED`
values and across processes.

---

## H-006 — Cross-lane work leaves dead or half-wired code

**Status:** guarded — `tests/hazards/test_h006_reachability.py`

Parallel work partitioned by file produces changes whose two halves land in different
lanes. Three occurrences:

1. the version bump fell between the lane owning `pyproject.toml` and the lane owning
   `__init__.py`; the tree built 2.0.0 while the changelog claimed 2.0.1
2. `requirements.txt` still carried both packaging defects already fixed in `pyproject.toml`
3. `search_batch()` shipped **2.5× faster and with no caller at all** — the call site was
   in another lane's file

**The check:** no new public symbol without a caller in `src/` or an explicit public-API
registration. Reconciliation after a parallel phase is an owned step, not a hope.

---

## H-007 — The noise band is a function of machine state, not a constant

**Status:** guarded — `benchmarks/optimization_ledger.py` machine-state precondition

Measured on this machine, identical code, three runs:

| machine state | match throughput band |
|---|---|
| 49.5% CPU busy | **30.6%** (276 → 193 → 272 fields/sec) |
| idle | **0.9%** |

Index throughput swung 407 → 528 entries/sec on identical code under load. A fixed
tolerance is therefore wrong in both directions: too loose when idle to catch a real
regression, too tight under load to avoid inventing one.

This repo has already recorded a false regression from exactly this — and then, correcting
itself, got the *direction* wrong too: measured idle, the pre-BM25 commit gives 550
entries/sec, so the 715 originally recorded was the outlier, not the 520.

**The check:** the ledger measures its own noise floor at run time and refuses to record a
timing above a CPU-busy threshold, marking it UNMEASURABLE rather than writing down noise.
Quality metrics measured **0.0%** spread and are exempt.
