# Defensibility

Every gate in this repository: what it catches, **the evidence it has been observed
failing**, and the holes that are still open.

A gate nobody has watched go red is a hypothesis, not a gate. Two shipped here that could
not fail under any circumstance — a publish step carrying `continue-on-error`, and
nineteen tests that did not notice a transposed matrix multiply. Both read as coverage.

## Gates

| Gate | Catches | Observed failing against |
|---|---|---|
| `scripts/museum_replay.py` | any historical defect returning | each entry's own replay — green → red → green, asserted per run |
| `tests/museum/NM-0005` | a field losing its governance classification | reverting the result key to `full_path` |
| `tests/museum/NM-0016` | an encoder returning noise | injecting seeded random vectors into `_encode` |
| `tests/museum/NM-0023` | a gate wired to only part of the repo | narrowing the CI lint command back to one directory |
| `tests/meta/test_ci_has_teeth.py` | `continue-on-error`, `\|\| true`, `\|\| echo`, `set +e`, gate steps under `always()`, and steps installing the dependency whose absence they test | **re-adding the literal 2.0.0 publish step** — fires on two independent grounds |
| `tests/hazards/test_h001` | a change raising P@1 while lowering auto-approve precision | a synthetic +0.02 / −0.05 candidate → `REGRESSION` |
| `tests/hazards/test_h004` | a transpose that both sides of a differential test share | non-square, non-symmetric fixtures with hand-verified absolute values |
| `tests/hazards/test_h005` | ranking that depends on `PYTHONHASHSEED` | four seeds × separate interpreters |
| `tests/hazards/test_h006` | a public symbol shipping with no caller | `search_batch`, pinned by name after it shipped dead |
| `tests/regression/` | retrieval quality silently degrading | query vectors truncated to 64 dims, noise added, doc dims zeroed past 128 — all three trip P@1, R@5 and MRR |
| `tests/unit/application/test_batched_dense_retrieval.py` | batched retrieval diverging from per-field | scatter-back off-by-one, trusting a short batch, and skipping batching entirely — all red |
| `scripts/release_preflight.py` | a wheel that does not work on a bare install | the **published 2.0.0 wheel** — fails 5 checks, exits 1 |

## Ratchets

| Ratchet | Value | Rule |
|---|---|---|
| accuracy tolerance | P@1 / R@5 / MRR, ±0.02 | may tighten, never loosen; baselines re-recorded from the test's own code path |
| guarded metrics | P@1 −0.005, auto-approve precision −0.010, R@10 −0.010, memory +10% | may tighten, never loosen |
| unreferenced public methods | 85 | may fall, may never rise |
| CI soft steps | 1 (advisory MyPy) | each needs a written reason in `.ci-exceptions.yaml` and a deletion condition |
| lint exemptions | 6 rules × `benchmarks/suite_*.py` | list may shrink, never grow |

## Holes still open

Stated plainly, because an unlisted hole is indistinguishable from a closed one.

| Hole | Severity | Why it is still open |
|---|---|---|
| **H-002 unguarded** — a threshold calibrated on one corpus does not transfer | MEDIUM | needs quality records to carry corpus identity and refuse cross-corpus comparison; not built |
| **H-003 hand-enforced** — optimizations that fix artifacts of their own measurement environment | MEDIUM | the 1-thread/N-thread rule rejected the small-corpus fallback by hand; nothing automates it |
| 8 museum entries not yet built (NM-0001..0004, 0006..0015, 0017..0022) | MEDIUM | the machinery and format are proven on 3; the rest is mechanical |
| 85 unreferenced public methods | LOW | a name count cannot prove reachability; needs a call graph |
| `presentation/api/__init__.py` imports fastapi eagerly | MEDIUM | NM-0007 one level down — `from nexus_matcher.presentation.api import ...` still raises on a bare install |
| `_EXTRA_FOR_MODULE` covers 8 of 15 extras | LOW | a missing optional dep in the other 7 gives a raw `ModuleNotFoundError` |
| README-link regression test never shipped | LOW | it exists only in a scratchpad; `release_preflight` covers the same ground |
| `parsers` / `loaders` extras declare deps nothing imports | LOW | dead extras; needs the extras-graph check |
| 140 mypy errors across 29 files | LOW | advisory, tracked in `.ci-exceptions.yaml` with a deletion condition |
| no mutation testing | **HIGH** | the only mechanical way to find tests invariant to the bug they claim to cover — which is precisely the 19/19 case. Not built |

## The protocol

When anything escapes:

1. reproduce it, and land the failing test in the same commit as the fix
2. add a museum entry with a replay, and **watch `museum_replay.py` go red**
3. name the gate that should have caught it, and say how that gate is now stronger
4. ask whether it was a **known hazard with no check** — if so that is a `HAZARDS.md`
   failure, which is worse than a missing test, because the knowledge was already here
