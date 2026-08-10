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

## Second-round adjudication (Waves B–D)

Six lanes built the machinery below. An independent adjudicator per lane then re-applied
every claimed proof and hunted for gates that *cannot* be made to fail.

**Every claimed proof reproduced — 100+ mutations, across all six lanes, zero unreproducible.**
The lanes were honest about what they had proven. What they missed is another matter, and
five of six came back `TOOTHLESS_GATES`.

That result is the system working. A first pass writes gates; only a second, adversarial
pass discovers which of them are decoration. Recording the gap is the point — an unlisted
hole is indistinguishable from a closed one.

### Toothless spots found — third round

| Where | What cannot fail | Severity |
|---|---|---|
| ~~`tests/properties/` — the `content_hash` oracle~~ | **CLOSED.** Now pins the documented contract absolutely — a governance-only edit must encode zero texts — instead of deriving its expectation from the hash it is testing. Proven: the mutation that survived now goes red | ~~HIGH~~ |
| ~~`test_sync_state_machine.py`~~ | **CLOSED.** A full re-embed produces an identical final state, so no state comparison could see it. The machine now counts WORK DONE — the multiset of texts actually encoded — and asserts `report.embedded` matches | ~~HIGH~~ |
| ~~`benchmarks/optimization_ledger.py`~~ | **CLOSED, 7/7 reproduced.** Two narrower spots remain: the WARNING (not the verdict) on the thread-rule path, and two message-text assertions in the cross-benchmark refusal | MEDIUM |
| ~~`setup.cfg` mutation ratchets~~ | **CLOSED.** Scope is now pinned alongside the score. One narrower hole remains: a three-way consistent shrink (paths + group paths + scope pin together) still satisfies it | MEDIUM |
| ~~`noxfile.py` stall detector~~ | **CLOSED.** A truncated run is now reported as TRUNCATED rather than scored | MEDIUM |
| ~~`tests/museum/NM-0009`~~ | **CLOSED.** Assertion now checks what the docstring promised | ~~LOW~~ |
| ~~`tests/museum/NM-0001`~~ | **CLOSED.** Renamed and documented as a premise guard, not a gate | ~~LOW~~ |

### Vacuous-pass risks found

Tests that silently do not run are worse than absent ones, because the count looks healthy.

- `pytest.importorskip("tomli")` in `tests/packaging/_load_pyproject()` — **every one of 13
  tests** calls it, so a missing `tomli` on the 3.10 CI leg disables the whole file
- `pytest.importorskip("hatchling")` silences 3 of 20 requirements tests
- `pytest.importorskip("fastapi")` silences the api-subpackage export test
- `test_the_api_subpackage_declares_nothing_that_needs_an_extra` iterates `api.__all__`,
  which the same lane deliberately set to `[]` — so it asserts over an empty list
- `test_every_optional_import_in_src_maps_to_an_extra` filters with
  `sys.stdlib_module_names` from the *running* interpreter, so it would fail spuriously on
  the 3.10 leg

### Caught and closed during reconciliation

- **The scope detector fired twice more.** `tests/properties` and `tests/packaging` were
  built and wired to no CI job — the fourth and fifth instances of a real suite nothing
  executes. Now wired.
- **NM-0020 and NM-0023 were missing from the CHANGELOG**, caught by the new version
  coherence check the packaging lane wrote. Now recorded.
- One out-of-lane edit (`requirements.txt`), disclosed by the lane itself with
  justification and verified byte-identical to the generated output.

## Fourth round — closing the toothless gates

All five lanes came back **CLOSED**, and every closure was reproduced independently by a
separate confirmer re-applying the mutation that used to survive: 3/3, 5/5, 7/7, 3/3.

### One real defect found by the new gates — NM-0024

Closing the `content_hash` oracle immediately exposed a live bug it had been hiding.
`content_hash` hashed a hand-written list of three fields; `to_searchable_text()` embedded
those three **and `synonyms`**. Editing an entry's synonyms changed the encoded text while
leaving the hash untouched, so `sync()` skipped the row and the stored vector silently
stopped matching its entry — no error, report says "unchanged".

It escaped because the property suite compared incremental sync against a full rebuild and
**both sides used the same hash**. H-004 again.

Fixed structurally rather than by adding `synonyms` to the list: the hash is now derived
*from* the embedded text, so the two cannot drift. Two hand-maintained lists of "the fields
that matter" are kept in step by discipline, and discipline is what failed. Museum entry
NM-0024, replay proven red. The audit-field guarantee is preserved and now follows from the
definition: a column that never reaches `to_searchable_text()` cannot reach the hash.

### The sixth "gate that nothing runs"

`noxfile.py` grew the `mutation` session — the highest-severity hole in this document — and
`nox` appeared in **no workflow and no Makefile**. Written, configured, ratcheted,
documented, and invoked by nobody.

That is the sixth occurrence: `tests/regression`, `tests/museum`, `tests/properties`,
`tests/packaging`, the CI lint scope, and now this. The scope detector has been extended to
nox sessions, and the mutation job is wired weekly and on demand — deliberately not on every
push, because a gate that makes every push painful is a gate people route around.

### Still open after this round

| Hole | Severity |
|---|---|
| A three-way consistent shrink of the mutation scope (paths + group paths + scope pin) still satisfies the ratchet | MEDIUM |
| `tests/packaging/conftest.py::pytest_runtest_logreport` has no test that can kill it, and the AST scanner matches attribute chains only — so `from pytest import importorskip` reopens the original finding | MEDIUM |
| Two ledger spots: the WARNING on the thread-rule path, and two message-text assertions in cross-benchmark refusal | LOW |
| `_expand()` drops `__init__.py` from a directory scope | LOW |
| `_STDLIB_REMOVED_BY_3_13` is hand-maintained and nothing fails when it goes stale | LOW |
| `TestFusionActuallyFuses::test_equal_scores_are_emitted_in_dense_retrieval_order` cannot fail under the mutation its class was written to close — it guards a different one (H-005 tie-break) | LOW |

## The protocol

When anything escapes:

1. reproduce it, and land the failing test in the same commit as the fix
2. add a museum entry with a replay, and **watch `museum_replay.py` go red**
3. name the gate that should have caught it, and say how that gate is now stronger
4. ask whether it was a **known hazard with no check** — if so that is a `HAZARDS.md`
   failure, which is worse than a missing test, because the knowledge was already here
