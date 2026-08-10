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
| `scripts/check_doc_numbers.py` + `tests/packaging/test_doc_numbers.py` | a documented number disagreeing with the artifact it names | 5 mutations of real documents — a wrong P@1, a drifting throughput, an **understated** P@1, a citation of a non-existent artifact, and a stale number being *fixed* — all green → red → green, tree restored byte-identical |

## Ratchets

| Ratchet | Value | Rule |
|---|---|---|
| accuracy tolerance | P@1 / R@5 / MRR, ±0.02 | may tighten, never loosen; baselines re-recorded from the test's own code path |
| guarded metrics | P@1 −0.005, auto-approve precision −0.010, R@10 −0.010, memory +10% | may tighten, never loosen |
| unreferenced public methods | 85 | may fall, may never rise |
| documented numbers contradicting their artifact | 141, in `KNOWN_MISMATCHES` | an **exact set**, not a count: a new one fails, and *fixing* one also fails until its line is deleted. Each entry pins the claim AND the artifact value, so re-running a benchmark breaks every entry for it |
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

## Fifth round — the documented numbers were never gated at all

The DX round's most valuable finding was not a defect but a **category**: this repository
has extensive machinery gating CODE and, until now, none whatsoever gating DOCUMENTED
NUMBERS. Two false statements shipped to PyPI as the wheel's `long_description` and no test
could have noticed, because no test read a document.

One of the two is worth internalising. The README said "Accuracy at 100k entries is
unmeasured"; `exp_scale_combined.json` measures it, at P@1 0.589 and 0.591. The README
**understated its own evidence**. Every human review process here is tuned to catch
exaggeration, and this was the opposite — it reads as commendable caution. Only a machine
comparing the sentence to the file finds it.

### What the gate found: the numbers are stale almost everywhere

`scripts/check_doc_numbers.py` extracts numeric claims from README.md, CHANGELOG.md and
`docs/**/*.md`, resolves each to the artifact its section names, and compares. On a clean
tree it verifies **96 claims and reports 141 mismatches**, across nine documents:

| Document | Mismatches | What is wrong |
|---|---|---|
| `docs/BENCHMARK_REGISTRY.md` | 69 | every A-grade number is **pre-leakage-fix**: P@1 0.6999, 652.3 fields/sec, the whole EXP-QUERY-REPR table, the whole EXP-CALIBRATION table, and a **+0.0555 reranker gain** the README explicitly retracts as a loss of 1.3 |
| `docs/ENHANCEMENT_JOURNEY.md` | 27 | the same era, same numbers |
| `CHANGELOG.md` | 11 | P@1 0.700 / bird 0.490 / omop 0.819, and "94.7% precise over 42.7% coverage" (artifact: 91.6% over 20.8%) |
| `docs/RESEARCH_ALIGNMENT.md` | 8 | "real measured performance P@1 0.700 / P@5 0.888 / MRR@10 0.781" |
| `README.md` | 6 | three real (bird 0.598 vs 0.601; index build ~1.8 s vs 2.07 s; two rounding-direction slips) and three that are quotations of retracted numbers |
| `docs/PROJECT_STATE.md` | 5 | the entire Headline table |
| `docs/modules/*.md` | 14 | context_enricher, cross_encoder_reranker, abbreviation_expansion |
| `docs/QUALITY_GATES.md` | 1 | 652 fields/sec |

The registry is the sharpest case. Its own opening line says every claim "should be
traceable to a row in this file", the README links to it as the place where each benchmark
appears "with its artifact or an explicit note that it has none" — and it disagrees with
its own named artifacts 69 times, including reporting as a **+5.5-point win** the reranker
the README documents as a measured loss.

None of it was fixed here: the documents belong to other lanes. Every mismatch is recorded
in `KNOWN_MISMATCHES` with both sides pinned, and listed as follow-ups.

### What this gate cannot do

Written down because a checker that implies total coverage is worse than none — it makes
the next person believe the numbers are guarded when most of them are not. In full in the
module docstring; the load-bearing four:

- **prose with no number**, and **behaviour claims** — "rank-bm25 is a core dependency" is
  the *other* escaped defect, and this file would not have caught it
- **numbers naming no artifact** — the README's Encoders table cites its three P@1 figures
  and three throughput figures against nothing, so all six are invisible to this gate even
  though a committed encoder-comparison artifact exists. (Reproducing those numbers here,
  next to that artifact's filename, made the gate go red on *this* file while the section
  was being written. That is the gate working, and the sentence was rewritten rather than
  added to the ledger.)
- **derived quantities** — "+19.3 points of P@1" is unchecked; the artifact's own
  `context − raw` is **+19.9**, and the registry says **+20.1**. Three numbers, one
  measurement, nothing to compare them against
- **mappings** — a document can transcribe every number correctly and pair them wrongly

### The controls, because a first pass writes gates and a second finds the decoration

Ten mutations of the checker itself, each required to turn a named control red: tolerance
widened to a fudge factor, a missing metric key counted as agreement, table recognition
disabled, unscoped numbers checked against everything, ordering derived from `hash()`,
correct documents rejected, a wallpaper ledger entry, transliteration removed, broken-ref
detection deleted, and the ledger key regrown a line number. **All ten went red.**

Two of them were holes when first written, and both were the familiar shape:

- the cp437 test asserted only that the process survived, and today's 141 findings happen
  to quote pure ASCII — so a checker with its transliteration ripped out passed. It now
  scans a document that really contains `≤`, `—` and `×` and renders the finding.
- the broken-reference test asserted `not broken` against an empty list. It could not fail.
  A control now proves the detector sees a fabricated citation.

A third was found by another lane rather than by design: line numbers were in the ledger
key, and a concurrent 44-line CHANGELOG edit made the gate report **eleven fixes and eleven
regressions that had not happened**. A ratchet that fires on churn is a ratchet somebody
disables, so the key is now content-only. The cost is stated in `Finding.key`: two identical
wrong claims in one document collapse to one entry.

### Still open after this round

| Hole | Severity |
|---|---|
| 141 documented numbers still contradict their artifacts; the gate records them, nobody has fixed them | **HIGH** |
| `docs/BENCHMARK_REGISTRY.md` is the designated source of truth for every claim in the repo and is stale throughout — a reader sent there by the README gets retracted numbers | **HIGH** |
| No museum entry for this class. The replay exists (`--report` plus a doc mutation) but was not built as `tests/museum/NM-XXXX/`, being outside this lane's files | MEDIUM |
| Derived quantities are unchecked, and "+19.3 / +19.9 / +20.1 points of P@1" are three different answers to one question | MEDIUM |
| The checker cannot distinguish a stale number from a deliberately-quoted retracted one, so ~5 ledger entries are quotations rather than errors | LOW |
| `scripts/check_doc_numbers.py` runs only via `pytest tests/packaging`; it is not a separate CI step, so a failure reads as a packaging-test failure | LOW |

## The protocol

When anything escapes:

1. reproduce it, and land the failing test in the same commit as the fix
2. add a museum entry with a replay, and **watch `museum_replay.py` go red**
3. name the gate that should have caught it, and say how that gate is now stronger
4. ask whether it was a **known hazard with no check** — if so that is a `HAZARDS.md`
   failure, which is worse than a missing test, because the knowledge was already here
