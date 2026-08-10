# DX baseline — measured, 2026-08-09

Measured against the **published** `nexus-matcher==2.0.1` wheel installed from PyPI into a
clean venv, run from a temp directory. Never against the repo tree, because what a stranger
gets is the thing under test.

Reproduce: `scripts/dx_measure.py` (see git history of this commit).

## Numbers

| Metric | Measured | Note |
|---|---|---|
| Lines of user code, simplest useful task | **8** | load glossary, match schema, print decisions |
| Concepts required | **8** | `NexusMatcher`, `from_config`, `load_dictionary`, `match_schema`, dict-of-tuples result shape, `.dictionary_entry`, `ProtectionLevel`, `MatchDecision` |
| Config values before first run | **0** | target met |
| Cold start, 1k entries | **1.22 s** | process start → first result, incl. index build |
| Cold start, 10k entries | **11.11 s** | ~110 s extrapolated at 100k |
| Time to first result (fresh-eyes, no prior knowledge) | **5 min** | governance task |
| Time to first *trusted* result | **10.5 min** | and **none of it came from documentation** |

Machine at 5.5% CPU busy during the latency runs, under the 10% precondition (H-007).

## Error actionability — 12 realistic mistakes, real messages captured

Scored 0–2: 0 = unactionable or silent, 1 = says what happened, 2 = says what, why, and the
next step.

**Three fail silently (score 0).** No exception, no warning, no signal:

| Case | Behaviour |
|---|---|
| Glossary is empty (header only) | loads "successfully" with zero entries |
| Glossary has a malformed row | the row disappears |
| Schema parses to zero fields (Python API) | returns empty; only the CLI reports it |

**Three are opaque (score 0)** — they name internals the user cannot act on:

```
NexusMatcher()                          TypeError: missing 2 required positional
                                        arguments: 'embedding_provider', 'vector_store'
from_config({"auto_approve_treshold": …}) TypeError: argument should be a str or
                                        os.PathLike … not 'dict'
from_config(build_index(...))           TypeError: argument should be a str or
                                        os.PathLike … not 'GlossaryIndex'
```

The last one is the API-shape problem in one line: `build_index()` returns an object its own
docstring calls "ready to search", and handing it to the matcher produces an error about
filesystem paths.

**One is a near-miss (score 1).** A latin-1 glossary raises a raw `UnicodeDecodeError` with
no mention that `encoding=` is accepted — which it is.

**Five are good (score 2)**: missing file, unrecognised columns, unknown extension,
`match_schema` before `load_dictionary`, unknown `schema_format`.

**Score: 10 / 24.**

## The finding that outranks all of the above

`MatchingSession.get_low_confidence_fields()` — the one API whose *name* answers "which of
these should I not trust?" — **returns an empty list by construction.**

Its default threshold is `0.6`. Measured on a 6-field schema where every field was below the
auto-approve bar:

```
default (0.6)   ->  0 fields flagged
0.87            ->  6 fields flagged
actual top-1 confidences: min 0.730  max 0.755
below auto_approve 0.87: 6 of 6
```

Nothing can fall below 0.6, because `semantic_score` is a *rank-normalised* fused retrieval
score pinned near 0.9 for every rank-1 match and carries weight 0.70. The structural floor is
≈0.63. A governance lead who calls this method and trusts its default is told there is
nothing to review, on a schema where nothing was trustworthy.

This is not an ergonomics defect. It is the same class as NM-0005 — a silent governance
failure — and it is the highest-severity finding in this baseline.
