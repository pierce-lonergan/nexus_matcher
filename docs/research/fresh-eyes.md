# Fresh-eyes report — 2026-08-09

Three agents with **no repository context**, forbidden from opening any file in this repo,
installed `nexus-matcher==2.0.1` from PyPI and were given a realistic task. Every wrong turn
is a DX defect, not a user error.

Reported unsoftened. The temptation to tidy these into a clean narrative is exactly what
would make the exercise worthless.

---

## DX-001 — `get_low_confidence_fields()` returns nothing, by construction · **CRITICAL**

The one API whose name answers "which of these should I not trust?" returns an empty list.

```
signature: get_low_confidence_fields(threshold: float = 0.6)
default (0.6)  ->  0 fields flagged
0.87           ->  6 fields flagged
actual top-1 confidences: min 0.730  max 0.755   (6 of 6 below the auto-approve bar)
```

Nothing can fall below 0.6. `semantic_score` is a **rank-normalised** fused retrieval score,
pinned near 0.9 for every rank-1 match, carrying weight 0.70 — so the structural floor of
`final_confidence` is ≈0.63.

A governance lead who calls this and trusts the default is told there is nothing to review,
on a schema where nothing was trustworthy. Same class as NM-0005: a silent governance
failure. **Verified independently.**

## DX-002 — the documented machine-readable output drops the governance payload · **HIGH**

`nexus-matcher match -f json` emits `dictionary_entry` as
`{id, business_name, logical_name, data_type}`. **No `protection_level`.** The one field the
entire stated use case depends on — "so the object inherits that entry's classification" —
is absent from the only documented machine-readable surface.

The `scores` block carries 3 of 5 components (drops `edit_distance` and `domain`), so the
emitted numbers cannot reproduce the emitted confidence. The governance task is impossible
from the documented interface; you must drop to the Python API.

## DX-003 — `semantic_score` is not a semantic score · **HIGH**

It is the min-max-normalised **fused retrieval** score. The agent proved this by moving
`fusion_alpha` 0.9 → 0.5 and watching the rank-1 floor move 0.858 → 0.454; a cosine would
not move at all. The recurring value 0.9 *is* `fusion_alpha`.

Consequence: it is rank-relative, so ~0.63 of the 0.70 semantic weight is near-constant
across every field and carries almost no evidence about correctness. Telling an auditor
"semantic score 0.9" implies 90% semantic similarity. It means "ranked first among 32
candidates". The absolute cosine is never surfaced anywhere.

## DX-004 — `pip install` fails on a normal Windows path · **MEDIUM**

`OSError [WinError 206] filename or extension is too long` on a numpy dist-info path, after
downloading ~38 packages. The agent had to relocate the venv to `C:\nmx`. Transitive, not
ours — but we advertise Windows 11 measurement and a stranger cannot tell the difference.

## DX-005 — `Score` is exported and is literally `float` · **LOW**

`Score` is in `__all__`; `nexus_matcher.Score` is `<class 'float'>` with float's docstring.
`nexus_matcher.annotations` also resolves, to `_Feature((3,7,0,'beta',1), None, 16777216)` —
`from __future__ import annotations` reachable through the package namespace.

*Correction to the agent's report:* it claimed `annotations` was in the export list. It is
not — `'annotations' in __all__` is `False`, so `import *` does not pull it in. Reachable,
not exported.

## DX-006 — docstrings crash the default Windows console · **MEDIUM**

`UnicodeEncodeError: 'charmap' codec can't encode character '→'` when printing the package's
own docstrings. Every introspection command thereafter needed `PYTHONIOENCODING=utf-8`.

We fixed this for CLI *output* (NM-0001) and did not consider docstring *content*. The
encoding matrix covers commands, not the text the package ships.

## DX-007 — the PyPI page does not render for a programmatic reader · **LOW**

WebFetch of `https://pypi.org/project/nexus-matcher/2.0.1/` returns "A required part of this
site couldn't load." The agent fell back to the JSON API to read the README at all. Affects
agents and scrapers, not humans in browsers — worth knowing given who reads package pages now.

## DX-008 — `MatchingConfig` looks like pydantic and is not · **LOW**

The package depends on pydantic *and* pydantic-settings, and the README shows
`MatchingConfig(auto_approve_threshold=0.85)`. `\.model_fields` raises `AttributeError`; it
is a plain dataclass. Minor, except it is the step that unlocks the whole audit answer and
there is no documented route to it.

---

## Two false claims in our own documentation, found by adjudication

Neither fresh-eyes report caught these; both are ours.

**README said `rank-bm25` is a core dependency.** It is not, and has not been since the
inverted-index rewrite stopped importing it. BM25 is built in and works on a bare install —
verified: `BM25Retriever` scores correctly with `rank_bm25` absent from `sys.modules`. The
pyproject comment was updated with the code; this paragraph was not. **My error. Fixed.**

**README said "Accuracy at 100k entries is unmeasured."** It is measured, by us, in
`benchmarks/results/exp_scale_combined.json`: P@1 **0.589** (in-memory) and **0.591** (HNSW)
at 100,000 entries. The README *understated* our own evidence. **Fixed.**

Both were live on PyPI, because the README is the long_description.

The pattern is worth naming: this repo has extensive machinery gating **code** and none
gating **documented numbers**. Every claim in the README names an artifact; nothing checks
that the artifact says what the claim says.
