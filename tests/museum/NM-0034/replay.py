"""
Reintroduce NM-0034: take the average IDF with an order-dependent reduction again.

The defect was one operator. `_term_idf` derived rank_bm25's epsilon floor from
`idf.sum()` -- numpy's pairwise reduction over an array laid out in first-seen term order,
which is to say in glossary ROW order. On a corpus whose raw IDFs cancel, the exact total
is 0.0 and the pairwise total is +2.220446e-16 or -2.220446e-16 depending on which row was
listed first, so the floor changed SIGN with the row order. The fix replaced that one call
with `math.fsum`, whose result is the correctly-rounded exact sum and is therefore
identical for every permutation of its input.

WHY THIS REPLAY REBINDS A NAME INSTEAD OF PATCHING THE LINE
-----------------------------------------------------------
A regex over `floor = self._epsilon * (math.fsum(idf.tolist()) / idf.size)` would pin the
whole expression: the epsilon factor, the `.tolist()` call, the division by `idf.size`, and
the exact spacing the formatter happened to choose. Every one of those is a legitimate
target for a future edit that does not reintroduce anything, and the first such edit would
turn this entry into a hole that still prints PASS -- which is precisely how NM-0016 rotted
(it pinned `batch_size: int = 64`, somebody tuned it to 512, and the entry reported PASS
while catching nothing).

What this entry is actually about is narrower than the line: the average must not be taken
by a reduction whose answer depends on input order. So the replay appends a module-level
rebinding of `math` that leaves every attribute alone except `fsum`, which it routes back
through numpy's pairwise sum over the same float64 values -- bit-identical to the historical
`idf.sum()`. Nothing inside `_term_idf` is touched, so the function may be reformatted,
re-annotated, re-documented or moved within the class and this entry still applies.

THE ANCHOR IS THE TOKEN `math.fsum(`, and its absence is a LOUD failure. If the exact
summation is ever spelled some other way -- `fsum` imported directly, a numpy Kahan helper,
a sorted sum -- `apply` raises and `museum_replay.py` reports "replay could not be applied",
which is a hole being announced rather than a gate quietly passing. If the exact summation
is REMOVED, the guard still finds nothing and the same announcement follows. The one thing
this cannot do is silently succeed at nothing.
"""

from __future__ import annotations

import pathlib

TARGET = "src/nexus_matcher/infrastructure/adapters/sparse_retrievers/bm25.py"

# The one token the repair cannot be spelled without. Deliberately not a full expression:
# see the module docstring for why anchoring on the arithmetic around it would rot.
GUARD = "math.fsum("

SHIM = '''

# ---------------------------------------------------------------------------
# NM-0034 replay. Route the module's `math.fsum` back through numpy's pairwise
# reduction, which is bit-identical to the historical `idf.sum()` and, like it,
# returns a different total for a different input ORDER. Every other attribute of
# `math` is passed straight through, so nothing else in this module changes.
# ---------------------------------------------------------------------------
import math as _nm0034_exact_math  # noqa: E402


class _NM0034PairwiseMath:
    """`math`, except that summing is order-dependent again."""

    def __getattr__(self, name):
        return getattr(_nm0034_exact_math, name)

    @staticmethod
    def fsum(values):
        return float(np.asarray(list(values), dtype=np.float64).sum())


math = _NM0034PairwiseMath()  # type: ignore[assignment]
'''


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    if GUARD not in text:
        raise LookupError(
            f"NM-0034 replay: {TARGET} no longer contains {GUARD!r}, so there is no "
            f"exact summation left for this replay to undo. Either the epsilon IDF floor "
            f"is now derived some other way -- in which case this entry needs rewriting "
            f"against that -- or the order-independent average has been removed and the "
            f"defect is back on its own."
        )
    path.write_text(text + SHIM, encoding="utf-8")
