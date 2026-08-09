"""
Reintroduce NM-0020: build the fused candidate list from a set again.

The arithmetic is preserved exactly -- same weights, same min-max normalization, same
stable sort. The ONLY difference is that the union of the two retrieval arms is iterated
as a set instead of as an insertion-ordered dict. That is the whole defect: str hashing is
salted per process, so a set of document ids enumerates differently in every interpreter,
and `list.sort` being stable faithfully preserves whichever order it was handed.

Injecting a whole replacement body rather than patching lines inside the existing one is
deliberate. The real body is six statements of arithmetic that a future optimisation may
well rewrite; a replay that patched one of those lines would rot into a hole. The
signature and the module-level `_score_of` sort key are the two things this depends on.

Anchored on the signature's CLOSING PAREN at column zero, not on the return annotation, so
retyping `list[tuple[T, float]]` does not silently disarm this entry.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/core/fusion.py"

ANCHOR_RE = re.compile(r"^def fuse_linear_ids\([\s\S]*?\n\) ->[^\n]*:\n", re.M)

BODY = """    # NM-0020 replay: union the two arms through a SET, whose iteration order follows
    # per-process string hashing. Every number below is unchanged; only the ORDER of
    # equal-scoring candidates moves, which is exactly enough to change the top match.
    _total = semantic_weight + lexical_weight
    _sem_w = semantic_weight / _total
    _lex_w = lexical_weight / _total
    _sem = dict(semantic_results)
    _lex = dict(lexical_results)
    if normalize_scores:
        _sem = _minmax_normalize(_sem)
        _lex = _minmax_normalize(_lex)
    _fused = [
        (_id, _sem_w * _sem.get(_id, 0.0) + _lex_w * _lex.get(_id, 0.0))
        for _id in set(_sem) | set(_lex)
    ]
    _fused.sort(key=_score_of, reverse=True)
    if top_k is not None:
        _fused = _fused[:top_k]
    return _fused
"""


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    match = ANCHOR_RE.search(text)
    if match is None:
        raise LookupError(f"NM-0020 replay: fuse_linear_ids signature not found in {TARGET}")
    path.write_text(text[: match.end()] + BODY + text[match.end() :], encoding="utf-8")
