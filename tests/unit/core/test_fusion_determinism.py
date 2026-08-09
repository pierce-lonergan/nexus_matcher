"""
tests.unit.core.test_fusion_determinism | Layer: TEST
Ranking must not depend on the value of PYTHONHASHSEED.

Why this exists
---------------
`fuse_linear` used to walk `set(dense) | set(sparse)`. Set iteration order for strings
follows their hashes, and CPython randomizes string hashing per process, so two runs of
IDENTICAL code ordered equal-scoring candidates differently and returned different
rank-1 matches.

Measured on the 1556-query FHIR benchmark, legacy ordering under six hash seeds:

    seed    0       1       2       3       4       5
    P@1   0.2339  0.2320  0.2333  0.2301  0.2307  0.2314

A 0.38-point band from nothing but the hash seed -- most of the way to the 0.5-point
tolerance that benchmarks/optimization_ledger.py guards P@1 with, and enough to make any
smaller optimization unmeasurable. It also means an auto-approve decision on a near-tie
could flip between two runs of the same matcher on the same data, which is not a property
a governance tool is allowed to have.

The fix is to iterate dense-retrieval order and then lexical-only candidates, so a tie
breaks toward the better dense rank. These tests pin both halves: the ORDER is the
documented one, and it survives a different hash seed in a real subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from nexus_matcher.core.fusion import (
    FusionConfig,
    FusionMethod,
    HybridFuser,
    fuse_linear,
    fuse_linear_ids,
)


class TestTieOrderIsRetrievalOrder:
    """Equal fused scores must resolve to dense rank, not to an arbitrary hash."""

    def test_all_tied_candidates_keep_dense_order(self):
        """A perfectly tied field ranks by dense retrieval order, not set order."""
        # Identical dense scores -> min-max normalization maps them all to 0.0, so every
        # candidate ties exactly. This is the degenerate case the hash order used to decide.
        dense = [(f"e{i}", 0.5) for i in range(12)]
        out = fuse_linear_ids(dense, [], semantic_weight=0.9, lexical_weight=0.1)

        assert [i for i, _ in out] == [i for i, _ in dense]

    def test_lexical_only_candidates_come_after_dense_ones(self):
        """A candidate only the sparse arm found never displaces a tied dense candidate."""
        dense = [("d1", 0.5), ("d2", 0.5)]
        sparse = [("s1", 0.0), ("s2", 0.0)]
        out = fuse_linear_ids(dense, sparse, semantic_weight=0.9, lexical_weight=0.1)

        ids = [i for i, _ in out]
        assert ids.index("d1") < ids.index("s1")
        assert ids.index("d2") < ids.index("s1")

    def test_score_order_still_beats_retrieval_order(self):
        """Retrieval order is the TIE-break only; a better fused score still wins."""
        dense = [("weak", 0.1), ("strong", 0.9)]
        out = fuse_linear_ids(dense, [], semantic_weight=0.9, lexical_weight=0.1)

        assert [i for i, _ in out] == ["strong", "weak"]


class TestLeanPathMatchesFullPath:
    """fuse_linear_ids is an optimization, so it must not be a second implementation."""

    def test_ids_and_scores_match_fuse_linear_exactly(self):
        """The lean (id, score) path agrees bit-for-bit with the ScoredItem path."""
        dense = [(f"e{i}", 1.0 - i * 0.03) for i in range(40)]
        sparse = [(f"e{i}", 12.0 - i * 0.5) for i in range(20, 55)]

        full = fuse_linear(dense, sparse, semantic_weight=0.9, lexical_weight=0.1)
        lean = fuse_linear_ids(dense, sparse, semantic_weight=0.9, lexical_weight=0.1)

        assert [(i.id, i.score) for i in full] == lean

    def test_hybrid_fuser_fuse_ids_matches_fuse(self):
        """HybridFuser.fuse_ids is a projection of fuse, for every fusion method."""
        dense = [(f"e{i}", 1.0 - i * 0.05) for i in range(15)]
        sparse = [(f"e{i}", 9.0 - i * 0.4) for i in range(8, 25)]

        for method in FusionMethod:
            cfg = FusionConfig(method=method, semantic_weight=0.9, lexical_weight=0.1)
            expected = [(i.id, i.score) for i in HybridFuser(config=cfg).fuse(dense, sparse)]
            got = HybridFuser(config=cfg).fuse_ids(dense, sparse)
            assert got == expected, f"{method.value} diverged between fuse and fuse_ids"

    def test_top_k_truncates_after_ranking(self):
        """top_k must cut the ranked list, not the pre-sort one."""
        dense = [("lo", 0.1), ("hi", 0.9), ("mid", 0.5)]
        out = fuse_linear_ids(dense, [], top_k=2)
        assert [i for i, _ in out] == ["hi", "mid"]


# Run in a subprocess so PYTHONHASHSEED is actually applied: it is read at interpreter
# start-up, so setting it from inside a test would do nothing at all.
_HASH_PROBE = textwrap.dedent(
    """
    from nexus_matcher.core.fusion import fuse_linear_ids
    dense = [("entry-%03d" % i, 0.5) for i in range(60)]
    sparse = [("entry-%03d" % i, 3.0) for i in range(30, 90)]
    print(",".join(i for i, _ in fuse_linear_ids(dense, sparse, 0.9, 0.1)))
    """
)


class TestHashSeedIndependence:
    """The real property: same input, different hash seed, same ranking."""

    def test_ranking_is_identical_across_hash_seeds(self):
        """Two interpreters with different PYTHONHASHSEED must rank tied candidates alike."""
        orders = []
        for seed in ("0", "1", "12345"):
            # Inherit the real environment and override one variable. A hand-built env
            # loses the interpreter's own PATH/venv wiring and fails for reasons that have
            # nothing to do with the property under test.
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run(
                [sys.executable, "-c", _HASH_PROBE],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            orders.append(proc.stdout.strip())

        assert all(orders), f"probe produced no output: {orders}"

        assert len(set(orders)) == 1, (
            "fusion ranked tied candidates differently under different hash seeds; "
            "ranking has become hash-order dependent again"
        )
