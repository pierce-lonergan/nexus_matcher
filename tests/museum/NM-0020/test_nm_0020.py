"""
NM-0020 -- the answer depended on how the interpreter was started.

The fused candidate list was built by unioning the dense and sparse arms through a SET.
Python salts string hashing per process, so a set of document ids enumerates differently
in every run, and `list.sort` is stable -- it faithfully preserves whatever order it was
handed. Equal-scoring candidates therefore came back in a different order in every
process, from byte-identical code.

That is not a cosmetic tie-break. Measured cosine to the gold entry is 0.7657 and to the
nearest wrong entry 0.7633 -- a margin of 0.0024 -- so near-ties are the normal operating
regime here, not an edge case. On the 1556-query FHIR benchmark P@1 spanned
0.2301-0.2339 across six hash seeds: a 0.38-point band from nothing but the seed, which is
most of the tolerance the optimization ledger guards P@1 with. Concretely, the same field
could be auto-approved in one run and sent to a human in the next, and inherit a different
glossary entry's protection level.

The tests run separate interpreters, because that is the only way to vary
`PYTHONHASHSEED`; an in-process test cannot observe this defect at all.

Both levels are asserted. The end-to-end ranking is the symptom a user meets. The fusion
order is the same property one layer down, and it is what makes a failure legible -- an
end-to-end divergence alone would leave you guessing which stage leaked the ordering.

Each probe carries a guard against passing vacuously: if the candidates stopped being
exactly tied, the sort would fix the order by score and these comparisons would hold
without proving anything about determinism.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

# Seeds are arbitrary but fixed; the point is that they DIFFER, and that "0" is included
# so a hash-order defect cannot hide behind a lucky choice.
SEEDS = ("0", "1", "42", "12345")

PROBE = textwrap.dedent(
    """
    import json

    from nexus_matcher.core.fusion import fuse_linear_ids

    # Deliberately tied scores within each arm, and an overlap between the arms, which is
    # the shape real retrieval produces at a 0.0024 margin. Min-max normalization maps a
    # constant arm to all-zeros, so every fused score here is identical and the ORDER is
    # decided entirely by how the union was built.
    dense = [(f"d-{i}", 0.80) for i in range(16)]
    sparse = [(f"d-{i}", 7.25) for i in range(8, 24)]
    fused = fuse_linear_ids(dense, sparse, semantic_weight=0.9, lexical_weight=0.1)

    from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
    from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
    from nexus_matcher.shared.types.base import DataType

    # Entries identical in every scored signal, so ranking is settled purely by ordering.
    entries = [
        DictionaryEntry(
            id=f"e-{i:02d}",
            business_name="Customer Email Address",
            logical_name="cust_email",
            definition="The email address of a customer.",
            data_type=DataType.STRING,
            domain="customer",
        )
        for i in range(12)
    ]
    field = SchemaField(
        name="email",
        data_type=DataType.STRING,
        full_path="customer.email",
        parent_path="customer",
        description="Customer email",
    )
    matcher = NexusMatcher.from_config(MatchingConfig())
    matcher._index_dictionary(entries)
    ranked = [r for results in matcher._match_fields([field]).values() for r in results]

    print(
        "RESULT "
        + json.dumps(
            {
                "fusion_order": [i for i, _ in fused],
                "fusion_distinct_scores": len({round(s, 12) for _, s in fused}),
                "ranking": [r.dictionary_entry.id for r in ranked],
                "ranking_distinct_confidences": len(
                    {round(r.final_confidence, 12) for r in ranked}
                ),
            }
        )
    )
    """
)


def _probe_under(seed: str) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode} under PYTHONHASHSEED={seed}\n{proc.stderr[-2000:]}"
    )
    marker = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert marker, f"probe printed no RESULT line under PYTHONHASHSEED={seed}:\n{proc.stdout}"
    return json.loads(marker[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def probes() -> dict[str, dict]:
    """One interpreter per seed. Module-scoped: starting them is the expensive part."""
    return {seed: _probe_under(seed) for seed in SEEDS}


def test_the_probe_really_is_measuring_a_tie(probes):
    """
    Guards the guards.

    If the fused candidates stopped being exactly equal-scoring, the sort would settle
    their order by score and every comparison below would hold no matter how the union was
    built -- this entry would keep printing PASS while covering nothing.
    """
    for seed, probe in probes.items():
        assert probe["fusion_distinct_scores"] == 1, (
            f"seed {seed}: fused candidates are no longer tied "
            f"({probe['fusion_distinct_scores']} distinct scores), so ordering no longer "
            f"decides the outcome and this test proves nothing"
        )
        assert probe["ranking_distinct_confidences"] == 1, (
            f"seed {seed}: returned matches are no longer tied "
            f"({probe['ranking_distinct_confidences']} distinct confidences)"
        )


def test_the_probe_returned_something_to_compare(probes):
    """An empty ranking would make every equality below trivially true."""
    for seed, probe in probes.items():
        assert probe["fusion_order"], f"seed {seed}: fusion returned no candidates"
        assert probe["ranking"], f"seed {seed}: the matcher returned no results"


@pytest.mark.parametrize("seed", SEEDS[1:])
def test_the_top_match_is_the_same_under_every_hash_seed(probes, seed):
    """
    The symptom a user meets: which glossary entry the field is mapped to.

    At a 0.0024 margin a different order is a different ENTRY, whose protection level the
    field then inherits -- so this is a governance outcome changing with an environment
    variable nobody set on purpose.
    """
    assert probes[seed]["ranking"] == probes[SEEDS[0]]["ranking"], (
        f"ranking changed under PYTHONHASHSEED={seed}. Ordering is leaking out of a set "
        f"or another hash-ordered container, so the mapping a field receives depends on "
        f"how the interpreter was started."
    )


@pytest.mark.parametrize("seed", SEEDS[1:])
def test_fusion_emits_equal_scoring_candidates_in_the_same_order(probes, seed):
    """
    The same property at the stage that owns it.

    Ranking order is defined in fusion: dense-retrieval order first, then lexical-only
    candidates. Pinning it here means a future ordering leak is reported against the stage
    that caused it rather than as an unexplained end-to-end difference.
    """
    assert probes[seed]["fusion_order"] == probes[SEEDS[0]]["fusion_order"], (
        f"fused candidate order changed under PYTHONHASHSEED={seed}"
    )


def test_a_tie_breaks_toward_the_better_dense_rank(probes):
    """
    Determinism alone is not enough -- a stable but arbitrary order would satisfy every
    test above.

    The defined rule is that equal-scoring candidates keep dense-retrieval order, and the
    lexical-only ones follow. That is a deliberate choice: it makes a tie resolve toward
    the arm that measured similarity, rather than toward whichever id happened to hash
    first.
    """
    order = probes[SEEDS[0]]["fusion_order"]
    dense_ids = [f"d-{i}" for i in range(16)]
    assert order[: len(dense_ids)] == dense_ids, (
        f"tied candidates are no longer emitted in dense-retrieval order: {order}"
    )
