"""
H-005 -- the 0.0024 margin means ties are everywhere.

Measured cosine to the gold entry 0.7657, to the nearest wrong entry 0.7633. At that
separation any iteration order that leaks into ranking changes the answer, and ranking did
depend on PYTHONHASHSEED until the deterministic tie-break landed.

This is not an edge case to be tidied away. It is the normal operating regime of this
system, and it is why "the top match" is frequently a coin toss dressed as a decision.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

PROBE = textwrap.dedent(
    """
    from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
    from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
    from nexus_matcher.shared.types.base import DataType

    # Entries deliberately near-identical, so ordering is decided by tie-breaking rather
    # than by any real difference in similarity.
    entries = [
        DictionaryEntry(id=f"d-{i}", business_name="Customer Email Address",
                        logical_name="cust_email", definition="The email address of a customer.",
                        data_type=DataType.STRING, domain="customer")
        for i in range(8)
    ]
    fields = [SchemaField(name="email", data_type=DataType.STRING, full_path="c.email",
                          parent_path="customer", description="Customer email")]
    m = NexusMatcher.from_config(MatchingConfig())
    m._index_dictionary(entries)
    out = m._match_fields(fields)
    print("|".join(r.dictionary_entry.id for v in out.values() for r in v))
    """
)


def _ranking_under(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    return proc.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("seed", ["0", "1", "42", "12345"])
def test_ranking_is_identical_across_hash_seeds(seed):
    """
    Four separate interpreters, four different hash seeds, one ranking.

    Set iteration order varies with PYTHONHASHSEED. If any set or dict ordering reaches
    the ranking, this diverges -- and with a 0.0024 margin it will reach a DIFFERENT
    glossary entry, whose protection level the field then inherits.
    """
    assert _ranking_under(seed) == _ranking_under("0"), (
        f"ranking changed under PYTHONHASHSEED={seed}. Ordering is leaking from a set or "
        f"dict, so the answer depends on how the interpreter was started."
    )


def test_ranking_is_stable_across_repeated_processes():
    """Same seed twice: catches nondeterminism that is not hash-seed related."""
    assert _ranking_under("0") == _ranking_under("0")
