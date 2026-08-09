"""
H-002 -- a threshold calibrated on one corpus does not transfer.

`auto_approve_threshold = 0.87` was calibrated on 688 labelled fields. Corpus size is not
a scaling factor on this task, it is a REGIME CHANGE: dictionary aliasing is worth
**+1.9** P@1 at 688 entries and **-18.8** at 30,000. The sign inverts, because max-pooling
over aliases gives every distractor N extra chances to beat the gold, so alias noise grows
with corpus size while alias signal does not. Anything validated at one size is
unvalidated at another -- not approximately right, backwards.

Until now this hazard had NO executable check. It was a paragraph in HAZARDS.md and a
warning comment in `MatchingConfig` saying these numbers "move with the retriever AND with
the benchmark", which is knowledge in the building that nothing in the building enforces.

These tests pin the two properties that make the hazard un-shippable:
  1. every recorded quality result carries its corpus identity, INCLUDING the size
  2. a comparison across different corpora is REFUSED, not performed and annotated

The second is the whole point. A ledger that diffs two corpora and mentions the difference
in a field is a ledger that will be read as "+1.9, ship it".
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))

ledger = pytest.importorskip("optimization_ledger", reason="benchmarks/ not importable")


# =============================================================================
# FIXTURES -- built from vectors, so these run in milliseconds and load no model
# =============================================================================


def _quality(*, n_entries: int, n_queries: int = 40, digest: str = "d0", corpus: str = "real"):
    correct = [1] * (n_queries // 2) + [0] * (n_queries - n_queries // 2)
    return ledger.QualityMetrics(
        benchmark="combined",
        corpus=corpus,
        n_queries=n_queries,
        n_entries=n_entries,
        p_at_1=sum(correct) / n_queries,
        r_at_5=1.0,
        r_at_10=1.0,
        mrr_at_10=0.8,
        auto_approve_precision=0.95,
        auto_approve_coverage=0.5,
        corpus_digest=digest,
        query_ids=[f"q{i}" for i in range(n_queries)],
        correct_at_1=correct,
        hit_at_5=[1] * n_queries,
        hit_at_10=[1] * n_queries,
        rr_at_10=[1.0 if c else 0.5 for c in correct],
        auto_approved=[1] * n_queries,
        auto_correct=list(correct),
    )


def _cost(fields_per_sec: float = 200.0):
    def stat(v: float):
        return ledger.CostStat(median=v, best=v, worst=v, iqr=0.0, values=[v, v, v])

    return [
        ledger.CostAtScale(
            entries=1000,
            fields=300,
            trials=3,
            stats={
                "match_fields_per_sec": stat(fields_per_sec),
                "index_entries_per_sec": stat(700.0),
                "latency_ms_p50": stat(6.0),
                "latency_ms_p95": stat(8.0),
                "latency_ms_p99": stat(9.0),
                "peak_memory_mb": stat(60.0),
                "rss_delta_mb": stat(160.0),
            },
        )
    ]


def _record(label: str, *, n_entries: int, n_queries: int = 40, digest: str = "d0", **kw):
    q = _quality(n_entries=n_entries, n_queries=n_queries, digest=digest, **kw)
    return ledger.Measurement(
        record_id=label.replace(" ", "-"),
        label=label,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        benchmark=q.benchmark,
        corpus=q.corpus,
        provenance={"git_sha": "abc1234", "git_dirty": False, "seed": 1},
        quality=q,
        cost=_cost(),
    )


# =============================================================================
# 1. EVERY QUALITY RECORD CARRIES ITS CORPUS IDENTITY
# =============================================================================


def test_quality_metrics_carry_corpus_identity_including_size():
    """
    The reporting shape is the control, exactly as in H-001.

    If a P@1 can be written down without the corpus it came from, it will be, and the
    next reader has no way to know the number is only true at 688 entries.
    """
    fields = set(ledger.QualityMetrics.__dataclass_fields__)
    for required in ("benchmark", "corpus", "n_entries", "n_queries", "corpus_digest"):
        assert required in fields, (
            f"QualityMetrics does not carry {required!r}. A quality number without its "
            f"corpus is a number nobody can check for transfer."
        )


def test_corpus_size_is_part_of_the_identity_not_metadata_beside_it():
    """
    The distinction that matters: `differences()` must SPEAK about a size mismatch, which
    is what makes the refusal below possible.
    """
    small = ledger.CorpusIdentity("combined", "real", 688, 688, "d0")
    large = ledger.CorpusIdentity("combined", "real", 30000, 688, "d0")
    diffs = small.differences(large)
    assert diffs, "688 entries and 30,000 entries were reported as the same corpus"
    assert any("SIZE" in d or "688" in d for d in diffs), diffs


def test_a_real_measurement_fills_in_the_content_digest():
    """
    End-to-end, against the real code path: a record produced by `measure_quality` must
    come back with a digest, not with the empty default that only exists so old ledger
    lines still load.

    Deliberately runs the matcher rather than asserting on the dataclass -- the field
    existing and the field being POPULATED are different claims, and only the second one
    stops a fresh record from being unverifiable.
    """
    q = ledger.measure_quality(benchmark="__no_such_benchmark__", limit=6)
    assert q.corpus_digest, (
        "measure_quality returned a record with no corpus digest. Sizes and names would "
        "still match a differently-labelled corpus of the same shape."
    )
    assert q.n_entries > 0 and q.n_queries > 0
    ident = q.identity()
    assert ident.has_digest and ident.n_entries == q.n_entries


def test_the_digest_is_stable_across_processes():
    """
    A fingerprint computed with the builtin `hash()` changes with PYTHONHASHSEED, so it
    would refuse every comparison including the legitimate ones -- and the natural fix for
    that is to delete the check. H-005 already caught ranking depending on the hash seed;
    the same trap is one line away here.
    """
    probe = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(REPO / "benchmarks")!r})
        from optimization_ledger import _synthetic_dataset, corpus_digest
        print(corpus_digest(_synthetic_dataset(25)))
        """
    )

    def run(seed: str) -> str:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-800:]
        return proc.stdout.strip().splitlines()[-1]

    first = run("0")
    for seed in ("1", "42", "12345"):
        assert run(seed) == first, (
            f"corpus digest changed under PYTHONHASHSEED={seed}. It is not a fingerprint, "
            f"it is a per-process accident, and every cross-run comparison would be refused."
        )


def test_the_digest_notices_a_relabelled_corpus_of_the_same_shape():
    """Same entry count, same query count, different gold labels. Sizes cannot see this."""
    ds = ledger._synthetic_dataset(30)
    before = ledger.corpus_digest(ds)
    ds.queries[0] = dataclasses.replace(ds.queries[0], gold_id=ds.entries[-1].id)
    assert ledger.corpus_digest(ds) != before, (
        "the digest is blind to which entry is the correct answer, so a corpus rebuilt "
        "with different labels would compare as identical."
    )


# =============================================================================
# 2. A CROSS-CORPUS COMPARISON IS REFUSED
# =============================================================================


def test_comparing_688_entries_against_30000_is_refused():
    """
    The hazard itself, executed.

    These are the two corpus sizes from the recorded occurrence: dictionary aliasing
    measured +1.9 P@1 at 688 entries and -18.8 at 30,000. A ledger that will diff those
    two records will report the +1.9 as a win.
    """
    calibration = _record("aliases on, 688-entry calibration set", n_entries=688)
    production = _record("aliases on, 30k production glossary", n_entries=30000)

    with pytest.raises(ledger.CorpusMismatch) as exc:
        ledger.compare(calibration, production, target="p_at_1")

    message = str(exc.value)
    assert "refusing to compare" in message
    assert "688" in message and "30000" in message, (
        f"the refusal does not say WHICH corpora were involved, so the reader cannot act "
        f"on it:\n{message}"
    )


def test_the_refusal_is_a_value_error_so_nothing_swallows_it_by_type():
    """CorpusMismatch stays a ValueError: callers that already catch broadly keep working."""
    assert issubclass(ledger.CorpusMismatch, ValueError)


def test_a_truncated_query_set_is_also_a_different_corpus():
    """
    `quality_limit` silently truncates the query set. P@1 over 400 queries and P@1 over
    1556 queries are different quantities, and diffing them looks exactly like a result.
    """
    full = _record("full run", n_entries=4598, n_queries=40)
    limited = _record("limit=20 run", n_entries=4598, n_queries=20)
    with pytest.raises(ledger.CorpusMismatch, match="query count"):
        ledger.compare(full, limited, target="p_at_1")


def test_same_shape_different_content_is_refused_by_the_digest():
    """The case sizes cannot catch: a corpus rebuilt with the same shape, new content."""
    a = _record("built monday", n_entries=4598, digest="aaaaaaaaaaaaaaaa")
    b = _record("rebuilt friday", n_entries=4598, digest="bbbbbbbbbbbbbbbb")
    with pytest.raises(ledger.CorpusMismatch, match="digest"):
        ledger.compare(a, b, target="p_at_1")


def test_a_synthetic_record_is_refused_against_a_real_one():
    """
    Retained from the previous, weaker check. `_load_dataset` falls back to a synthetic
    generator whose queries are built from the same tokens as their gold entry, so its
    absolute scores are near-ceiling and not comparable to anything.
    """
    real = _record("real", n_entries=4598, corpus="real")
    synth = _record("synthetic", n_entries=4598, corpus="synthetic")
    with pytest.raises(ledger.CorpusMismatch, match="refusing to compare"):
        ledger.compare(real, synth, target="p_at_1")


def test_a_record_with_no_quality_block_cannot_launder_its_corpus_size():
    """
    An unknown size is not a matching size. A cost-only record has nothing that says how
    big its corpus was, so letting it compare against a sized one would reopen the hazard
    through the one door that looks like an absence rather than a mismatch.
    """
    sized = _record("sized", n_entries=4598)
    unsized = _record("unsized", n_entries=4598)
    unsized.quality = None
    with pytest.raises(ledger.CorpusMismatch):
        ledger.compare(sized, unsized, target="match_fields_per_sec")


# =============================================================================
# 3. THE OPPOSITE FAILURE -- a gate that refuses everything is also broken
# =============================================================================


def test_two_records_from_the_same_corpus_still_compare():
    """
    Over-correcting after an escape produces a check that rejects every comparison, which
    is indistinguishable from having no ledger at all.
    """
    base = _record("baseline", n_entries=4598, digest="aaaaaaaaaaaaaaaa")
    cand = _record("candidate", n_entries=4598, digest="aaaaaaaaaaaaaaaa")
    result = ledger.compare(base, cand, target="p_at_1")
    assert result.verdict in {"WIN", "NEUTRAL", "INCONCLUSIVE", "REGRESSION"}


def test_a_missing_digest_is_reported_rather_than_treated_as_verified():
    """
    Records written before the digest existed carry "". Those still compare -- refusing
    them would strand the history -- but the comparison must SAY it could not check the
    content, otherwise "no digest" quietly reads as "digests matched".
    """
    base = _record("old baseline", n_entries=4598, digest="")
    cand = _record("new candidate", n_entries=4598, digest="aaaaaaaaaaaaaaaa")
    result = ledger.compare(base, cand, target="p_at_1")
    assert any("digest missing" in w for w in result.warnings), (
        f"a comparison against a record with no corpus digest produced no warning: "
        f"{result.warnings}"
    )
