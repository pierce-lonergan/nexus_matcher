"""
tests/packaging/test_calibration_provenance.py | Env: ALL

The published calibration corpus is a claim about a measurement, so it is checked against
the measurement.

NM-V2-03 SC-7 asks for the shipped default thresholds to be labelled with the corpus they
were fitted on, in a form a consumer can read without this library's source. `GET
/api/v1/status` now answers that at `calibration.corpus` -- and the moment a number lives in
two places, one of them starts being wrong. The corpus block is a dict in
`presentation/api/introspect.py`; the measurement is a set of committed JSON artifacts under
`benchmarks/results/`. This file is what keeps them the same numbers.

That direction matters. `check_doc_numbers.py` already does this for PROSE in documents;
nothing did it for a number the SERVICE publishes over HTTP, and a wrong number on the wire
is worse than a wrong number in a README: a consumer deciding whether the shipped 0.87 was
fitted on anything resembling their data has no other source to check it against.

What is deliberately NOT checked here
-------------------------------------
`fieldNaming` and `ambiguity` are prose, and prose is what this block uses to describe the
DOMAIN and NAMING-STYLE dimensions of NM-V2-01 AR-4 that no arithmetic in this repository
can honestly compare. They are asserted to be non-empty and to be labelled as descriptions
on the published schema, and nothing more. A gate that pretended to check them would be
claiming a capability the surface itself declines to claim.

The measured accuracy figures move with the encoder and the benchmark -- the module comment
on `auto_approve_threshold` says so at length, and this repository has already been through
one leakage fix that moved every number downstream. When they move, the artifacts are
re-run and this gate turns the corpus block red until it is re-copied, which is the whole
point: a re-measurement that silently left a stale number on the wire is the failure this
file exists to make impossible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexus_matcher.application.use_cases.match_schema import MatchingConfig
from nexus_matcher.presentation.api.introspect import (
    _CALIBRATION_CORPUS,
    _UNCALIBRATED_SIZE_RATIO,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "benchmarks" / "results"

# The artifact each split's size is read out of. A split's size is not recorded as a count
# anywhere, but every field gets exactly one decision, so the decision histogram sums to it.
_SPLIT_ARTIFACTS = {
    "bird": "eval_pipeline_bird.json",
    "omop": "eval_pipeline_omop.json",
}

# Where the corpus-wide accuracy figure comes from. Separate from the calibration artifact,
# which measures the auto-approve boundary rather than rank-1 accuracy.
_PIPELINE_ARTIFACT = "eval_pipeline_combined.json"

# Six decimals, matching `introspect._PRECISION`. The corpus block rounds for the wire, so
# the comparison has to round the artifact the same way or it fails on the seventeenth
# digit of a float nobody published.
_PRECISION = 6


def _artifact(name: str) -> dict:
    path = RESULTS / name
    assert path.is_file(), f"{path} is missing -- the corpus block cites a measurement that is gone"
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# VACUITY GUARD -- first, so nothing below can pass by measuring nothing
# =============================================================================


def test_the_corpus_block_is_populated_and_names_a_real_artifact():
    """
    Every check below compares the block against a file. An empty block, or one naming a
    file that is not there, would make all of them vacuous rather than red.
    """
    assert _CALIBRATION_CORPUS, "the published corpus block is empty"

    named = _CALIBRATION_CORPUS["artifact"]
    assert named.startswith("benchmarks/results/"), (
        f"the corpus names {named!r}, which this gate cannot resolve to an artifact"
    )
    assert (REPO / named).is_file(), f"the corpus cites {named}, which does not exist"


# =============================================================================
# THE SIZES
# =============================================================================


def test_the_published_field_count_is_the_one_the_calibration_was_run_over():
    """
    688 is the number a consumer compares their own corpus against, and it is the single
    most load-bearing figure in the block: `dictionarySizeRatio` and the `UNCALIBRATED_SIZE`
    warning are both computed from it.
    """
    assert _CALIBRATION_CORPUS["fields"] == _artifact("exp_calibration_combined.json")["n"]


def test_each_split_size_is_the_one_its_own_run_reports():
    """
    Every field in a run gets exactly one decision, so the decision histogram sums to the
    number of fields. Read that way rather than from the builder, because the builder can be
    re-run against a re-downloaded corpus while the committed measurement is what the
    published threshold was actually fitted against.
    """
    for split, artifact in _SPLIT_ARTIFACTS.items():
        measured = sum(_artifact(artifact)["decisions"].values())
        assert _CALIBRATION_CORPUS["splits"][split] == measured, (
            f"the corpus block says the {split} split is "
            f"{_CALIBRATION_CORPUS['splits'][split]} fields; {artifact} reports {measured}"
        )


def test_the_splits_account_for_the_whole_corpus():
    """
    A block whose parts do not sum to its whole is describing two different corpora, and a
    consumer judging similarity would be comparing against neither.
    """
    assert sum(_CALIBRATION_CORPUS["splits"].values()) == _CALIBRATION_CORPUS["fields"]


def test_the_dictionary_is_the_pool_every_field_competed_against():
    """
    Both benchmarks pool their entries, so each query competes against every other field's
    gold entry. That is why the two counts are equal, and why publishing only one of them
    would leave a consumer unable to tell whether 688 was a query count or a corpus size.
    """
    assert _CALIBRATION_CORPUS["dictionaryEntries"] == _CALIBRATION_CORPUS["fields"]


# =============================================================================
# THE OPERATING POINT
# =============================================================================


def test_the_published_threshold_is_the_one_this_library_actually_ships():
    """
    The corpus block describes the SHIPPED default. If `auto_approve_threshold` is retuned
    and this block is not, the service would publish a precision measured at a boundary it
    no longer runs -- a true number about the wrong deployment, which is the hardest kind of
    wrong to notice.
    """
    assert _CALIBRATION_CORPUS["autoApproveThreshold"] == MatchingConfig().auto_approve_threshold


def test_the_precision_and_coverage_are_the_curve_row_for_that_threshold():
    """
    Pairing, not presence. The checker `scripts/check_doc_numbers.py` names this exact hole
    in its own limitations: a document can transcribe both numbers correctly and pair them
    wrongly, because every value is present somewhere in the artifact. Here the row is
    selected by the threshold first, so the pairing is what is asserted.
    """
    curve = _artifact("exp_calibration_combined.json")["curve"]
    threshold = _CALIBRATION_CORPUS["autoApproveThreshold"]
    rows = [row for row in curve if abs(row["threshold"] - threshold) < 1e-9]
    assert len(rows) == 1, f"the calibration curve has {len(rows)} rows at threshold {threshold}"

    assert _CALIBRATION_CORPUS["autoApprovePrecision"] == round(rows[0]["precision"], _PRECISION)
    assert _CALIBRATION_CORPUS["autoApproveCoverage"] == round(rows[0]["coverage"], _PRECISION)


def test_the_rank_one_accuracy_is_the_pipeline_run_on_the_same_corpus():
    """
    P@1 is the accuracy the auto-approve coverage sits inside: 12.4% of fields reaching
    AUTO_APPROVE means little without knowing what the other 87.6% were doing.
    """
    pipeline = _artifact(_PIPELINE_ARTIFACT)

    assert pipeline["benchmark"] == "combined", (
        "the pipeline artifact is no longer the combined run, so it is not measuring the "
        "corpus this block describes"
    )
    assert _CALIBRATION_CORPUS["precisionAtRank1"] == round(pipeline["p_at_1"], _PRECISION)
    assert _CALIBRATION_CORPUS["autoApprovePrecision"] == round(
        pipeline["auto_approve_precision"], _PRECISION
    )


# =============================================================================
# THE WARNING'S OWN EVIDENCE
# =============================================================================


def test_the_size_ratio_that_warns_is_one_the_repository_has_measured_across():
    """
    `_UNCALIBRATED_SIZE_RATIO` is not a tuned constant; it is the smallest ratio at which
    this repository can point at a measurement showing the score distribution has moved. The
    evidence is `exp_alias_scale.json`, where a retrieval setting worth +1.9 points at the
    calibration corpus size is worth -13.7 at ten times it.

    If that artifact stops containing a run at `corpus size x ratio`, the constant has lost
    its justification and this gate says so rather than letting a number with no evidence
    behind it keep raising a warning on other people's deployments.
    """
    corpus = _CALIBRATION_CORPUS["dictionaryEntries"]
    scale = _artifact("exp_alias_scale.json")
    measured_sizes = {row["n"] for row in scale}

    assert corpus in measured_sizes, (
        f"the scale experiment no longer includes the calibration corpus size ({corpus}), "
        f"so there is no baseline for the ratio to be measured against"
    )
    assert max(measured_sizes) >= corpus * _UNCALIBRATED_SIZE_RATIO, (
        f"the largest measured corpus is {max(measured_sizes)} entries, below the "
        f"{corpus * _UNCALIBRATED_SIZE_RATIO:.0f} the warning ratio claims evidence at"
    )

    at_corpus = {row["aliases"]: row["p_at_1"] for row in scale if row["n"] == corpus}
    beyond = {
        row["aliases"]: row["p_at_1"]
        for row in scale
        if row["n"] >= corpus * _UNCALIBRATED_SIZE_RATIO
    }
    shared = sorted(set(at_corpus) & set(beyond))
    assert len(shared) >= 2, "the scale experiment no longer varies a setting across sizes"

    # The claim: a setting's SIGN inverts between the calibration corpus and ten times it.
    # Asserted, not quoted, so the docstring in `introspect.py` cannot outlive its evidence.
    baseline, alternative = shared[0], shared[-1]
    assert (at_corpus[alternative] - at_corpus[baseline]) > 0
    assert (beyond[alternative] - beyond[baseline]) < 0


# =============================================================================
# THE PROSE, AND THE LIMIT OF WHAT THIS GATE CLAIMS
# =============================================================================


@pytest.mark.parametrize("member", ["fieldNaming", "ambiguity", "name", "measuredBy"])
def test_the_descriptive_members_are_present_and_say_something(member):
    """
    These are for a human comparing their own schema against the corpus, which is the part
    of AR-4 this library will not fake with an invented similarity metric. All this can
    check is that they exist and are not blank -- said out loud in the module docstring so
    a green run here is not mistaken for a checked description.
    """
    value = _CALIBRATION_CORPUS[member]
    assert isinstance(value, str) and value.strip(), f"{member} is blank"


def test_the_script_that_produced_the_fit_is_in_this_repository():
    """A `measuredBy` naming a script nobody can run is a citation, not provenance."""
    assert (REPO / _CALIBRATION_CORPUS["measuredBy"]).is_file()
