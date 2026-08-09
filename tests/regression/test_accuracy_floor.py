"""
tests.regression.test_accuracy_floor | Layer: TEST
The repo's defence against silently getting worse at its actual job.

Why this exists
---------------
Every other test here asserts behaviour: this function returns that value, this bug does
not recur. None of them would notice if a change left the whole system ranking 10 points
worse. That is the failure mode this project is most exposed to, because almost every
change in it touches retrieval:

  * swapping the encoder for static embeddings cost 8.9 points of P@1 and broke nothing
  * turning dictionary aliasing on cost 13.7 points at 10k entries and broke nothing
  * the abbreviation expander collapsed enriched text into one token, giving 787 of 793
    queries zero BM25 hits, and broke nothing

All three were caught by running a benchmark on purpose. This makes it automatic.

Why a fixture rather than the real corpus
-----------------------------------------
`data/` is gitignored and the full FHIR benchmark is built from a download, so CI cannot
see it. `tests/fixtures/fhir_regression.json` is a committed 1500-entry / 300-query subset
of it -- HL7 FHIR R5 is CC0 1.0, so redistributing a slice is fine.

THE FIXTURE SCORE IS NOT THIS SYSTEM'S ACCURACY. A 1500-entry pool is an easier task than
the real 4598-entry one, and far easier than a 30k-entry enterprise glossary, so P@1 here
reads higher than the honest number. Quoting it anywhere as a headline figure would repeat
exactly the mistake that made the OMOP benchmark degenerate. Its one job is to fail when
ranking breaks.

What it does NOT cover
----------------------
Dense retrieval only -- no BM25, no fusion, no reranking, no multi-signal scoring. That is
deliberate: it keeps the test fast (~5s) and stable, and the encoder plus query
construction are where the large regressions have historically come from. A fusion-only
regression would slip past this; benchmarks/optimization_ledger.py is the tool for that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
    bundled_model_available,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "fhir_regression.json"

pytestmark = [
    pytest.mark.skipif(not bundled_model_available(), reason="bundled ONNX model not present"),
    pytest.mark.skipif(not FIXTURE.exists(), reason="regression fixture missing"),
]


@pytest.fixture(scope="module")
def scored() -> dict:
    """Encode the fixture once and derive every metric from the same similarity matrix."""
    from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
        BundledOnnxProvider,
    )

    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    entries, queries = data["entries"], data["queries"]
    position = {e["id"]: i for i, e in enumerate(entries)}

    provider = BundledOnnxProvider()
    doc_vectors = provider.embed_documents(
        [f"{e['business_name']} {e['description']}" for e in entries]
    )
    query_vectors = provider.embed_queries(
        [
            f"{q['parent_path']} {q['field_name'].replace('__', ' ').replace('_', ' ')} {q['doc']}".strip()
            for q in queries
        ]
    )

    sims = query_vectors @ doc_vectors.T
    gold = np.array([position[q["gold_id"]] for q in queries])
    order = np.argsort(-sims, axis=1)
    ranks = np.array([int(np.where(order[i] == gold[i])[0][0]) + 1 for i in range(len(gold))])

    return {
        "baseline": data["_baseline"],
        "p_at_1": float((ranks == 1).mean()),
        "r_at_5": float((ranks <= 5).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "ranks": ranks,
    }


def _assert_not_worse(scored: dict, metric: str) -> None:
    baseline = scored["baseline"]
    expected, tolerance = baseline[metric], baseline["tolerance"]
    actual = scored[metric]
    assert actual >= expected - tolerance, (
        f"\n{metric} fell to {actual:.4f} from a recorded {expected:.4f} "
        f"(tolerance {tolerance}).\n"
        f"Retrieval quality regressed. Find the change that caused it before proceeding.\n"
        f"If the drop is intentional and justified, re-measure with "
        f"benchmarks/optimization_ledger.py and update '_baseline' in\n  {FIXTURE}\n"
        f"deliberately, in the same commit, with the reason in the message. Do not widen "
        f"the tolerance to make this pass."
    )


def test_precision_at_1_has_not_regressed(scored):
    """The headline retrieval metric. An encoder or query-construction break lands here."""
    _assert_not_worse(scored, "p_at_1")


def test_recall_at_5_has_not_regressed(scored):
    """
    Guards the case P@1 alone would miss: the gold entry sliding from rank 1 to rank 4
    looks like a P@1 collapse, but if R@5 held, the retriever is fine and the ORDERING
    broke -- a different bug with a different fix.
    """
    _assert_not_worse(scored, "r_at_5")


def test_mrr_has_not_regressed(scored):
    """
    The whole-distribution check. P@1 and R@5 are both thresholds and are blind to
    movement inside and beyond them; MRR notices the gold entry drifting from rank 6 to
    rank 40 across the corpus, which is what a slow degradation looks like before it
    becomes a visible one.
    """
    _assert_not_worse(scored, "mrr")


def test_the_task_is_still_hard_enough_to_measure_anything(scored):
    """
    Guards the FIXTURE rather than the code.

    A benchmark everything solves measures nothing. If someone regenerates this fixture
    without distractors -- a pool of only correct answers -- every test above would pass
    forever while detecting nothing. This project has already shipped one degenerate
    benchmark (OMOP, where the entry's business name was derived from the field name,
    giving token overlap of 1.000), so the trap is not hypothetical.
    """
    assert scored["p_at_1"] < 0.95, (
        f"P@1 {scored['p_at_1']:.3f} is saturated -- the fixture has stopped being a test. "
        f"Check it still contains distractors."
    )
    assert (scored["ranks"] > 1).sum() >= 30, (
        "too few queries are non-trivial to be a useful signal"
    )
