"""
NM-0030 -- refreshing an index stripped the protection code off every entry, and reported
"unchanged".

`build_index(source, governance=VOCABULARY)` reads a glossary against the caller's
controlled vocabulary and attaches the code each row declares. `sync(index, source)` --
the second of the three lines the module's docstring promises -- re-read the same file with
none of that. `GlossaryIndex` remembered the embedding provider and nothing else, and
`sync`'s own docstring example passes no options at all, so the vocabulary went out of
scope the moment `build_index` returned.

Measured on a byte-identical file: 30 entries, 27 of them coded, became 30 entries and 0
coded, and the report read `+0 added ~0 updated -0 removed =30 unchanged`. Nothing errored,
because there is nothing wrong with a glossary loaded without a vocabulary -- that is the
documented default. The refresh loop that keeps a reclassified row's entry object current
is what made the loss total: it replaces every entry, so every code went.

## The half that is worse than the loss

The vocabulary is not only where codes come from. It is where the DERIVATION INVARIANT is
enforced -- a row whose stated tier contradicts the tier its own code derives is a data
defect, and `load_entries` refuses the whole load rather than index it (NM-0028). With no
vocabulary there is nothing to contradict, so the identical row that NM-0028 refuses on
build loaded silently on refresh, and the index a caller then matched against carried a
classification its own code disowns.

NM-0028's gate was real and stayed green the entire time. It asked `load_entries`. Nothing
asked `sync`: of the 24 places in the suite that called `build_index`, not one passed
`governance=`, so every sync test ran over a glossary with no codes to lose.

## What this asserts

The OBSERVABLE SYMPTOM at the boundary a caller uses -- what the entries in the index carry
after a sync, and what a self-contradicting row makes `sync` do -- rather than the presence
of any particular field on `GlossaryIndex`. If the remembering is restructured, this test
should still be right.

## The vocabulary is FICTIONAL

Thornbury Water Authority does not exist. This library ships no taxonomy: the vocabulary is
supplied by the caller, and this file supplies one exactly as a caller would. What is under
test is the STRUCTURE -- a closed set of codes, each deriving a tier and two flags.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application import ingest

# USAGE derives "Guarded", PUBMAP derives "Open". Pinned as literals and asserted below, so
# the expectation does not come from the code under test (H-004).
VOCABULARY = {
    "open_classification": "Open",
    "classes": [
        {
            "code": "USAGE",
            "name": "Metered Consumption Reading",
            "classification": "Guarded",
            "personal_information": True,
            "direct_identifier": False,
        },
        {
            "code": "PUBMAP",
            "name": "Published Network Map Reference",
            "classification": "Open",
            "personal_information": False,
            "direct_identifier": False,
        },
    ],
}

READING = {
    "Term": "Quarterly Reading",
    "Definition": "Volume of water drawn at a supply point over a billing quarter.",
    "Protection Class": "USAGE",
    "Classification": "Guarded",
}
TRUNK_MAIN = {
    "Term": "Trunk Main Reference",
    "Definition": "Identifier of a trunk main on the published network map.",
    "Protection Class": "PUBMAP",
    "Classification": "Open",
}

# The NM-0028 shape: the code says USAGE, which derives "Guarded"; the row says the field
# sits at the open tier. Honouring it publishes household consumption data.
CONTRADICTING_READING = {**READING, "Classification": "Open"}


class _ConstantProvider:
    """
    An encoder-shaped object returning one fixed vector.

    This entry is about which OPTIONS a refresh re-reads with; no assertion below looks at
    a vector. Loading the bundled model to produce numbers nothing reads would make the
    entry slower and no stronger.
    """

    dimension = 4
    model_name = "constant"

    def embed_documents(self, texts):
        return np.ones((len(list(texts)), self.dimension), dtype="float32")


def _codes(index: ingest.GlossaryIndex) -> dict[str, str | None]:
    return {e.business_name: e.governance_code for e in index.entries.values()}


def test_the_premise_the_build_really_does_attach_the_codes():
    """
    Guards this file against passing vacuously.

    Every assertion below is about what a SYNC preserves. If `build_index` stopped
    attaching codes at all, "the codes survived the sync" would be true of nothing, and
    this entry would report coverage it does not have.
    """
    index = ingest.build_index(
        [READING, TRUNK_MAIN], provider=_ConstantProvider(), governance=VOCABULARY
    )

    assert _codes(index) == {"Quarterly Reading": "USAGE", "Trunk Main Reference": "PUBMAP"}


def test_a_no_change_sync_preserves_every_protection_code():
    """THE DEFECT. Same rows, no edit at all, and the index came back stripped."""
    rows = [READING, TRUNK_MAIN]
    index = ingest.build_index(rows, provider=_ConstantProvider(), governance=VOCABULARY)

    report = ingest.sync(index, rows)

    assert _codes(index) == {"Quarterly Reading": "USAGE", "Trunk Main Reference": "PUBMAP"}
    assert report.unchanged == 2


def test_sync_refuses_a_self_contradicting_row_exactly_as_load_entries_does():
    """
    THE INVARIANT, and the reason this is not merely a lost-metadata bug. NM-0028's gate
    is the vocabulary; a refresh that dropped the vocabulary dropped the gate with it.

    The refusal is compared against the one `load_entries` gives for the same rows, not
    merely required to be "a ValueError". A `sync` that refuses for some unrelated reason
    -- because the glossary carries codes nothing is configured to read, say -- has still
    bypassed the derivation check, and an assertion that could not tell those apart would
    go green on the bypass.
    """
    rows = [TRUNK_MAIN, READING]
    index = ingest.build_index(rows, provider=_ConstantProvider(), governance=VOCABULARY)
    edited = [TRUNK_MAIN, CONTRADICTING_READING]

    with pytest.raises(ValueError) as from_load:
        ingest.load_entries(edited, governance=VOCABULARY)
    with pytest.raises(ValueError) as from_sync:
        ingest.sync(index, edited)

    assert "contradicts itself" in str(from_sync.value)
    assert str(from_sync.value) == str(from_load.value)


def test_no_entry_survives_in_the_index_carrying_the_contradicted_tier():
    """
    The consequence, stated as the thing that must not exist. A refresh that logged and
    carried on would satisfy nothing here: the point is that the index a caller goes on to
    match against does not hold the row.
    """
    rows = [TRUNK_MAIN, READING]
    index = ingest.build_index(rows, provider=_ConstantProvider(), governance=VOCABULARY)

    with pytest.raises(ValueError):
        ingest.sync(index, [TRUNK_MAIN, CONTRADICTING_READING])

    assert _codes(index)["Quarterly Reading"] == "USAGE"


def test_a_legitimate_edit_still_syncs():
    """
    The control. A `sync` that refused everything, or one that re-embedded the whole
    glossary every time, would satisfy all of the above while making the function useless
    -- and nobody would notice until a real glossary was pointed at it.
    """
    rows = [READING, TRUNK_MAIN]
    index = ingest.build_index(rows, provider=_ConstantProvider(), governance=VOCABULARY)
    edited = [
        {**READING, "Definition": "Volume of water drawn over a billing quarter."},
        TRUNK_MAIN,
    ]

    report = ingest.sync(index, edited)

    assert report.embedded == 1
    assert report.unchanged == 1
    assert _codes(index) == {"Quarterly Reading": "USAGE", "Trunk Main Reference": "PUBMAP"}
