"""
tests.unit.application.test_governance_through_load_dictionary | Layer: TEST
The documented loading path, read through the matcher's own controlled vocabulary.

## Relationships
# TESTS → application/use_cases/match_schema :: NexusMatcher.load_dictionary,
#         _attach_governance, _refuse_an_unreadable_code_column, _merge_governance

`tests/museum/NM-0033` holds the headline claim: `from_config(governance=...)` plus
`load_dictionary(...)` produces coded entries and a match that carries a class. This file
covers the edges around it, each of which has its own silent failure mode:

  1. `governance_strict=False` is a real escape hatch, not a way to lose the reason a
     code is missing. An entry whose code was REFUSED must not come back looking like an
     entry that never declared one -- that is the defect this whole file is about, one
     row deep instead of one glossary deep.
  2. A vocabulary that was CONFIGURED and declares nothing is not the same as no
     vocabulary. The first says "every code in this glossary is one nobody defined"; the
     second says "read no codes". `_load_governance_vocabulary` collapses both to
     `GovernanceVocabulary.empty()`, so the matcher has to remember which happened.
  3. An explicit `column_mapping` must reach the governance read as well as the loader.
     Two readings of one file that disagree about which column holds the business name
     are two glossaries, and lining up rows between them would be a coin toss.
  4. When they DO disagree, the load must refuse rather than attach nothing.
  5. The path is not CSV-only. The Excel loader is the other one this package ships and
     the one an enterprise glossary actually arrives in.

The vocabulary is fictional. This library ships no taxonomy: it is supplied by the
caller, and this file supplies one exactly as a caller would.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application import ingest
from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.governance import GovernanceVocabulary
from nexus_matcher.domain.models.entities import DictionaryEntry
from nexus_matcher.domain.ports.dictionary_loader import ColumnMapping
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.dictionary_loaders.excel import (
    CsvDictionaryLoader,
    ExcelDictionaryLoader,
)
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import DataType, Result

# HULLNUM derives "Yard Confidential"; SLIPWAY derives "Open". Legacy spellings are
# declared as aliases, because a real glossary carries a decade of them.
VOCABULARY = {
    "open_classification": "Open",
    "aliases": {"OLD-HULL-REF": "HULLNUM", "n/a": None},
    "classes": [
        {
            "code": "HULLNUM",
            "name": "Vessel Hull Registration",
            "classification": "Yard Confidential",
            "personal_information": False,
            "direct_identifier": True,
        },
        {
            "code": "SLIPWAY",
            "name": "Published Slipway Reference",
            "classification": "Open",
            "personal_information": False,
            "direct_identifier": False,
        },
    ],
}

HEADER = "id,business_name,definition,protection_class,classification\n"
CODED = (
    HEADER + "Y-1,Hull Registration Number,The registration number stamped on a vessel hull,"
    "HULLNUM,Yard Confidential\n"
    "Y-2,Slipway Reference,Identifier of a slipway on the published yard plan,"
    "SLIPWAY,Open\n"
)
UNCODED = (
    "id,business_name,definition,classification\n"
    "Y-1,Hull Registration Number,The registration number stamped on a vessel hull,Internal\n"
)


class _ConstantProvider:
    """
    An encoder-shaped object returning one fixed vector.

    Nothing in this file reads a score or a ranking; loading the bundled model to produce
    numbers nobody looks at would make the file slower and no stronger.
    """

    dimension = 4
    model_name = "constant"

    def embed(self, texts):
        rows = np.ones((len(list(texts)), self.dimension), dtype=np.float32)

        class _Batch:
            embeddings = rows

        return Result.success(_Batch())

    def embed_single(self, text):
        return Result.success(np.ones(self.dimension, dtype=np.float32))


def _matcher(governance=VOCABULARY) -> NexusMatcher:
    return NexusMatcher(
        embedding_provider=_ConstantProvider(),
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=4)
        ),
        dictionary_loader_registry={"csv": CsvDictionaryLoader(), "excel": ExcelDictionaryLoader()},
        config=MatchingConfig(results_per_field=3),
        governance=governance,
    )


def _csv(tmp_path, text=CODED, name="glossary.csv") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _by_name(matcher: NexusMatcher) -> dict[str, DictionaryEntry]:
    return {e.business_name: e for e in matcher._dictionary_entries.values()}


# =============================================================================
# 1. governance_strict=False -- load anyway, and keep the evidence
# =============================================================================


class TestTheEscapeHatchKeepsTheEvidence:
    def test_a_refused_code_leaves_the_reason_on_the_entry(self, tmp_path):
        """
        The whole defect, one row deep. Under the escape hatch a row whose code the
        vocabulary rejects is loaded with `governance_code=None` -- which is exactly what
        an unclassified row looks like. What separates them is
        `source_metadata['governance_problems']`, and it has to survive the hand-off from
        the governance read to the loader's entry or the escape hatch becomes a quieter
        version of the bug it is an escape from.
        """
        text = HEADER + "Y-9,Mystery Column,A column nobody has classified yet,NOSUCHCODE,Open\n"
        matcher = _matcher()

        matcher.load_dictionary(_csv(tmp_path, text), governance_strict=False)

        entry = _by_name(matcher)["Mystery Column"]
        assert entry.governance_code is None
        assert entry.source_metadata["governance_code_raw"] == "NOSUCHCODE"
        assert "not in the configured vocabulary" in entry.source_metadata["governance_problems"][0]

    def test_a_contradicted_tier_loads_with_the_code_winning_and_the_problem_recorded(
        self, tmp_path
    ):
        """
        The catalog wins over the row, which is the derivation invariant -- so no entry
        inherits a tier its own code disowns even here. The contradiction is still
        recorded, because a load that swallowed it would leave the source file wrong and
        nobody looking.
        """
        text = HEADER + "Y-1,Hull Registration Number,The number stamped on a hull,HULLNUM,Open\n"
        matcher = _matcher()

        matcher.load_dictionary(_csv(tmp_path, text), governance_strict=False)

        entry = _by_name(matcher)["Hull Registration Number"]
        assert entry.governance_code == "HULLNUM"
        assert "contradicts itself" in entry.source_metadata["governance_problems"][0]

    def test_a_clean_row_gains_no_governance_metadata_keys(self, tmp_path):
        """
        The control for the two above. If the merge attached `governance_problems` to
        everything, "this row has a problem" would stop meaning anything.
        """
        matcher = _matcher()

        matcher.load_dictionary(_csv(tmp_path), governance_strict=False)

        entry = _by_name(matcher)["Hull Registration Number"]
        assert entry.governance_code == "HULLNUM"
        assert "governance_problems" not in entry.source_metadata

    def test_with_no_vocabulary_it_also_switches_off_the_code_column_refusal(self, tmp_path):
        """
        The documented opt-out of the OTHER refusal: read the protection-code column as
        plain metadata and load the glossary anyway. The entries then carry no codes, and
        the caller asked for that in writing.
        """
        matcher = _matcher(governance=None)

        matcher.load_dictionary(_csv(tmp_path), governance_strict=False)

        assert [e.governance_code for e in _by_name(matcher).values()] == [None, None]


# =============================================================================
# 2. "configured and empty" is not "not configured"
# =============================================================================


class TestAConfiguredVocabularyThatDeclaresNothing:
    def test_an_empty_vocabulary_rejects_every_code_rather_than_ignoring_the_column(self, tmp_path):
        """
        `GovernanceVocabulary.empty()` means a vocabulary WAS configured and declares
        nothing, so every code in the glossary is one nobody defined. Silently reading
        that as "governance off" is how an installation with a half-written catalog
        indexes a glossary of codes it cannot resolve and reports success.
        """
        matcher = _matcher(governance=GovernanceVocabulary.empty())

        with pytest.raises(ValueError) as refusal:
            matcher.load_dictionary(_csv(tmp_path))

        assert "defective governance" in str(refusal.value)

    def test_no_vocabulary_gives_the_other_refusal_for_the_same_file(self, tmp_path):
        """
        The pair to the test above, on a byte-identical glossary. Two different states,
        two different refusals, each naming the fix that actually applies -- wire a
        vocabulary, versus declare the codes your glossary already uses.
        """
        matcher = _matcher(governance=None)

        with pytest.raises(ValueError) as refusal:
            matcher.load_dictionary(_csv(tmp_path))

        assert "no vocabulary to interpret it" in str(refusal.value)

    def test_a_glossary_with_no_code_column_is_not_refused(self, tmp_path):
        """
        The control that keeps the refusal above from being a blanket one. Most glossaries
        carry no protection code, and a matcher with no vocabulary must go on loading them
        exactly as it did before this fix existed.
        """
        matcher = _matcher(governance=None)

        stats = matcher.load_dictionary(_csv(tmp_path, UNCODED, "uncoded.csv"))

        assert stats.valid_entries == 1
        assert _by_name(matcher)["Hull Registration Number"].governance_code is None


# =============================================================================
# 3. What the vocabulary resolves on this path
# =============================================================================


class TestWhatTheVocabularyResolves:
    def test_a_legacy_spelling_is_stored_as_the_code_the_catalog_declares(self, tmp_path):
        """
        The stored code is the CANONICAL one, so a glossary carrying a decade of legacy
        spellings resolves to one class rather than several. Attaching the raw token
        instead would produce entries whose codes no consumer can look up.
        """
        text = HEADER + "Y-1,Hull Registration Number,The number stamped on a hull,OLD-HULL-REF,\n"
        matcher = _matcher()

        matcher.load_dictionary(_csv(tmp_path, text))

        assert _by_name(matcher)["Hull Registration Number"].governance_code == "HULLNUM"

    def test_rows_sharing_a_business_name_keep_their_own_codes_in_order(self, tmp_path):
        """
        Two rows, one name -- a real glossary has them, usually because two systems
        describe the same term differently. The two readings are lined up by name, so a
        merge that collapsed duplicates would hand both rows the first one's class, and
        `HULLNUM` on a slipway reference is a wrong answer rather than a missing one.
        """
        text = (
            HEADER
            + "Y-1,Yard Reference,The registration number stamped on a vessel hull,HULLNUM,\n"
            + "Y-2,Yard Reference,Identifier of a slipway on the published yard plan,SLIPWAY,\n"
        )
        matcher = _matcher()

        matcher.load_dictionary(_csv(tmp_path, text))

        by_id = matcher._dictionary_entries
        assert by_id["Y-1"].governance_code == "HULLNUM"
        assert by_id["Y-2"].governance_code == "SLIPWAY"

    def test_the_load_statistics_are_the_loaders_own_and_unchanged(self, tmp_path):
        """
        The governance read is a second reading of the file, and it must not leak into
        what the caller is told about the first. `LoadStatistics` describes the load the
        dictionary loader performed; a caller who logs it is logging rows, not readings.
        """
        matcher = _matcher()

        stats = matcher.load_dictionary(_csv(tmp_path))

        assert (stats.total_rows, stats.valid_entries, stats.error_rows) == (2, 2, 0)


# =============================================================================
# 4. The two readings must line up, and say so when they do not
# =============================================================================


class TestTheTwoReadingsAreLinedUp:
    def test_an_explicit_column_mapping_reaches_the_governance_read(self, tmp_path):
        """
        A caller who names their columns names them for the loader; the governance read
        has to be told the same thing or it infers its own business-name column, lines up
        rows that are not the same rows, and either refuses a healthy glossary or -- worse
        -- attaches one row's class to another.

        The header here is deliberately outside the alias table, so inference cannot
        rescue it: nothing in it normalises to a known business-name or definition
        spelling, and only the explicit mapping makes the file readable at all.
        """
        text = (
            "Ref,Yard Wording,Yard Meaning,protection_class\n"
            "Y-1,Hull Registration Number,The number stamped on a vessel hull,HULLNUM\n"
        )
        mapping = ColumnMapping(
            id_column="Ref",
            business_name_column="Yard Wording",
            definition_column="Yard Meaning",
        )
        matcher = _matcher()

        matcher.load_dictionary(_csv(tmp_path, text, "renamed.csv"), column_mapping=mapping)

        entry = _by_name(matcher)["Hull Registration Number"]
        assert entry.id == "Y-1"
        assert entry.governance_code == "HULLNUM"

    def test_two_readings_that_disagree_refuse_the_load(self, tmp_path, monkeypatch):
        """
        The guard on the join itself. If the governance read ever produces a different set
        of rows from the loader's, the honest answer is a refusal naming the entry that
        could not be matched -- because the alternative is attaching no class to it, which
        is indistinguishable from the entry having none, which is this defect.

        Provoked directly rather than through a file, because every file that reaches this
        point lines up: the disagreement this guards against is a future change to either
        reader, and a test that could only be written once such a change existed would be
        written after the damage.
        """
        matcher = _matcher()
        monkeypatch.setattr(ingest, "load_entries", lambda *args, **kwargs: [])

        with pytest.raises(ValueError) as refusal:
            matcher.load_dictionary(_csv(tmp_path))

        assert "Hull Registration Number" in str(refusal.value)
        assert "disagree about which entries exist" in str(refusal.value)


# =============================================================================
# 5. Not CSV-only
# =============================================================================


class TestTheOtherLoaderThisPackageShips:
    def test_an_excel_glossary_is_read_through_the_vocabulary_too(self, tmp_path):
        """
        The Excel loader is the one an enterprise glossary actually arrives in, and it is
        a separate implementation reached by a separate branch of the loader registry. A
        fix wired only into the CSV path would leave the commonest real source uncoded and
        every test above green.
        """
        openpyxl = pytest.importorskip("openpyxl")
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["id", "business_name", "definition", "protection_class", "classification"])
        sheet.append(
            [
                "Y-1",
                "Hull Registration Number",
                "The registration number stamped on a vessel hull",
                "HULLNUM",
                "Yard Confidential",
            ]
        )
        path = tmp_path / "glossary.xlsx"
        workbook.save(path)
        matcher = _matcher()

        matcher.load_dictionary(str(path))

        entry = _by_name(matcher)["Hull Registration Number"]
        assert entry.governance_code == "HULLNUM"
        assert entry.data_type is DataType.UNKNOWN or entry.data_type is DataType.STRING
