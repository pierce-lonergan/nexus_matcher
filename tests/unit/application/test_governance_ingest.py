"""
tests.unit.application.test_governance_ingest | Layer: TEST
Loading a glossary against a caller-supplied vocabulary.

Three properties, each of which has a matching failure mode that is silent:

  * the governance code rides in on the EXISTING alias machinery, so a caller who never
    heard of this feature gets it from the column they already have
  * a code the vocabulary does not define is REJECTED, never stored -- a stored code
    nobody defined reads as governance and is not
  * a row whose tier contradicts its code REFUSES the load, because indexing it would let
    a field inherit a tier its own code disowns

The vocabulary below is fictional (see tests/unit/domain/test_governance.py for why). It
is repeated rather than imported because each file bends it differently and a shared
fixture that every file has to mutate is harder to read than the twenty lines it saves.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from nexus_matcher.application.ingest import content_hash, load_entries, map_columns

THORNBURY = {
    "open_classification": "Open",
    "aliases": {"MTR#": "METERID", "n/a": None},
    "classes": [
        {
            "code": "METERID",
            "name": "Meter Serial Identifier",
            "classification": "Sealed",
            "personal_information": True,
            "direct_identifier": True,
        },
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


def _rows(**overrides):
    """Three good rows, plus whatever the caller wants to break."""
    rows = [
        {
            "Term": "Meter Serial",
            "Definition": "The serial number stamped on a customer's water meter.",
            "Protection Class": "METERID",
            "Classification": "Sealed",
        },
        {
            "Term": "Quarterly Reading",
            "Definition": "Volume of water drawn at a supply point over a billing quarter.",
            "Protection Class": "USAGE",
            "Classification": "Guarded",
        },
        {
            "Term": "Trunk Main Reference",
            "Definition": "Identifier of a trunk main on the published network map.",
            "Protection Class": "PUBMAP",
            "Classification": "Open",
        },
    ]
    if overrides:
        rows.append(
            {
                "Term": "Broken Row",
                "Definition": "A row whose governance columns disagree with each other.",
                **overrides,
            }
        )
    return rows


def _by_name(entries):
    return {e.business_name: e for e in entries}


# =============================================================================
# COLUMN MAPPING
# =============================================================================


class TestTheCodeColumnRidesTheExistingMachinery:
    @pytest.mark.parametrize(
        "column",
        ["Protection Class", "protection_class", "GOVERNANCE-CODE", "Classification Code"],
    )
    def test_a_governance_code_column_is_recognised(self, column):
        assert map_columns(["Term", "Definition", column])["governance_code"] == column

    def test_the_free_text_tier_column_keeps_its_existing_meaning(self):
        """
        A glossary carrying both must not have them swapped. `map_columns` takes the first
        unclaimed column per field and `protection_level` is resolved first, so
        "Classification" stays the tier and "Classification Code" becomes the code.
        """
        mapping = map_columns(["Term", "Classification", "Classification Code"])
        assert mapping["protection_level"] == "Classification"
        assert mapping["governance_code"] == "Classification Code"

    @pytest.mark.parametrize(
        "header",
        [
            ["Term", "Definition", "Protection Class", "Classification"],
            ["Term", "Definition", "Classification", "Classification Code"],
            ["Term", "Definition", "Governance Code", "Sensitivity"],
            ["Term", "Definition", "Classification"],
            ["Term", "Definition", "Protection Class"],
            ["Term", "Definition"],
            # Every column an earlier field could plausibly steal, all at once.
            ["ID", "Term", "Column", "Definition", "Type", "Category", "PII", "Protection Code"],
        ],
    )
    def test_the_two_resolvers_agree(self, header):
        """
        THE DRIFT GATE. `problems_with()` takes a raw row and resolves the code and tier
        columns itself; `load_entries()` resolves them through `map_columns`. They share
        the alias tuples, but not the resolution CODE -- `map_columns` also enforces a
        first-come "taken" rule that lets an earlier field claim a column.

        If those two ever answer differently, the validator and the loader read the same
        file differently: a row could be validated against one column and indexed from
        another, and every message would name the wrong one. Today the alias sets are
        pairwise disjoint so the rules cannot diverge; this pins that, so a future alias
        that broke it fails here rather than silently.
        """
        from nexus_matcher.domain.governance import _governance_columns

        mapped = map_columns(header)
        resolved = _governance_columns(tuple(header))

        assert resolved.code == mapped.get("governance_code")
        assert resolved.classification == mapped.get("protection_level")


# =============================================================================
# ATTACHING A CODE
# =============================================================================


class TestLoading:
    def test_a_valid_code_lands_on_the_entry(self):
        entries = load_entries(_rows(), governance=THORNBURY)
        assert [e.governance_code for e in entries] == ["METERID", "USAGE", "PUBMAP"]

    def test_the_vocabulary_can_be_a_path_to_the_caller_s_json_file(self, tmp_path):
        """The shape a real caller uses: a file they own, next to their glossary."""
        path = tmp_path / "protection_classes.json"
        path.write_text(json.dumps(THORNBURY), encoding="utf-8")

        entries = load_entries(_rows(), governance=path)

        assert _by_name(entries)["Meter Serial"].governance_code == "METERID"

    def test_a_legacy_spelling_is_stored_as_the_code_it_maps_to(self):
        """
        Mapped, never stored raw. The entry carries "METERID" even though the glossary
        said "MTR#", so a consumer comparing codes across two glossaries can compare them.
        """
        rows = _rows()
        rows[0]["Protection Class"] = "MTR#"

        entry = _by_name(load_entries(rows, governance=THORNBURY))["Meter Serial"]

        assert entry.governance_code == "METERID"
        assert entry.source_metadata["governance_code_raw"] == "MTR#"

    def test_without_a_vocabulary_no_code_is_attached_at_all(self):
        """
        The default. Nothing validated the token, so nothing may present it as governance
        -- but the raw value is still preserved, because dropping the caller's own column
        is what NM-0005's sibling defect did.
        """
        entries = load_entries(_rows())

        assert [e.governance_code for e in entries] == [None, None, None]
        assert entries[0].source_metadata["governance_code_raw"] == "METERID"

    def test_a_row_with_no_code_carries_none(self):
        rows = _rows()
        rows[2]["Protection Class"] = ""

        assert (
            _by_name(load_entries(rows, governance=THORNBURY))[
                "Trunk Main Reference"
            ].governance_code
            is None
        )

    def test_a_declared_junk_token_is_dropped_rather_than_stored(self):
        rows = _rows()
        rows[2]["Protection Class"] = "n/a"

        entry = _by_name(load_entries(rows, governance=THORNBURY))["Trunk Main Reference"]

        assert entry.governance_code is None
        assert entry.source_metadata["governance_code_raw"] == "n/a"

    def test_governance_is_not_embedded_and_does_not_move_the_content_hash(self):
        """
        Pins the documented incremental-sync contract from the other side. Governance is
        metadata ABOUT a term, not a description of it; if it reached
        `to_searchable_text()` then re-classifying a glossary would re-embed every row in
        it -- the exact cost `content_hash` exists to avoid.
        """
        plain = _by_name(load_entries(_rows()))["Meter Serial"]
        coded = _by_name(load_entries(_rows(), governance=THORNBURY))["Meter Serial"]

        assert coded.governance_code == "METERID"
        assert plain.governance_code is None
        assert coded.to_searchable_text() == plain.to_searchable_text()
        assert content_hash(coded) == content_hash(plain)
        assert "METERID" not in coded.to_searchable_text()


# =============================================================================
# REJECTION AND REFUSAL
# =============================================================================


class TestUnknownCodesAreRejected:
    def test_an_undefined_code_refuses_the_load(self):
        with pytest.raises(ValueError, match="not in the configured vocabulary"):
            load_entries(_rows(**{"Protection Class": "SUPERSECRET"}), governance=THORNBURY)

    def test_and_is_never_stored_even_when_the_load_is_allowed_through(self):
        """
        The hard requirement: a code nobody defined must not end up on an entry. It
        survives only under a diagnostic key, as evidence for whoever fixes the source.
        """
        entries = load_entries(
            _rows(**{"Protection Class": "SUPERSECRET"}),
            governance=THORNBURY,
            governance_strict=False,
        )
        broken = _by_name(entries)["Broken Row"]

        assert broken.governance_code is None
        assert broken.source_metadata["governance_code_raw"] == "SUPERSECRET"
        assert broken.source_metadata["governance_problems"]

    def test_an_empty_vocabulary_rejects_every_code(self):
        """
        Configuring nothing is not configuring permissiveness. `empty()` defines no codes,
        so a glossary full of them is a glossary full of codes nobody defined.
        """
        from nexus_matcher.domain.governance import GovernanceVocabulary

        with pytest.raises(ValueError, match="no vocabulary is configured"):
            load_entries(_rows(), governance=GovernanceVocabulary.empty())


class TestTheDerivationInvariant:
    """
    A row whose tier contradicts its own code is a DATA DEFECT and the loader refuses it.

    This is the museum entry NM-0028 case, asserted here at the loader boundary.
    """

    CONTRADICTION: ClassVar[dict[str, str]] = {
        "Protection Class": "USAGE",
        "Classification": "Open",
    }

    def test_the_load_is_refused(self):
        with pytest.raises(ValueError) as excinfo:
            load_entries(_rows(**self.CONTRADICTION), governance=THORNBURY)

        message = str(excinfo.value)
        assert "'Open'" in message and "'Guarded'" in message, (
            f"the refusal must name both the stated tier and the derived one: {message!r}"
        )

    def test_nothing_at_all_is_returned_not_just_the_bad_row(self):
        """
        Returning the good rows is the silent failure this replaces. The caller would
        index a dictionary that looks healthy, match a field against it, and inherit
        nothing where they should have inherited a class -- with no error anywhere.
        """
        with pytest.raises(ValueError):
            load_entries(_rows(**self.CONTRADICTION), governance=THORNBURY)

    def test_the_softer_mode_still_refuses_to_honour_the_row(self):
        """
        `governance_strict=False` loads the glossary; it does NOT make the row's claim
        true. The catalog wins, so the entry carries USAGE -- which derives "Guarded" --
        and the contradiction travels with the entry as evidence.
        """
        entries = load_entries(
            _rows(**self.CONTRADICTION), governance=THORNBURY, governance_strict=False
        )
        broken = _by_name(entries)["Broken Row"]

        assert broken.governance_code == "USAGE"
        assert len(broken.source_metadata["governance_problems"]) == 1
        assert "contradicts itself" in broken.source_metadata["governance_problems"][0]

    def test_a_row_that_agrees_with_its_code_is_untouched(self):
        """Guards the vacuous pass: a loader that refused everything would satisfy the above."""
        entries = load_entries(_rows(), governance=THORNBURY)
        assert len(entries) == 3
        assert all("governance_problems" not in e.source_metadata for e in entries)

    def test_a_disagreeing_flag_column_is_reported_too(self):
        """The catalog wins for the PI and direct-identifier flags on the same terms."""
        with pytest.raises(ValueError, match="personal_information"):
            load_entries(
                _rows(**{"Protection Class": "USAGE", "Personal Data": "N"}),
                governance=THORNBURY,
            )
