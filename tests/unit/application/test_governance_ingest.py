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
            # A tier column and a PII column together. These are two different columns
            # saying two different things, and "PII" used to be an alias of the first.
            ["Term", "Definition", "Protection Class", "Classification", "PII"],
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

        # The PI and direct-identifier columns belong to the domain resolver alone.
        # `map_columns` must never claim one, because anything it maps is EXCLUDED from
        # `source_metadata` -- so a mapped PI column would delete the caller's own column
        # from the entry in exchange for a value the catalog overrides anyway.
        claimed = set(mapped.values())
        assert resolved.personal_information not in claimed
        assert resolved.direct_identifier not in claimed


class TestThePIIColumnIsAFlagNotATier:
    """
    "PII" was a member of the free-text TIER alias tuple, inherited unchanged from
    `COLUMN_ALIASES["protection_level"]` where it had been harmless -- as a free-text tier
    alias, a column named "PII" was just another place a tier might be written.

    Under the derivation invariant the same membership means something else entirely:
    "this row asserts its tier is 'Yes'". The header `Term,Definition,Protection
    Class,PII` therefore refused every coded row with "row states classification 'Yes'
    but code 'METERID' derives 'Sealed'" -- a contradiction the row never stated. Whether
    the load survived depended on whether some OTHER tier column happened to be present to
    mask it.

    The silent half is why this is more than noise. When an earlier tier alias IS present,
    "PII" is not the tier and was then never consulted at all: a row claiming personal
    information against a class declaring `personal_information: false` reported nothing,
    while `personal_data`, `is_personal_data` and `contains_personal_data` all reported it.
    """

    def test_a_pii_column_does_not_contradict_the_tier_the_code_derives(self):
        """The loud half. The row states a FLAG; it states no tier at all."""
        entries = load_entries(
            _rows(**{"Protection Class": "METERID", "PII": "Yes"}), governance=THORNBURY
        )

        assert _by_name(entries)["Broken Row"].governance_code == "METERID"

    def test_a_pii_column_disagreeing_with_the_catalog_is_reported(self):
        """PUBMAP declares personal_information false; this row says otherwise."""
        with pytest.raises(ValueError, match="personal_information"):
            load_entries(
                _rows(**{"Protection Class": "PUBMAP", "PII": "Yes"}), governance=THORNBURY
            )

    def test_the_disagreement_is_reported_even_when_a_tier_column_is_present(self):
        """
        THE SILENT HALF. With "Classification" present, "PII" lost the tier race and was
        then read by nothing -- so this row, which contradicts the catalog outright, loaded
        clean. Under every other spelling of the same column it was refused.
        """
        with pytest.raises(ValueError, match="personal_information"):
            load_entries(
                _rows(**{"Protection Class": "PUBMAP", "Classification": "Open", "PII": "Yes"}),
                governance=THORNBURY,
            )

    def test_a_pii_column_is_no_longer_mapped_onto_the_free_text_tier(self):
        assert "protection_level" not in map_columns(["Term", "Definition", "PII"])

    def test_a_pii_column_is_left_where_the_caller_put_it(self):
        """
        The documented breaking change. Anything `map_columns` claims is excluded from
        `source_metadata`; the PI columns are advisory and the catalog overrides them, so
        mapping one deleted the caller's column in exchange for a value we then ignore.
        """
        rows = [
            {
                "Term": "Household Name",
                "Definition": "The name of the account holder at a supply point.",
                "PII": "Yes",
            }
        ]

        entry = load_entries(rows)[0]

        assert entry.source_metadata["PII"] == "Yes"
        assert "governance_raw" not in entry.source_metadata


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
        The default reading of an unreadable code column. Nothing validated the token, so
        nothing may present it as governance -- but the raw value is still preserved,
        because dropping the caller's own column is what NM-0005's sibling defect did.

        `governance_strict=False` is required to reach this at all now: a glossary that
        carries protection codes and hands over no vocabulary to read them produces
        entries a consumer cannot distinguish from a glossary with no classes, so the load
        refuses unless the caller says out loud that that is what they want. See
        `TestACodeColumnNobodyCanRead`.
        """
        entries = load_entries(_rows(), governance_strict=False)

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
        plain = _by_name(load_entries(_rows(), governance_strict=False))["Meter Serial"]
        coded = _by_name(load_entries(_rows(), governance=THORNBURY))["Meter Serial"]

        assert coded.governance_code == "METERID"
        assert plain.governance_code is None
        assert coded.to_searchable_text() == plain.to_searchable_text()
        assert content_hash(coded) == content_hash(plain)
        assert "METERID" not in coded.to_searchable_text()


# =============================================================================
# REJECTION AND REFUSAL
# =============================================================================


class TestACodeColumnNobodyCanRead:
    """
    A glossary that carries protection codes, loaded with no vocabulary to interpret them.

    The silence is circular, which is why neither layer could catch it before: a code is
    attached only when a vocabulary is configured, and every consumer refuses codes it
    cannot resolve -- so with NO vocabulary there are no codes, nothing to refuse, and a
    glossary whose header plainly says "Protection Class" produces entries carrying
    nothing at all. Over HTTP that is `"governance": null` on every field of every
    response, which a caller cannot distinguish from a glossary that declares no classes.

    The check used to live in the HTTP app alone, which is one caller out of every caller,
    and it paid a second full read of the glossary to run it. Here the column mapping is
    already built.
    """

    def test_the_default_load_refuses_it(self):
        with pytest.raises(ValueError, match="protection-code column"):
            load_entries(_rows())

    def test_the_refusal_names_the_column_and_both_ways_out(self):
        """
        A refusal that does not name the column sends the reader to compare two files by
        eye, and one that does not name the escape hatch reads as "this file cannot be
        loaded".
        """
        with pytest.raises(ValueError) as excinfo:
            load_entries(_rows())

        message = str(excinfo.value)
        assert "'Protection Class'" in message
        assert "governance=" in message
        assert "governance_strict=False" in message

    def test_the_file_is_named_when_the_source_has_a_name(self, tmp_path):
        """
        This message is read at deployment time. "Some glossary has a column" sends an
        operator looking through however many they configured.
        """
        path = tmp_path / "glossary.csv"
        path.write_text(
            "Term,Definition,Protection Class\nMeter Serial,The serial number,METERID\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=r"glossary\.csv has a protection-code column"):
            load_entries(path)

    def test_the_opt_out_reads_the_column_as_plain_metadata(self):
        entries = load_entries(_rows(), governance_strict=False)

        assert [e.governance_code for e in entries] == [None, None, None]
        assert entries[0].source_metadata["governance_code_raw"] == "METERID"

    def test_a_glossary_with_no_code_column_is_untouched(self):
        """
        The control, and the reason this is not simply "refuse every ungoverned load".
        Most glossaries have no controlled vocabulary at all and must keep loading exactly
        as they did.
        """
        rows = [{"Term": "Household Name", "Definition": "The account holder's name."}]

        assert len(load_entries(rows)) == 1

    def test_a_free_text_tier_column_alone_is_not_a_code_column(self):
        """
        "Classification" is the free-text tier, not the controlled code. A glossary that
        classifies in prose has adopted no vocabulary and is not missing one.
        """
        rows = [
            {
                "Term": "Household Name",
                "Definition": "The account holder's name.",
                "Classification": "Sealed",
            }
        ]

        assert len(load_entries(rows)) == 1

    def test_a_configured_vocabulary_loads_the_same_glossary_normally(self):
        """The other control: the guard must fire on the missing vocabulary, not the column."""
        assert len(load_entries(_rows(), governance=THORNBURY)) == 3


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


class TestTheRefusalNamesTheVocabularyOnce:
    """
    The declared codes belong in the SUMMARY, which is built once per load.

    `problems_with()` used to interpolate the whole list into every rejected row's message.
    Ten displayed defects therefore carried ten copies of it -- a ~70,000-character
    refusal, of which one copy was information and nine were scrolling -- and under
    `governance_strict=False` all 30,000 of those strings were retained in
    `source_metadata`. The per-row message now names the token and the SIZE of the
    vocabulary; this is the one place the vocabulary itself is spelled out.
    """

    def _unknown_code_rows(self, count: int):
        return [
            {
                "Term": f"Broken Row {i}",
                "Definition": "A row whose protection code nobody declared.",
                "Protection Class": "SUPERSECRET",
                "Classification": "Sealed",
            }
            for i in range(count)
        ]

    def test_the_declared_codes_appear_exactly_once(self):
        """
        Twelve defective rows, one copy of the vocabulary. Counting occurrences is the
        assertion, not merely finding one: the defect was that the list scaled with the
        number of displayed defects.
        """
        with pytest.raises(ValueError) as excinfo:
            load_entries(self._unknown_code_rows(12), governance=THORNBURY)

        message = str(excinfo.value)
        assert message.count("METERID") == 1
        assert message.count("PUBMAP") == 1
        assert "declares 3 code(s): METERID, PUBMAP, USAGE" in message

    def test_only_the_first_ten_defects_are_quoted_and_the_rest_are_counted(self):
        with pytest.raises(ValueError) as excinfo:
            load_entries(self._unknown_code_rows(12), governance=THORNBURY)

        message = str(excinfo.value)
        assert message.startswith("12 row(s) carry defective governance")
        assert message.count("SUPERSECRET") == 10
        assert "... and 2 more" in message

    def test_a_row_that_only_contradicts_its_own_tier_is_not_shown_the_code_list(self):
        """
        The code list answers "what could I have written instead", which is a question
        only a rejected code asks. On a glossary full of tier contradictions it would push
        the two tiers that actually disagree off the top of the message.
        """
        with pytest.raises(ValueError) as excinfo:
            load_entries(
                _rows(**{"Protection Class": "USAGE", "Classification": "Open"}),
                governance=THORNBURY,
            )

        assert "code(s):" not in str(excinfo.value)

    def test_an_empty_vocabulary_lists_nothing_and_keeps_saying_so(self):
        """
        `empty()` declares no codes, so there is no list to print -- and "declares 0
        code(s)" would be a second, worse answer to the question the per-row message
        already answers by name.
        """
        from nexus_matcher.domain.governance import GovernanceVocabulary

        with pytest.raises(ValueError) as excinfo:
            load_entries(_rows(), governance=GovernanceVocabulary.empty())

        message = str(excinfo.value)
        assert "no vocabulary is configured" in message
        assert "code(s):" not in message


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
