"""
tests.unit.domain.test_governance | Layer: TEST
The caller-supplied controlled vocabulary, and the derivation invariant it enforces.

## The vocabulary used here is FICTIONAL, and deliberately so

Thornbury Water Authority does not exist. Its codes, tiers and legacy tokens were invented
for this file. That is not decoration: a library that assigns governance classifications is
most likely to be pointed at a real organisation's internal catalog, and a test suite that
hard-codes one organisation's taxonomy both leaks it and quietly asserts that everyone
else's is wrong. The library ships NO taxonomy; these tests supply one, exactly as a caller
would.

What is being tested is the STRUCTURE, and the structure is what a real catalog has:

  * a small closed set of protection codes
  * each code implying a tier, a personal-information flag and a direct-identifier flag
  * the tier DERIVED from the code, never read from the row -- a row where the two
    disagree is a data defect
  * an uncoded case that lands on the most open tier
  * legacy and junk tokens that must be mapped onto a code or dropped, never stored raw
"""

from __future__ import annotations

import json

import pytest

from nexus_matcher.domain.governance import (
    OPEN_CLASSIFICATION,
    GovernanceVocabulary,
    ProtectionClass,
)

# --- the fictional catalog ---------------------------------------------------
#
# Three tiers, five codes, and both kinds of legacy token: "MTR#" and "LEGACY-METER" are
# old spellings that MAP onto a current code, while "n/a" and "TBD" are junk the source
# system emits and the caller declares droppable.
THORNBURY = {
    "open_classification": "Open",
    "aliases": {"MTR#": "METERID", "n/a": None, "TBD": None},
    "classes": [
        {
            "code": "METERID",
            "name": "Meter Serial Identifier",
            "classification": "Sealed",
            "personal_information": True,
            "direct_identifier": True,
            "enhancement": "tokenise",
            "aliases": ["LEGACY-METER"],
        },
        {
            "code": "HOUSEHOLD",
            "name": "Household Name On Account",
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
            "code": "TARIFF",
            "name": "Tariff Band Code",
            "classification": "Guarded",
            "personal_information": False,
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


@pytest.fixture(scope="module")
def vocabulary() -> GovernanceVocabulary:
    return GovernanceVocabulary.from_json(THORNBURY)


# =============================================================================
# LOADING
# =============================================================================


class TestLoading:
    def test_a_json_file_on_disk_is_the_supported_path(self, tmp_path):
        """The documented entry point. A caller has a file, not a dict literal."""
        path = tmp_path / "protection_classes.json"
        path.write_text(json.dumps(THORNBURY), encoding="utf-8")

        loaded = GovernanceVocabulary.from_json(path)

        assert loaded.codes == {"METERID", "HOUSEHOLD", "USAGE", "TARIFF", "PUBMAP"}
        assert loaded.get("METERID") == ProtectionClass(
            code="METERID",
            name="Meter Serial Identifier",
            classification="Sealed",
            personal_information=True,
            direct_identifier=True,
            enhancement="tokenise",
        )

    def test_a_missing_file_is_a_FileNotFoundError_not_an_empty_vocabulary(self, tmp_path):
        """
        The dangerous alternative is degrading to `empty()`: every code then becomes
        unknown, every row gets refused, and the message blames the glossary for a typo in
        a path.
        """
        with pytest.raises(FileNotFoundError, match="not found"):
            GovernanceVocabulary.from_json(tmp_path / "absent.json")

    def test_the_library_ships_no_taxonomy(self):
        """`empty()` has no codes. There is no built-in catalog to fall back on."""
        assert GovernanceVocabulary.empty().codes == frozenset()

    @pytest.mark.parametrize(
        ("document", "expected"),
        [
            pytest.param(THORNBURY["classes"], 5, id="bare-array"),
            pytest.param({"classes": THORNBURY["classes"]}, 5, id="classes-key"),
            pytest.param(
                {"protection_classes": THORNBURY["classes"]}, 5, id="protection-classes-key"
            ),
            pytest.param(
                {
                    "TARIFF": {
                        "name": "Tariff Band Code",
                        "classification": "Guarded",
                        "personal_information": False,
                        "direct_identifier": False,
                    }
                },
                1,
                id="code-keyed-mapping",
            ),
        ],
    )
    def test_the_three_document_shapes_all_load(self, document, expected):
        assert len(GovernanceVocabulary.from_json(document).codes) == expected

    def test_a_misspelled_classes_key_fails_rather_than_loading_one_class_called_clases(self):
        """
        The bare `{code: attrs}` shape is only recognised when every value is an object,
        so a typo in the wrapper key cannot be read as a catalog. Half-loading is the worst
        outcome available: the codes it silently dropped become "unknown", and every row
        using them is refused for a reason that appears nowhere in the caller's file.
        """
        with pytest.raises(ValueError, match="no protection classes found"):
            GovernanceVocabulary.from_json({"clases": THORNBURY["classes"]})

    def test_a_class_without_a_classification_is_refused(self):
        with pytest.raises(ValueError, match="no 'classification'"):
            GovernanceVocabulary.from_json(
                [
                    {
                        "code": "TARIFF",
                        "name": "Tariff Band Code",
                        "personal_information": False,
                        "direct_identifier": False,
                    }
                ]
            )

    def test_an_undeclared_flag_is_refused_rather_than_defaulted_to_false(self):
        """
        False is the PERMISSIVE answer -- "this class is not personal information" -- and
        defaulting to it would have the library assert that on the caller's behalf, in the
        one place where being wrong is expensive.
        """
        with pytest.raises(ValueError, match="does not declare 'direct_identifier'"):
            GovernanceVocabulary.from_json(
                [
                    {
                        "code": "TARIFF",
                        "name": "Tariff Band Code",
                        "classification": "Guarded",
                        "personal_information": False,
                    }
                ]
            )

    def test_two_classes_cannot_share_a_code(self):
        """Otherwise a row's tier depends on which of the two won the dict."""
        with pytest.raises(ValueError, match="duplicate protection code"):
            GovernanceVocabulary.from_json(
                [
                    {
                        "code": "TARIFF",
                        "name": "Tariff Band Code",
                        "classification": "Guarded",
                        "personal_information": False,
                        "direct_identifier": False,
                    },
                    {
                        "code": "tariff",
                        "name": "Tariff Band Code (old)",
                        "classification": "Open",
                        "personal_information": False,
                        "direct_identifier": False,
                    },
                ]
            )

    def test_an_alias_pointing_at_nothing_is_refused(self):
        with pytest.raises(ValueError, match="not a declared code"):
            GovernanceVocabulary.from_json({**THORNBURY, "aliases": {"OLD": "NOSUCHCODE"}})

    def test_an_alias_cannot_shadow_a_declared_code(self):
        with pytest.raises(ValueError, match="collides with the declared code"):
            GovernanceVocabulary.from_json({**THORNBURY, "aliases": {"tariff": "USAGE"}})


# =============================================================================
# LOOKUP
# =============================================================================


class TestLookup:
    def test_a_code_carries_its_whole_class_not_just_a_tier(self, vocabulary):
        """
        The flags are the point. "Sealed" alone does not tell a consumer whether the field
        identifies a person on its own, which is the fact that decides masking.
        """
        usage = vocabulary.get("USAGE")
        assert (usage.classification, usage.personal_information, usage.direct_identifier) == (
            "Guarded",
            True,
            False,
        )

    @pytest.mark.parametrize("spelling", ["METERID", "meterid", "Meter-Id", " meter id "])
    def test_lookup_survives_the_spelling_a_glossary_actually_uses(self, vocabulary, spelling):
        assert vocabulary.get(spelling).code == "METERID"

    def test_a_declared_legacy_token_maps_onto_its_current_code(self, vocabulary):
        """Mapped, not stored raw: both old spellings resolve to the same class."""
        assert vocabulary.get("MTR#").code == "METERID"
        assert vocabulary.get("LEGACY-METER").code == "METERID"

    @pytest.mark.parametrize("junk", ["n/a", "N/A", "TBD"])
    def test_a_declared_junk_token_is_dropped(self, vocabulary, junk):
        assert vocabulary.get(junk) is None

    @pytest.mark.parametrize("nothing", [None, "", "   ", "--"])
    def test_absent_means_absent(self, vocabulary, nothing):
        assert vocabulary.get(nothing) is None
        assert vocabulary.classification_for(nothing) == "Open"

    def test_an_uncoded_field_lands_on_the_declared_open_tier(self, vocabulary):
        assert vocabulary.classification_for(None) == "Open"

    def test_an_unconfigured_vocabulary_uses_a_sentinel_not_a_plausible_tier(self):
        """
        Any real-sounding default would be this library inventing one organisation's
        policy, and would read in an audit as somebody's decision rather than as a gap in
        configuration.
        """
        assert GovernanceVocabulary.empty().classification_for(None) == OPEN_CLASSIFICATION
        assert OPEN_CLASSIFICATION not in {"Open", "Guarded", "Sealed"}

    def test_the_tier_comes_from_the_code_and_from_nowhere_else(self, vocabulary):
        """`classification_for` reads the catalog. There is no row argument to override it."""
        assert vocabulary.classification_for("USAGE") == "Guarded"
        assert vocabulary.classification_for("METERID") == "Sealed"

    def test_aliases_are_not_codes(self, vocabulary):
        """`codes` is what the caller declared, so it can be shown to a human verbatim."""
        assert "MTR#" not in vocabulary.codes
        assert vocabulary.codes == {"METERID", "HOUSEHOLD", "USAGE", "TARIFF", "PUBMAP"}


# =============================================================================
# THE DERIVATION INVARIANT
# =============================================================================


class TestProblemsWith:
    """
    A row is valid when it asserts nothing the vocabulary contradicts.

    Absolute expectations are pinned against the FICTIONAL catalog above rather than
    derived from the vocabulary under test (H-004): a check that computed the expected
    tier by calling `classification_for` would agree with any implementation, including
    one that read the tier straight off the row.
    """

    def test_a_consistent_row_has_no_problems(self, vocabulary):
        assert (
            vocabulary.problems_with(
                {
                    "Term": "Meter Serial",
                    "Protection Class": "METERID",
                    "Classification": "Sealed",
                    "Personal Data": "Y",
                    "Direct Identifier": "yes",
                }
            )
            == []
        )

    def test_a_row_whose_tier_contradicts_its_code_is_a_defect(self, vocabulary):
        """
        THE INVARIANT. USAGE derives "Guarded"; this row claims "Open". Honouring the row
        would let a field inherit a tier its own code disowns -- which is the whole class
        of bug this library exists to prevent (NM-0005).
        """
        problems = vocabulary.problems_with(
            {"Term": "Quarterly Reading", "Protection Class": "USAGE", "Classification": "Open"}
        )

        assert len(problems) == 1
        assert "'Open'" in problems[0] and "'Guarded'" in problems[0], (
            f"the message must name BOTH sides so a reader knows which file to fix: {problems[0]!r}"
        )

    def test_the_contradiction_is_reported_in_the_permissive_direction_too(self, vocabulary):
        """
        A row that OVER-states its tier is equally a defect. It is the easier one to wave
        through -- nobody is harmed by extra protection today -- and it is how a catalog
        and a glossary drift apart until neither can be trusted.
        """
        assert vocabulary.problems_with(
            {"Term": "Tariff Band", "Protection Class": "TARIFF", "Classification": "Sealed"}
        )

    def test_the_comparison_ignores_case_and_padding_only(self, vocabulary):
        """ "  guarded " is the same tier written sloppily. "Open" is a different tier."""
        assert (
            vocabulary.problems_with(
                {"Term": "x", "Protection Class": "USAGE", "Classification": "  guarded "}
            )
            == []
        )
        assert vocabulary.problems_with(
            {"Term": "x", "Protection Class": "USAGE", "Classification": "Guarded-ish"}
        )

    def test_an_undefined_code_is_reported(self, vocabulary):
        problems = vocabulary.problems_with({"Term": "x", "Protection Class": "SUPERSECRET"})
        assert len(problems) == 1
        assert "SUPERSECRET" in problems[0]
        assert "not in the configured vocabulary" in problems[0]

    def test_an_empty_vocabulary_reports_every_code_as_undefined(self):
        """
        Configuring nothing is not the same as configuring permissiveness. With no
        vocabulary, no code has been defined by anybody, and saying so is the honest answer.
        """
        problems = GovernanceVocabulary.empty().problems_with(
            {"Term": "x", "Protection Class": "TARIFF"}
        )
        assert problems and "no vocabulary is configured" in problems[0]

    def test_the_catalog_wins_on_the_personal_information_flag(self, vocabulary):
        problems = vocabulary.problems_with(
            {"Term": "x", "Protection Class": "USAGE", "Personal Data": "N"}
        )
        assert len(problems) == 1
        assert "personal_information" in problems[0]
        assert "the catalog wins" in problems[0]

    def test_the_catalog_wins_on_the_direct_identifier_flag(self, vocabulary):
        problems = vocabulary.problems_with(
            {"Term": "x", "Protection Class": "USAGE", "Direct Identifier": "true"}
        )
        assert len(problems) == 1
        assert "direct_identifier" in problems[0]

    def test_every_disagreement_in_one_row_is_reported_together(self, vocabulary):
        """
        Reporting the first and stopping makes fixing a glossary an N-round game against
        a tool that reveals one problem per run.
        """
        problems = vocabulary.problems_with(
            {
                "Term": "x",
                "Protection Class": "TARIFF",
                "Classification": "Sealed",
                "Personal Data": "Y",
                "Direct Identifier": "Y",
            }
        )
        assert len(problems) == 3

    def test_a_blank_cell_is_not_a_claim(self, vocabulary):
        """
        Half-filled glossaries are the norm. Treating a blank as False would bury the real
        contradictions under noise from every sparsely-populated source.
        """
        assert (
            vocabulary.problems_with(
                {
                    "Term": "x",
                    "Protection Class": "USAGE",
                    "Classification": "",
                    "Personal Data": "",
                    "Direct Identifier": None,
                }
            )
            == []
        )

    def test_a_row_with_no_code_column_is_not_this_check_s_business(self, vocabulary):
        """
        Plenty of glossaries classify in prose and have adopted no controlled vocabulary.
        Refusing them would make this feature mandatory, which it is not.
        """
        assert vocabulary.problems_with({"Term": "x", "Classification": "Whatever"}) == []

    def test_a_dropped_junk_token_is_not_a_defect(self, vocabulary):
        """The caller DECLARED that "n/a" is noise, so it is handled, not a surprise."""
        assert vocabulary.problems_with({"Term": "x", "Protection Class": "n/a"}) == []

    def test_a_legacy_token_is_validated_against_the_code_it_maps_to(self, vocabulary):
        """The alias resolves first, so a row using an old spelling is held to the same tier."""
        assert (
            vocabulary.problems_with(
                {"Term": "x", "Protection Class": "MTR#", "Classification": "Sealed"}
            )
            == []
        )
        assert vocabulary.problems_with(
            {"Term": "x", "Protection Class": "MTR#", "Classification": "Guarded"}
        )
