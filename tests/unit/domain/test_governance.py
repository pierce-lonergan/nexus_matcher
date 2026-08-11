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
import unicodedata
from pathlib import Path

import pytest

from nexus_matcher.domain.governance import (
    OPEN_CLASSIFICATION,
    GovernanceVocabulary,
    ProtectionClass,
    _governance_columns,
)

REPO = Path(__file__).resolve().parents[3]

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


# =============================================================================
# THE CATALOG REFUSES TO GUESS AT A STRING
# =============================================================================


def _one_class(**overrides):
    """A complete, valid class, plus whatever the caller wants to break."""
    return [
        {
            "code": "TARIFF",
            "name": "Tariff Band Code",
            "classification": "Guarded",
            "personal_information": False,
            "direct_identifier": False,
            **overrides,
        }
    ]


class TestTheCatalogRefusesToGuessAtAString:
    """
    `str()` coercion applied BEFORE the emptiness test, on the two fields that carry the
    classification.

    `str(None)` is `"None"`, which is not empty, so a `"code": null` produced the code
    `'None'` -- which then matched every glossary cell spelling "none" or "NONE", handing a
    field asserting it has NO code the whole class. A `"classification": null` derived the
    literal tier `'None'`, which shipped over the wire as a tier nobody defined AND turned
    every honest row stating its real tier into a self-contradiction that refused the load,
    blaming the glossary for a defect in the catalog.

    `_required_bool` has refused to guess since it was written. These two fields matter
    more and had no such check.
    """

    def test_a_null_code_is_refused_rather_than_becoming_the_string_None(self):
        with pytest.raises(ValueError, match="'code' of null"):
            GovernanceVocabulary.from_json(_one_class(code=None))

    def test_a_null_classification_is_refused_rather_than_deriving_the_tier_None(self):
        with pytest.raises(ValueError, match="'classification' of null"):
            GovernanceVocabulary.from_json(_one_class(classification=None))

    @pytest.mark.parametrize(
        "bad", [{"a": 1}, ["A", "B"], 7, 1.5], ids=["dict", "list", "int", "float"]
    )
    def test_a_code_that_is_not_a_string_is_refused_rather_than_stringified(self, bad):
        """`str({'a': 1})` is a perfectly good dict key and a nonsense protection code."""
        with pytest.raises(ValueError, match="must be a string"):
            GovernanceVocabulary.from_json(_one_class(code=bad))

    @pytest.mark.parametrize("bad", [{"a": 1}, ["A", "B"], 7], ids=["dict", "list", "int"])
    def test_a_classification_that_is_not_a_string_is_refused(self, bad):
        with pytest.raises(ValueError, match="must be a string"):
            GovernanceVocabulary.from_json(_one_class(classification=bad))

    @pytest.mark.parametrize("absent", [None, {}, []], ids=["null", "dict", "list"])
    def test_nothing_that_is_not_a_string_can_become_the_code_None(self, absent):
        """
        The symptom stated as the property, over the inputs `str()` used to launder. `'None'`
        is the spelling a glossary cell meaning "no code at all" is most likely to use, so a
        class carrying it inherits every such row.

        Note what this does NOT claim. A caller who literally writes `"code": "None"` has
        declared a class called None, and the library takes the caller's vocabulary as
        given. The defect was inventing that token from something that was not one.
        """
        try:
            vocabulary = GovernanceVocabulary.from_json(_one_class(code=absent))
        except ValueError:
            return
        assert "None" not in vocabulary.codes, f"code={absent!r} produced a class called 'None'"

    def test_a_null_name_falls_back_to_the_code_rather_than_the_string_None(self):
        """`name` is optional; null means "unnamed", not a class named None."""
        assert GovernanceVocabulary.from_json(_one_class(name=None)).get("TARIFF").name == "TARIFF"

    def test_a_name_that_is_not_a_string_is_refused(self):
        with pytest.raises(ValueError, match="must be a string or null"):
            GovernanceVocabulary.from_json(_one_class(name={"en": "Tariff"}))

    def test_a_non_string_enhancement_is_refused_rather_than_stringified(self):
        """`enhancement` is passed through to the caller untouched, so a dict would ship a
        Python repr into whatever consumes it."""
        with pytest.raises(ValueError, match="must be a string or null"):
            GovernanceVocabulary.from_json(_one_class(enhancement={"mask": True}))

    def test_a_null_alias_token_on_a_class_is_refused(self):
        """`str(None)` was becoming the alias "NONE", pointing at a real class."""
        with pytest.raises(ValueError, match="alias"):
            GovernanceVocabulary.from_json(_one_class(aliases=[None]))

    def test_a_non_string_open_classification_is_refused(self):
        with pytest.raises(ValueError, match="must be a string or null"):
            GovernanceVocabulary.from_json(
                {"open_classification": {"tier": "Open"}, "classes": _one_class()}
            )

    def test_a_null_open_classification_still_means_use_the_sentinel(self):
        """The existing `or OPEN_CLASSIFICATION` already handled null correctly."""
        loaded = GovernanceVocabulary.from_json(
            {"open_classification": None, "classes": _one_class()}
        )
        assert loaded.classification_for(None) == OPEN_CLASSIFICATION

    def test_a_non_string_alias_target_is_refused(self):
        with pytest.raises(ValueError, match="must be a declared code or null"):
            GovernanceVocabulary.from_json({"aliases": {"OLD": 7}, "classes": _one_class()})

    def test_a_null_alias_target_is_still_how_a_token_is_declared_droppable(self):
        """The load-bearing half: null stays meaningful in the top-level alias map."""
        loaded = GovernanceVocabulary.from_json({"aliases": {"n/a": None}, "classes": _one_class()})
        assert loaded.get("n/a") is None
        assert loaded.problems_with({"Term": "x", "Protection Class": "n/a"}) == []

    def test_the_constructor_closes_the_same_door_as_the_loader(self):
        """
        `from_json` gives the friendly per-index message; `__init__` is the invariant
        nothing walks past. It accepted `ProtectionClass(code=None)` because
        `_norm_code(None)` is 'NONE' and the empty-code check therefore never fired.
        """
        for bad in (
            ProtectionClass(None, "x", "Guarded", False, False),
            ProtectionClass("TARIFF", "x", None, False, False),
        ):
            with pytest.raises(ValueError, match="non-blank string"):
                GovernanceVocabulary(classes=[bad])


class TestTheShippedExamplePackStillLoads:
    """
    The regression that keeps the two helpers from becoming one.

    A single "no nulls anywhere" rule, applied to the field list this fix started from,
    REJECTS this repo's own example pack: five of its nine classes declare
    `"enhancement": null`, and its alias map declares `"n/a": null` and `"tbc": null`.
    Both are documented features, and both are how a caller says "nothing here" out loud.
    """

    PACK = REPO / "examples" / "governance" / "protection_classes.json"

    def test_the_pack_loads(self):
        assert len(GovernanceVocabulary.from_json(self.PACK).codes) == 9

    def test_a_null_enhancement_survives_as_None(self):
        loaded = GovernanceVocabulary.from_json(self.PACK)
        assert loaded.get("CREW_ROSTER").enhancement is None
        assert loaded.get("MANIFEST_NAME").enhancement == "MASK_IN_LOGS"

    def test_a_null_alias_target_survives_as_a_dropped_token(self):
        loaded = GovernanceVocabulary.from_json(self.PACK)
        assert loaded.get("n/a") is None
        assert loaded.get("tbc") is None
        assert loaded.get("GBF-LEGACY-NAME").code == "MANIFEST_NAME"

    def test_the_packs_declared_tier_ordering_is_read_and_kept_in_its_declared_order(self):
        """
        The pack has always declared `tiers_most_open_first`, and it was read by NOTHING --
        one occurrence in the whole repository, the declaration itself. An adopter copies
        this file to start their own vocabulary, so the no-op propagates with it.
        """
        assert GovernanceVocabulary.from_json(self.PACK).tiers_most_open_first == (
            "OPEN_DECK",
            "CREW_ONLY",
            "BRIDGE_SENSITIVE",
            "SEALED_RESTRICTED",
        )


# =============================================================================
# THE DECLARED TIER ORDERING
# =============================================================================


def _ordered(**overrides) -> dict:
    """A two-tier vocabulary, so a test can vary one thing about the ordering."""
    document = {
        "open_classification": "Open",
        "tiers_most_open_first": ["Open", "Sealed"],
        "classes": [
            {
                "code": "METERID",
                "classification": "Sealed",
                "personal_information": True,
                "direct_identifier": True,
            }
        ],
    }
    document.update(overrides)
    return document


class TestTheDeclaredTierOrderingIsCheckedAgainstTheClasses:
    """
    The caller's own ladder, checked against the caller's own classes.

    Enforcing this is not the library forming an opinion about a taxonomy: the list is
    theirs, the tiers in it are theirs, and the only comparison made is their file against
    itself. What is refused is a file that DISAGREES WITH ITSELF -- an ordering that cannot
    place a tier the vocabulary actually derives, which is a ladder with a missing rung
    rather than a ranking anyone here chose.

    The alternative considered was deleting the key. Enforcing is the smaller change and
    keeps something real the caller can say; deleting throws away the only thing in the
    file that can rank two classifications.
    """

    def test_a_vocabulary_that_declares_no_ordering_still_loads_and_says_so(self):
        """
        OPTIONAL, and empty is a real answer. Every vocabulary in this file and every one
        in the test suite predates the key; none of them may start failing, and none of
        them may acquire an ordering nobody declared.
        """
        loaded = GovernanceVocabulary.from_json(THORNBURY)
        assert loaded.tiers_most_open_first == ()
        assert loaded.codes

    def test_a_tier_a_class_derives_but_the_ordering_omits_refuses_the_load(self):
        """
        The message must name BOTH sides. A refusal that says only "inconsistent" sends
        the reader back to two parts of one file to work out which half is wrong -- and
        the answer is genuinely either, since the library has no view on which.
        """
        with pytest.raises(ValueError) as raised:
            GovernanceVocabulary.from_json(_ordered(tiers_most_open_first=["Open", "Guarded"]))

        message = str(raised.value)
        assert "METERID" in message and "'Sealed'" in message, message
        assert "'Open', 'Guarded'" in message, message

    def test_the_declared_open_tier_must_be_placeable_too(self):
        """
        The open tier is where every UNCODED field sits -- typically the most common answer
        the vocabulary gives, and the meaning of a null on the wire. An ordering that
        cannot place it cannot rank the majority of a schema.
        """
        with pytest.raises(ValueError, match="open_classification"):
            GovernanceVocabulary.from_json(
                _ordered(open_classification="Public", tiers_most_open_first=["Open", "Sealed"])
            )

    def test_the_sentinel_open_tier_is_exempt_because_the_caller_never_wrote_it(self):
        """
        `UNCLASSIFIED` is what an unset `open_classification` DEFAULTS to. Requiring it in
        the caller's list would refuse a file over a value this library supplied -- our own
        default failing our own check.
        """
        loaded = GovernanceVocabulary.from_json(
            {
                "tiers_most_open_first": ["Open", "Sealed"],
                "classes": _ordered()["classes"],
            }
        )
        assert loaded.classification_for(None) == OPEN_CLASSIFICATION
        assert loaded.tiers_most_open_first == ("Open", "Sealed")

    def test_a_declared_tier_no_class_uses_is_legal(self):
        """
        An organisation's ladder is allowed rungs this vocabulary does not reach. Refusing
        would make the key unusable for the thing it is for -- declaring a policy, not
        enumerating today's classes.
        """
        loaded = GovernanceVocabulary.from_json(
            _ordered(tiers_most_open_first=["Open", "Guarded", "Sealed", "Sealed Plus"])
        )
        assert loaded.tiers_most_open_first[-1] == "Sealed Plus"

    def test_one_tier_declared_twice_refuses_because_an_order_needs_one_position(self):
        with pytest.raises(ValueError, match="same tier"):
            GovernanceVocabulary.from_json(
                _ordered(tiers_most_open_first=["Open", "Sealed", "Open"])
            )

    def test_two_spellings_of_one_tier_are_the_same_tier_here_too(self):
        """
        Compared through `_norm_tier`, the same normalisation a glossary row is compared
        through. A ladder where "Sealed " and "sealed" were two rungs would rank a tier
        against itself; and the class below must still find its rung despite the padding.
        """
        with pytest.raises(ValueError, match="same tier"):
            GovernanceVocabulary.from_json(
                _ordered(tiers_most_open_first=["Open", "Sealed", "sealed "])
            )

        loaded = GovernanceVocabulary.from_json(
            _ordered(tiers_most_open_first=["Open", "  SEALED  "])
        )
        assert loaded.tiers_most_open_first == ("Open", "  SEALED  ")

    def test_an_ordering_given_as_a_bare_string_is_refused(self):
        """
        A string is iterable, so it would be read one letter per rung -- the same defect
        that read `"aliases": "LEGACY-METER"` as seven single-character aliases.
        """
        with pytest.raises(ValueError, match="list of tokens"):
            GovernanceVocabulary.from_json(_ordered(tiers_most_open_first="Open"))

    def test_a_null_inside_the_ordering_is_refused_rather_than_stringified(self):
        with pytest.raises(ValueError, match="non-blank string"):
            GovernanceVocabulary.from_json(_ordered(tiers_most_open_first=["Open", None]))

    def test_the_key_is_reserved_in_the_code_keyed_mapping_shape(self):
        """
        A regression for a refusal that was real before the key was read at all: the
        code-keyed `{code: {...}}` shape is recognised only when EVERY unreserved value is
        an object, so a document declaring the ordering held one list value and the whole
        file came back "no protection classes found" -- a message pointing at the classes,
        which were fine.
        """
        loaded = GovernanceVocabulary.from_json(
            {
                "open_classification": "Open",
                "tiers_most_open_first": ["Open", "Sealed"],
                "METERID": {
                    "classification": "Sealed",
                    "personal_information": True,
                    "direct_identifier": True,
                },
            }
        )
        assert loaded.get("METERID").classification == "Sealed"
        assert loaded.tiers_most_open_first == ("Open", "Sealed")


# =============================================================================
# ONE TOKEN, ONE MEANING
# =============================================================================


class TestDuplicateAliasesAreRefused:
    """
    A token declared twice used to resolve positionally, to whichever declaration came
    last, and class-level aliases were applied after the top-level map -- so the file's
    most explicit statement was the one that lost.

    The direction that matters is the SILENT DOWNGRADE: a restricted class and an open
    class both claiming one legacy spelling, and the field loading as the open one with no
    problems reported and a strict load not refused. That is NM-0005's harm -- a field
    losing the classification it should have inherited -- reached through the catalog
    instead of through the row.

    Note what an `__init__`-only duplicate check cannot see. `from_json` used to accumulate
    into a dict keyed by the RAW token, so byte-identical spellings collapsed before the
    constructor ever ran -- and byte-identical is the likeliest duplicate of all, a
    maintainer copying a token verbatim from one part of the file to the other. The
    byte-identical cases below are the ones measured to survive that fix.
    """

    @staticmethod
    def _two_classes(sealed_alias, open_alias):
        return {
            "classes": [
                {
                    "code": "SEALEDTHING",
                    "name": "Sealed",
                    "classification": "Sealed",
                    "personal_information": True,
                    "direct_identifier": True,
                    "aliases": [sealed_alias],
                },
                {
                    "code": "OPENTHING",
                    "name": "Open",
                    "classification": "Open",
                    "personal_information": False,
                    "direct_identifier": False,
                    "aliases": [open_alias],
                },
            ]
        }

    def test_two_classes_cannot_claim_one_alias_byte_identically(self):
        """The spelling an `__init__`-only fix misses: the dict collapsed these."""
        with pytest.raises(ValueError, match="declared twice"):
            GovernanceVocabulary.from_json(self._two_classes("LEGACY-NAME", "LEGACY-NAME"))

    def test_two_classes_cannot_claim_one_alias_in_two_spellings(self):
        with pytest.raises(ValueError, match="declared twice"):
            GovernanceVocabulary.from_json(self._two_classes("LEGACY_NAME", "legacy-name"))

    def test_the_message_names_the_token_and_both_targets(self):
        """
        A normalised key is not searchable in the caller's own file, and "duplicate alias"
        without the two targets does not tell the reader which declaration to delete.
        """
        with pytest.raises(ValueError) as excinfo:
            GovernanceVocabulary.from_json(self._two_classes("LEGACY-NAME", "LEGACY-NAME"))

        message = str(excinfo.value)
        assert "LEGACY-NAME" in message, f"the token is not named as spelled: {message!r}"
        assert "SEALEDTHING" in message and "OPENTHING" in message, (
            f"the message must name both targets: {message!r}"
        )

    def test_a_droppable_token_cannot_be_quietly_promoted_to_a_class(self):
        """
        The module docstring's promise: "we quietly dropped something" is not a thing that
        can happen unnoticed. The inverse -- we quietly UNdropped something -- was.
        """
        with pytest.raises(ValueError, match="declared twice"):
            GovernanceVocabulary.from_json(
                {
                    "aliases": {"n/a": None},
                    "classes": [
                        {
                            "code": "SEALEDTHING",
                            "name": "Sealed",
                            "classification": "Sealed",
                            "personal_information": True,
                            "direct_identifier": True,
                            "aliases": ["n/a"],
                        }
                    ],
                }
            )

    def test_a_top_level_alias_cannot_be_silently_overridden_by_a_class(self):
        """Class aliases are applied last, so the more explicit declaration lost."""
        with pytest.raises(ValueError, match="declared twice"):
            GovernanceVocabulary.from_json(
                {
                    "aliases": {"MTR#": "OPENTHING"},
                    "classes": [
                        {
                            "code": "OPENTHING",
                            "name": "Open",
                            "classification": "Open",
                            "personal_information": False,
                            "direct_identifier": False,
                        },
                        {
                            "code": "SEALEDTHING",
                            "name": "Sealed",
                            "classification": "Sealed",
                            "personal_information": True,
                            "direct_identifier": True,
                            "aliases": ["mtr"],
                        },
                    ],
                }
            )

    def test_declaring_the_same_alias_twice_at_the_same_target_is_fine(self):
        """
        Guards the vacuous pass. A refusal that fired on every repeat would make a
        vocabulary unmaintainable: restating a mapping is not a conflict, and a rule that
        cannot tell the two apart gets worked around by deleting the restatement that
        documented the intent.
        """
        loaded = GovernanceVocabulary.from_json(
            {
                "aliases": {"MTR#": "OPENTHING"},
                "classes": [
                    {
                        "code": "OPENTHING",
                        "name": "Open",
                        "classification": "Open",
                        "personal_information": False,
                        "direct_identifier": False,
                        "aliases": ["mtr#"],
                    }
                ],
            }
        )
        assert loaded.get("MTR#").code == "OPENTHING"

    def test_a_class_alias_given_as_a_bare_string_is_refused(self):
        """
        `"aliases": "LEGACY-TARIFF"` is iterable, so the loop walked it CHARACTER BY
        CHARACTER and declared L, E, G, A, C, Y, T, I, R and F as aliases of the class.
        Single-character aliases match nothing a glossary carries, so the spelling the
        caller was trying to declare stays unknown and every row using it is refused, for a
        reason that appears nowhere in their file.
        """
        with pytest.raises(ValueError, match="list of tokens"):
            GovernanceVocabulary.from_json(_one_class(aliases="LEGACY-TARIFF"))


# =============================================================================
# ONE CODE, ONE UNICODE SPELLING
# =============================================================================


class TestUnicodeSpellingsOfOneCode:
    """
    `_norm_code` strips non-alphanumerics, and a combining accent is not alphanumeric. So
    the composed spelling of an accented code (what an editor types) and the decomposed one
    (what macOS's filesystem and several exporters emit) normalised to DIFFERENT codes --
    with no visible difference between them anywhere a human looks.
    """

    NFC = unicodedata.normalize("NFC", "MÉTÉRID")
    NFD = unicodedata.normalize("NFD", "MÉTÉRID")

    def test_the_two_spellings_really_are_different_bytes(self):
        """Guards the vacuous pass: if these were equal the tests below prove nothing."""
        assert self.NFC != self.NFD

    def test_one_code_written_two_ways_is_a_duplicate_not_two_classes(self):
        """
        The catalog used to load both, with two different tiers, and `codes` then showed
        the same word twice. Which tier a row inherited depended on which byte sequence its
        exporter happened to emit.
        """
        with pytest.raises(ValueError, match="duplicate protection code"):
            GovernanceVocabulary.from_json(
                [
                    {
                        "code": self.NFC,
                        "name": "a",
                        "classification": "Sealed",
                        "personal_information": True,
                        "direct_identifier": True,
                    },
                    {
                        "code": self.NFD,
                        "name": "b",
                        "classification": "Open",
                        "personal_information": False,
                        "direct_identifier": False,
                    },
                ]
            )

    def test_a_row_reaches_its_class_whichever_spelling_it_carries(self):
        loaded = GovernanceVocabulary.from_json(
            [
                {
                    "code": self.NFC,
                    "name": "a",
                    "classification": "Sealed",
                    "personal_information": True,
                    "direct_identifier": True,
                }
            ]
        )
        assert loaded.get(self.NFD) is not None
        assert loaded.classification_for(self.NFD) == "Sealed"

    def test_two_spellings_of_one_tier_are_not_a_contradiction(self):
        """
        `casefold()` does not normalise either, so the row and the catalog spelling one
        tier two ways would be reported as a row contradicting itself -- and under the
        default `governance_strict` that REFUSES the whole glossary over two words that
        render identically.
        """
        loaded = GovernanceVocabulary.from_json(
            [
                {
                    "code": "TARIFF",
                    "name": "a",
                    "classification": unicodedata.normalize("NFC", "Sealéd"),
                    "personal_information": False,
                    "direct_identifier": False,
                }
            ]
        )
        assert (
            loaded.problems_with(
                {
                    "Term": "x",
                    "Protection Class": "TARIFF",
                    "Classification": unicodedata.normalize("NFD", "Sealéd"),
                }
            )
            == []
        )


# =============================================================================
# THE PER-ROW COST OF VALIDATION
# =============================================================================

# The three governance columns the rows below carry, so the equivalence test can build the
# filtered key without reaching into the module's own private frozenset to do it.
_GOVERNANCE_COLUMN_NAMES = {"Protection Class", "Classification", "Personal Data"}


class TestTheColumnCacheIsKeyedOnTheGovernanceColumns:
    """
    The cache key was the row's WHOLE key tuple. CSV and Excel hand every row the same
    shape so it worked; JSON, JSONL and iterable-of-dicts sources hand each row back
    verbatim, so a sparse exporter yields one cache entry per combination of present
    columns. Measured on a 30k-row glossary: 64 shapes cost 53.6 ms, 65 shapes cost
    247.5 ms with ZERO hits -- the LRU is 64 wide, so past the cliff every lookup evicts
    the entry it is about to need.
    """

    def test_noise_columns_do_not_multiply_cache_entries(self, vocabulary):
        _governance_columns.cache_clear()

        for i in range(200):
            vocabulary.problems_with(
                {"Term": "x", "Protection Class": "USAGE", f"optional_field_{i}": "value"}
            )

        misses = _governance_columns.cache_info().misses
        assert misses == 1, (
            f"200 row shapes differing only in columns governance cannot read produced "
            f"{misses} cache misses; the key is still the whole row"
        )

    @pytest.mark.parametrize(
        "noise",
        [
            {},
            {"Owner": "a", "System": "b"},
            {"Classification Code": "USAGE"},
            {"Notes": None, "Steward": "", "Reviewed": "2026-01-01"},
        ],
    )
    def test_filtering_does_not_change_a_single_answer(self, vocabulary, noise):
        """
        The filtered key must resolve exactly what the whole-row key resolved. A row's
        governance columns are the only ones `_first` can ever select, so dropping the
        others is information-free -- pinned here rather than argued.
        """
        row = {
            "Term": "x",
            "Protection Class": "USAGE",
            "Classification": "Open",
            "Personal Data": "N",
            **noise,
        }

        assert _governance_columns(tuple(str(k) for k in row)) == _governance_columns(
            tuple(str(k) for k in row if k in _GOVERNANCE_COLUMN_NAMES)
        )
        assert len(vocabulary.problems_with(row)) == 2

    def test_a_row_whose_columns_are_reordered_is_the_same_shape(self, vocabulary):
        """JSON sources preserve each object's own key order, and it varies between rows."""
        _governance_columns.cache_clear()

        vocabulary.problems_with({"Protection Class": "USAGE", "Term": "x"})
        vocabulary.problems_with({"Term": "x", "Protection Class": "USAGE"})

        assert _governance_columns.cache_info().misses == 1


class TestTheUnknownCodeMessageDoesNotCarryTheVocabulary:
    """
    `", ".join(sorted(self.codes))` sat inside the PER-ROW unknown-code branch, producing
    an identical message for every defective row. Measured: the valid-code path is flat at
    34 ms across 9 to 800 classes; the unknown-code path went 60 ms to 1802 ms. Worse, under
    `governance_strict=False` -- the escape hatch the refusal message itself recommends --
    all 30,000 of those strings are RETAINED in `source_metadata`, holding one distinct
    value in 218 MB.
    """

    @staticmethod
    def _vocabulary(size):
        return GovernanceVocabulary.from_json(
            [
                {
                    "code": f"BAND{n:04d}",
                    "name": f"Band {n}",
                    "classification": "Guarded",
                    "personal_information": False,
                    "direct_identifier": False,
                }
                for n in range(size)
            ]
        )

    def test_the_message_names_the_token_and_the_count_not_the_list(self):
        problems = self._vocabulary(400).problems_with(
            {"Term": "x", "Protection Class": "SUPERSECRET"}
        )

        assert len(problems) == 1
        assert "SUPERSECRET" in problems[0]
        assert "400" in problems[0], (
            f"the message must say how big the vocabulary is: {problems[0]!r}"
        )
        assert "BAND0399" not in problems[0], (
            f"the whole vocabulary is still inside a per-row message: {problems[0]!r}"
        )

    def test_the_message_does_not_grow_with_the_vocabulary(self):
        """The property, not a byte count: nine classes and eight hundred cost the same."""
        small = self._vocabulary(9).problems_with({"Term": "x", "Protection Class": "NOPE"})
        large = self._vocabulary(800).problems_with({"Term": "x", "Protection Class": "NOPE"})

        assert len(large[0]) - len(small[0]) <= 2, (
            f"the message grew by {len(large[0]) - len(small[0])} characters between a "
            f"9-class and an 800-class vocabulary"
        )

    def test_the_empty_vocabulary_still_says_so_in_the_words_two_tests_assert(self):
        """`no vocabulary is configured` is asserted by name here and in
        tests/unit/application/test_governance_ingest.py; it is not free to reword."""
        problems = GovernanceVocabulary.empty().problems_with(
            {"Term": "x", "Protection Class": "TARIFF"}
        )
        assert problems and "no vocabulary is configured" in problems[0]
