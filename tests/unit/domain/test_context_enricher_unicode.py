"""
tests.unit.domain.test_context_enricher_unicode | Layer: TEST
Tests: ContextEnricher._humanize | Target: src/nexus_matcher/domain/services/context_enricher.py

`_humanize` used to identify separators by exclusion -- `[^0-9A-Za-z]+` -- which made
every character outside ASCII alphanumerics a separator. A field named in any non-Latin
script was therefore deleted from its own query. Reproduced before the fix:

    'cafe' with an acute accent  ->  'caf'      (accented letter dropped)
    two CJK characters           ->  ''         (the WHOLE name became empty)
    Cyrillic + '_id'             ->  'id'       (only the ASCII suffix survived)

An empty name is not a degraded query, it is a different query: the field then retrieves
on its parent-path context alone, or on nothing. No benchmark in this repo can catch it,
because both committed corpora are ASCII English -- which is exactly why it needs a test.

The second half of this file is the safety half. `_humanize` feeds parent-path context,
the largest single accuracy factor in the pipeline, so the fix has to be provably inert
on ASCII. `test_ascii_behaviour_is_identical_to_the_previous_implementation` brute-forces
every string over a mixed alphabet against the pre-fix implementation kept verbatim
below. If someone later "simplifies" the splitter, that test is what fails.
"""

from __future__ import annotations

import itertools
import re

import pytest

from nexus_matcher.domain.services.context_enricher import ContextEnricher


def _previous_implementation(name: str) -> str:
    """
    The pre-fix `_humanize`, verbatim, as an equivalence oracle for ASCII.

    Kept as a literal copy rather than imported from history so the ASCII contract is
    pinned by something that cannot drift when the real implementation changes.
    """
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    name = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", name)
    name = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", name)
    name = re.sub(r"[^0-9A-Za-z]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


@pytest.fixture
def humanize():
    return ContextEnricher()._humanize


class TestNonAsciiNamesSurvive:
    """A field named outside ASCII must still be present in its own query."""

    def test_accented_latin_letter_is_kept(self, humanize):
        assert humanize("café_id") == "café id"
        # Regression form: the accented letter used to be dropped outright.
        assert _previous_implementation("café_id") == "caf id"

    def test_cjk_name_does_not_become_the_empty_string(self, humanize):
        name = "金額"  # two CJK ideographs
        assert humanize(name) == name
        assert _previous_implementation(name) == ""

    def test_cyrillic_name_is_not_reduced_to_its_ascii_suffix(self, humanize):
        name = "сумма_id"
        assert humanize(name) == "сумма id"
        assert _previous_implementation(name) == "id"

    def test_non_ascii_digits_are_kept(self, humanize):
        # Arabic-Indic digits are `\d` but were not `[0-9]`.
        assert humanize("٣٤_code") == "٣٤ code"

    def test_a_named_field_is_never_erased(self, humanize):
        """The property that actually matters: a non-empty name yields a non-empty query."""
        for name in (
            "金額",
            "сумма",
            "ποσό",
            "café",
            "naïve",
            "金額_合計",
        ):
            assert humanize(name) != "", f"{name!r} was erased"


class TestCaseBoundariesInScriptsThatHaveCase:
    """camelCase splitting is not an ASCII privilege."""

    def test_cyrillic_camel_case_splits(self, humanize):
        # Before the fix this returned '' -- no split, then every letter stripped.
        assert humanize("суммаПлатежа") == "сумма платежа"

    def test_greek_camel_case_splits(self, humanize):
        assert humanize("ποσόΠληρωμής") == "ποσό πληρωμής"

    def test_uncased_script_yields_one_token(self, humanize):
        """CJK has no case, so there is no boundary to find -- and nothing to delete."""
        assert humanize("金額合計") == "金額合計"


class TestExistingAsciiBehaviourIsPinned:
    """The documented ASCII examples, spelled out so a regression names itself."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("first_name", "first name"),
            ("firstName", "first name"),
            ("HTTPResponse", "http response"),
            ("enroll12", "enroll 12"),
            ("K12Grade", "k 12 grade"),
            ("FRPM Count (K-12)", "frpm count k 12"),
            ("enrollment (1st-12nd grade)", "enrollment 1 st 12 nd grade"),
            ("customer-account-balance", "customer account balance"),
            ("", ""),
            ("___", ""),
        ],
    )
    def test_ascii_examples(self, humanize, raw, expected):
        assert humanize(raw) == expected

    def test_underscore_still_separates(self, humanize):
        """`\\w` counts '_' as a word character; identifiers do not. Easy fix to get wrong."""
        assert humanize("txn_amt") == "txn amt"
        assert "_" not in humanize("a_b_c")


class TestAsciiEquivalenceWithPreviousImplementation:
    """
    The safety proof. Every ASCII string over a mixed alphabet must humanize identically
    to the pre-fix implementation -- the fix is meant to add scripts, not change English.
    """

    def test_ascii_behaviour_is_identical_to_the_previous_implementation(self, humanize):
        # lower, upper, upper, lower, digit, and four separator kinds -- enough to reach
        # every branch: camelCase seams, acronym/word seams, letter/digit boundaries in
        # both directions, and separator runs.
        alphabet = "aBCd1_-( "
        checked = 0
        for length in range(1, 6):
            for chars in itertools.product(alphabet, repeat=length):
                candidate = "".join(chars)
                assert humanize(candidate) == _previous_implementation(candidate), (
                    f"ASCII divergence on {candidate!r}"
                )
                checked += 1
        assert checked == sum(len(alphabet) ** n for n in range(1, 6))

    def test_real_corpus_identifier_shapes_are_unchanged(self, humanize):
        """Shapes drawn from the committed corpora, including the awkward one."""
        for name in (
            "County Name",
            "frpm.County Name",
            "Account.billingStatus",
            "activityDefinition",
            "enrollment (1st-12nd grade)",
            "AcademicYear",
            "NumTstTakr",
            "person_source_value",
            "care_site_id",
        ):
            assert humanize(name) == _previous_implementation(name)
