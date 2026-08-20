"""
tests.unit.domain.test_review_evidence | Layer: TEST
The two evidence passes: the contrast between rank 1 and rank 2, and the cross-field
consistency check.

## Relationships
# TESTS → domain/services/review_evidence :: grouping, agreement, and the contrast arithmetic

## What is actually load-bearing here

Both passes are READ-ONLY, so the usual worry -- did it change an answer -- does not
apply. The worry that replaces it is that they say something that is not true:

  A CAUSE THAT IS NOT VISIBLE. A signal whose two scores differ below the resolution of
  the published numbers must never be named. `TestResolution` is that assertion, and it
  matters because a review surface that produces reasons which are not reasons is worse
  than one that produces none -- the reviewer stops reading it, and then it protects
  nothing.

  A DISAGREEMENT THAT WAS NEVER THERE. A column with no answer is silent, not
  dissenting. Counting its blank would report a contradiction in a group where one
  column was answered, which is the noise a report like this dies of.

  AN ARITHMETIC THAT DOES NOT CLOSE. The per-signal weighted differences must sum to the
  gap between the two confidences. Checked here on the domain object and again at the
  wire, where the response is refused rather than sent.

The GROUPING's precision and recall are measured separately, against a fixture whose
answers are known by construction: `test_review_evidence_grouping.py`.
"""

from __future__ import annotations

import pytest

from nexus_matcher.domain.models.entities import DictionaryEntry, MatchResult, SchemaField
from nexus_matcher.domain.services.review_evidence import (
    ARRAY_BOUNDARY,
    Agreement,
    ConceptGroup,
    GroupingPolicy,
    Separation,
    SignalSpec,
    assess_consistency,
    concept_of,
    contrast_top_two,
    group_by_concept,
    normalise_tokens,
    path_segments,
    result_key,
)
from nexus_matcher.shared.types.base import (
    DataType,
    MatchDecision,
    PerformanceMetrics,
    ProtectionLevel,
    ScoreBreakdown,
)

# The five weighted signals as the wire names them, paired with the breakdown attribute
# each reads. Written out rather than imported from the presentation layer: this module
# takes the naming from its caller, and a test that borrowed the caller's table would be
# checking that one table equals itself.
SIGNALS = (
    SignalSpec("fusedRetrieval", "fused_retrieval_score", 0.70),
    SignalSpec("lexical", "lexical_score", 0.05),
    SignalSpec("editDistance", "edit_distance_score", 0.05),
    SignalSpec("type", "type_compatibility_score", 0.05),
    SignalSpec("domain", "domain_score", 0.15),
)


def field(path: str, name: str = "", data_type: DataType = DataType.STRING) -> SchemaField:
    """A field the way the HTTP boundary builds one: the caller's path is the address."""
    parent, _, leaf = path.rpartition(".")
    return SchemaField(
        name=name or leaf or path,
        data_type=data_type,
        full_path=path,
        parent_path=parent,
        source_metadata={"flattened_name": path},
    )


def entry(entry_id: str, code: str | None = "CODE-A", domain: str = "ALPHA") -> DictionaryEntry:
    return DictionaryEntry(
        id=entry_id,
        business_name=f"Term {entry_id}",
        logical_name=entry_id.lower(),
        definition="A term.",
        data_type=DataType.STRING,
        protection_level=ProtectionLevel.INTERNAL,
        governance_code=code,
        domain=domain,
    )


def match(
    scores: dict[str, float],
    rank: int = 1,
    entry_id: str = "T-1",
    code: str | None = "CODE-A",
    domain: str = "ALPHA",
) -> MatchResult:
    """A real MatchResult with hand-chosen components, so the arithmetic is stated."""
    weights = {spec.name: spec.weight for spec in SIGNALS}
    confidence = min(max(sum(scores[k] * weights[k] for k in weights), 0.0), 1.0)
    return MatchResult(
        schema_field=field("record.column"),
        dictionary_entry=entry(entry_id, code, domain),
        rank=rank,
        final_confidence=confidence,
        score_breakdown=ScoreBreakdown(
            fused_retrieval_score=scores["fusedRetrieval"],
            lexical_score=scores["lexical"],
            edit_distance_score=scores["editDistance"],
            type_compatibility_score=scores["type"],
            domain_score=scores["domain"],
        ),
        decision=MatchDecision.REVIEW,
        performance=PerformanceMetrics(latency_ms=1.0),
    )


BASE = {
    "fusedRetrieval": 0.9,
    "lexical": 0.5,
    "editDistance": 0.5,
    "type": 1.0,
    "domain": 1.0,
}


# =============================================================================
# NORMALISATION
# =============================================================================


class TestNormalisation:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("CUST_ACCT_NBR", ("cust", "acct", "nbr")),
            ("customerAccountNumber", ("customer", "account", "number")),
            ("HTTPResponseCode", ("http", "response", "code")),
            ("enroll12", ("enroll", "12")),
            ("account-balance amount", ("account", "balance", "amount")),
            ("", ()),
        ],
    )
    def test_an_identifier_becomes_its_word_tokens(self, text, expected):
        """
        Literal expectations, not a comparison against the splitter this module calls.
        Comparing the two would be an identity: `normalise_tokens` IS that splitter, and
        the point of pinning it here is that a change over there turns up as a red test
        in the module that depends on it rather than as a silently re-partitioned
        grouping.
        """
        assert normalise_tokens(text) == expected

    def test_a_non_latin_name_is_not_erased(self):
        """The splitter this module reuses was fixed once because it deleted any field
        named in a non-Latin script. A private copy here would have re-introduced that."""
        assert normalise_tokens("Kundennummer") == ("kundennummer",)
        assert normalise_tokens("cafe_nom") == ("cafe", "nom")
        assert normalise_tokens("счёт_nomer") == ("счёт", "nomer")

    def test_the_array_boundary_matches_the_flattener(self):
        """
        This module restates the flattener's array boundary rather than importing it, so
        that the domain does not depend on an infrastructure adapter. That trade is only
        safe while the two agree, and this is what makes a divergence a red test.
        """
        from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import (
            ARRAY_BOUNDARY as PARSER_BOUNDARY,
        )

        assert ARRAY_BOUNDARY == PARSER_BOUNDARY


class TestPathSegments:
    def test_dots_win_where_present(self):
        assert path_segments("a.b.c") == ("a", "b", "c")

    def test_the_array_boundary_is_the_next_authority(self):
        assert path_segments("CUST_ACCT__BAL_AMT") == ("CUST_ACCT", "BAL_AMT")

    def test_a_single_underscore_is_never_a_segment_boundary(self):
        """
        The load-bearing restriction. Every segment of a flattened name is itself
        underscore-joined, so splitting on it would promote the first TOKEN of a name to
        a path segment -- and for many naming standards that token is exactly the
        qualifier that distinguishes two concepts. A grouping keyed on it reproduces the
        naming convention instead of measuring anything, which is the trap
        `test_review_evidence_grouping` demonstrates with a control that scores 1.000 and
        never moves.
        """
        assert path_segments("CUST_ACCT_BAL_AMT") == ("CUST_ACCT_BAL_AMT",)

    def test_an_empty_key_has_no_segments(self):
        assert path_segments("") == ()


class TestResultKey:
    def test_the_matchers_own_key_wins(self):
        f = SchemaField(
            name="c",
            data_type=DataType.STRING,
            full_path="a.b.c",
            parent_path="a.b",
            source_metadata={"flattened_name": "A_B__C"},
        )
        assert result_key(f) == "A_B__C"

    def test_a_field_with_no_recorded_key_falls_back_to_its_path(self):
        f = SchemaField(name="c", data_type=DataType.STRING, full_path="a.b.c", parent_path="a.b")
        assert result_key(f) == "a.b.c"

    def test_the_concept_does_not_depend_on_which_parser_built_the_field(self):
        """
        A flattened-name parser splits `A_B__C_D` into four path segments and leaves
        `name` as `D`; the HTTP boundary leaves the whole string as the path. Both are the
        same column and must be the same concept, or a deployment's groups would change
        with the shape of its ingest rather than with its data.
        """
        from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import (
            field_from_flattened,
        )

        parsed = field_from_flattened("CUST_ACCT__BAL_AMT", data_type="string", is_array=True)
        over_the_wire = field("CUST_ACCT__BAL_AMT", name="BAL_AMT")
        policy = GroupingPolicy()
        assert concept_of(parsed, policy) == concept_of(over_the_wire, policy)


# =============================================================================
# GROUPING
# =============================================================================


class TestGroupingPolicy:
    def test_the_default_is_the_parent_aware_key_not_the_leaf_alone(self):
        """
        Pinned here as well as in the measurement file, because this is the number a
        caller who constructs `GroupingPolicy()` gets without reading anything. It is 1
        because 0 was measured and produced four findings of which four were collisions;
        `test_review_evidence_grouping.py` carries the table and the search.
        """
        assert GroupingPolicy().qualifier_segments == 1

    def test_a_negative_qualifier_depth_is_refused(self):
        with pytest.raises(ValueError, match="qualifier_segments"):
            GroupingPolicy(qualifier_segments=-1)

    def test_a_group_of_one_is_refused_as_a_policy(self):
        with pytest.raises(ValueError, match="min_group_size"):
            GroupingPolicy(min_group_size=1)


class TestGrouping:
    # The loose leaf-only key, named explicitly rather than taken from the default.
    #
    # It STOPPED being the default because it was measured and failed -- see
    # `test_review_evidence_grouping.py` -- but it is still the key whose composition
    # these tests are about, and a test that reaches it through `GroupingPolicy()` would
    # go green or red on a decision about defaults rather than on the behaviour it names.
    LOOSE = GroupingPolicy(qualifier_segments=0)

    def test_the_same_leaf_under_different_parents_is_one_concept_only_at_the_loose_key(self):
        fields = [field("account.cust_nm"), field("billing.cust_nm")]
        groups = group_by_concept(fields, self.LOOSE)
        assert [g.paths for g in groups] == [("account.cust_nm", "billing.cust_nm")]
        # ... and NOT at the shipped default, which is the whole point of moving it: two
        # columns are one concept only when they share the record they hang off.
        assert group_by_concept(fields) == ()

    def test_a_qualifier_segment_splits_them(self):
        fields = [field("account.cust_nm"), field("billing.cust_nm")]
        assert group_by_concept(fields, GroupingPolicy(qualifier_segments=1)) == ()

    def test_the_qualifier_is_the_nearest_parent_not_the_outermost(self):
        """A column's immediate record says more about which concept it is than the
        namespace at the root of the schema does."""
        fields = [field("alpha.account.cust_nm"), field("beta.account.cust_nm")]
        assert len(group_by_concept(fields, GroupingPolicy(qualifier_segments=1))) == 1
        assert group_by_concept(fields, GroupingPolicy(qualifier_segments=2)) == ()

    def test_a_lone_column_is_not_a_group(self):
        assert group_by_concept([field("account.cust_nm")], self.LOOSE) == ()

    def test_the_data_type_separates_two_spellings_of_one_name(self):
        fields = [field("a.eff_dt"), field("b.eff_dt", data_type=DataType.DATE)]
        assert group_by_concept(fields, self.LOOSE) == ()
        assert (
            len(
                group_by_concept(
                    fields, GroupingPolicy(qualifier_segments=0, include_data_type=False)
                )
            )
            == 1
        )

    def test_token_order_is_ignored_by_default_but_the_class_word_is_not(self):
        reordered = [field("a.acct_status"), field("b.status_acct")]
        assert group_by_concept(reordered, self.LOOSE) == ()  # different class word
        same_class_word = [field("a.cust_acct_nbr"), field("b.acct_cust_nbr")]
        assert len(group_by_concept(same_class_word, self.LOOSE)) == 1
        assert (
            group_by_concept(
                same_class_word, GroupingPolicy(qualifier_segments=0, order_sensitive=True)
            )
            == ()
        )

    def test_groups_and_members_keep_the_order_they_were_sent(self):
        fields = [
            field("z.cust_nm"),
            field("a.bal_amt"),
            field("b.cust_nm"),
            field("c.bal_amt"),
        ]
        groups = group_by_concept(fields, self.LOOSE)
        assert [g.paths for g in groups] == [
            ("z.cust_nm", "b.cust_nm"),
            ("a.bal_amt", "c.bal_amt"),
        ]


# =============================================================================
# CONSISTENCY
# =============================================================================


class TestConsistency:
    def test_one_answer_across_the_group_agrees(self):
        group = ConceptGroup("c", ("a", "b"))
        finding = assess_consistency([group], {"a": "T-1", "b": "T-1"})[0]
        assert finding.agreement is Agreement.AGREE
        assert finding.distinct_answers == 1
        assert (finding.majority_answer, finding.majority_count) == ("T-1", 2)

    def test_two_answers_across_the_group_disagree(self):
        group = ConceptGroup("c", ("a", "b", "d"))
        finding = assess_consistency([group], {"a": "T-1", "b": "T-1", "d": "T-2"})[0]
        assert finding.agreement is Agreement.DISAGREE
        assert finding.distinct_answers == 2
        assert (finding.majority_answer, finding.majority_count) == ("T-1", 2)

    def test_a_blank_is_silence_and_not_a_dissenting_answer(self):
        group = ConceptGroup("c", ("a", "b"))
        finding = assess_consistency([group], {"a": "T-1", "b": None})[0]
        assert finding.agreement is Agreement.UNDECIDED
        assert finding.answers == (("a", "T-1"), ("b", None))

    def test_a_group_nobody_answered_is_undecided(self):
        group = ConceptGroup("c", ("a", "b"))
        assert assess_consistency([group], {})[0].agreement is Agreement.UNDECIDED

    def test_a_tie_for_the_mode_is_not_a_majority(self):
        """Calling a coin toss a majority puts one on a governance artifact."""
        group = ConceptGroup("c", ("a", "b"))
        finding = assess_consistency([group], {"a": "T-1", "b": "T-2"})[0]
        assert finding.agreement is Agreement.DISAGREE
        assert finding.majority_answer is None
        assert finding.majority_count == 0

    def test_the_majority_does_not_depend_on_insertion_order(self):
        group = ConceptGroup("c", ("a", "b", "d", "e"))
        forwards = assess_consistency([group], {"a": "T-2", "b": "T-1", "d": "T-1", "e": "T-2"})
        assert forwards[0].majority_answer is None


# =============================================================================
# CONTRAST
# =============================================================================


class TestContrastBoundaries:
    def test_a_field_with_one_candidate_has_no_contrast(self):
        assert contrast_top_two([match(BASE)], SIGNALS) is None

    def test_a_field_with_no_candidates_has_no_contrast(self):
        assert contrast_top_two([], SIGNALS) is None

    def test_an_unreadable_component_raises_rather_than_reading_as_zero(self):
        """
        A zero delta on a component nobody could read is a report that two candidates
        were identical on a signal that was never looked at.
        """
        matches = [match(BASE), match(BASE, rank=2, entry_id="T-2")]
        with pytest.raises(ValueError, match="colbert"):
            contrast_top_two(matches, (*SIGNALS, SignalSpec("colbert", "colbert_score", 0.1)))


class TestContrastArithmetic:
    def test_the_weighted_differences_sum_to_the_confidence_gap(self):
        loser = {**BASE, "domain": 0.2, "lexical": 0.1}
        contrast = contrast_top_two([match(BASE), match(loser, 2, "T-2")], SIGNALS)
        assert contrast is not None
        assert contrast.signal_gap == pytest.approx(contrast.confidence_gap, abs=1e-6)
        assert sum(d.weighted_delta for d in contrast.differences) == pytest.approx(
            contrast.confidence_gap, abs=1e-6
        )

    def test_the_only_differing_signal_is_the_deciding_one(self):
        loser = {**BASE, "domain": 0.2}
        contrast = contrast_top_two([match(BASE), match(loser, 2, "T-2")], SIGNALS)
        assert contrast is not None
        assert contrast.separation is Separation.SEPARATED
        assert contrast.largest_difference == "domain"
        assert contrast.deciding_signals == ("domain",)

    def test_a_margin_no_single_signal_carries_names_none(self):
        """
        Empty is a real answer, and the common one on a wide margin: two signals each
        worth less than the gap means neither decided it.
        """
        winner = {**BASE, "lexical": 1.0, "editDistance": 1.0}
        loser = {**BASE, "lexical": 0.0, "editDistance": 0.0}
        contrast = contrast_top_two([match(winner), match(loser, 2, "T-2")], SIGNALS)
        assert contrast is not None
        assert contrast.separation is Separation.SEPARATED
        assert contrast.deciding_signals == ()
        assert {d.signal for d in contrast.differences if d.separating} == {
            "lexical",
            "editDistance",
        }

    def test_signals_are_ordered_largest_difference_first(self):
        loser = {**BASE, "domain": 0.5, "lexical": 0.0}
        contrast = contrast_top_two([match(BASE), match(loser, 2, "T-2")], SIGNALS)
        assert contrast is not None
        assert [d.signal for d in contrast.differences][:2] == ["domain", "lexical"]

    def test_the_order_is_total_so_two_runs_agree(self):
        """Two signals with the same weighted difference fall back to the order the
        caller declared them in, which is what makes the response deterministic."""
        loser = {**BASE, "lexical": 0.0, "editDistance": 0.0}
        contrast = contrast_top_two([match(BASE), match(loser, 2, "T-2")], SIGNALS)
        assert contrast is not None
        assert [d.signal for d in contrast.differences][:2] == ["lexical", "editDistance"]

    def test_the_entries_own_facts_are_reported_not_the_resolved_class(self):
        contrast = contrast_top_two(
            [
                match(BASE, entry_id="T-1", code="CODE-A", domain="ALPHA"),
                match(BASE, 2, "T-2", code="CODE-B", domain="BETA"),
            ],
            SIGNALS,
        )
        assert contrast is not None
        assert contrast.governance_differs is True
        assert contrast.domain_differs is True

    def test_two_entries_of_one_class_do_not_read_as_differing(self):
        contrast = contrast_top_two(
            [match(BASE, code="CODE-A"), match(BASE, 2, "T-2", code="CODE-A")], SIGNALS
        )
        assert contrast is not None
        assert contrast.governance_differs is False


class TestResolution:
    def test_a_difference_below_the_resolution_is_not_separating(self):
        loser = {**BASE, "fusedRetrieval": 0.9000001}
        contrast = contrast_top_two([match(BASE), match(loser, 2, "T-2")], SIGNALS)
        assert contrast is not None
        assert not any(d.separating for d in contrast.differences)
        assert contrast.largest_difference is None
        assert contrast.deciding_signals == ()

    def test_level_candidates_name_no_cause_but_keep_the_evidence(self):
        """
        Two signals that disagree and cancel is exactly what a reviewer wants to see, and
        exactly where a naive contrast would invent a winner. The ordering came from the
        matcher's sort; nothing here dresses that up as a finding.
        """
        loser = {**BASE, "lexical": 0.1, "domain": 1.0 + (0.05 * 0.4) / 0.15}
        contrast = contrast_top_two([match(BASE), match(loser, 2, "T-2")], SIGNALS)
        assert contrast is not None
        assert contrast.confidence_gap == pytest.approx(0.0, abs=1e-6)
        assert contrast.separation is Separation.TIED
        assert contrast.largest_difference is None
        assert contrast.deciding_signals == ()
        assert {d.signal for d in contrast.differences if d.separating} == {"lexical", "domain"}

    def test_the_published_deltas_are_the_subtraction_of_the_published_scores(self):
        """
        A reviewer subtracts two rounded component scores by hand. The delta printed
        beside them has to BE that subtraction: computing it at full precision and
        rounding afterwards would disagree in the last place and read as the tool getting
        its own arithmetic wrong.
        """
        winner = {**BASE, "lexical": 0.1234565}
        loser = {**BASE, "lexical": 0.1000004}
        contrast = contrast_top_two([match(winner), match(loser, 2, "T-2")], SIGNALS)
        assert contrast is not None
        lexical = next(d for d in contrast.differences if d.signal == "lexical")
        assert lexical.delta == round(lexical.top_score - lexical.runner_up_score, 6)

    def test_a_coarser_precision_widens_what_counts_as_indistinguishable(self):
        """`precision` is the resolution the evidence is stated at, and it has to move the
        answer -- a threshold that changed nothing would be decoration."""
        loser = {**BASE, "domain": 0.9999}
        fine = contrast_top_two([match(BASE), match(loser, 2, "T-2")], SIGNALS, precision=6)
        coarse = contrast_top_two([match(BASE), match(loser, 2, "T-2")], SIGNALS, precision=3)
        assert fine is not None and coarse is not None
        assert fine.largest_difference == "domain"
        assert coarse.largest_difference is None
