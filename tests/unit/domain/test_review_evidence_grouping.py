"""
tests.unit.domain.test_review_evidence_grouping | Layer: TEST
How good the concept grouping actually is, measured against schemas whose answers are
known by construction -- and the SEARCH that says no setting of it is good enough.

## Relationships
# TESTS → domain/services/review_evidence :: group_by_concept, on real generated schemas

## Why this file exists rather than a paragraph of confidence

The consistency pass reports that N columns look like one concept and got M different
answers. That finding is only worth reading if the grouping is right, and grouping is the
whole difficulty:

  TOO LOOSE  distinct concepts merge, and the pass MANUFACTURES disagreement. Every one
             of those findings is noise, and a reviewer who dismisses three in a row
             stops opening the fourth.
  TOO TIGHT  nothing groups, and the pass finds nothing at all.

So the grouping is measured, not asserted. The synthetic pack makes the term first and
the column that should match it second, so concept identity is a fact about the generator
rather than a judgement about the data -- which is what makes precision and recall of the
GROUPING computable without a labelled corpus and without running the matcher at all.

## The oracle, and what it deliberately excludes

Two columns are the same concept when both carry an EXACT truth row naming the SAME
single governed element. Rows with several defensible answers are excluded from the
oracle entirely, in both directions: two columns that could each match any member of one
cluster are not thereby the same concept, and counting them as agreeing or disagreeing
would be reading an answer the fixture does not contain. Rows with no correct answer are
excluded for the same reason. Precision and recall below are therefore over the pairs the
fixture actually knows about, which is the honest denominator.

Pairs, not clusters: precision is the share of pairs the grouping puts together that
belong together, recall the share of pairs that belong together which it finds. Grouping
runs over EVERY column of the schema, which is what a request actually sends; only the
scoring is restricted to the oracle.

## THE MEASUREMENT (rows=600, schema_scale=0.15, seed 7, repeats_per_domain=2)

`nested-repeated` is the shape the roadmap's own example describes: one leaf name governed
separately in each of ~30 domains, so the leaf carries NO information and the parent
carries all of it.

    policy                       precision   recall   groups   pure   collisions
    qualifier_segments=0            0.0175   1.0000        4      0            4
    qualifier_segments>=1              n/a   0.0000        0      -            -
    CONTROL: leaf + first token     1.0000   1.0000       --      -            -

`mixed-production`, the proportioned mixture of all five shapes:

    qualifier_segments=0            1.0000   0.0435        9      1            0
    qualifier_segments>=1              n/a   0.0000        0      -            -

At `repeats_per_domain=3` the same run gives 0.0233/1.0000 with four groups and four
collisions on `nested-repeated`, and 1.0000/0.1212 with fourteen groups and none on the
mixture. At rows=1200 / scale=0.5 over seeds 7, 11 and 23 the loose key holds
0.0169-0.0235 precision on `nested-repeated` and 0.857-1.000 at 0.065-0.137 recall on the
mixture. `flat-english`, `flat-contracted`, `nested-deep` and `no-doc` produce no scorable
pairs under any policy.

## What the numbers say, including the part that ends the feature

THE LOOSE KEY IS NOT MERELY IMPRECISE ON THE SHAPE THAT MOTIVATES THIS FEATURE -- IT IS
WRONG EVERY TIME IT SPEAKS. Four groups, zero concepts, four collisions: 87 columns
spanning 29 distinct correct answers merged into one "concept" and reported to a reviewer
as a contradiction. Precision 0.0233 is 97.7% of asserted pairs being false. On the
mixture the same key is right about everything it says and finds a tenth of the
repetition, so the failure is not "the key is bad" but "the key is a bet on whether the
parent is decorative, and nothing in the names says which".

THE SEGMENT DIAL CANNOT REACH AN OPERATING POINT, AND THAT IS SEARCHED, NOT OBSERVED.
`TestNoPolicyIsDefensible` enumerates the entire published policy space -- 684 policies,
every `qualifier_segments` x `include_data_type` x `order_sensitive` x `min_group_size`
combination -- on both profiles at two scales and two repetition depths, and looks for one
that reports at least one pair on `nested-repeated` at precision >= 0.5. There is none:
the best precision reached by any policy that reports anything there is 0.0235. The same
search is run against the reconstruction control, which does reach 1.0, so the negative
result is a fact about the space rather than a broken harness.

So the shipped default is `qualifier_segments=1`, which emits NO GROUP on any generated
profile. A default that reports nothing is a poor feature; a default that is wrong four
times out of four is a liability, and between the two the honest one wins. The loose key
stays reachable, with its precision published on the request parameter that selects it.

## The control, and why it is a control rather than a result

`CONTROL` groups on the leaf plus the FIRST UNDERSCORE TOKEN of the parent. It scores a
perfect 1.000/1.000 -- and it scores exactly 1.000/1.000 at `repeats_per_domain` 2 and 3,
at 600 rows and at 1,200, on two seeds. A number that does not move when the difficulty
moves is not measuring difficulty. It is reconstruction: the generator's key for this
fixture IS (leaf, domain), the domain IS the first underscore token, so that grouping
recomputes the answer instead of retrieving it. `test_the_control_is_reconstruction_not_retrieval`
pins both halves -- the perfect score and the fact that it never changes -- so nobody
adopts it as a shipped policy on the strength of the 1.000.

The shipped policy refuses to split inside a segment for exactly this reason. The
measured precision of 0.0175 is a worse number and a real one.

## One part of the key earns nothing here, and that is worth saying

`include_data_type` is on by default and changes NOTHING on any fixture in this file:
removing the data type from the concept key leaves every number above untouched. It stays
because refusing to merge a date column with the string that spells it is defensible on
its face and costs nothing -- but its value is unmeasured, and this file is not evidence
for it. `test_the_data_type_separates_two_spellings_of_one_name` in
`test_review_evidence.py` pins the behaviour; nothing pins that the behaviour is worth
having.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

import pytest

from benchmarks.synthetic.pack import PackSpec, SyntheticPack
from nexus_matcher.domain.services.review_evidence import (
    GroupingPolicy,
    concept_of,
    group_by_concept,
    max_qualifier_segments,
    normalise_tokens,
    path_segments,
)
from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import (
    field_from_flattened,
)

# Small enough to generate in well under a second, large enough that the wide clusters
# are at full width -- `wide_cluster_size` does not scale with the row count, so the
# adversarial construction is intact at 600 rows.
ROWS = 600
SCALE = 0.15
SEED = 7

# What "good enough to report" would have to mean at a bare minimum: more than half of
# what the grouping asserts is true. Deliberately a low bar. The point of the search is
# that nothing clears even this on the shape that motivates the feature.
DEFENSIBLE_PRECISION = 0.5


def _pack(rows: int = ROWS, seed: int = SEED, repeats: int = 2, scale: float = SCALE):
    return SyntheticPack.generate(
        PackSpec(
            rows=rows,
            seed=seed,
            schema_scale=scale,
            feedback_events=0,
            repeats_per_domain=repeats,
        )
    )


def _oracle(schema) -> dict[str, str]:
    """Column -> its single correct id, for the rows whose concept identity the fixture
    actually knows: EXACT, one answer."""
    seen: set[str] = set()
    answers: dict[str, str] = {}
    for row in schema.truth:
        if row.flattened_name in seen:
            continue
        seen.add(row.flattened_name)
        if row.truth_class.value == "EXACT" and len(row.correct_ids) == 1:
            answers[row.flattened_name] = row.correct_ids[0]
    return answers


def _fields(schema) -> list:
    """EVERY column, built by the library's own flattened-name parser -- which is what a
    pipeline sending a flattened export actually produces, and what a request carries. The
    oracle restricts the SCORING, not the grouping: a collision with an unlabelled column
    is still a collision, it just cannot be counted."""
    seen: set[str] = set()
    out = []
    for row in schema.flattened:
        name = row["flattenedName"]
        if name in seen:
            continue
        seen.add(name)
        out.append(
            field_from_flattened(
                name,
                doc=row.get("doc", ""),
                data_type=row.get("dataType"),
                is_array=row.get("isArraySerialized", False),
            )
        )
    return out


# =============================================================================
# SCORING -- two routes to the same numbers, pinned against each other
# =============================================================================


def _pairs(partition: dict[str, list[str]]) -> set[frozenset[str]]:
    out: set[frozenset[str]] = set()
    for members in partition.values():
        for a, b in itertools.combinations(sorted(set(members)), 2):
            out.add(frozenset((a, b)))
    return out


def _truth_pairs(schema) -> set[frozenset[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for name, answer in _oracle(schema).items():
        buckets[answer].append(name)
    return _pairs(buckets)


def _predicted_pairs(fields, policy: GroupingPolicy, oracle: dict[str, str]) -> set[frozenset[str]]:
    groups = group_by_concept(fields, policy)
    pairs = _pairs({group.concept: list(group.paths) for group in groups})
    return {pair for pair in pairs if all(member in oracle for member in pair)}


def _score(predicted: set[frozenset[str]], truth: set[frozenset[str]]) -> tuple[float, float]:
    hit = len(predicted & truth)
    precision = hit / len(predicted) if predicted else float("nan")
    recall = hit / len(truth) if truth else float("nan")
    return precision, recall


def _choose_2(n: int) -> int:
    return n * (n - 1) // 2


def _counted_score(fields, policy: GroupingPolicy, oracle: dict[str, str]) -> tuple[float, float]:
    """
    The same precision and recall, counted instead of enumerated.

    Identical by construction -- the grouping is a partition, so no pair is produced
    twice -- and it is what the 684-policy search runs, because materialising 15,000
    frozensets per policy turns a search into a coffee break.
    `test_the_counted_score_agrees_with_the_enumerated_one` pins the two together.
    """
    predicted = hit = 0
    for group in group_by_concept(fields, policy):
        labelled = [oracle[path] for path in set(group.paths) if path in oracle]
        predicted += _choose_2(len(labelled))
        counts: dict[str, int] = defaultdict(int)
        for answer in labelled:
            counts[answer] += 1
        hit += sum(_choose_2(count) for count in counts.values())

    totals: dict[str, int] = defaultdict(int)
    for answer in oracle.values():
        totals[answer] += 1
    truth = sum(_choose_2(count) for count in totals.values())

    precision = hit / predicted if predicted else float("nan")
    recall = hit / truth if truth else float("nan")
    return precision, recall


def _group_purity(fields, policy: GroupingPolicy, oracle: dict[str, str]) -> tuple[int, int, int]:
    """
    (pure, collisions, unlabelled) over the emitted groups.

    The finding-level view, which is what a reviewer actually receives: a group is PURE
    when every labelled member shares one answer, a COLLISION when they do not, and
    unlabelled when the oracle knows none of its members. Pair precision can be quoted as
    a small number and shrugged off; "four findings, zero concepts" cannot.
    """
    pure = collisions = unlabelled = 0
    for group in group_by_concept(fields, policy):
        answers = {oracle[path] for path in group.paths if path in oracle}
        if not answers:
            unlabelled += 1
        elif len(answers) == 1:
            pure += 1
        else:
            collisions += 1
    return pure, collisions, unlabelled


def _control_pairs(schema, oracle: dict[str, str]) -> set[frozenset[str]]:
    """
    THE RECONSTRUCTION CONTROL, and it is not a candidate policy.

    Leaf plus the first underscore token of the parent, which for this fixture is the
    generator's own key. Present so the perfect score it earns can be labelled as
    reconstruction instead of being mistaken for a result -- and so the search below has
    something that DOES clear its bar, which is what makes the negative result mean
    anything.
    """
    buckets: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in schema.flattened:
        name = row["flattenedName"]
        if name in seen:
            continue
        seen.add(name)
        parent, _, leaf = name.rpartition("__")
        first = parent.split("_")[0] if parent else ""
        buckets[f"{first}|{'.'.join(normalise_tokens(leaf))}"].append(name)
    pairs = _pairs({key: members for key, members in buckets.items() if len(members) >= 2})
    return {pair for pair in pairs if all(member in oracle for member in pair)}


@pytest.fixture(scope="module")
def repeated():
    schema = _pack().schema("nested-repeated")
    oracle = _oracle(schema)
    return schema, _fields(schema), _truth_pairs(schema), oracle


@pytest.fixture(scope="module")
def mixture():
    schema = _pack().schema("mixed-production")
    oracle = _oracle(schema)
    return schema, _fields(schema), _truth_pairs(schema), oracle


# =============================================================================
# THE FIXTURE IS THE ONE THE MEASUREMENT NEEDS
# =============================================================================


class TestFixture:
    def test_the_oracle_has_pairs_to_find(self, repeated):
        _schema, fields, truth, _oracle_map = repeated
        assert len(fields) > 100
        assert len(truth) > 50

    def test_one_leaf_name_really_is_repeated_across_many_parents(self, repeated):
        _schema, fields, _truth, _oracle_map = repeated
        leaves = defaultdict(set)
        for f in fields:
            key = f.source_metadata["flattened_name"]
            parent, _, leaf = key.rpartition("__")
            leaves[leaf].add(parent)
        widest = max(len(parents) for parents in leaves.values())
        assert widest >= 20, (
            f"the widest leaf appears under only {widest} parents; this fixture is "
            f"supposed to be the adversarial one and it no longer is"
        )

    def test_the_counted_score_agrees_with_the_enumerated_one(self, repeated, mixture):
        """The search below trusts `_counted_score`. If the two ever disagree, the search
        is measuring something other than what this file's table reports."""
        for _schema, fields, truth, oracle in (repeated, mixture):
            for segments in (0, 1):
                policy = GroupingPolicy(qualifier_segments=segments)
                enumerated = _score(_predicted_pairs(fields, policy, oracle), truth)
                counted = _counted_score(fields, policy, oracle)
                assert repr(enumerated) == repr(counted), (segments, enumerated, counted)


# =============================================================================
# THE SHIPPED DEFAULT -- IT REPORTS NOTHING, ON PURPOSE
# =============================================================================


class TestDefaultPolicy:
    @pytest.mark.parametrize("profile", ["nested-repeated", "mixed-production"])
    def test_the_default_emits_no_group_on_any_generated_profile(self, request, profile):
        """
        The shipped promise, and it is a modest one: at the default the grouping makes no
        claim about this corpus at all. That is the point. A grouping nobody has shown to
        work on the repeated-leaf shape should not be handing a reviewer findings, and the
        only setting that never hands them a wrong one is the one that hands them none.
        """
        _schema, fields, _truth, _oracle_map = request.getfixturevalue(
            "repeated" if profile == "nested-repeated" else "mixture"
        )
        assert GroupingPolicy().qualifier_segments == 1
        assert group_by_concept(fields, GroupingPolicy()) == ()

    def test_the_default_cannot_manufacture_a_disagreement_it_did_not_find(self, repeated, mixture):
        """
        The property that matters more than recall: every pair the default asserts is a
        pair the loose key also asserts, because raising `qualifier_segments` REFINES the
        partition rather than rearranging it. So the default can lose a true finding but
        can never invent one the loose key would not also have made -- which is why moving
        the default in this direction is safe and moving it back is not.
        """
        for _schema, fields, _truth, oracle in (repeated, mixture):
            loose = _predicted_pairs(fields, GroupingPolicy(qualifier_segments=0), oracle)
            default = _predicted_pairs(fields, GroupingPolicy(), oracle)
            assert default <= loose, sorted(default - loose)[:5]


# =============================================================================
# THE LOOSE KEY -- WHAT IT COSTS, MEASURED
# =============================================================================


class TestTheLooseKey:
    LOOSE = GroupingPolicy(qualifier_segments=0)

    def test_it_finds_every_true_pair_on_the_repeated_leaf_fixture(self, repeated):
        """
        Recall is 1.000 here BY CONSTRUCTION and not as an achievement: every pair that
        belongs together shares a leaf, and the leaf-only key groups everything that
        shares a leaf, so the predicted partition is a superset of the truth. It is
        asserted anyway, because losing it would mean the normalisation stopped seeing
        two spellings of one leaf as one -- and because a recall that cannot move is
        exactly the kind of number this repository has published as an achievement before.
        """
        _schema, fields, truth, oracle = repeated
        _precision, recall = _score(_predicted_pairs(fields, self.LOOSE, oracle), truth)
        assert recall == 1.0

    def test_its_precision_on_that_fixture_is_bad_and_that_is_the_finding(self, repeated):
        _schema, fields, truth, oracle = repeated
        precision, _recall = _score(_predicted_pairs(fields, self.LOOSE, oracle), truth)
        assert precision < 0.05, precision

    def test_every_finding_it_emits_on_that_fixture_is_a_collision(self, repeated):
        """
        The number a reviewer actually experiences. Pair precision of 0.0175 is abstract;
        "four findings, none of which is a concept" is not, and it is the reason this key
        is no longer the default. Each of those four groups merges columns that SHOULD
        have different answers, so the pass reports a contradiction where correctness
        requires divergence.
        """
        _schema, fields, _truth, oracle = repeated
        pure, collisions, unlabelled = _group_purity(fields, self.LOOSE, oracle)
        assert (pure, collisions, unlabelled) == (0, 4, 0)

    def test_precision_moves_with_the_difficulty(self):
        """
        The check that this is a measurement rather than an identity. More repeats of
        each (leaf, parent) pair means more pairs that genuinely belong together inside a
        group whose size grows more slowly, so precision must rise. A number that stayed
        put under this sweep would be arithmetic about the fixture, not about the policy.
        """
        scores = []
        for repeats in (2, 3):
            schema = _pack(repeats=repeats).schema("nested-repeated")
            oracle = _oracle(schema)
            precision, _recall = _score(
                _predicted_pairs(_fields(schema), self.LOOSE, oracle), _truth_pairs(schema)
            )
            scores.append(precision)
        assert scores[1] > scores[0], scores

    def test_it_is_precise_on_a_realistic_mixture(self, mixture):
        """
        The other half of the story, and the reason this key was ever the default: on a
        proportioned mixture of schema shapes it puts nothing together that does not
        belong together. Both facts are real, and neither is available to the grouping at
        request time -- which is exactly why the key cannot be chosen for the caller.
        """
        _schema, fields, truth, oracle = mixture
        precision, recall = _score(_predicted_pairs(fields, self.LOOSE, oracle), truth)
        assert precision == 1.0, precision
        assert 0.0 < recall < 0.3, recall
        pure, collisions, _unlabelled = _group_purity(fields, self.LOOSE, oracle)
        assert (pure, collisions) == (1, 0)


# =============================================================================
# THE SEARCH -- NO POLICY IN THE PUBLISHED SPACE IS DEFENSIBLE
# =============================================================================


def _published_space() -> list[GroupingPolicy]:
    """
    Every policy a caller can select, enumerated.

    `qualifier_segments` is swept to 8 rather than to the published ceiling of 511
    because `concept_of` slices the segment list: once the qualifier is the whole parent
    path, every larger value produces a byte-identical key.
    `test_values_past_a_keys_depth_are_inert` pins that, so sweeping further would add
    runtime and no coverage. `min_group_size` is swept to 20, past the largest group any
    of these policies emits on the mixture.
    """
    return [
        GroupingPolicy(
            qualifier_segments=segments,
            include_data_type=data_type,
            order_sensitive=order,
            min_group_size=size,
        )
        for segments in range(0, 9)
        for data_type in (True, False)
        for order in (True, False)
        for size in range(2, 21)
    ]


class TestNoPolicyIsDefensible:
    """
    The claim this feature turns on, established by SEARCHING for a counter-example rather
    than by sampling a corpus and reporting the best it happened to see.

    A policy is defensible on a profile when it reports at least one pair AND more than
    half of what it reports is true. The search asks whether any policy in the published
    space is defensible on the repeated-leaf shape. If one were, the right fix would be to
    default to it; none is, so the right fix is to default to reporting nothing.
    """

    @pytest.mark.parametrize("repeats", [2, 3])
    def test_nothing_in_the_space_reports_anything_true_on_the_repeated_leaf_shape(self, repeats):
        schema = _pack(repeats=repeats).schema("nested-repeated")
        fields, oracle = _fields(schema), _oracle(schema)

        best = 0.0
        reaching: list[GroupingPolicy] = []
        for policy in _published_space():
            precision, _recall = _counted_score(fields, policy, oracle)
            if math.isnan(precision):  # this policy reported no pair at all
                continue
            best = max(best, precision)
            if precision >= DEFENSIBLE_PRECISION:
                reaching.append(policy)

        assert not reaching, (
            f"the search found {len(reaching)} policies that report a pair on the "
            f"repeated-leaf shape at precision >= {DEFENSIBLE_PRECISION}; the first is "
            f"{reaching[0]}. If that is real, the grouping is fixable and this feature "
            f"should default to that policy instead of to silence."
        )
        assert best < 0.05, (
            f"the best precision any reporting policy reaches is {best:.4f}, which is "
            f"materially better than the 0.0235 this file documents; re-derive the table."
        )

    def test_the_search_is_not_vacuous_because_the_control_clears_its_bar(self):
        """
        The guard on the negative result. A search that finds nothing proves nothing
        unless something CAN be found, so the same bar is applied to the reconstruction
        control -- a key outside the published space that splits inside a path segment.
        It scores 1.0, so "nothing in the space reaches 0.5" is a fact about the space and
        not about a harness that cannot score anything.
        """
        schema = _pack().schema("nested-repeated")
        oracle = _oracle(schema)
        precision, recall = _score(_control_pairs(schema, oracle), _truth_pairs(schema))
        assert precision >= DEFENSIBLE_PRECISION
        assert precision == 1.0 and recall == 1.0

    def test_nothing_in_the_space_is_defensible_on_both_profiles_at_once(self):
        """
        The two-sided form, which is the decision that was actually in front of us: is
        there a single setting a deployment could be given that is right on a
        parent-diverse schema AND on a repeated-leaf one? A feature whose correct setting
        depends on a property of the schema that the request does not carry cannot have a
        safe default.
        """
        pack = _pack()
        profiles = [
            (name, _fields(schema), _oracle(schema))
            for name, schema in (
                ("nested-repeated", pack.schema("nested-repeated")),
                ("mixed-production", pack.schema("mixed-production")),
            )
        ]
        both: list[GroupingPolicy] = []
        for policy in _published_space():
            scores = [_counted_score(fields, policy, oracle) for _n, fields, oracle in profiles]
            if all(not math.isnan(p) and p >= DEFENSIBLE_PRECISION for p, _r in scores):
                both.append(policy)
        assert not both, both[:3]


# =============================================================================
# THE DIAL, AND WHERE IT CANNOT REACH
# =============================================================================


class TestQualifierSegments:
    @pytest.mark.parametrize("depth", [1, 2])
    def test_one_declared_boundary_leaves_the_dial_with_nothing_to_grip(self, repeated, depth):
        """
        These names carry one declared boundary, and the qualifier that would separate
        the domains lives inside the segment beside a record noun that varies between
        repeats. So the first turn of the dial splits every group and finds nothing.
        Stated rather than tuned away: the fix would be to split inside a segment, which
        is what the control does, and the control reconstructs.
        """
        _schema, fields, _truth, _oracle_map = repeated
        assert group_by_concept(fields, GroupingPolicy(qualifier_segments=depth)) == ()

    def test_the_dial_does_work_where_a_boundary_was_declared(self):
        """The dial is not inert in general -- it is inert on that SHAPE. A caller who
        sends a dotted path gets the split they asked for."""
        from nexus_matcher.domain.models.entities import SchemaField
        from nexus_matcher.shared.types.base import DataType

        def dotted(path: str) -> SchemaField:
            parent, _, leaf = path.rpartition(".")
            return SchemaField(
                name=leaf,
                data_type=DataType.STRING,
                full_path=path,
                parent_path=parent,
                source_metadata={"flattened_name": path},
            )

        fields = [dotted("alpha.cust_nm"), dotted("beta.cust_nm"), dotted("beta.cust_nm2")]
        assert len(group_by_concept(fields, GroupingPolicy(qualifier_segments=0))) == 1
        assert group_by_concept(fields, GroupingPolicy(qualifier_segments=1)) == ()


# =============================================================================
# THE PUBLISHED BOUND, DERIVED AND PROVED TIGHT
# =============================================================================


class TestTheDerivedBound:
    """
    `max_qualifier_segments` replaces a literal ceiling of 8. A bound is only worth
    deriving if the derivation is checked, so both halves are: nothing can exceed it, and
    something reaches it.
    """

    def test_no_key_within_the_budget_beats_the_derived_bound(self):
        """
        The SEARCH half. Adversarial keys are constructed across the whole length range
        and every separator this splitter honours -- dots, array boundaries, mixtures,
        runs of empty segments -- and none yields more segments than the derivation
        allows. A bound proved by trying to break it, not by measuring one string.
        """
        for budget in (1, 2, 3, 7, 8, 63, 64, 1024):
            ceiling = max_qualifier_segments(budget)
            candidates = [
                "." * budget,
                "a" * budget,
                ("a." * budget)[:budget],
                (".a" * budget)[:budget],
                ("a__" * budget)[:budget],
                ("__a" * budget)[:budget],
                ("a.b__" * budget)[:budget],
                ("..a." * budget)[:budget],
                ("é." * budget)[:budget],
            ]
            for key in candidates:
                segments = path_segments(key)
                assert len(key) <= budget
                assert max(len(segments) - 1, 0) <= ceiling, (budget, key, len(segments))

    def test_the_published_qualifier_bound_is_exactly_reachable(self):
        """
        The TIGHTNESS half, which is what stops the derivation from being a safe-looking
        overestimate. A key at exactly the path budget declares one more segment than the
        ceiling, so the ceiling is the largest value that can still change a key -- one
        less would refuse a value this endpoint can distinguish.
        """
        from nexus_matcher.presentation.api.schemas import _MAX_PATH, MAX_QUALIFIER_SEGMENTS

        assert max_qualifier_segments(_MAX_PATH) == MAX_QUALIFIER_SEGMENTS
        key = ".".join("a" * ((_MAX_PATH + 1) // 2))
        assert len(key) <= _MAX_PATH
        assert len(path_segments(key)) == MAX_QUALIFIER_SEGMENTS + 1

    def test_values_past_a_keys_depth_are_inert(self):
        """
        Why the ceiling is a bound and not a refusal: once the qualifier has consumed the
        whole parent path, `concept_of` slices the same list and a larger number produces
        a byte-identical key. So an over-large value costs a caller nothing but the
        illusion of a tighter grouping, which is why the response echoes the number back.
        """
        from nexus_matcher.domain.models.entities import SchemaField
        from nexus_matcher.shared.types.base import DataType

        field = SchemaField(
            name="cust_nm",
            data_type=DataType.STRING,
            full_path="alpha.beta.cust_nm",
            parent_path="alpha.beta",
            source_metadata={"flattened_name": "alpha.beta.cust_nm"},
        )
        keys = {
            segments: concept_of(field, GroupingPolicy(qualifier_segments=segments))
            for segments in (2, 3, 8, 511)
        }
        assert len(set(keys.values())) == 1, keys
        assert concept_of(field, GroupingPolicy(qualifier_segments=1)) != keys[2]


# =============================================================================
# THE CONTROL
# =============================================================================


class TestControl:
    def test_the_control_is_reconstruction_not_retrieval(self):
        """
        A perfect score that does not move when the difficulty moves is the signature of
        a measurement that is recomputing its own answer. This control keys on the
        generator's own key, scores 1.000/1.000, and scores exactly that at every setting
        swept here.

        It is pinned so that the 1.000 can never be quoted as evidence for a policy. The
        shipped grouping refuses to split inside a segment, which is precisely what gives
        up this number, and the 0.0175 it gets instead is the honest one.
        """
        seen = []
        for rows, seed, repeats in ((600, 7, 2), (600, 7, 3), (1200, 11, 2)):
            schema = _pack(rows=rows, seed=seed, repeats=repeats).schema("nested-repeated")
            oracle = _oracle(schema)
            truth = _truth_pairs(schema)
            control = _score(_control_pairs(schema, oracle), truth)
            shipped = _score(
                _predicted_pairs(_fields(schema), GroupingPolicy(qualifier_segments=0), oracle),
                truth,
            )
            seen.append((control, shipped))

        assert [c for c, _s in seen] == [(1.0, 1.0)] * 3, seen
        # ... while the loose key's precision is not the same number twice.
        precisions = {round(s[0], 6) for _c, s in seen}
        assert len(precisions) > 1, precisions
