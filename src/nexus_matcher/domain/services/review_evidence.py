"""
nexus_matcher.domain.services.review_evidence | Layer: DOMAIN
Evidence a reviewer needs that a decision does not carry: what separated the top two
candidates, and where the same concept got two different answers.

## Relationships
# DEPENDS_ON → domain/models/entities :: SchemaField in, MatchResult in
# USED_BY    → presentation/api/matching :: projected onto the wire, on request only

## Why one module for two things

Both answer a question ABOUT a set of results rather than a question about one result,
and both are read-only: nothing here changes a rank, a confidence, a decision or a
governance id. They are evidence, not adjudication. A caller that ignores everything in
this module gets exactly the answers it got before it existed, which is the property the
presentation layer proves byte-for-byte.

## CONTRAST -- why the runner-up lost

A confidence and a weight breakdown answer "why did the winner score what it did". The
question a reviewer actually has is "why not the other one", and answering it means
subtracting two candidates rather than describing one.

The value is in the DIFFERENCE, and in NAMING WHAT DECIDED IT. Re-stating the published
per-signal weights beside a candidate adds bytes and no information: the weights are
already on the response when `explain` is asked for, and they are the same weights for
every candidate. What is not derivable from them is which signal actually moved this
pair apart, and whether any single one of them accounts for the whole margin.

TWO THINGS ARE DELIBERATELY NOT CLAIMED.

  * A signal whose two scores differ by less than the RESOLUTION of the emitted numbers
    is reported as not separating, and can never be named as a cause. The response
    rounds every number it publishes; a "cause" below that rounding is a difference the
    reviewer cannot see in the artifact they are holding, and naming one would be
    inventing a reason. `Contrast.resolution` states the threshold that was applied.
  * When the two candidates are level -- the whole margin is at or below the resolution
    -- nothing is named at all. The ordering then came from the matcher's own sort, not
    from a signal, and `separation` says `TIED` rather than dressing a sort order up as
    a finding. The per-signal differences are still reported, because two signals that
    disagree and cancel is exactly the case a reviewer wants to see.

`deciding_signals` is arithmetic, not judgement: a signal is decisive when removing its
contribution would leave the runner-up level with or ahead of the winner. It can be
empty, and empty is a real answer -- it means no single signal carried the margin.

## CONSISTENCY -- the same concept, answered twice

Fields are matched one at a time and independently, which throws away a constraint that
costs nothing to check: two columns that are the same business concept should get the
same answer. Nothing enforces that today and, more to the point, NOTHING NOTICES when it
fails.

DISAGREEMENT IS CHECKABLE WITHOUT GROUND TRUTH, which is what made this look immediately
deployable. Knowing that six columns look like one concept and received three different
answers does not require knowing which of the three is right, and reporting it cannot be
wrong in a way that changes a classification.

THAT ARGUMENT HAS A HOLE AND THE MEASUREMENT FOUND IT. "Look like one concept" is not
free: it is the grouping's claim, it can be false, and when it is, the report is a
confident contradiction about columns that were never related. A reviewer handed four such
findings in a row stops opening the fifth, which costs more than the feature was ever
going to return. So the grouping is the gate, and the grouping is measured below rather
than assumed.

So this module REPORTS and does not override. `ConsistencyFinding.majority_answer` is
computed and published as evidence; nothing here applies it, and the presentation layer
does not either. Promoting a group's majority is a decision that can be wrong in a NEW
way -- it can move a correct answer to an incorrect one, which reporting cannot -- and
the measurement that would justify it does not exist yet. What does exist is a
measurement of the GROUPING, which is the prerequisite -- and it came back negative. See
`tests/unit/domain/test_review_evidence_grouping.py`.

## Grouping is the whole difficulty, and the measurement says it is NOT SOLVED

Too loose and distinct concepts merge, which MANUFACTURES disagreement: a group that
should never have existed reports its members as contradicting each other, and every one
of those findings is noise. Too tight and nothing groups, and the feature finds nothing.

THE LOOSE KEY WAS THE DEFAULT AND IT FAILED THE ONE SHAPE THIS FEATURE EXISTS FOR. On the
generated repeated-leaf schema -- one leaf name governed separately in each of ~30
domains, which is the construction the roadmap's own example describes -- grouping on the
leaf alone scores pair-precision **0.0233** at recall 1.0000, and the four groups it emits
contain **zero** concepts and four collisions: 87 columns spanning 29 genuinely distinct
answers merged into one "concept", reported to a reviewer as a contradiction. On a
parent-diverse mixture the same key scores 0.86-1.00 precision at 0.06-0.14 recall. It is
right where the parent is decorative and catastrophic where the parent is the answer, and
NOTHING IN THE NAMES DISTINGUISHES THE TWO CASES.

That is not a tuning problem, and it was not established by looking at a fixture.
`tests/unit/domain/test_review_evidence_grouping.py` SEARCHES the whole published policy
space -- 684 policies, every `qualifier_segments` x `include_data_type` x
`order_sensitive` x `min_group_size` combination -- across two profiles, two scales and
two repetition depths, and finds that the best precision reached by ANY policy that
reports anything at all on the repeated-leaf schema is 0.0235. There is no operating
point. The one key that does separate those domains -- leaf plus the first underscore
token of the parent -- is the generator's own key and scores a frozen 1.000/1.000 at every
difficulty, which is reconstruction, not retrieval; it is kept as a labelled control and
refused as a policy.

SO THE DEFAULT IS THE SETTING THAT REPORTS NOTHING RATHER THAN THE ONE THAT REPORTS
WRONGLY. `qualifier_segments` defaults to 1: the concept key includes the nearest DECLARED
path segment, and two columns are one concept only when they share a leaf AND the record
they hang off. On every generated profile that setting emits no group at all, so it makes
no false claim; the loose key remains reachable at 0, with its precision published on the
parameter itself, for a deployment that has measured its own schemas. An off-by-default
feature with an honest limitation is a fine thing to ship. Four findings out of four wrong
is not.

THE KEY IS DERIVED FROM THE RESULT KEY, NOT FROM `SchemaField.name`. The result key is
the caller's own address for the column and is the same string whichever parser built
the field; `name` is not. A flattened-name parser splits `A_B__C_D_E` into five path
segments and leaves `name` as `E`, so grouping on `name` would put every column whose
last token happens to be `E` into one concept. That is the too-loose failure arriving
through the back door, and it depends on which parser the caller used, which is not
something a concept should depend on.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from nexus_matcher.domain.services.context_enricher import ContextEnricher

if TYPE_CHECKING:
    from nexus_matcher.domain.models.entities import MatchResult, SchemaField

# =============================================================================
# NORMALISATION
# =============================================================================

# The separator a flattener puts at an array boundary. Restated here rather than imported
# because the importer is an INFRASTRUCTURE adapter and this is the domain: a domain
# module that imports an adapter inverts the dependency the hexagon exists to keep. The
# two are pinned together by `test_the_array_boundary_matches_the_flattener`, so a change
# to one is a red test rather than a silent divergence.
ARRAY_BOUNDARY = "__"

# The identifier splitter this module normalises with.
#
# `ContextEnricher._humanize` rather than a second splitter, and the private access is
# deliberate. It is the same layer, and it is the function whose Unicode behaviour was
# validated against 10,687 identifiers when it was last changed; a private copy here
# would be a second, unvalidated splitter that drifts the first time either is touched.
# `test_normalisation_tracks_the_enricher` pins the two together over the awkward cases,
# so a change in that module is a red test in this one.
_SPLITTER = ContextEnricher()


def normalise_tokens(text: str) -> tuple[str, ...]:
    """
    An identifier as its lower-case word tokens: `CUST_ACCT_NBR` -> `(cust, acct, nbr)`.

    No expansion, no stemming and no vocabulary of any kind. Two columns group because
    they are spelled the same way after case, separators and digit boundaries are taken
    out, and for no other reason -- this library ships no taxonomy, and a grouping that
    depended on one would group differently in every deployment.
    """
    if not text:
        return ()
    return tuple(_SPLITTER._humanize(text).split())


def result_key(field: SchemaField) -> str:
    """
    The caller's own address for this column: the key its results come back under.

    Read from `source_metadata['flattened_name']` first because that is what the matcher
    keys its result map by and what every parser writes there, so it is the one string
    that is the same for a given column however the field was built.
    """
    flattened = field.source_metadata.get("flattened_name")
    if isinstance(flattened, str) and flattened:
        return flattened
    return field.full_path or field.name


def path_segments(key: str) -> tuple[str, ...]:
    """
    A result key as its path segments, most general first.

    Dots win where present, because a dotted path is unambiguous and is what a caller
    who has a hierarchy sends. Otherwise the array boundary is used, because it is the
    one separator a flattener guarantees is a boundary and not part of a name.

    A SINGLE UNDERSCORE IS NOT SPLIT ON HERE, and that is the load-bearing restriction.
    Every segment of a flattened name is itself underscore-joined, so splitting on it
    would make the first token of `CUST_ACCT__BAL_AMT` a path segment -- which for many
    naming standards is the very token that distinguishes two concepts, and treating it
    as a qualifier would let the grouping reproduce a naming convention rather than
    measure one. Segments are boundaries the caller declared; tokens are not.

    THE CONSEQUENCE, SAID OUT LOUD BECAUSE IT IS WHY THE DIAL LOOKS INERT. This library's
    own flattener joins nested RECORD levels with a single underscore and emits `__` only
    at an ARRAY boundary. So a flattened name from a nested-but-array-free schema is ONE
    segment, and `qualifier_segments` has nothing to grip: every value of it produces the
    same key. The dial only has reachable positions for a caller who sends a dotted path,
    or whose columns sit under an array. That is a property of the input, not a bug here,
    and `GroupingPolicy` says what it means for the default.
    """
    if not key:
        return ()
    if "." in key:
        return tuple(p for p in key.split(".") if p)
    if ARRAY_BOUNDARY in key:
        return tuple(p for p in key.split(ARRAY_BOUNDARY) if p)
    return (key,)


def max_qualifier_segments(key_chars: int) -> int:
    """
    The largest `qualifier_segments` that can change ANY key of at most `key_chars`
    characters -- the ceiling a publisher should bound the dial at, derived rather than
    picked.

    `path_segments` splits on a ONE-character separator (`.`) or a two-character one
    (`__`), and drops empty parts. The one-character separator is what maximises the
    segment count: `s` segments need `s - 1` separators and at least one character each,
    so `2s - 1 <= key_chars` and therefore `s <= (key_chars + 1) // 2`. `concept_of` never
    puts the LEAF in the qualifier, so at most `s - 1` segments can ever join it, and any
    larger value slices the same list and yields a byte-identical key.

    A bound that follows from the length budget is worth having because the alternative is
    a literal: the previous ceiling was 8, which admitted five values that no key reachable
    through this library can distinguish and refused none that it can.
    `test_the_published_qualifier_bound_is_exactly_reachable` constructs the extremal key
    and searches for one that beats it.
    """
    if key_chars < 1:
        return 0
    return (key_chars + 1) // 2 - 1


# =============================================================================
# GROUPING
# =============================================================================


@dataclass(frozen=True, slots=True)
class GroupingPolicy:
    """
    How loosely two columns are allowed to be called the same concept.

    `qualifier_segments` is the dial that matters, and THE DEFAULT IS 1 BECAUSE 0 WAS
    MEASURED AND FAILED -- see this module's docstring for the numbers and the search
    behind them. 1 puts the nearest declared parent segment in the key, so two columns are
    one concept only when they share a leaf AND the record they hang off; 0 is the leaf
    alone, the loosest key and the one the roadmap proposes, which on a repeated-leaf
    schema merges 87 columns spanning 29 answers and reports the result as a contradiction.

    Raising it further trades recall for precision and neither direction is free. Values
    above the deepest key in the request are INERT rather than wrong: `concept_of` slices
    the segment list, so once the qualifier is the whole parent path a larger number
    changes nothing. `max_qualifier_segments` derives the point past which that is true
    for any key of a given length, and is what a publisher should bound this at.

    `include_data_type` costs almost nothing and refuses to merge a date with the string
    that spells it. Its value is UNMEASURED: removing it moves no number on any generated
    profile. `order_sensitive` keeps token ORDER in the key, which separates `STATUS_CODE`
    from `CODE_STATUS`; off by default because the class word is in the key separately and
    carries most of that distinction already.

    `min_group_size` is 2 because a group of one is not evidence about anything.
    """

    qualifier_segments: int = 1
    include_data_type: bool = True
    order_sensitive: bool = False
    min_group_size: int = 2

    def __post_init__(self) -> None:
        if self.qualifier_segments < 0:
            raise ValueError(f"qualifier_segments must be >= 0, got {self.qualifier_segments}")
        if self.min_group_size < 2:
            raise ValueError(
                f"min_group_size must be >= 2, got {self.min_group_size}: a group of one "
                f"column is not evidence that two columns disagree."
            )


@dataclass(frozen=True, slots=True)
class ConceptGroup:
    """Columns this policy calls one concept, in the order they were sent."""

    concept: str
    paths: tuple[str, ...]


def concept_of(field: SchemaField, policy: GroupingPolicy) -> str:
    """
    The concept label this policy gives one column.

    Printable and stable: it is emitted on the wire, so it has to be something a reviewer
    can read in a ticket and something a diff can compare. The pieces are separated by a
    character that cannot occur in a normalised token, so two different keys cannot
    render as one label.
    """
    key = result_key(field)
    segments = path_segments(key)
    leaf_tokens = normalise_tokens(segments[-1]) if segments else ()

    qualifier: tuple[str, ...] = ()
    if policy.qualifier_segments:
        # The NEAREST parents, not the outermost: a column's immediate record says more
        # about which concept it is than the namespace at the root of the schema.
        parents = segments[:-1][-policy.qualifier_segments :]
        qualifier = tuple(token for parent in parents for token in normalise_tokens(parent))

    ordered = leaf_tokens if policy.order_sensitive else tuple(sorted(leaf_tokens))
    class_word = leaf_tokens[-1] if leaf_tokens else ""
    data_type = field.data_type.value if policy.include_data_type else ""
    return "|".join((" ".join(qualifier), " ".join(ordered), class_word, data_type))


def group_by_concept(
    fields: Sequence[SchemaField],
    policy: GroupingPolicy | None = None,
) -> tuple[ConceptGroup, ...]:
    """
    Partition columns into concepts, keeping only the groups with something to say.

    Groups come back in the order their FIRST member was sent, and members in the order
    they were sent, so two identical requests produce the same list. A column that shares
    its concept with nothing else is not returned at all: it cannot disagree with anyone.
    """
    policy = policy or GroupingPolicy()
    buckets: dict[str, list[str]] = {}
    for field in fields:
        buckets.setdefault(concept_of(field, policy), []).append(result_key(field))
    return tuple(
        ConceptGroup(concept=concept, paths=tuple(paths))
        for concept, paths in buckets.items()
        if len(paths) >= policy.min_group_size
    )


# =============================================================================
# CONSISTENCY
# =============================================================================


class Agreement(str, Enum):
    """Whether a group's columns received the same answer."""

    AGREE = "AGREE"
    DISAGREE = "DISAGREE"
    # Fewer than two members of the group have an answer at all, so there is nothing to
    # agree or disagree about. Distinct from AGREE: one answer and five blanks is not
    # five columns confirming each other.
    UNDECIDED = "UNDECIDED"


@dataclass(frozen=True, slots=True)
class ConsistencyFinding:
    """
    One concept, the answers its columns got, and whether they contradict each other.

    `majority_answer` IS NOT AN INSTRUCTION. It is the modal answer within the group,
    published so a reviewer can see where the weight of evidence sits, and None when no
    single answer holds a plurality. Nothing in this library applies it.
    """

    concept: str
    paths: tuple[str, ...]
    answers: tuple[tuple[str, str | None], ...]
    distinct_answers: int
    agreement: Agreement
    majority_answer: str | None
    majority_count: int


def assess_consistency(
    groups: Sequence[ConceptGroup],
    answers: Mapping[str, str | None],
) -> tuple[ConsistencyFinding, ...]:
    """
    Compare each group's rank-1 answers.

    `answers` maps a result key to the governance id its rank-1 candidate would confer,
    or None for a column that matched nothing. A None is NOT counted as an answer that
    disagrees: a column with no candidate is silent, not contradictory, and counting it
    would report a disagreement on a group where only one column was answered at all.
    """
    findings: list[ConsistencyFinding] = []
    for group in groups:
        pairs = tuple((path, answers.get(path)) for path in group.paths)
        decided = [answer for _path, answer in pairs if answer]
        counts = Counter(decided)
        distinct = len(counts)

        if len(decided) < 2:
            agreement = Agreement.UNDECIDED
        elif distinct == 1:
            agreement = Agreement.AGREE
        else:
            agreement = Agreement.DISAGREE

        majority: str | None = None
        majority_count = 0
        if counts:
            top = counts.most_common()
            # A tie for the mode is not a majority, and saying it is would put a coin
            # toss on a governance artifact. Ordered by count then by id so the tie test
            # itself does not depend on dict ordering.
            top.sort(key=lambda item: (-item[1], item[0]))
            if len(top) == 1 or top[0][1] > top[1][1]:
                majority, majority_count = top[0]

        findings.append(
            ConsistencyFinding(
                concept=group.concept,
                paths=group.paths,
                answers=pairs,
                distinct_answers=distinct,
                agreement=agreement,
                majority_answer=majority,
                majority_count=majority_count,
            )
        )
    return tuple(findings)


# =============================================================================
# CONTRAST
# =============================================================================


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """
    One weighted signal, named by whoever is going to publish it.

    The name is the CALLER'S, not this module's. The wire vocabulary belongs to the
    presentation layer and the attribute name belongs to the score breakdown; pairing
    them is a single table in one place there, and a second copy of that pairing here is
    exactly the drift that produces a self-consistently wrong audit surface.
    """

    name: str
    score_attr: str
    weight: float


class Separation(str, Enum):
    """Whether the top two candidates are actually apart."""

    SEPARATED = "SEPARATED"
    # The margin is at or below the resolution of the published numbers. The order came
    # from the matcher's own sort, not from anything a reviewer can see.
    TIED = "TIED"


@dataclass(frozen=True, slots=True)
class SignalDifference:
    """One signal's contribution to the margin between the top two candidates."""

    signal: str
    top_score: float
    runner_up_score: float
    delta: float
    weight: float
    weighted_delta: float
    # False when the two scores are within the resolution of the published numbers, in
    # which case this signal is not reported as a cause of anything.
    separating: bool
    # True when removing this signal's contribution would leave the runner-up level with
    # or ahead of the winner. Arithmetic, not judgement, and possibly true of none of
    # them: a margin can be carried collectively.
    deciding: bool


@dataclass(frozen=True, slots=True)
class Contrast:
    """
    Rank 1 against rank 2: what separated them, by how much, and what decided it.

    `signal_gap` is the sum of the weighted differences and `confidence_gap` is the
    difference of the two published confidences. They are two routes to one number and
    the publisher checks them against each other, exactly as it checks that a candidate's
    components reproduce its confidence -- an explanation whose arithmetic does not close
    is worse than none, because it is the one that gets used as evidence.
    """

    top_governance_id: str
    runner_up_governance_id: str
    top_confidence: float
    runner_up_confidence: float
    confidence_gap: float
    signal_gap: float
    differences: tuple[SignalDifference, ...]
    largest_difference: str | None
    deciding_signals: tuple[str, ...]
    separation: Separation
    resolution: float
    # Facts about the two ENTRIES rather than about the scoring, and usually the ones
    # that settle a review. Read from the entries' own codes, not from the resolved
    # `governance` on the result, so a rank-1 REJECT -- which carries no class by
    # design -- does not read as "the two entries are classified differently".
    governance_differs: bool
    domain_differs: bool


def _score(breakdown: object, attr: str) -> float | None:
    value = getattr(breakdown, attr, None)
    return None if value is None else float(value)


def contrast_top_two(
    matches: Sequence[MatchResult],
    signals: Sequence[SignalSpec],
    precision: int = 6,
) -> Contrast | None:
    """
    Contrast the first two matches, or None when there is no runner-up to contrast.

    None is a real answer and the caller must publish it as one: a field with a single
    candidate has nothing it lost to, and that is different from a field this pass forgot.

    THE NUMBERS ARE ROUNDED TO `precision` BEFORE ANYTHING IS SUBTRACTED, deliberately.
    A reviewer holding the response can subtract two published component scores and must
    get the delta printed beside them; computing the contrast at full precision and
    publishing it at six decimals would disagree with that subtraction in the last place
    and read as the tool getting its own arithmetic wrong. `resolution` is the smallest
    difference these numbers can express, and no cause is named below it.

    `matches` is taken in the order the matcher returned it, which is the same ordering
    the rest of the response already trusts for rank 1 and for truncation to `top_k`.
    Introducing a second, independent notion of "first" here would be a way for one
    response to disagree with itself about which candidate won.

    Every signal must be readable off both breakdowns. A missing one raises rather than
    defaulting to zero: a zero delta on a component that could not be read is a report
    that two candidates were identical on a signal nobody looked at.
    """
    if len(matches) < 2:
        return None

    top, runner_up = matches[0], matches[1]
    resolution = 10.0**-precision

    top_confidence = round(float(top.final_confidence), precision)
    runner_up_confidence = round(float(runner_up.final_confidence), precision)
    confidence_gap = round(top_confidence - runner_up_confidence, precision)

    differences: list[SignalDifference] = []
    signal_gap = 0.0
    for spec in signals:
        top_score = _score(top.score_breakdown, spec.score_attr)
        runner_up_score = _score(runner_up.score_breakdown, spec.score_attr)
        if top_score is None or runner_up_score is None:
            raise ValueError(
                f"the {spec.name!r} component is missing from a score breakdown, so a "
                f"contrast would report the two candidates as identical on a signal that "
                f"was never read."
            )
        top_score = round(top_score, precision)
        runner_up_score = round(runner_up_score, precision)
        delta = round(top_score - runner_up_score, precision)
        weight = round(float(spec.weight), precision)
        weighted_delta = round(delta * weight, precision)
        signal_gap += weighted_delta
        differences.append(
            SignalDifference(
                signal=spec.name,
                top_score=top_score,
                runner_up_score=runner_up_score,
                delta=delta,
                weight=weight,
                weighted_delta=weighted_delta,
                separating=abs(delta) > resolution,
                deciding=False,
            )
        )

    signal_gap = round(signal_gap, precision)
    separation = Separation.SEPARATED if confidence_gap > resolution else Separation.TIED

    # Decisiveness is only meaningful while there IS a margin. On a tie nothing decided
    # the order, and marking a signal as deciding one would be dressing the matcher's
    # sort order up as a finding.
    if separation is Separation.SEPARATED:
        differences = [
            replace(
                difference,
                deciding=(
                    difference.weighted_delta > 0.0
                    and confidence_gap - difference.weighted_delta <= resolution
                ),
            )
            for difference in differences
        ]

    # Largest first, and the sort is total: ties fall back to the order the caller
    # declared the signals in, so two identical requests order this list identically.
    order = {spec.name: position for position, spec in enumerate(signals)}
    differences.sort(key=lambda d: (-abs(d.weighted_delta), order[d.signal]))

    separators = [d for d in differences if d.separating]
    largest = separators[0].signal if separators and separation is Separation.SEPARATED else None
    deciding = tuple(d.signal for d in differences if d.deciding and d.separating)

    entries = (top.dictionary_entry, runner_up.dictionary_entry)
    return Contrast(
        top_governance_id=str(top.governance_id or top.dictionary_entry.id),
        runner_up_governance_id=str(runner_up.governance_id or runner_up.dictionary_entry.id),
        top_confidence=top_confidence,
        runner_up_confidence=runner_up_confidence,
        confidence_gap=confidence_gap,
        signal_gap=signal_gap,
        differences=tuple(differences),
        largest_difference=largest,
        deciding_signals=deciding,
        separation=separation,
        resolution=resolution,
        governance_differs=entries[0].governance_code != entries[1].governance_code,
        domain_differs=entries[0].domain != entries[1].domain,
    )
