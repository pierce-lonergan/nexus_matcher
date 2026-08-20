"""
nexus_matcher.application.use_cases.match_schema | Layer: APPLICATION
Core schema matching use case - the main orchestrator.

## Relationships
# DEPENDS_ON → domain/ports/* :: all port interfaces
# DEPENDS_ON → domain/models/* :: domain models
# DEPENDS_ON → infrastructure/config :: configuration
# USED_BY    → presentation/api :: REST endpoints
# USED_BY    → presentation/cli :: CLI commands
# USED_BY    → external :: library API

## Attributes
# Security: No direct access to sensitive data, uses ports
# Performance: Caches embeddings, batch processing
# Reliability: Graceful degradation if components unavailable
"""

from __future__ import annotations

import dataclasses
import json
import math
import operator
import re
import sys
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from nexus_matcher.core.fusion import fuse_linear_ids
from nexus_matcher.domain.governance import GovernanceVocabulary
from nexus_matcher.domain.models.entities import (
    DictionaryEntry,
    MatchingSession,
    MatchResult,
    Schema,
    SchemaField,
)
from nexus_matcher.domain.ports import (
    DictionaryLoader,
    EmbeddingProvider,
    Reranker,
    SchemaParser,
    SparseRetriever,
    VectorStore,
)
from nexus_matcher.domain.ports.dictionary_loader import ColumnMapping, LoadStatistics
from nexus_matcher.domain.ports.retrieval import RerankCandidate, SparseDocument
from nexus_matcher.domain.ports.vector_store import SearchResult, VectorDocument, VectorStoreConfig
from nexus_matcher.domain.services.abbreviation import (
    AbbreviationDictionary,
    AbbreviationExpander,
)
from nexus_matcher.domain.services.alias_generation import expand_dictionary
from nexus_matcher.domain.services.context_enricher import (
    HIERARCHY_SEPARATOR,
    ContextEnricher,
)
from nexus_matcher.domain.services.domain_hierarchy import DomainMatcher
from nexus_matcher.shared.types.base import (
    EntityId,
    MatchDecision,
    PerformanceMetrics,
    ScoreBreakdown,
)

# Optional acceleration for the per-candidate string similarity on the hot path.
try:  # pragma: no cover - trivial import guard
    from rapidfuzz.distance import Levenshtein as _LEVENSHTEIN
except ImportError:  # pragma: no cover
    _LEVENSHTEIN = None  # type: ignore[assignment]

# C-level sort keys. These run once per candidate per field, where a lambda's frame setup
# is measurable against the handful of float operations it guards.
_second = operator.itemgetter(1)
_first = operator.itemgetter(0)


# =============================================================================
# MATCHING CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MatchingConfig:
    """Configuration for the matching process."""

    # Retrieval
    dense_top_k: int = 100
    sparse_top_k: int = 100
    # Weight for dense scores in linear fusion. 0.9 is the measured optimum for P@1 on
    # the combined BIRD+OMOP benchmark (see _fuse_results and benchmarks/exp_fusion.py):
    # dense=0.5 -> 0.657, 0.7 -> 0.682, 0.8 -> 0.686, 0.9 -> 0.702.
    # The lexical arm still earns its 10%: sparse-only reaches 0.542 P@1 on its own and
    # rescues exact-token matches that the embedding model misses.
    fusion_alpha: float = 0.90

    # Reranking
    colbert_top_k: int = 50
    cross_encoder_top_k: int = 20

    # Scoring weights
    semantic_weight: float = 0.70
    lexical_weight: float = 0.05
    edit_distance_weight: float = 0.05
    type_weight: float = 0.05
    domain_weight: float = 0.15

    # Thresholds
    #
    # auto_approve_threshold is CALIBRATED, not arbitrary. Measured on the combined
    # BIRD+OMOP benchmark (688 labelled fields, benchmarks/exp_calibration.py):
    #
    #   threshold   coverage   auto-approve precision
    #   0.85          20.8%       0.916
    #   0.87          12.4%       0.953      <- default: ~95% precision target
    #
    # These are for the shipped configuration (dictionary_alias_count = 0). Turning
    # aliasing on also caps achievable auto-approve precision at ~91%, because max-pool
    # reports an entry's BEST fabricated spelling and a wrong entry can therefore look
    # confident -- a second, independent reason to leave it off.
    #
    # Auto-approving a wrong mapping is far more expensive than sending a field to
    # review, so the default targets ~95% auto-approve precision and accepts low
    # coverage; everything below the bar goes to a human rather than being guessed.
    # If your cost balance differs, 0.85 buys +8 points of coverage for -4 of precision.
    #
    # IMPORTANT: these numbers move with the retriever AND with the benchmark. They were
    # re-derived after a leakage fix in the OMOP split (its business name used to be
    # derived from the field name, making the task string-identity and inflating every
    # number downstream). Improving retrieval also SHIFTS THE SCORE DISTRIBUTION UPWARD,
    # pushing more candidates over a fixed bar and LOWERING precision -- that happened
    # twice during tuning. Re-run exp_calibration.py after any change to the model,
    # fusion, query representation, or benchmark.
    auto_approve_threshold: float = 0.87
    review_threshold: float = 0.50
    min_confidence_gap: float = 0.10

    # The absolute-score floor beneath which a field is reported NO_MATCH.
    #
    # OFF BY DEFAULT, AND THAT IS NOT A PLACEHOLDER. A threshold is a statement about a
    # score distribution, and the distribution is a property of the caller's dictionary
    # and the caller's field names -- neither of which this library has seen. Shipping a
    # number here would be inventing a calibration for somebody else's corpus, which is
    # the same mistake as shipping a taxonomy. `auto_approve_threshold` above at least
    # names the corpus it was measured on; there is no corpus at all behind this one.
    #
    # WHY IT IS NEEDED. `review_threshold` cannot express "nothing matched". The rank-1
    # confidence has a structural floor of `semantic_weight * fusion_alpha` = 0.63 (see
    # `minimum_achievable_confidence`), which sits above `review_threshold` = 0.50, so a
    # rank-1 candidate can never be REJECT on score alone -- every field comes back at
    # least REVIEW, however irrelevant its best candidate is. No setting of
    # `review_threshold` recovers the missing state, because the floor moves with the
    # WEIGHTS and the thresholds do not move with it.
    #
    # WHAT IT IS COMPARED AGAINST. `ScoreBreakdown.absolute_cosine` on rank 1: the raw
    # dense-retrieval score, which has no per-field normalisation and therefore no floor.
    # Under the shipped wiring that is a cosine similarity in [-1, 1]; see
    # `NexusMatcher.absolute_score_metric` for when it is not.
    #
    # HOW TO CHOOSE ONE. Match a labelled sample of YOUR fields against YOUR dictionary,
    # look at the absolute score of the rank-1 candidate for the fields you know have no
    # correct entry, and put the floor below the lowest absolute score among the fields
    # that DO. There is no substitute for that measurement, and a value copied from
    # another deployment is a guess wearing a number.
    #
    # `None` means off. `0.0` means ON with a floor at zero, which is a different thing
    # and refuses only candidates the retriever scored at or below zero.
    absolute_score_floor: float | None = None

    # Results
    results_per_field: int = 5

    # Whether to run abbreviation expansion over the QUERY text.
    #
    # Off by default, but the justification is WEAKER than it used to read here. The old
    # comment cited "0.691 -> 0.671" from an unpaired comparison against a 793-pair
    # dataset that is no longer the committed one. Re-measured paired on the full
    # committed combined corpus (688/688, exact McNemar):
    #     OFF 0.5814   ON 0.5654   delta -1.60 pts   b=24 c=13  p=0.099
    # Negative point estimate, NOT a significant regression. Read this as "inconclusive,
    # point estimate negative" -- not as "expansion is proven harmful". The default is
    # still right: an unproven change to the largest accuracy factor in the pipeline is
    # not one to enable for everybody by default.
    #
    # A SECOND paired full-corpus measurement of a NARROWER quantity exists and reads
    # -1.16 pts, p = 0.2559 (15 gained, 23 lost, 38 discordant). It is dense retrieval
    # only, on `parent + field` text, with no fusion, no lexical or type signals and no
    # decision layer -- benchmarks/exp_governed_abbrev.py, condition
    # `guessing_on_original`. Both are honest; they measure different things, and the
    # -1.60 above is the one that describes this flag end to end. Recorded so the two do
    # not drift into being quoted as disagreeing measurements of the same quantity.
    #
    # The mechanism behind the negative point estimate is the DICTIONARY, not the idea.
    # `AbbreviationExpander.default()` carries a generic hardcoded list that fires on
    # ambiguous short tokens and asserts the wrong long form -- expanding "ST" to "state"
    # inside a street field, "NO" to "number" inside a negation flag. A wrong expansion
    # is worse than no expansion, because the query only gets one vector.
    #
    # Callers whose schemas follow a GOVERNED abbreviation standard are the case this
    # flag exists for, and the whole path is already plumbed -- pass your own catalog:
    #
    #     catalog = {"txn": "transaction", "amt": "amount"}   # your approved list
    #     NexusMatcher(
    #         ...,
    #         abbreviation_expander=AbbreviationExpander(
    #             AbbreviationDictionary.from_dict(catalog)
    #         ),
    #         config=MatchingConfig(expand_query_abbreviations=True),
    #     )
    #
    # The expander is exact-lookup with passthrough: a token absent from the catalog is
    # left alone, so a catalog is only as dangerous as its WRONG rows. Before enabling
    # this, measure your catalog's wrong-rate against your own fields -- the fraction of
    # rows asserting the wrong long form IN YOUR SCHEMA'S CONTEXT -- and watch for
    # colliding short forms ("ST" as both state and street). Then re-run
    # benchmarks/exp_calibration.py: a large P@1 shift moves the score distribution and
    # can LOWER auto-approve precision at a fixed threshold.
    #
    # The expander is also used to enrich the DICTIONARY side, where entries are longer
    # and more predictable; that use is unaffected by this flag.
    #
    # docs/guides/governed_abbreviations.md is the full write-up: the wiring above in
    # runnable form, the measured shape of the trade (recovery tracks catalog coverage
    # roughly linearly; a 5%-wrong catalog still recovers ~91%; break-even near 75%
    # wrong; a 100%-wrong catalog is ~7 points WORSE than doing nothing), and the reason
    # those recovery figures are an UPPER BOUND -- they come from a synthetic experiment
    # in which the abbreviation was generated from the gold text, so expanding it
    # reconstructs a near-identical string (683/688 and 1556/1556 caselessly identical).
    # `from_config()` cannot reach this flag's catalog; it takes no expander.
    expand_query_abbreviations: bool = False

    # Index-time enrichment: how many fabricated technical spellings to index per
    # dictionary entry, max-pooled at query time. See domain/services/alias_generation.
    #
    # This is the inverse of query expansion, and unlike query expansion it works: a
    # wrong alias loses the max-pool instead of corrupting the single query vector.
    # OFF BY DEFAULT. It helps on a small dictionary and is CATASTROPHIC on a large one.
    #
    # Measured end-to-end, aliases=0 vs aliases=6 (benchmarks/results/exp_alias_scale.json):
    #     entries    off      on       delta
    #        688   0.5814  0.6003     +1.9
    #     10,000   0.5044  0.3677    -13.7
    #     30,000   0.4666  0.2791    -18.8
    #
    # The gain inverts between 688 and 10k entries and then keeps falling. The mechanism
    # is inherent to max-pooling, not a tuning problem: every DISTRACTOR also receives
    # `dictionary_alias_count` extra chances to beat the gold entry, so alias noise grows
    # with corpus size while the useful signal does not. A 6x larger index is a 6x larger
    # opportunity for a wrong entry to land inside the ~0.002 similarity margin that
    # separates right from wrong on this task.
    #
    # Enable ONLY if your dictionary is genuinely small (order 1000 entries) and you have
    # re-measured on your own data. On the 361-entry BIRD split it is worth +3.9 points.
    # Do not enable it on an enterprise glossary.
    dictionary_alias_count: int = 0

    @property
    def minimum_achievable_confidence(self) -> float:
        """
        The structural floor: the lowest `final_confidence` a rank-1 match can carry.

        For the shipped configuration this is **0.63**, and knowing it is not optional
        trivia. `MatchingSession.get_low_confidence_fields()` shipped with a default
        threshold of 0.6 -- below this floor -- so it returned an empty list on every
        schema ever matched, and told a governance lead there was nothing to review on
        a schema where nothing was trustworthy (DX-001, tests/museum/NM-0027). The floor
        was folklore; nothing computed it, so nothing could contradict the number.

        Why it exists. `fused_retrieval_score` is min-max normalised over the candidates
        retrieved for one field, so the best candidate's dense score normalises to
        exactly 1.0, and `fuse_linear_ids` renormalises the two arm weights to sum to 1 --
        so that candidate's fused score is at least `fusion_alpha`. It carries
        `semantic_weight` of the final confidence, and the other four signals cannot be
        negative. Hence:

            floor = semantic_weight * fusion_alpha = 0.70 * 0.90 = 0.63

        Move `fusion_alpha` and the floor moves with it, which is the proof that this
        score is rank-relative rather than a similarity: at fusion_alpha=0.5 the rank-1
        fused score drops to ~0.45 for the SAME embeddings.

        Two preconditions, stated because a bound that quietly does not hold is worse
        than no bound:

        1. **No reranker.** A reranker REPLACES the fused score with a squashed reranker
           score, which has no floor. `NexusMatcher.minimum_achievable_confidence`
           returns None when one is wired, rather than reporting a number that is wrong.
        2. **At least two distinct dense scores.** Min-max maps a constant map to all
           zeros, so a one-entry dictionary (or a perfect tie across every candidate)
           produces a fused score of 0.0 and there is no floor at all.

        Clamped to [0, 1] because `_weighted_confidence` clamps too: with a weight set
        that oversums, confidences pile up at 1.0 and so does the floor.
        """
        return min(max(self.semantic_weight * self.fusion_alpha, 0.0), 1.0)


def _load_matching_config(source: MatchingConfig | str | Path | None) -> MatchingConfig:
    """
    Coerce a config argument into a MatchingConfig.

    Accepts an instance (returned as-is), a path to a JSON or TOML file, or None for the
    calibrated defaults.

    An UNKNOWN KEY IS AN ERROR rather than a warning. Every field here is a tuned number
    whose default was measured; a typo like `auto_approve_treshold` would otherwise be
    dropped in silence and leave the user believing they had raised the bar while the
    matcher went on auto-approving at 0.87. A loud failure at startup is far cheaper than
    a mis-governed field discovered in an audit.
    """
    if source is None:
        return MatchingConfig()
    if isinstance(source, MatchingConfig):
        return source

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Matching config not found: {path}")

    if path.suffix.lower() == ".toml":
        # tomllib is stdlib from 3.11 only, and this package supports 3.10.
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - version-dependent
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"Reading {path.name} needs a TOML parser on Python "
                    f"{sys.version_info.major}.{sys.version_info.minor}. Either install "
                    f"one (pip install tomli) or use a JSON config file, which needs "
                    f"nothing extra."
                ) from None
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    # Tolerate a wrapping table so one project file can hold several sections.
    if "matching" in data and isinstance(data["matching"], dict):
        data = data["matching"]

    known = {f.name for f in dataclasses.fields(MatchingConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"Unknown matching config option(s) in {path}: {', '.join(unknown)}. "
            f"Valid options: {', '.join(sorted(known))}"
        )
    return MatchingConfig(**data)


def _load_governance_vocabulary(
    source: GovernanceVocabulary | str | Path | None,
) -> GovernanceVocabulary:
    """
    Coerce a `governance=` argument into a vocabulary.

    None becomes `empty()` rather than a permissive stand-in. An installation that
    configured no vocabulary defines no codes, and `_index_dictionary` refuses entries
    carrying codes it cannot resolve -- so "I forgot to wire the vocabulary" surfaces at
    index time with the fix in the message, instead of as matches whose governance is
    quietly None.
    """
    if source is None:
        return GovernanceVocabulary.empty()
    if isinstance(source, GovernanceVocabulary):
        return source
    return GovernanceVocabulary.from_json(source)


# The `ColumnMapping` fields that name a column `ingest.load_entries` also reads, as
# {ColumnMapping attribute: load_entries field}. `parent_table`, `sample_values` and
# `synonyms` are absent because `load_entries` does not map them -- and the second read is
# only ever asked for a governance code, so mapping them would buy nothing and, for
# `synonyms`, be refused outright (it is a frozenset, and embedding it unordered makes one
# process's vectors differ from another's).
_GOVERNANCE_READ_COLUMNS: tuple[tuple[str, str], ...] = (
    ("id_column", "id"),
    ("business_name_column", "business_name"),
    ("logical_name_column", "logical_name"),
    ("definition_column", "definition"),
    ("data_type_column", "data_type"),
    ("domain_column", "domain"),
    ("protection_level_column", "protection_level"),
)


def _governance_read_columns(column_mapping: ColumnMapping | None) -> dict[str, str] | None:
    """
    Translate a `ColumnMapping` into the `columns=` overrides `load_entries` takes.

    None passes through as None, which is the common case and means "infer from the
    header" -- the same alias table the loader's own `ColumnMapping.detect` uses, so the
    two reads agree by construction.

    When the caller DID name their columns, the second read must be told, or it infers a
    different business-name column and the join below sees two glossaries that disagree
    about which entries exist. It is a faithful translation, defaults included: the caller
    gave those literals to the loader and the loader used them, so the second read using
    them too is the only reading that keeps the two files describing the same rows.

    The protection CODE column is deliberately not in the mapping and cannot be: the port
    has no field for it, so it is always resolved from the header by alias.
    """
    if column_mapping is None:
        return None
    named = {
        field_name: getattr(column_mapping, attribute)
        for attribute, field_name in _GOVERNANCE_READ_COLUMNS
    }
    return {field_name: column for field_name, column in named.items() if column}


def _merge_governance(
    entries: Sequence[DictionaryEntry],
    governed: Sequence[DictionaryEntry],
    source: str | Path,
) -> list[DictionaryEntry]:
    """
    Copy each governed entry's protection code onto the loader's entry for the same row.

    The two lists come from two readers over one file, both in source order, so they are
    joined on `business_name` through a queue per name: duplicates -- a real glossary has
    them -- keep their order of appearance instead of collapsing onto one another.

    They are NOT joined on `id`. The two readers derive an id differently when the source
    has no id column (`auto_1`, versus a digest of the identifying content), so an id join
    silently matches nothing on exactly the glossaries that are commonest.

    A name with no counterpart REFUSES the load. It means the two readers disagree about
    which rows became entries, and the alternative -- attaching no code to that entry --
    is once again a class that reads as "this entry has no class". A wrong governance
    answer must not be reachable by a code path that had every chance to know better.
    """
    pending: dict[str, deque[DictionaryEntry]] = defaultdict(deque)
    for entry in governed:
        pending[entry.business_name].append(entry)

    merged: list[DictionaryEntry] = []
    for entry in entries:
        queue = pending.get(entry.business_name)
        if not queue:
            raise ValueError(_governance_join_failure(entry, source))
        merged.append(_with_governance_of(entry, queue.popleft()))
    return merged


def _governance_join_failure(entry: DictionaryEntry, source: str | Path) -> str:
    return (
        f"Reading {source} for its protection codes produced no row for "
        f"{entry.business_name!r}, which the dictionary loader did produce, so the two "
        f"readings of this file disagree about which entries exist and no code can be "
        f"attached to it.\n"
        f"Pass an explicit column_mapping=ColumnMapping(...) naming the business-name "
        f"column, or load the glossary with nexus_matcher.application.ingest.load_entries "
        f"and index it directly."
    )


def _with_governance_of(entry: DictionaryEntry, governed: DictionaryEntry) -> DictionaryEntry:
    """
    One entry, plus the code the governed reading gave it, and nothing else changed.

    `governance_code_raw` and `governance_problems` travel with it when they exist. They
    only exist under `governance_strict=False`, which is the mode that loads a defective
    glossary anyway -- and there an entry whose code was REFUSED would otherwise carry
    `governance_code=None`, indistinguishable from a row that never declared one. The
    reason a code is missing is the one thing that must survive that mode.
    """
    carried = {
        key: governed.source_metadata[key]
        for key in ("governance_code_raw", "governance_problems")
        if key in governed.source_metadata
    }
    if governed.governance_code is None and not carried:
        return entry
    return dataclasses.replace(
        entry,
        governance_code=governed.governance_code,
        source_metadata={**entry.source_metadata, **carried} if carried else entry.source_metadata,
    )


# =============================================================================
# TOKENIZATION
# =============================================================================

# Splits identifiers on case transitions, digit boundaries and any punctuation, so
# "NumTstTakr", "num_tst_takr" and "enroll12" all yield comparable token sets.
_IDENT_SPLIT = re.compile(
    r"""
    (?<=[a-z0-9])(?=[A-Z])      # camelCase   -> camel | Case
  | (?<=[A-Z])(?=[A-Z][a-z])    # HTTPResponse -> HTTP | Response
  | (?<=[A-Za-z])(?=[0-9])      # enroll12    -> enroll | 12
  | (?<=[0-9])(?=[A-Za-z])      # 12grade     -> 12 | grade
    """,
    re.VERBOSE,
)


def _tokenize_identifier(name: str) -> set[str]:
    """Split a schema identifier into a lowercase token set."""
    if not name:
        return set()
    spaced = _IDENT_SPLIT.sub(" ", name)
    return {t for t in re.split(r"[^0-9A-Za-z]+", spaced.lower()) if t}


# =============================================================================
# SCORE NORMALIZATION
# =============================================================================


# The five scored signals, in weight order: fused retrieval, lexical, edit distance,
# type, domain. This is NOT ScoreBreakdown's field order -- that dataclass declares two
# reranker fields between type and domain -- so widening one into the other goes through
# `_breakdown`, never through positional expansion.
#
# Signal 0 is weighted by `MatchingConfig.semantic_weight`, whose name is a survivor of
# the same confusion as the old `ScoreBreakdown.semantic_score`: it weights the fused
# RETRIEVAL score. The config field keeps its name because it is a calibrated knob that
# users set in JSON/TOML files, and renaming it would break those files silently.
#
# Carried as a plain tuple on the hot path so a candidate that never reaches the result
# list does not cost a dataclass; see _match_field.
Signals = tuple[float, float, float, float, float]


def _signal_weights(config: MatchingConfig) -> Signals:
    """The configured weight of each signal, in signal order."""
    return (
        config.semantic_weight,
        config.lexical_weight,
        config.edit_distance_weight,
        config.type_weight,
        config.domain_weight,
    )


def _breakdown(signals: Signals, absolute_cosine: float | None = None) -> ScoreBreakdown:
    """
    Widen the five signals into the public ScoreBreakdown.

    Keyword arguments are not stylistic here: ScoreBreakdown declares `colbert_score` and
    `cross_encoder_score` between the type and domain fields, so positional expansion
    would quietly land the domain score in `colbert_score` and report a reranker result
    that never ran.

    `absolute_cosine` is carried ALONGSIDE the five signals and never inside them: it is
    reported to the caller and is deliberately not an input to the weighted confidence,
    so adding it cannot move a ranking. See `_match_field` for where it comes from.
    """
    sem, lex, edit, type_, domain = signals
    return ScoreBreakdown(
        fused_retrieval_score=sem,
        lexical_score=lex,
        edit_distance_score=edit,
        type_compatibility_score=type_,
        domain_score=domain,
        absolute_cosine=absolute_cosine,
    )


def _weighted_confidence(signals: Signals, weights: Signals) -> float:
    """
    The final confidence: a weighted sum of the five signals, clamped to [0, 1].

    One definition, two callers -- the per-candidate ranking loop, which holds signals as
    a tuple, and `_calculate_final_confidence`, which holds them as a ScoreBreakdown.
    """
    sem, lex, edit, type_, domain = signals
    w_sem, w_lex, w_edit, w_type, w_domain = weights
    total = w_sem * sem + w_lex * lex + w_edit * edit + w_type * type_ + w_domain * domain
    return min(max(total, 0.0), 1.0)


def _squash_score(score: float) -> float:
    """
    Map an arbitrary reranker score into [0, 1].

    Scores already in [0, 1] (ColBERT cosine MaxSim, normalized rerankers) pass through
    unchanged so that well-behaved rerankers keep their calibration. Anything outside
    that range is assumed to be a raw logit and goes through a logistic sigmoid, which
    is monotonic -- so it never changes the candidate ORDER, only the magnitudes that
    feed the weighted confidence and the auto-approve threshold.

    A per-query min-max normalization was deliberately rejected here: it would force
    the top candidate to exactly 1.0 for every field, making confidence meaningless
    across fields and defeating the whole point of a calibrated threshold.
    """
    if 0.0 <= score <= 1.0:
        return float(score)
    # Guard against overflow warnings on extreme logits.
    if score < -30.0:
        return 0.0
    if score > 30.0:
        return 1.0
    return float(1.0 / (1.0 + math.exp(-score)))


# =============================================================================
# RESULT IDENTITY
# =============================================================================


def field_result_key(field: SchemaField) -> str:
    """
    The name a caller uses to look this field up in a match result.

    Results come back as a dict, so every field needs a handle -- and the handle has to
    be the string the CALLER supplied, not one we derived. The flattened parser
    deliberately rewrites `cust_addr__city` into the dotted path `cust.addr.city`,
    because recovering the parent path is worth +19.3 P@1; keyed by that path, a caller
    asking for their own column name got a KeyError from a result set that did contain
    their field, and a caller iterating the keys saw names their schema never used.

    `flattened_name` is that original string, and both flattened entry points set it.
    Every other parser -- raw Avro, JSON Schema, SQL DDL -- has no such name and keeps
    its dotted `full_path`, which is what those callers have always addressed.
    """
    flattened = field.source_metadata.get("flattened_name")
    if isinstance(flattened, str) and flattened:
        return flattened
    # __post_init__ guarantees full_path falls back to the bare field name.
    return field.full_path


# =============================================================================
# QUERY-SIDE SIGNAL CHANNEL
# =============================================================================

# The canonical name of each signal THIS library interprets. Everything else a caller
# sends through the channel is carried and ignored -- see `QuerySignals.from_mapping`.
#
# Published as constants rather than spelled out at each use, because the presentation
# layer documents these names on the wire and a divergence between "what the schema says
# we read" and "what we read" is one of the failures this channel exists to remove.
QUERY_SIGNAL_ABBREVIATIONS = "abbreviations"
QUERY_SIGNAL_ENTITY = "entity"
QUERY_SIGNAL_DOMAIN = "domain"

# Where a single field's own signals live, when they differ from the request's.
#
# NAMESPACED ON PURPOSE. `SchemaField.source_metadata` is a shared bag -- it already holds
# `flattened_name`, and `_calculate_domain_score` has always read a bare `domain` key out
# of it. Reading signals from bare keys would silently reinterpret data an existing caller
# already puts there, which is precisely the class of change that cannot be shown to be
# behaviour-preserving. One nested key can be.
QUERY_SIGNALS_METADATA_KEY = "query_signals"

# Every name the three signals answer to, canonical first, in the order a lookup tries
# them. Aliases exist because the channel's purpose is to let a deployment supply context
# it already has under the name it already calls it; refusing a request over a synonym
# would be the same 422 this design exists to remove, one level down. Precedence is
# declared rather than incidental: a caller who sends both `domain` and `namespace` meant
# the more specific one.
QUERY_SIGNAL_ALIASES: dict[str, tuple[str, ...]] = {
    QUERY_SIGNAL_ABBREVIATIONS: ("abbreviations", "abbreviation_overlay"),
    QUERY_SIGNAL_ENTITY: ("entity", "parent_record"),
    QUERY_SIGNAL_DOMAIN: ("domain", "domain_prior", "namespace"),
}

# Flattened for a caller (or a test) that wants to know what is read and what is merely
# carried, without walking the table above.
INTERPRETED_SIGNAL_NAMES: frozenset[str] = frozenset(
    name for names in QUERY_SIGNAL_ALIASES.values() for name in names
)


@dataclass(frozen=True)
class QuerySignals:
    """
    Caller-supplied, per-request context about the QUERY side of a match. AR-6.

    ## What this is

    A declared extension point, not three fields. Every retrieval signal this library has
    is derived from what the caller sent -- the field name, the path, the doc. This
    channel is for what the caller KNOWS and the library cannot derive: which live
    abbreviation catalog is authoritative today, which record a flattened column came out
    of, which business domain the schema belongs to. A deployment may put anything else in
    it; this library carries those keys and reads none of them.

    ## Three rules, and the reason for each

    1. **Every signal is optional, and absence is the shipped behaviour.** `EMPTY` is not
       a special case handled somewhere -- it is the object every existing call site gets,
       and `is_empty` short-circuits every code path this channel adds. A caller who sends
       nothing must get results identical to a caller on the previous release.

    2. **An unrecognised key is carried, never refused.** `extra="forbid"` on a request
       model is right for a field a typo can silently drop; it is wrong for an extension
       point, where it turns "this deployment knows something the library does not" into a
       422. Ignored keys are kept in `carried`, so a deployment that extends the matcher
       can read them and a diagnostic can say what was sent but not acted on.

    3. **A malformed value is ignored, not raised on.** These arrive from a live
       reference-data service. A feed that returns `null` for the overlay one morning must
       cost that request its overlay, not its answer.

    ## Where each signal is applied

    `abbreviations` is REQUEST-scoped: it is a catalog, and merging it per field would
    copy it once per column. `entity` and `domain` are per-FIELD facts with a per-request
    default -- one schema usually has one namespace, but a flattened export can carry
    columns from several parent records. A field overrides the request by putting the same
    key in its own `SchemaField.source_metadata`.

    ## What supplying nothing costs

    Nothing, and that is measured rather than asserted. The matcher at HEAD and this one,
    both with no signals, over the full committed corpus: 688 of 688 rank-1 entries
    identical, P@1 0.5814 on both -- and the same for `signals={}`, for an overlay of `{}`,
    and for a map of keys this library does not recognise. The reference side is a
    different BUILD of the code rather than a recorded expectation, so the comparison
    cannot pass by both sides drifting together.

    The per-signal measurements live beside the code that applies each one --
    `_request_expander`, `_with_entity_context`, `_calculate_domain_score` -- because a
    number kept away from the decision it justifies is a number nobody re-checks.
    """

    # A READ-ONLY empty mapping by default, not a fresh dict. `EMPTY_QUERY_SIGNALS` below
    # is a module-level singleton every no-signal call receives, and a mutable default on
    # a shared object is one stray `.update()` away from giving every request in the
    # process an overlay nobody sent. Per-request instances carry an ordinary dict.
    abbreviations: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    entity: str = ""
    domain: str = ""
    # Keys the caller sent that this library did not interpret. Sorted, so two equal
    # signal sets compare equal and render identically.
    carried: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """
        True when this channel asks the matcher to do nothing differently.

        `carried` deliberately does NOT count: a key the library ignores cannot change an
        answer, so a request carrying only those must still take the untouched path.
        """
        return not self.abbreviations and not self.entity and not self.domain

    @classmethod
    def coerce(cls, source: QuerySignals | Mapping[str, Any] | None) -> QuerySignals:
        """Accept an already-built object, a raw mapping, or nothing."""
        if source is None:
            return EMPTY_QUERY_SIGNALS
        if isinstance(source, QuerySignals):
            return source
        return cls.from_mapping(source)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> QuerySignals:
        """
        Parse a caller's signal map. NEVER RAISES -- see rule 3 in the class docstring.

        A value of the wrong shape is dropped and its key recorded in `carried`, which is
        the honest report: the library did not interpret it. That is deliberately the same
        outcome as a key the library has never heard of, because from the caller's side
        the two are one fact -- "this did not take effect".
        """
        if not raw:
            return EMPTY_QUERY_SIGNALS

        supplied: dict[str, Any] = {}
        consumed: set[str] = set()
        for canonical, names in QUERY_SIGNAL_ALIASES.items():
            for name in names:
                if name in raw:
                    consumed.add(name)
                    if canonical not in supplied:
                        supplied[canonical] = raw[name]

        abbreviations = _coerce_overlay(supplied.get(QUERY_SIGNAL_ABBREVIATIONS))
        entity = _coerce_signal_text(supplied.get(QUERY_SIGNAL_ENTITY))
        domain = _coerce_signal_text(supplied.get(QUERY_SIGNAL_DOMAIN))

        carried = {name for name in raw if name not in consumed}
        # A key whose value could not be used is reported as carried rather than as
        # interpreted. `carried` answers "what did this server not act on?", and a
        # malformed overlay belongs in that answer.
        for canonical, value in (
            (QUERY_SIGNAL_ABBREVIATIONS, abbreviations),
            (QUERY_SIGNAL_ENTITY, entity),
            (QUERY_SIGNAL_DOMAIN, domain),
        ):
            if not value:
                carried |= {n for n in QUERY_SIGNAL_ALIASES[canonical] if n in raw}

        if not abbreviations and not entity and not domain and not carried:
            return EMPTY_QUERY_SIGNALS

        return cls(
            abbreviations=abbreviations,
            entity=entity,
            domain=domain,
            carried=tuple(sorted(carried)),
        )

    def merged_over(self, base: QuerySignals) -> QuerySignals:
        """
        These signals layered on top of `base`, field-level winning KEY BY KEY.

        Key by key, not object by object: a field that names its own `entity` must not
        also have to restate the request's domain. `abbreviations` is request-scoped, so
        it comes from `base` unless this object supplies its own.
        """
        if self.is_empty and not self.carried:
            return base
        if base.is_empty and not base.carried:
            return self
        return QuerySignals(
            abbreviations=self.abbreviations or base.abbreviations,
            entity=self.entity or base.entity,
            domain=self.domain or base.domain,
            carried=tuple(sorted(set(self.carried) | set(base.carried))),
        )


# The one object every caller who supplies nothing gets. Shared rather than rebuilt, so
# `is_empty` on the hot path is an attribute read and so identity comparison is available
# to a test that wants to prove the no-signal path built nothing.
EMPTY_QUERY_SIGNALS = QuerySignals()


def _coerce_signal_text(value: Any) -> str:
    """A scalar signal as text, or "" when it is not usable as one."""
    if isinstance(value, str):
        return value.strip()
    # An integer is a plausible thing for a feed to send as a namespace id and str() of it
    # is unambiguous. Everything else -- list, dict, bool, None -- is not text, and is
    # dropped rather than stringified into something like "['a', 'b']".
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return ""


def _coerce_overlay(value: Any) -> Mapping[str, str]:
    """
    A `{short -> long}` overlay, filtered to the rows that are usable.

    Row-level filtering rather than all-or-nothing: a live feed that returns one null
    expansion among 7,839 good rows should cost that row, not the catalog. The expander
    filters again on admission (`AbbreviationDictionary.merged_with`); this pass exists so
    `is_empty` is honest about whether anything will actually be merged.
    """
    if not isinstance(value, Mapping):
        return {}
    return {
        short: long
        for short, long in value.items()
        if isinstance(short, str) and isinstance(long, str) and short.strip() and long.strip()
    }


# =============================================================================
# NEXUS MATCHER - Main class
# =============================================================================


class NexusMatcher:
    """
    Main schema matching orchestrator.

    Coordinates all components to match schema fields to dictionary entries.
    Can be used as a library, backend service, or CLI tool.

    Example (Library Mode):
        ```python
        from nexus_matcher import NexusMatcher

        matcher = NexusMatcher.from_config()
        matcher.load_dictionary("data/dictionary.xlsx")

        results = matcher.match_schema("schemas/customer.avsc")
        for field_path, matches in results.items():
            print(f"{field_path} -> {matches[0].dictionary_entry.business_name}")
        ```

    Example (With custom components):
        ```python
        from nexus_matcher import NexusMatcher
        from nexus_matcher.domain.ports import EmbeddingProvider

        matcher = NexusMatcher(
            embedding_provider=my_custom_provider,
            vector_store=my_vector_store,
        )
        ```
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        sparse_retriever: SparseRetriever | None = None,
        reranker: Reranker | None = None,
        schema_parser_registry: dict[str, SchemaParser] | None = None,
        dictionary_loader_registry: dict[str, DictionaryLoader] | None = None,
        abbreviation_expander: AbbreviationExpander | None = None,
        context_enricher: ContextEnricher | None = None,
        domain_matcher: DomainMatcher | None = None,
        config: MatchingConfig | None = None,
        governance: GovernanceVocabulary | str | Path | None = None,
    ) -> None:
        """
        Initialize the matcher.

        Args:
            embedding_provider: Provider for generating embeddings
            vector_store: Store for vector similarity search
            sparse_retriever: Optional BM25/sparse retriever for hybrid search
            reranker: Optional neural reranker
            schema_parser_registry: Schema parsers by format name
            dictionary_loader_registry: Dictionary loaders by source type
            abbreviation_expander: Expander for abbreviations (uses default if None)
            context_enricher: Enricher for nested schema context (uses default if None)
            domain_matcher: Matcher for domain hierarchy scoring (uses default if None)
            config: Matching configuration
            governance: The caller's controlled vocabulary, or a path to its JSON file.
                It is what turns an indexed entry's `governance_code` into the
                `ProtectionClass` on every MatchResult. Defaults to
                `GovernanceVocabulary.empty()`, which defines nothing -- so indexing
                entries that carry codes without wiring one is refused rather than
                silently producing matches with no class. This library ships no taxonomy.
        """
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._sparse_retriever = sparse_retriever
        self._reranker = reranker
        self._schema_parsers = schema_parser_registry or {}
        self._dictionary_loaders = dictionary_loader_registry or {}
        self._abbreviation_expander = abbreviation_expander or AbbreviationExpander.default()
        self._context_enricher = context_enricher or ContextEnricher()
        self._domain_matcher = domain_matcher or DomainMatcher.default()
        self._config = config or MatchingConfig()
        self._governance = _load_governance_vocabulary(governance)
        # Whether a vocabulary was CONFIGURED, which `self._governance` cannot express:
        # `_load_governance_vocabulary` turns None into `empty()`, and `empty()` is also
        # what a caller gets by handing over a vocabulary that declares nothing. The two
        # mean opposite things to a load -- "read no codes and check nothing" against
        # "every code in this glossary is one nobody defined, say so" -- and
        # `load_dictionary` has to tell them apart to give the right refusal.
        self._governance_configured = governance is not None

        # State
        self._dictionary_entries: dict[str, DictionaryEntry] = {}
        self._indexed_ids: set[str] = set()
        # Maps a synthetic alias document id back to the entry that owns it.
        self._alias_owner: dict[str, str] = {}
        self._is_initialized = False

    @classmethod
    def from_config(
        cls,
        config: MatchingConfig | str | Path | None = None,
        governance: GovernanceVocabulary | str | Path | None = None,
    ) -> NexusMatcher:
        """
        Build a fully wired matcher, with no components to assemble by hand.

        Args:
            config: A `MatchingConfig`, or a path to a JSON/TOML file holding its fields,
                or None for the calibrated defaults.
            governance: The caller's controlled vocabulary, or a path to its JSON file.
                None means none is configured; this library ships no taxonomy.

        Returns:
            Configured NexusMatcher instance.

        Example:
            matcher = NexusMatcher.from_config()
            matcher = NexusMatcher.from_config(MatchingConfig(auto_approve_threshold=0.85))
            matcher = NexusMatcher.from_config("matching.json")
            matcher = NexusMatcher.from_config(governance="our_protection_classes.json")

        Note:
            This parameter used to be named `config_path` and was accepted and then never
            read, so passing a tuned config file silently produced default thresholds --
            a failure that is invisible precisely because the matcher still works.
        """
        matching_config = _load_matching_config(config)
        # Import infrastructure components
        from nexus_matcher.infrastructure.adapters.dictionary_loaders.excel import (
            CsvDictionaryLoader,
            ExcelDictionaryLoader,
        )
        from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
            default_embedding_provider,
        )
        from nexus_matcher.infrastructure.adapters.schema_parsers.avro import AvroSchemaParser
        from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import (
            FlattenedAvroParser,
        )
        from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
        from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore

        # Resolve the best provider AVAILABLE rather than hardcoding one.
        #
        # This used to construct SentenceTransformersProvider() directly, which needs
        # torch. The result was that every documented entry point -- the README
        # quickstart, `nexus-matcher match`, `nexus-matcher sync` -- raised
        # ImportError on a plain `pip install nexus-matcher`, while the 33.8 MB encoder
        # bundled into the wheel specifically to make that work was never reached.
        # default_embedding_provider() prefers the bundled offline encoder and falls
        # back to sentence-transformers when the extra is installed.
        embedding_provider = default_embedding_provider()
        vector_store = InMemoryVectorStore(
            VectorStoreConfig(
                collection_name="dictionary",
                dimension=embedding_provider.dimension,
            )
        )
        sparse_retriever = BM25Retriever()

        # Register parsers and loaders
        schema_parsers = {
            "avro": AvroSchemaParser(),
            # The production input shape: a flattened schema, not a nested .avsc.
            "flattened_avro": FlattenedAvroParser(),
        }
        dictionary_loaders = {
            "excel": ExcelDictionaryLoader(),
            "csv": CsvDictionaryLoader(),
        }

        return cls(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            sparse_retriever=sparse_retriever,
            schema_parser_registry=schema_parsers,
            dictionary_loader_registry=dictionary_loaders,
            config=matching_config,
            governance=governance,
        )

    def load_dictionary(
        self,
        source: str | Path,
        column_mapping: ColumnMapping | None = None,
        source_type: str | None = None,
        governance_strict: bool = True,
    ) -> LoadStatistics:
        """
        Load a data dictionary into the matcher, READ THROUGH its vocabulary.

        Args:
            source: Path to dictionary file
            column_mapping: Custom column mapping
            source_type: Force specific loader (auto-detect if None)
            governance_strict: REFUSE the load when the source's governance is defective
                -- with a vocabulary configured, any row carrying an undefined code or a
                tier that contradicts its own code; with none configured, a source that
                HAS a protection-code column, which nothing can then read. Set False to
                load anyway: the catalog still wins, so no entry inherits a contradicted
                tier, and each offending row carries `governance_problems` in its
                `source_metadata`. It is also the switch that skips the extra read of the
                source described below.

        Returns:
            Loading statistics

        Raises:
            ValueError: If no suitable loader found; if the source's governance is
                defective under `governance_strict`; if the two readings of the source
                described below disagree about which entries exist; or if the source
                carries a protection-code column and no vocabulary was configured to read
                it.

        ## The defect this method carried, recorded as NM-0033

        `from_config(governance=...)` accepted a vocabulary and this method never applied
        it. `DictionaryLoader` -- the port every loader implements -- hands back finished
        `DictionaryEntry` objects, and the two loaders this package ships build them
        without reading a protection-code column at all, so every indexed entry carried
        `governance_code=None`, every match came back `governance=None`, and a field the
        glossary marks as a direct identifier was auto-approved carrying no class. Nothing
        errored, because a glossary loaded without a vocabulary is a documented mode: the
        result is indistinguishable from a glossary that declares no classes. That is
        NM-0005's failure -- a field silently losing the classification it should have
        inherited -- on the documented Python happy path, in the commit that cut 2.1.0
        (`git show 36ffc1d:` this file: `load_dictionary` says "governance" zero times).
        CHANGELOG.md records that 2.1.0 was never published, so it reached no user;
        `release_preflight.py` declared the wheel built from that tree fit to publish, so
        nothing between there and a publish was going to stop it.

        It is also the second time this class accepted a parameter and never read it; the
        first is recorded on `from_config`'s `config_path`. A parameter that is read
        nowhere is worse than one that does not exist, because the caller has evidence
        they configured the thing.

        ## Why the governance is attached AFTER the loader, not instead of it

        `application.ingest.load_entries` is the reader in this layer that understands
        governance: it resolves the code column, canonicalises the code through the
        caller's vocabulary, and enforces the derivation invariant. The obvious fix is to
        route this method through it and delete the loader call.

        Measured against `examples/governance/glossary.csv`, that swap is not a subset of
        this method's behaviour -- it also drops `parent_table`, `sample_values` and
        `synonyms` (which `load_entries` deliberately never maps), turns `auto_N` ids into
        content digests, and coerces types and tiers through a different table
        (`double` -> `decimal`, `INTERNAL` -> `RESTRICTED` on the same 30 rows). Several
        of those are improvements. All of them are changes, and shipping them inside a P0
        governance fix would mean nobody can tell which change moved a match.

        So the loader still builds the entries, unchanged, and the governance is attached
        to them afterwards from the same file read through `load_entries`. The cost is one
        extra read of the source; it is measured in `_attach_governance`.

        ## Why there is no per-call `governance=` here, when `load_entries` has one

        Because a matcher resolves `MatchResult.governance` through `self._governance` at
        MATCH time. A per-call vocabulary that differs from it can only do one of two
        things: attach codes this matcher cannot resolve, which `_index_dictionary`
        refuses outright, or attach fewer than it could, which is the silent nothing this
        fix exists to remove. The one spelling that would work -- letting the argument
        also replace `self._governance` -- means a load silently reconfigures the matcher,
        and leaves the next load with no argument holding an ambiguous vocabulary.

        `load_entries` keeps its parameter because it has no matcher and no match-time
        resolver; it is a function that returns entries. Here the vocabulary is a property
        of the object, set once where it can be seen, in exactly one place. Two answers to
        "which vocabulary is this matcher's" is worse than one, which is the argument
        `MatchResult.governance_id` already makes about which entry a field inherits from.
        """
        path = Path(source)

        # Auto-detect loader
        if source_type is None:
            for loader_type, loader in self._dictionary_loaders.items():
                if path.suffix.lower() in loader.supported_extensions:
                    source_type = loader_type
                    break

        if source_type is None:
            raise ValueError(f"No loader found for extension {path.suffix}")

        loader = self._dictionary_loaders.get(source_type)
        if loader is None:
            raise ValueError(f"Unknown dictionary source type: {source_type}")

        # Load entries
        result = loader.load(path, column_mapping)
        if result.is_failure:
            raise ValueError(result.error)

        entries, stats = result.unwrap()

        # The step whose absence is NM-0033. Everything above produces entries with
        # no governance whatever the caller configured.
        entries = self._attach_governance(path, entries, column_mapping, governance_strict)

        # Index entries
        self._index_dictionary(entries)

        return stats

    def _attach_governance(
        self,
        source: Path,
        entries: Sequence[DictionaryEntry],
        column_mapping: ColumnMapping | None,
        governance_strict: bool,
    ) -> list[DictionaryEntry]:
        """
        Give the loader's entries the protection code the source declares.

        Reads the source a SECOND time, through `ingest.load_entries`, and takes exactly
        one thing from it: the canonical `governance_code` per entry (plus, under
        `governance_strict=False`, the evidence for a code that was refused). Nothing else
        about an entry is touched, so a reviewer can check the claim "this fix changes
        governance and nothing else" by reading this method alone.

        Delegating rather than resolving the code here is deliberate. The vocabulary
        lookup, the alias table, the derivation invariant and the refusal messages all
        live in `load_entries`; a second implementation of any of them is a second set of
        rules, and the last time this repository had two readers with two notions of which
        column was which, one of them rejected files the other read without complaint.

        ## What the second read costs

        Measured 2026-08-19 on a generated 30,000-row CSV (12 columns, 5.4 MB, a
        protection-code column resolving against a 12-code vocabulary), median of five
        runs on this machine:

            loader.load() alone                  0.281 s
            + load_entries() for the codes       1.154 s   (+0.873 s, +310% of the parse)
            load_dictionary() end to end        70.64  s   (one run, bundled encoder)

        So the second read triples the parse and costs **1.2%** of the call, because
        embedding dominates by two orders of magnitude. Peak traced allocation rises
        46.9 -> 79.4 MB while both entry lists are alive; `load_entries` returns a list,
        so this cannot be streamed without changing its signature.

        The guard-only path -- no vocabulary configured -- pays `read_source` alone at
        0.137 s, because all it wants is the header.

        `governance_strict=False` with no vocabulary configured skips the read entirely --
        there is then nothing to attach and nothing to check.

        The read disappears altogether the day `DictionaryLoader` can carry a protection
        code, which is the honest fix and is not this method's to make: the port is in
        `domain/ports/dictionary_loader.py` and its `ColumnMapping` has no field for a
        code column, which is also why a caller whose code column is spelled outside
        `CODE_COLUMN_ALIASES` cannot map it on this path.
        """
        from nexus_matcher.application import ingest

        if not self._governance_configured:
            if governance_strict:
                self._refuse_an_unreadable_code_column(source)
            return list(entries)

        governed = ingest.load_entries(
            source,
            columns=_governance_read_columns(column_mapping),
            governance=self._governance,
            governance_strict=governance_strict,
        )
        return _merge_governance(entries, governed, source)

    def _refuse_an_unreadable_code_column(self, source: Path) -> None:
        """
        Refuse a glossary that carries protection codes when no vocabulary was configured.

        `load_entries` has had this guard since NM-0030 and this path bypassed it, which
        is how a deployment reached "every entry carries no class" by the other door: not
        by mis-wiring the vocabulary, but by never wiring one, against a glossary whose
        header plainly says `protection_class`. The silence is circular -- with no
        vocabulary there are no codes, so there is nothing for any later layer to refuse.

        The refusal is `ingest`'s own, quoted rather than rewritten, so a caller who hits
        it from `load_dictionary` and from `load_entries` reads one message and gets one
        instruction. Only the header is wanted; the reader has no header-only mode, which
        is the whole of the cost here.

        A source `ingest` cannot read at all is left alone rather than refused. That is a
        custom loader reading something this package's reader has never heard of, and
        turning a load that works today into an error over a check we could not run would
        be a regression dressed as a safeguard. Nothing is lost: with no vocabulary
        configured there is no code to attach either way.
        """
        from nexus_matcher.application import ingest

        try:
            _rows, header = ingest.read_source(source)
        except Exception:
            # Deliberately broad, and deliberately silent: see the docstring. Whatever a
            # reader this package does not own raises, it is not evidence about this
            # source's governance, and it must not fail a load that works today.
            return

        code_column = ingest.map_columns(header).get("governance_code")
        if code_column:
            raise ingest._unread_code_column_error(source, code_column)

    def _index_dictionary(self, entries: Sequence[DictionaryEntry]) -> None:
        """
        Index dictionary entries for search.

        Refuses to index an entry whose `governance_code` this matcher's vocabulary does
        not define. That is a WIRING error -- a glossary validated against one vocabulary
        handed to a matcher holding another, or none -- and it is checked once here rather
        than per match. Left unchecked it degrades silently: every MatchResult comes back
        with `governance=None`, which is indistinguishable from "this entry has no class",
        and a field inherits nothing where it should have inherited something. Same shape
        as NM-0005, one layer up.
        """
        self._reject_unresolvable_governance(entries)

        # Store entries
        self._dictionary_entries = {e.id: e for e in entries}

        # Index-time enrichment: extra rows holding fabricated technical spellings of
        # each entry's business name. They are indexed as ordinary vectors but carry a
        # synthetic id, and `_alias_owner` maps them back so retrieval can max-pool per
        # entry. See domain/services/alias_generation for why this is done on the
        # dictionary side rather than by expanding the query.
        self._alias_owner = {}
        alias_rows: list[tuple[str, str]] = []
        if self._config.dictionary_alias_count > 0:
            alias_rows = expand_dictionary(
                [(e.id, e.business_name) for e in entries],
                max_aliases=self._config.dictionary_alias_count,
            )

        # Generate embeddings for the primary texts and the aliases in ONE batch.
        texts = [e.to_searchable_text() for e in entries]
        texts.extend(alias for _, alias in alias_rows)

        embed_result = self._embedding_provider.embed(texts)
        if embed_result.is_failure:
            raise RuntimeError(f"Embedding failed: {embed_result.error}")

        embeddings = embed_result.unwrap()

        # Index in vector store
        docs = [
            VectorDocument(
                id=entry.id,
                embedding=embeddings.embeddings[i],
                payload={
                    "business_name": entry.business_name,
                    "logical_name": entry.logical_name,
                    "data_type": entry.data_type.value,
                    "domain": entry.domain,
                },
            )
            for i, entry in enumerate(entries)
        ]

        offset = len(entries)
        for j, (owner_id, alias) in enumerate(alias_rows):
            alias_id = f"{owner_id}\x00alias{j}"
            self._alias_owner[alias_id] = owner_id
            owner = self._dictionary_entries[owner_id]
            docs.append(
                VectorDocument(
                    id=alias_id,
                    embedding=embeddings.embeddings[offset + j],
                    payload={
                        "business_name": owner.business_name,
                        "logical_name": owner.logical_name,
                        "data_type": owner.data_type.value,
                        "domain": owner.domain,
                        "alias_of": owner_id,
                        "alias_text": alias,
                    },
                )
            )

        # Drop any previously indexed vectors first. `_dictionary_entries` is REPLACED
        # above, but the vector store was only ever upserted into, so loading a second
        # dictionary left the first one's vectors searchable while their entries were no
        # longer resolvable -- producing silent misses and stale matches.
        clear_result = self._vector_store.delete(list(self._indexed_ids))
        if clear_result.is_failure:
            raise RuntimeError(f"Failed to clear previous index: {clear_result.error}")

        upsert_result = self._vector_store.upsert(docs)
        if upsert_result.is_failure:
            raise RuntimeError(f"Vector indexing failed: {upsert_result.error}")

        self._indexed_ids = {entry.id for entry in entries} | set(self._alias_owner)

        # Index in sparse retriever
        if self._sparse_retriever:
            sparse_docs = [
                SparseDocument(
                    id=entry.id,
                    text=entry.to_searchable_text(),
                    metadata={"domain": entry.domain},
                )
                for entry in entries
            ]
            # index() REPLACES the sparse index, so no explicit clear is needed here.
            # The Result was previously discarded: a failed sparse build left the
            # matcher running dense-only with no indication anything had gone wrong.
            sparse_result = self._sparse_retriever.index(sparse_docs)
            if sparse_result.is_failure:
                raise RuntimeError(f"Sparse indexing failed: {sparse_result.error}")

        self._is_initialized = True

    def _reject_unresolvable_governance(self, entries: Sequence[DictionaryEntry]) -> None:
        """Fail loudly, once, naming the codes and the fix."""
        unresolved = sorted(
            {
                e.governance_code
                for e in entries
                if e.governance_code and self._governance.get(e.governance_code) is None
            }
        )
        if not unresolved:
            return
        known = ", ".join(sorted(self._governance.codes)) or "none -- no vocabulary is wired"
        raise ValueError(
            f"{len(unresolved)} protection code(s) on the entries being indexed are not "
            f"in this matcher's governance vocabulary: {', '.join(unresolved)}.\n"
            f"Vocabulary declares: {known}.\n"
            f"Pass the same vocabulary the glossary was validated against, e.g. "
            f"NexusMatcher.from_config(governance='protection_classes.json'). Indexing "
            f"anyway would return every match with governance=None, which reads as 'this "
            f"entry has no class' rather than 'nobody told me what its class means'."
        )

    def match_schema(
        self,
        schema_source: str | Path | dict[str, Any],
        schema_format: str | None = None,
        signals: QuerySignals | Mapping[str, Any] | None = None,
    ) -> dict[str, tuple[MatchResult, ...]]:
        """
        Match a schema against the loaded dictionary.

        Args:
            schema_source: Schema file path or content
            schema_format: Force specific parser (auto-detect if None)
            signals: Per-request query-side context (AR-6). Optional in every sense: None
                is the shipped behaviour, and every recognised signal is individually
                optional within it. See `QuerySignals`.

        Returns:
            One entry per parsed field, in schema order, keyed by the name the caller
            used for it -- the original flattened column name for a flattened schema,
            the dotted `full_path` for every other format. See `field_result_key`.
        """
        if not self._is_initialized:
            raise RuntimeError("Dictionary not loaded. Call load_dictionary() first.")

        # Parse schema
        schema = self._parse_schema(schema_source, schema_format)

        return self._match_fields(schema.fields, signals=signals)

    def _request_expander(self, signals: QuerySignals) -> tuple[AbbreviationExpander, bool]:
        """
        The expander and the on/off decision for ONE request. Never mutates the matcher.

        Three cases, and the middle one is the one worth reading:

        * **No overlay.** The configured expander and the configured flag, unchanged. This
          is the path every existing caller takes and it allocates nothing.
        * **Overlay, `expand_query_abbreviations` OFF.** Expansion runs for this request
          with the OVERLAY ALONE. Sending a live approved-abbreviation catalog is an
          unambiguous request to expand with it; ignoring it because a start-up flag is
          off would make the channel unreachable on the exact deployment it exists for
          (`from_config()` builds the matcher and takes no expander, so its catalog is
          always the bundled generic list). And the flag being off is the deployment
          saying it does NOT vouch for the configured catalog as a query-side source.

          That is not a stylistic reading of the flag. Measured paired on the full
          committed corpus (688/688, exact McNemar), with contracted field names and a
          caller's catalog supplied through this channel: overlay alone 0.5785 P@1,
          overlay merged with the bundled generic list 0.5625 -- **-1.60 points, 13
          gained, 24 lost, p = 0.0989**. Inconclusive on its own, point estimate
          negative, and it reproduces the independently recorded -1.60 / p = 0.099 for
          that list. So merging a catalog the operator declined to trust into one they
          just vouched for would import a measured wrong-rate into every overlay request,
          for nothing the caller asked for. (On FHIR the same comparison is -0.13 points,
          2 discordant -- the bundled list contains almost none of those tokens, which is
          the same reason its damage is invisible on the committed benchmarks and shows up
          the moment a caller's own catalog is in play.)
        * **Overlay, flag ON.** Configured catalog merged with the overlay, overlay
          winning row by row. The operator has vouched for both, so both are used.

        WHAT THE OVERLAY IS WORTH, and what that number does not say. On the same
        measurement, contracting every field name through a generated naming standard
        costs 48.3 points of P@1 on combined (0.5814 -> 0.0988) and 23.4 on FHIR (0.2461
        -> 0.0122); a FULL catalog recovers 99.4% and 99.5% of that. Those two recovery
        figures are an upper bound established BY CONSTRUCTION -- the catalog was
        generated from those field names, so expanding a contracted name reconstructs the
        original string. They measure this plumbing, not the idea, and the giveaway is
        that they do not move.

        The informative arms are the degraded ones, and they agree across both corpora
        (combined / FHIR, mean of 3 and 2 seeds):

            coverage 75%   72.7% / 65.1% of the gap recovered
            coverage 50%   47.0% / 36.8%
            coverage 25%   23.5% / 16.6%
            wrong  5%      92.9% / 90.8%
            wrong 25%      62.8% / 58.4%
            wrong 75%       4.8% /  4.0%   (p = 0.70 / 0.22 -- break-even)
            wrong 100%    -14.9% / -4.7%   (WORSE than sending no overlay)

        Recovery tracks coverage roughly linearly and tolerates staleness well; a catalog
        that is mostly wrong is worse than none. So the thing a deployment must measure
        before turning this on is its catalog's WRONG-RATE against its own field names,
        not its size.

        Returns:
            `(expander, expand)`. `expander` is `self._abbreviation_expander` itself in
            the no-overlay case -- identity, not a copy.
        """
        if not signals.abbreviations:
            return self._abbreviation_expander, self._config.expand_query_abbreviations
        if self._config.expand_query_abbreviations:
            return self._abbreviation_expander.with_overlay(signals.abbreviations), True
        return AbbreviationExpander(AbbreviationDictionary.from_dict(signals.abbreviations)), True

    @staticmethod
    def _field_signals(field: SchemaField, request: QuerySignals) -> QuerySignals:
        """
        The effective signals for one field: its own, layered over the request's.

        A flattened export can carry columns from several parent records in one request,
        so `entity` cannot be request-scoped only. Returns `request` itself when the field
        adds nothing, so the common case costs one dict lookup.
        """
        raw = field.source_metadata.get(QUERY_SIGNALS_METADATA_KEY)
        if not isinstance(raw, Mapping) or not raw:
            return request
        return QuerySignals.from_mapping(raw).merged_over(request)

    def _match_fields(
        self,
        fields: Sequence[SchemaField],
        signals: QuerySignals | Mapping[str, Any] | None = None,
    ) -> dict[str, tuple[MatchResult, ...]]:
        """
        Match a batch of fields, encoding all queries in a single call.

        Transformer encoders are throughput-bound by batch size on CPU: encoding one
        text at a time reaches ~128 texts/sec where a batch of 128 reaches ~1690.
        Building every query string first and embedding them together is what turns a
        per-field loop into a batched pipeline.

        Returns exactly one entry per field, in input order. That count is the contract:
        see `_unique_result_key` for what used to happen when it did not hold.

        `signals` is the per-request query-side channel (AR-6). It is resolved ONCE here
        and passed down, so the abbreviation overlay is merged once per request rather
        than once per field, and so nothing on `self` is touched: the matcher is shared
        across concurrent requests, and a per-request mutation would let one caller's
        catalog decide another caller's query text.
        """
        if not fields:
            return {}

        request_signals = QuerySignals.coerce(signals)
        expander, expand = self._request_expander(request_signals)
        per_field = [self._field_signals(f, request_signals) for f in fields]

        query_texts = [
            self._build_query_text(f, expander=expander, expand=expand, entity=s.entity)
            for f, s in zip(fields, per_field, strict=True)
        ]

        embeddings: list[np.ndarray] | None = None
        embed_result = self._embedding_provider.embed(query_texts)
        if embed_result.is_success:
            batch = embed_result.unwrap()
            embeddings = [batch.embeddings[i] for i in range(len(query_texts))]

        dense_per_field = self._search_dense_batch(embeddings)

        results: dict[str, tuple[MatchResult, ...]] = {}
        for i, field in enumerate(fields):
            field_results = self._match_field(
                field,
                query_text=query_texts[i],
                query_embedding=embeddings[i] if embeddings is not None else None,
                dense_candidates=dense_per_field[i] if dense_per_field is not None else None,
                signals=per_field[i],
            )
            results[self._unique_result_key(field, results)] = tuple(field_results)

        return results

    def _search_dense_batch(
        self, embeddings: list[np.ndarray] | None
    ) -> list[list[SearchResult]] | None:
        """
        Retrieve dense candidates for every field in ONE call, when the store supports it.

        The encoder was already batched; dense retrieval was not. Scoring one query is a
        matrix-VECTOR product that streams the whole corpus matrix out of RAM, so a
        688-field schema read a 153 MB matrix 688 times. Batching turns those into one
        matrix-MATRIX product per chunk, which reads the corpus once and reuses each
        cache line across every query in the block: measured 2.98x on the real FHIR
        dictionary (363 us -> 122 us per query).

        Returns None when batching is unavailable or fails, and the caller falls back to
        per-field search. That fallback is not decoration -- only InMemoryVectorStore
        implements search_batch today; Qdrant and HNSW do not, and a store supplied by a
        caller certainly need not. Degrading quietly here is right because the per-field
        path returns identical results, just slower.
        """
        if embeddings is None:
            return None
        search_batch = getattr(self._vector_store, "search_batch", None)
        if search_batch is None:
            return None

        batched = search_batch(embeddings, top_k=self._config.dense_top_k)
        if batched.is_failure:
            return None
        rows = batched.unwrap()
        # One row per query is the contract; anything else means the store disagrees with
        # us about it, and silently zipping mismatched lists would hand a field another
        # field's candidates -- the same class of defect as the result-key collision.
        if len(rows) != len(embeddings):
            return None
        return rows

    @staticmethod
    def _unique_result_key(field: SchemaField, taken: Mapping[str, Any]) -> str:
        """
        A key for `field` that cannot displace one already in `taken`.

        Results were keyed by `full_path`, which is NOT unique. The flattened parser maps
        both `contact__email` (an array of contacts) and `contact_email` (a scalar
        column) -- two legal, distinct fields of a single Avro record -- onto
        `contact.email`, so the second overwrote the first. `match_schema` then returned
        fewer entries than it was given: no exception, no warning, just a column absent
        from the results, inheriting no protection level, in a library whose entire job
        is to make a field inherit one. The only visible symptom was a count nobody had
        reason to check.

        Keying by the caller's own name removes the common case. A genuine duplicate --
        the same column listed twice in an export -- still has to go somewhere, so it
        takes a `#2`, `#3`, ... suffix. `#` occurs in neither an Avro name nor a dotted
        path, so a suffixed key reads as synthetic rather than as a real column, and each
        MatchResult still carries the SchemaField it belongs to.
        """
        key = field_result_key(field)
        if key not in taken:
            return key

        n = 2
        while f"{key}#{n}" in taken:
            n += 1
        return f"{key}#{n}"

    def match_schema_session(
        self,
        schema_source: str | Path | dict[str, Any],
        schema_format: str | None = None,
        signals: QuerySignals | Mapping[str, Any] | None = None,
    ) -> MatchingSession:
        """
        Match schema and return full session with metadata.

        Args:
            schema_source: Schema file path or content
            schema_format: Force specific parser
            signals: Per-request query-side context (AR-6); see `QuerySignals`.

        Returns:
            Complete MatchingSession with all results and metrics
        """
        if not self._is_initialized:
            raise RuntimeError("Dictionary not loaded. Call load_dictionary() first.")

        start_time = time.time()

        # Parse once and match the parsed fields directly. Calling match_schema() here
        # re-parsed the same source a second time, doubling parse cost and risking a
        # mismatch between the returned schema and the results computed from it.
        schema = self._parse_schema(schema_source, schema_format)
        results = self._match_fields(schema.fields, signals=signals)

        duration_ms = (time.time() - start_time) * 1000

        return MatchingSession(
            session_id=EntityId(),
            schema=schema,
            results=results,
            total_duration_ms=duration_ms,
            # Carried into the session so `get_low_confidence_fields()` can REFUSE a
            # threshold no match could ever fall below, instead of returning [] and
            # reading as "nothing to review". The session is a domain object and must
            # not import MatchingConfig, so it gets the derived number, not the config.
            minimum_achievable_confidence=self._session_confidence_floor(results),
            # Carried so `MatchingSession.field_decisions()` applies the floor this
            # matcher was configured with, rather than making every caller pass it back
            # in. Unlike the confidence floor above it needs no verification against the
            # data: it is not a derived bound that can quietly stop holding, it is the
            # number the caller typed.
            absolute_score_floor=self._config.absolute_score_floor,
        )

    def _session_confidence_floor(
        self, results: Mapping[str, Sequence[MatchResult]]
    ) -> float | None:
        """
        The floor for THIS session, or None when the derivation's preconditions did not
        hold for it.

        `MatchingConfig.minimum_achievable_confidence` is a bound, and a bound has
        preconditions: no reranker, and at least two distinct dense scores so that min-max
        normalisation maps the top candidate to 1.0 rather than to 0.0. The second one is
        easy to violate by ordinary means -- a one-entry dictionary, `dense_top_k=1`, or a
        perfect tie across every candidate -- and when it is violated the real confidences
        sit around 0.13 while the config still reports 0.63.

        Handing that number to the session was worse than not having it. Every threshold in
        the range where the fields actually were got REFUSED, with a message asserting no
        match could fall below it, on precisely the sessions where they all did. That is
        NM-0027's own failure -- an API telling a reviewer there is nothing to see -- coming
        back as an exception instead of an empty list.

        So the bound is checked against the data it claims to bound. If any top match sits
        below it, the precondition demonstrably did not hold and the session reports no
        floor. A self-verifying claim cannot be wrong about its own session.
        """
        floor = self.minimum_achievable_confidence
        if floor is None:
            return None
        tops = [matches[0].final_confidence for matches in results.values() if matches]
        if not tops:
            # Nothing matched, so there is nothing to bound and nothing to refuse against.
            return None
        return floor if min(tops) >= floor else None

    def _parse_schema(
        self,
        source: str | Path | dict[str, Any],
        format_name: str | None = None,
    ) -> Schema:
        """Parse schema from source."""
        # If dict, try to detect parser
        if isinstance(source, dict):
            for parser in self._schema_parsers.values():
                if parser.can_parse(source):
                    result = parser.parse(source)
                    if result.is_success:
                        return result.unwrap()
            raise ValueError("Could not parse schema dict")

        # If path
        path = Path(source)

        # Auto-detect parser
        if format_name is None:
            for name, parser in self._schema_parsers.items():
                if path.suffix.lower() in parser.file_extensions:
                    format_name = name
                    break

        if format_name is None:
            raise ValueError(f"No parser found for extension {path.suffix}")

        parser = self._schema_parsers.get(format_name)
        if parser is None:
            raise ValueError(f"Unknown schema format: {format_name}")

        result = parser.parse_file(path)
        if result.is_failure:
            raise ValueError(result.error)

        return result.unwrap()

    @staticmethod
    def _with_entity_context(field: SchemaField, entity: str) -> SchemaField:
        """
        A copy of `field` whose path is prefixed with the caller-supplied parent record.

        Injected into the PATH rather than pasted onto the enriched string, so the entity
        goes through exactly the same treatment every other parent level gets: the same
        humanisation, the same namespace-part skipping, the same depth limit, the same
        `HIERARCHY_SEPARATOR` that per-level abbreviation expansion splits on. A separate
        formatting path here would be a second, quietly different definition of "parent
        context" in a pipeline where parent context is the largest measured factor.

        Skipped when the path already leads with this entity, so a caller who sends both
        `parent.field` and `entity=parent` gets "parent field" and not "parent, parent
        field". Compared on token sets, because the caller's spelling of the record
        ("BookingPassenger") need not match the path's ("booking_passenger").

        Measured, paired, on both full corpora, by stripping the parent out of every path
        and handing that same parent back through this signal:

            combined (688)  path 0.5814  stripped 0.3953  signal 0.5814   0 discordant
            fhir    (1556)  path 0.2461  stripped 0.2500  signal 0.2461   0 discordant

        Read those for what they are. The zero-discordant column is a check on THIS design
        choice, not evidence that an entity is worth anything: the query text is
        reconstructed, so an exact match is what a correct injection should produce. What
        it establishes is that the signal enters at the level the parent-context effect
        lives at -- an implementation that pasted the entity onto the end of the enriched
        string, or joined it with a different separator, would not reproduce it.

        The stripped column is the honest half, and it does NOT say the same thing on both
        corpora. Removing the parent costs 18.6 points on combined (24 gained, 152 lost,
        p = 6.0e-24) and NOTHING measurable on FHIR (+0.4 points, 34 gained, 28 lost,
        p = 0.53), where a resource-name parent apparently adds no signal the leaf did not
        already carry. So what this signal is worth is a property of the caller's paths,
        not of the mechanism, and a deployment quoting the 18.6 without measuring its own
        corpus is quoting somebody else's schema.

        The copy is local to query building. Scoring still sees the caller's own field --
        `parent_path` feeds domain inference, and rewriting it here would make a supplied
        entity change a signal the caller did not ask it to change.
        """
        first, _, rest = field.full_path.partition(".")
        # `rest` guards the check, and the guard is load-bearing. Without it a path with no
        # parent at all compares the entity against the FIELD NAME, so `entity="Account"`
        # on a bare column called `account` would be silently dropped -- the caller's
        # parent context discarded because the leaf happened to repeat it. That is a
        # narrow re-entry of the level-wise dedup `EnrichmentConfig` documents as MEASURED
        # AND REMOVED: it improved ranking and cost auto-approve precision, and this
        # method is not the place to reopen that decision. Measured: it fires on 3 of 1556
        # FHIR queries, all of them leaves whose name repeats their resource.
        if rest and _tokenize_identifier(first) == _tokenize_identifier(entity):
            return field
        return dataclasses.replace(field, full_path=f"{entity}.{field.full_path}")

    def _build_query_text(
        self,
        field: SchemaField,
        *,
        expander: AbbreviationExpander | None = None,
        expand: bool | None = None,
        entity: str = "",
    ) -> str:
        """
        Build the retrieval query text for a field.

        `expander`, `expand` and `entity` are the per-request query-signal channel's three
        entry points into query building (AR-6). All three default to the matcher's own
        configuration, so `_build_query_text(field)` -- which is how `/diag/retrieval` and
        the property tests call it -- is exactly what it was.

        Hierarchical context is injected first (GAP-006): a bare field name like
        `sname` carries almost no signal, while `satscores sname` is unambiguous. On the
        combined BIRD+OMOP benchmark this parent-path context is worth +20 points of
        P@1 -- by far the largest single accuracy factor in the pipeline.

        Abbreviation expansion stays OFF by default (see `expand_query_abbreviations`).
        When a caller does turn it on, it runs PER PARENT-PATH LEVEL rather than over the
        whole enriched string, because the enriched string is not one identifier -- it is
        several, separated by `HIERARCHY_SEPARATOR`. Expanding it whole let the separator
        ride along on the token in front of it: the expander looked up `"accn,"`, comma
        and all, missed, and passed it through raw, while the very same `accn` expanded
        to `account` everywhere else in the same query. The token that failed to expand
        was always a parent-path level -- precisely the part of the query carrying the
        most signal. Measured on the FHIR corpus with a governed catalog: 619 of 1556
        queries came out different, 96.3% of every query that has any parent-path
        structure. (The shipped generic dictionary happens to contain none of these
        tokens, so the defect is invisible on the committed benchmarks -- 0 of 2244
        queries differ with `AbbreviationExpander.default()`. It fires with the governed
        catalog that makes the flag worth enabling in the first place.)

        The result is lowercased explicitly rather than left to the encoder. The expander
        restores the case style of each token it replaces, so a descriptive phrase can
        put capitals back into the query; today's BGE tokenizer is uncased and cannot
        see them, but that is a property of the current encoder, not of this contract,
        and swapping the encoder should not silently change the query text.
        """
        if entity:
            field = self._with_entity_context(field, entity)

        enriched_query = self._context_enricher.enrich(field)

        if expand is None:
            expand = self._config.expand_query_abbreviations
        if not expand:
            return enriched_query

        if expander is None:
            expander = self._abbreviation_expander

        levels = enriched_query.split(HIERARCHY_SEPARATOR)
        expanded = [expander.expand(level).expanded if level else level for level in levels]
        return HIERARCHY_SEPARATOR.join(expanded).lower()

    def _resolve_query(
        self,
        field: SchemaField,
        query_text: str | None,
        signals: QuerySignals | Mapping[str, Any] | None,
    ) -> tuple[QuerySignals, str]:
        """
        This field's effective signals, and the query text they produce.

        The merge happens here as well as in `_match_fields` because `_match_field` is also
        a single-field entry point (`/diag/retrieval`, the property tests, three museum
        entries): a field carrying its own signals must behave the same whichever door it
        came in by. `merged_over` is idempotent, so repeating it on the batch path costs one
        dict lookup and cannot change an answer.

        `query_text` is returned unchanged when the caller already built it, which is the
        batch path -- the batch built it from the same signals a moment earlier.
        """
        field_signals = self._field_signals(field, QuerySignals.coerce(signals))
        if query_text is not None:
            return field_signals, query_text
        expander, expand = self._request_expander(field_signals)
        return field_signals, self._build_query_text(
            field, expander=expander, expand=expand, entity=field_signals.entity
        )

    def _match_field(
        self,
        field: SchemaField,
        query_text: str | None = None,
        query_embedding: np.ndarray | None = None,
        dense_candidates: list[SearchResult] | None = None,
        signals: QuerySignals | Mapping[str, Any] | None = None,
    ) -> list[MatchResult]:
        """
        Match a single field against the dictionary.

        `query_text` and `query_embedding` may be supplied by the caller so that a whole
        schema's embeddings can be produced in one batched encoder call; encoding fields
        one at a time costs roughly 13x throughput on CPU.

        `signals` are this field's EFFECTIVE query signals -- the request's, already
        merged with the field's own by `_match_fields`. Passing a raw mapping here is
        supported for the single-field path, which has no batch to merge against.
        """
        start_time = time.time()

        field_signals, query_text = self._resolve_query(field, query_text, signals)

        if query_embedding is None:
            embed_result = self._embedding_provider.embed_single(query_text)
            if embed_result.is_failure:
                return []
            query_embedding = embed_result.unwrap()

        # Dense retrieval. `dense_candidates` arrives pre-computed when a whole schema was
        # retrieved in one batched call; searching per field is the fallback, and the
        # single-field public path.
        if dense_candidates is None:
            dense_results = self._vector_store.search(
                query_embedding,
                top_k=self._config.dense_top_k,
            )
            if dense_results.is_failure:
                return []
            dense_candidates = dense_results.unwrap()

        # Collapse alias hits onto the entry that owns them, keeping the BEST score per
        # entry (max-pool). Without this an entry could occupy several of the top-k slots
        # with different spellings of itself, crowding out real alternatives.
        # Primary-document score per entry, tracked separately from the pooled score
        # so the stricter confidence policy documented below remains available.
        primary_scores: dict[str, float] = {}
        if self._alias_owner:
            best: dict[str, SearchResult] = {}
            for r in dense_candidates:
                owner = self._alias_owner.get(r.id, r.id)
                if r.id == owner:
                    # This hit is the entry's own text, not a fabricated alias.
                    primary_scores[owner] = r.score
                current = best.get(owner)
                if current is None or r.score > current.score:
                    best[owner] = SearchResult(
                        id=owner, score=r.score, payload=r.payload, embedding=r.embedding
                    )
            dense_candidates = sorted(best.values(), key=lambda r: r.score, reverse=True)

        # The RAW dense score per entry, captured before fusion normalises it away.
        #
        # Under the shipped wiring this is the cosine similarity between the query vector
        # and the entry vector, and it is the only number in the breakdown an auditor can
        # compare ACROSS fields -- `fused_retrieval_score` is rescaled per field, so its
        # 0.9 says "won this field's shortlist", not "is 90% similar". It was computed on
        # every match and then discarded, so "how similar were they really?" had no
        # answer anywhere in the API. Reported only; never an input to the confidence.
        absolute_cosines = {r.id: r.score for r in dense_candidates}

        # Sparse retrieval (if available)
        sparse_candidates: dict[str, float] = {}
        if self._sparse_retriever:
            sparse_result = self._sparse_retriever.search(
                query_text,
                top_k=self._config.sparse_top_k,
            )
            if sparse_result.is_success:
                for sr in sparse_result.unwrap():
                    sparse_candidates[sr.id] = sr.score

        # Fuse results
        fused = self._fuse_results(dense_candidates, sparse_candidates)

        # Rerank if available
        if self._reranker and fused:
            candidates = [
                RerankCandidate(
                    id=doc_id,
                    text=self._dictionary_entries[doc_id].to_searchable_text(),
                    initial_score=score,
                )
                for doc_id, score in fused[: self._config.colbert_top_k]
            ]

            rerank_result = self._reranker.rerank(
                query_text,
                candidates,
                top_k=self._config.cross_encoder_top_k,
            )

            if rerank_result.is_success:
                reranked = rerank_result.unwrap()
                # Reranker scores are UNBOUNDED (cross-encoder logits can be negative or
                # far above 1; ColBERT MaxSim is a sum over query tokens). They feed
                # `fused_retrieval_score`, which carries 70% of the confidence weight, so
                # they must be squashed into [0, 1] first. Without this, min(score, 1.0)
                # saturated every plausible candidate to the same value and turned
                # `final_confidence` into a sign test on the raw logit.
                fused = [(r.id, _squash_score(r.score)) for r in reranked]

        # Score every surviving candidate, then rank by the multi-signal confidence.
        # Previously the loop scored only the first `results_per_field` entries and
        # emitted them in retrieval order, so the lexical/edit/type/domain signals
        # (30% of the weight) could never change which entry came back first.
        candidate_pool = fused[
            : max(self._config.cross_encoder_top_k, self._config.results_per_field)
        ]

        weights = _signal_weights(self._config)

        scored: list[tuple[float, Signals, DictionaryEntry, float | None]] = []
        for doc_id, retrieval_score in candidate_pool:
            entry = self._dictionary_entries.get(doc_id)
            if entry is None:
                continue

            # NOTE on aliasing and confidence. Ranking uses the max-pooled score, which
            # reports an entry's best-matching fabricated alias. Confidence uses it too,
            # and that is a deliberate compromise rather than an oversight:
            #
            #   pooled score -> confidence : ~91% max auto-approve precision, 15% coverage
            #   primary score -> confidence: ~95% precision but only 2.9% coverage
            #
            # Routing confidence through the primary text alone is more principled -- an
            # entry that only won via a fabricated spelling genuinely deserves less
            # confidence -- but it suppresses precisely the fields aliasing rescued, and
            # coverage collapses. `primary_scores` is computed and kept available for
            # callers that want the stricter behaviour.
            #
            # If a >=95% auto-approve guarantee matters more than ranking quality, set
            # dictionary_alias_count = 0: that restores 0.953 precision at 12.4% coverage
            # at the cost of ~2 points of P@1 (~4 on abbreviation-heavy schemas).
            signals = self._score_signals(
                field, entry, retrieval_score, domain_prior=field_signals.domain
            )
            scored.append(
                (
                    _weighted_confidence(signals, weights),
                    signals,
                    entry,
                    absolute_cosines.get(doc_id),
                )
            )

        # Stable sort by confidence descending; ties keep upstream retrieval order, which
        # `fuse_linear_ids` defines as dense rank rather than leaving it to string hashing.
        scored.sort(key=_first, reverse=True)

        latency_ms = (time.time() - start_time) * 1000
        results: list[MatchResult] = []

        for rank, (final_confidence, signals, entry, absolute_cosine) in enumerate(
            scored[: self._config.results_per_field], 1
        ):
            runner_up = scored[rank][0] if rank < len(scored) else None
            decision = self._determine_decision(final_confidence, rank, runner_up)

            results.append(
                MatchResult(
                    schema_field=field,
                    dictionary_entry=entry,
                    rank=rank,
                    final_confidence=final_confidence,
                    # Widened here, for the results that are actually returned.
                    score_breakdown=_breakdown(signals, absolute_cosine),
                    decision=decision,
                    performance=PerformanceMetrics(
                        latency_ms=latency_ms,
                        cache_hit=False,
                        retrieval_stage="reranked" if self._reranker else "fused",
                        candidates_evaluated=len(fused),
                        reranking_applied=self._reranker is not None,
                    ),
                    # Resolved for EVERY returned candidate, not just rank 1, because
                    # "rank 1 is a direct identifier and rank 2 is not" is usually the
                    # fact a reviewer decides on. `_index_dictionary` has already proven
                    # every indexed code resolves, so this cannot quietly return None for
                    # a code the vocabulary simply does not know. MatchResult drops it on
                    # a REJECTED RANK 1 only -- a novel field inherits nothing. A rejected
                    # runner-up KEEPS its class, because no field inherits from rank 2 and
                    # blanking it deleted the rank-1-versus-rank-2 comparison this line
                    # exists to provide: 66 of 104 runner-ups on the 26-field pack came
                    # back classless although their entry carried a real code.
                    governance=self._governance.get(entry.governance_code),
                )
            )

        return results

    def _fuse_results(
        self,
        dense: list[SearchResult],
        sparse: dict[str, float],
    ) -> list[tuple[str, float]]:
        """
        Fuse dense and sparse retrieval results.

        Uses weighted linear combination over MIN-MAX normalized scores, delegating to
        core.fusion so the arithmetic lives in one place. `fuse_linear_ids` is the same
        fusion as `HybridFuser`/`fuse_linear` with the per-item `ScoredItem` provenance
        left unbuilt -- this call site only ever kept the id and the score.

        Two deliberate choices, both measured on the combined BIRD+OMOP benchmark
        (benchmarks/exp_fusion.py, 793 labelled queries):

        1. NOT Reciprocal Rank Fusion, despite RRF being the conventional default. RRF
           keeps only rank and throws away score magnitude, so a confidently-correct
           dense hit gets averaged against a confidently-wrong lexical hit. Measured
           P@1: rrf 0.610, dense-alone 0.691, linear 0.702. RRF was the WORST method
           tried and was worse than using no fusion at all.

        2. Min-max rather than max-only normalization. Dividing by the max alone leaves
           a floor that varies per query, so scores are not comparable across the two
           arms; BM25 scores in particular never approach zero for a matched query.

        The 0.9 dense weight is the sweep optimum for P@1. If you tune for recall
        instead (e.g. to feed a reranker), balanced combsum scored higher R@10 (0.923
        vs 0.911) -- re-run the sweep rather than assuming.
        """
        dense_results = [(r.id, r.score) for r in dense]
        sparse_results = sorted(sparse.items(), key=_second, reverse=True)

        alpha = self._config.fusion_alpha
        return fuse_linear_ids(
            dense_results,
            sparse_results,
            semantic_weight=alpha,
            lexical_weight=1.0 - alpha,
            normalize_scores=True,
        )

    def _score_signals(
        self,
        field: SchemaField,
        entry: DictionaryEntry,
        retrieval_score: float,
        domain_prior: str = "",
    ) -> Signals:
        """
        The five raw signals for one (field, entry) pair, in `Signals` order.

        Returns a tuple rather than a ScoreBreakdown because this runs for every
        candidate in the pool while only `results_per_field` of them are ever returned to
        a caller -- three quarters of the dataclasses built here used to be discarded
        unread a few lines later. `_breakdown` widens the survivors.

        The arithmetic is deliberately the same work, in the same order, as before that
        change: a scoring change that silently moves rankings does not belong in a
        refactor that was only meant to stop building objects nobody reads.
        """
        # The fused RETRIEVAL score, not a semantic similarity. It is min-max normalised
        # per field, so it says how this entry placed among this field's candidates and
        # says almost nothing about how similar the two texts are. Named for what it is:
        # calling it `semantic_score` is what let "0.9" be read as 90% similarity for two
        # releases (DX-003). The absolute cosine is reported separately.
        fused_retrieval_score = min(retrieval_score, 1.0)

        # Lexical score (token overlap).
        # Identifiers must be split on case and digit boundaries, not just underscores:
        # `.replace("_", " ")` leaves "NumTstTakr" and "enroll12" as single tokens, so
        # every camelCase schema (Avro, JSON Schema, most SQL Server DDL) scored 0
        # lexical overlap no matter how well it actually matched.
        field_tokens = _tokenize_identifier(field.name)
        entry_tokens = _tokenize_identifier(entry.logical_name)
        entry_tokens |= _tokenize_identifier(entry.business_name)

        if field_tokens and entry_tokens:
            intersection = field_tokens & entry_tokens
            lexical_score = len(intersection) / max(len(field_tokens), 1)
        else:
            lexical_score = 0.0

        # Edit distance score. Compare on the normalized token strings so that
        # "NumTstTakr" and "num_tst_takr" are not penalized for punctuation alone.
        edit_distance_score = self._edit_distance_score(
            " ".join(sorted(field_tokens)),
            " ".join(sorted(_tokenize_identifier(entry.logical_name))),
        )

        # Type compatibility
        type_score = entry.matches_type(field.data_type)

        # Domain score using domain matcher
        domain_score = self._calculate_domain_score(field, entry, domain_prior=domain_prior)

        return (
            fused_retrieval_score,
            lexical_score,
            edit_distance_score,
            type_score,
            domain_score,
        )

    def _calculate_scores(
        self,
        field: SchemaField,
        entry: DictionaryEntry,
        query_embedding: np.ndarray,
        retrieval_score: float,
        domain_prior: str = "",
    ) -> ScoreBreakdown:
        """Calculate detailed score breakdown."""
        return _breakdown(self._score_signals(field, entry, retrieval_score, domain_prior))

    def _calculate_domain_score(
        self,
        field: SchemaField,
        entry: DictionaryEntry,
        domain_prior: str = "",
    ) -> float:
        """
        Calculate domain compatibility score.

        Uses domain hierarchy matching when domain info is available,
        falls back to neutral score otherwise.

        ## The caller-supplied prior

        `domain_prior` is the request-level signal (AR-6): the namespace or domain hint
        the caller knows and this library cannot derive. When present it REPLACES the
        inferred field domain, because inference from a path or a name is a guess and the
        prior is a statement.

        It also gets one rule the inference path does not have, and the reason is
        specific. `DomainMatcher` scores through a shipped hierarchy: two domains it has
        never heard of are `UNRELATED` and score `unknown_score` -- even when they are the
        SAME STRING. That is correct for two guesses, and useless for a prior, because an
        enterprise's own domain names are exactly the ones the shipped hierarchy does not
        contain. So a prior that CONTAINS the entry's declared domain scores 1.0 outright,
        and everything else falls through to the hierarchy as before.

        Containment rather than equality, on token sets, so that a schema namespace does
        the job a caller expects: `com.example.bookings` contains `Bookings`. It is
        one-directional on purpose -- an entry whose domain is "Customer Account" is NOT
        matched by a prior of "customer", because half a domain name is not the domain.

        The effect is a pure BOOST: a matching entry moves from the neutral 0.5 to 1.0,
        which at the shipped `domain_weight = 0.15` is +0.075 of pre-squash confidence.
        That is also the tie-break the signal is asked for -- it is several times the
        sub-0.02 margin that separates a near-tied top-1 and top-2 -- and it is one
        mechanism rather than two, so a domain hit cannot be counted twice.

        Nothing here fires when `domain_prior` is empty, which is the shipped path.

        MEASURED, AND THE MEASUREMENT'S OWN LIMIT. Paired on both full corpora, supplying
        each schema's own database or resource name as the prior:

            combined (688)   0.5814 -> 0.5930    8 gained, 0 lost, p = 0.0078
            fhir    (1556)   0.2461 -> 0.2738   43 gained, 0 lost, p = 2.3e-13

        ZERO LOSSES on both is the part that transfers: a prior that matches promotes, and
        one that does not leaves the ranking where the retriever put it. The MAGNITUDE does
        not transfer at all. Those corpora declare 47 domains over 688 entries and 158 over
        4,598 -- a median of 12 and 22 entries each -- and every query's gold entry sits in
        the domain its own schema names, so the prior there nearly PARTITIONS the answer
        key. A real glossary's namespaces do not. Read these as "the mechanism works and
        does not hurt", never as a number to expect.
        """
        if domain_prior:
            entry_domain = entry.domain
            if entry_domain:
                entry_tokens = _tokenize_identifier(entry_domain)
                if entry_tokens and entry_tokens <= _tokenize_identifier(domain_prior):
                    return 1.0
                return self._domain_matcher.score(domain_prior, entry_domain)
            return 0.5

        # Get entry domain (from dictionary)
        entry_domain = entry.domain

        # Try to infer field domain from:
        # 1. source_metadata if available
        # 2. parent_path inference
        # 3. field name patterns
        field_domain = field.source_metadata.get("domain")

        if not field_domain:
            # Try to infer from parent path
            field_domain = self._infer_domain_from_path(field.parent_path)

        if not field_domain:
            # Try to infer from field name
            field_domain = self._infer_domain_from_name(field.name)

        # If we have both domains, compute score
        if field_domain and entry_domain:
            return self._domain_matcher.score(field_domain, entry_domain)

        # No usable pair -- neutral score.
        #
        # This used to be two branches, "only the entry has a domain" and "neither side
        # has one", both returning 0.5. They are written as one return because the split
        # implied a partial-credit case that never existed, not because merging them is
        # faster: this is NOT a performance change. Both forms are one predicted branch
        # per candidate, and the whole collapse can save at most 0.16 ms of a 7.04 s
        # match. Do not cite it as an optimization.
        return 0.5

    def _infer_domain_from_path(self, path: str) -> str | None:
        """Infer domain from field path."""
        if not path:
            return None

        # Common path patterns that indicate domain
        path_lower = path.lower()

        domain_patterns = {
            "account": "ACCOUNTS",
            "transaction": "TRANSACTIONS",
            "txn": "TRANSACTIONS",
            "balance": "BALANCES",
            "customer": "CUSTOMER",
            "cust": "CUSTOMER",
            "payment": "PAYMENTS",
            "card": "CARDS",
            "loan": "LOANS",
            "mortgage": "MORTGAGES",
            "address": "ADDRESS",
            "contact": "CONTACT",
            "phone": "PHONE",
            "email": "EMAIL",
            "product": "PRODUCT",
            "compliance": "COMPLIANCE",
            "kyc": "KYC",
            "aml": "AML",
        }

        for pattern, domain in domain_patterns.items():
            if pattern in path_lower:
                return domain

        return None

    def _infer_domain_from_name(self, name: str) -> str | None:
        """Infer domain from field name."""
        if not name:
            return None

        name_lower = name.lower()

        # Strong indicators in field names
        name_patterns = {
            "acct": "ACCOUNTS",
            "account": "ACCOUNTS",
            "txn": "TRANSACTIONS",
            "transaction": "TRANSACTIONS",
            "bal": "BALANCES",
            "balance": "BALANCES",
            "cust": "CUSTOMER",
            "customer": "CUSTOMER",
            "addr": "ADDRESS",
            "address": "ADDRESS",
            "payment": "PAYMENTS",
            "pmt": "PAYMENTS",
        }

        for pattern, domain in name_patterns.items():
            if pattern in name_lower:
                return domain

        return None

    def _edit_distance_score(self, s1: str, s2: str) -> float:
        """
        Calculate normalized edit distance score.

        Uses rapidfuzz's C++ Levenshtein when available (~145x faster than the pure
        Python DP below, bit-identical results), falling back to the DP so the package
        still works without the optional dependency. This is on the hot path: it runs
        once per candidate per field.
        """
        if not s1 or not s2:
            return 0.0

        if _LEVENSHTEIN is not None:
            return float(_LEVENSHTEIN.normalized_similarity(s1, s2))

        return self._edit_distance_score_fallback(s1, s2)

    @staticmethod
    def _edit_distance_score_fallback(s1: str, s2: str) -> float:
        """Pure-Python Levenshtein, used only when rapidfuzz is unavailable."""
        m, n = len(s1), len(s2)

        # Single-row DP: O(min(m, n)) memory instead of a full (m+1)x(n+1) matrix.
        if m < n:
            s1, s2 = s2, s1
            m, n = n, m

        previous = list(range(n + 1))
        for i in range(1, m + 1):
            current = [i] + [0] * n
            c1 = s1[i - 1]
            for j in range(1, n + 1):
                if c1 == s2[j - 1]:
                    current[j] = previous[j - 1]
                else:
                    current[j] = 1 + min(previous[j], current[j - 1], previous[j - 1])
            previous = current

        max_len = max(m, n)
        return 1.0 - (previous[n] / max_len) if max_len > 0 else 1.0

    def _calculate_final_confidence(self, scores: ScoreBreakdown) -> float:
        """Calculate weighted final confidence score."""
        return _weighted_confidence(
            (
                scores.fused_retrieval_score,
                scores.lexical_score,
                scores.edit_distance_score,
                scores.type_compatibility_score,
                scores.domain_score,
            ),
            _signal_weights(self._config),
        )

    def _determine_decision(
        self,
        confidence: float,
        rank: int,
        runner_up_confidence: float | None,
    ) -> MatchDecision:
        """
        Determine match decision from confidence and the margin over the next candidate.

        Only the top-ranked match can be auto-approved, and only when it is BOTH
        confident enough and clearly ahead of the runner-up. `min_confidence_gap`
        exists to catch the ambiguous case where two dictionary entries score almost
        identically -- exactly the situation a human should adjudicate. Previously the
        gap was compared against already-emitted lower-ranked results, so it could
        never prevent an ambiguous rank-1 auto-approval; a near-tie was silently
        auto-approved.
        """
        config = self._config

        if rank == 1 and confidence >= config.auto_approve_threshold:
            margin = (
                confidence - runner_up_confidence
                if runner_up_confidence is not None
                else float("inf")
            )
            if margin >= config.min_confidence_gap:
                return MatchDecision.AUTO_APPROVE
            # Confident but ambiguous: send it to a human rather than guessing.
            return MatchDecision.REVIEW

        if confidence >= config.review_threshold:
            return MatchDecision.REVIEW

        return MatchDecision.REJECT

    @property
    def minimum_achievable_confidence(self) -> float | None:
        """
        The lowest `final_confidence` a rank-1 match can carry, for THIS matcher.

        Ask this before choosing any confidence threshold. A threshold at or below it
        selects nothing, however bad the matches are -- which is how
        `get_low_confidence_fields()` came to answer "nothing to review" on a schema
        where nothing was trustworthy (DX-001).

        Returns None when a reranker is wired, because the reranker replaces the fused
        retrieval score with its own squashed output and the floor no longer holds. None
        means "unknown for this configuration", not "zero"; see
        `MatchingConfig.minimum_achievable_confidence` for the derivation and the second
        precondition.
        """
        if self._reranker is not None:
            return None
        return self._config.minimum_achievable_confidence

    @property
    def absolute_score_floor(self) -> float | None:
        """
        The absolute-score floor beneath which a field is reported `NO_MATCH`, or None
        when none is configured.

        Public because the HTTP surface has to publish it: a consumer who cannot see the
        floor cannot tell an emitted `NO_MATCH` from a field the matcher simply had
        nothing for. See `MatchingConfig.absolute_score_floor` for why the default is off.
        """
        return self._config.absolute_score_floor

    @property
    def absolute_score_metric(self) -> str:
        """
        The distance metric the wired vector store declares, so nobody has to ASSUME the
        absolute score is a cosine.

        `ScoreBreakdown.absolute_cosine` is the raw number the store returned. Under the
        shipped wiring the store's metric is `cosine` and the name is accurate. A caller
        who supplies their own store configured for `dot` or `euclidean` gets a number
        that is monotone in similarity but is NOT a cosine, is not bounded to [-1, 1], and
        must not be compared against a floor chosen for one. Naming the metric is what
        turns that from a trap into a fact the caller can read.

        Returns `"unknown"` when the store declares no metric. That is not a synonym for
        cosine: it means this matcher cannot state what the number is, and a caller
        setting an absolute floor against an unknown metric is guessing.
        """
        store_config = getattr(self._vector_store, "_config", None)
        metric = getattr(store_config, "distance_metric", None)
        if isinstance(metric, str) and metric:
            return metric
        return "unknown"

    @property
    def dictionary_size(self) -> int:
        """Get number of loaded dictionary entries."""
        return len(self._dictionary_entries)

    @property
    def is_ready(self) -> bool:
        """Check if matcher is ready for queries."""
        return self._is_initialized
