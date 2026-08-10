"""
tests.unit.presentation._fakes | Layer: TEST
CLI test doubles, built from REAL domain objects.

## Relationships
# USED_BY → tests/unit/presentation/conftest :: fixtures
# USED_BY → tests/unit/presentation/test_cli_regressions :: CLI rendering defects
# USED_BY → tests/unit/presentation/test_cli_json_governance :: DX-002

Why real `DictionaryEntry` / `MatchResult` / `ScoreBreakdown` and not duck-typed stubs:

The CLI tests used to carry hand-written stand-ins declaring exactly the attributes the
CLI happened to read -- four on the entry, three on the score breakdown. That is a stub
shaped like the DEFECT. `protection_level` was missing from the JSON output and also
missing from the stub, so no test in this directory could have noticed; a stub that omits
what the code omits is a mirror, not an oracle.

Real domain objects cost nothing here (no model loading, no I/O) and make this suite fail
loudly if another lane renames a field the CLI reads. The matcher itself is still stubbed,
because that is the part that loads an embedding model.
"""

from __future__ import annotations

from typing import Any, ClassVar

from nexus_matcher.application.use_cases.match_schema import MatchingConfig
from nexus_matcher.domain.models.entities import DictionaryEntry, MatchResult, SchemaField
from nexus_matcher.shared.types.base import (
    DataType,
    MatchDecision,
    PerformanceMetrics,
    ProtectionLevel,
    ScoreBreakdown,
)

Signals = tuple[float, float, float, float, float]

# The five signals in weight order: semantic, lexical, edit distance, type, domain.
# Deliberately five DIFFERENT values, and none of them equal to another's weight: a
# fixture with repeated numbers cannot tell a component paired with the correct weight
# from one paired with the wrong weight.
DEFAULT_SIGNALS: Signals = (0.9012, 0.5, 0.4211, 0.8, 0.25)

# Hand-computed from DEFAULT_SIGNALS and the shipped weights (0.70/0.05/0.05/0.05/0.15):
#   0.7*0.9012 + 0.05*0.5 + 0.05*0.4211 + 0.05*0.8 + 0.15*0.25
#   = 0.630840 + 0.025000 + 0.021055 + 0.040000 + 0.037500 = 0.754395
# Written out rather than computed, so this file states an ABSOLUTE expectation. A fixture
# that derives its expected value from the code under test proves nothing (H-004).
EXPECTED_CONFIDENCE = 0.754395


def signal_weights(config: MatchingConfig) -> Signals:
    """The five weights in signal order, read off the public config fields."""
    return (
        config.semantic_weight,
        config.lexical_weight,
        config.edit_distance_weight,
        config.type_weight,
        config.domain_weight,
    )


def weighted_confidence(signals: Signals, config: MatchingConfig) -> float:
    """
    The weighted sum, reimplemented here rather than imported from the application.

    Independent arithmetic on purpose: a fixture that calls the production function to
    build the number the production function is then asserted against is an identity, and
    an identity holds just as well when both sides are wrong.
    """
    # strict=True: if the signal count and the weight count ever diverge, the fixture has
    # to say so rather than silently dropping the last term and still looking plausible.
    total = sum(s * w for s, w in zip(signals, signal_weights(config), strict=True))
    return min(max(total, 0.0), 1.0)


def make_entry(**overrides: Any) -> DictionaryEntry:
    """A dictionary entry carrying a real classification, definition and domain."""
    base: dict[str, Any] = {
        "id": "dict_042",
        "business_name": "Customer Email Address",
        "logical_name": "cust_email",
        "definition": "The email address used to contact a customer.",
        "data_type": DataType.STRING,
        "protection_level": ProtectionLevel.PII,
        "domain": "customer",
    }
    base.update(overrides)
    return DictionaryEntry(**base)


def make_match(
    *,
    entry: DictionaryEntry | None = None,
    signals: Signals = DEFAULT_SIGNALS,
    rank: int = 1,
    field_path: str = "customer.email",
    config: MatchingConfig | None = None,
    decision: MatchDecision = MatchDecision.REVIEW,
) -> MatchResult:
    """A MatchResult whose confidence really is the weighted sum of its own components."""
    config = config or MatchingConfig()
    sem, lex, edit, type_, domain = signals
    return MatchResult(
        schema_field=SchemaField(
            name=field_path.rsplit(".", 1)[-1],
            data_type=DataType.STRING,
            full_path=field_path,
        ),
        dictionary_entry=entry if entry is not None else make_entry(),
        rank=rank,
        final_confidence=weighted_confidence(signals, config),
        score_breakdown=ScoreBreakdown(
            # `fused_retrieval_score`, not the deprecated `semantic_score` alias: the
            # alias warns on construction and on read, and a fixture that trips a
            # DeprecationWarning on every test buries the ones that mean something.
            fused_retrieval_score=sem,
            lexical_score=lex,
            edit_distance_score=edit,
            type_compatibility_score=type_,
            domain_score=domain,
        ),
        decision=decision,
        performance=PerformanceMetrics(latency_ms=1.0),
    )


class FakeStats:
    """What `load_dictionary` returns; only the fields the CLI renders."""

    total_rows = 2
    valid_entries = 2
    skipped_rows = 0
    error_rows = 0
    success_rate = 1.0
    errors: ClassVar[list[str]] = []


class FakeMatcher:
    """
    Stands in for NexusMatcher without loading an embedding model.

    `_config` is present because that is where the CLI reads the scoring weights it emits
    -- see `_MATCHER_CONFIG_ATTR` in the CLI. A stub without it would send every test down
    the weights-unavailable error path instead of the real one.
    """

    def __init__(
        self,
        results: dict[str, list[MatchResult]] | None = None,
        config: MatchingConfig | None = None,
    ) -> None:
        self._results = {"customer.email": [make_match()]} if results is None else results
        self._config = config or MatchingConfig()

    def load_dictionary(self, path: Any) -> FakeStats:
        return FakeStats()

    def match_schema(self, path: Any) -> dict[str, list[MatchResult]]:
        return self._results
