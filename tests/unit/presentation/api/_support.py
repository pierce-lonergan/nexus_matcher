"""
tests.unit.presentation.api._support | Layer: TEST
Fixtures for the HTTP matching surface: a fictional controlled vocabulary, the glossary
that references it, and the two matchers the tests drive.

## Relationships
# USED_BY → tests/unit/presentation/api/test_match_endpoint
# USED_BY → tests/unit/presentation/api/test_degradation
# USED_BY → tests/unit/presentation/api/test_feedback_endpoint
# USED_BY → tests/unit/presentation/api/test_governance_contract

## The vocabulary here is INVENTED, and obviously so

Everything below describes **Lumenport Water & Power**, a municipal utility that does not
exist. Its protection classes, tiers, glossary entries and column names were made up for
this file. Nothing here is drawn from any real organisation's catalogue, and nothing
should be: the library deliberately hard-codes NO taxonomy, because the controlled
vocabulary is CALLER-SUPPLIED, and a test fixture is exactly where somebody else's real
one would leak in if it ever leaked in.

The STRUCTURE is what is under test, and it is preserved faithfully:

  * a small closed set of protection-class codes
  * each code implies a classification tier, a personal-information flag, and a
    direct-identifier flag -- all three derived from the code, never free text
  * an "unclassified" case that maps to the most open tier, represented here by an entry
    with no code at all, which must serialise as an explicit `null`

## Two matchers, because they answer different questions

`build_api_matcher` is a REAL `NexusMatcher` over a real vector store, a real BM25 index
and a real `GovernanceVocabulary`, with only the encoder swapped for a deterministic
bag-of-tokens one. It produces real `MatchResult` objects carrying real `ProtectionClass`
values, so conservation, ordering, the score arithmetic and the governance passthrough
are all tested against the thing that ships rather than against a mirror of it.

`FakeMatcher` returns results this file builds. It is how a request is made to fail, to
hang, or to come back malformed in the three specific ways the conservation law forbids
-- none of which a correct matcher will produce on demand.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.governance import GovernanceVocabulary, ProtectionClass
from nexus_matcher.domain.models.entities import DictionaryEntry, MatchResult, SchemaField
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import (
    DataType,
    MatchDecision,
    PerformanceMetrics,
    ProtectionLevel,
    ScoreBreakdown,
)
from tests.properties._support import DIMENSION, BagOfTokensProvider

# =============================================================================
# THE FICTIONAL CONTROLLED VOCABULARY
# =============================================================================

# The three tiers of the invented utility, most open last. Named so that nobody can
# mistake them for a real classification scheme.
TIER_SEALED = "LUMENPORT_SEALED"
TIER_GUARDED = "LUMENPORT_GUARDED"
TIER_OPEN = "LUMENPORT_OPEN"

# The caller-supplied JSON a Lumenport operator would hand the library, in exactly the
# shape `GovernanceVocabulary.from_json` accepts. Written as the FILE FORMAT rather than
# as constructed objects, because that is the surface an adopter actually fills in, and a
# fixture that skipped it would leave the loader untested from this side.
#
# A small CLOSED set. Each code implies its tier and its two flags; a glossary row whose
# stated tier disagreed with its code is a data defect the loader refuses, which is why
# nothing here stores a tier independently of a code.
FICTIONAL_VOCABULARY_JSON: dict[str, Any] = {
    "notice": (
        "FICTIONAL EXAMPLE DATA. Lumenport Water & Power does not exist and neither does "
        "this taxonomy. Invented for this test file; it is not any organisation's catalog."
    ),
    "open_classification": TIER_OPEN,
    "aliases": {
        # A legacy spelling that must be MAPPED, and a junk token that must be DROPPED.
        # Both declared, so "we quietly dropped something" cannot happen unnoticed.
        "LEGACY-METER": "METERKEY",
        "n/a": None,
    },
    "classes": [
        {
            "code": "METERKEY",
            "name": "Meter Access Key",
            "classification": TIER_SEALED,
            "personal_information": False,
            "direct_identifier": False,
            "enhancement": "ROTATE_QUARTERLY",
        },
        {
            "code": "RESIDENT",
            "name": "Resident Name",
            "classification": TIER_GUARDED,
            "personal_information": True,
            "direct_identifier": True,
            "enhancement": "MASK_IN_LOGS",
        },
        {
            "code": "OUTAGENOTE",
            "name": "Outage Note",
            "classification": TIER_GUARDED,
            "personal_information": True,
            "direct_identifier": False,
            "enhancement": None,
        },
        {
            "code": "USAGEAGG",
            "name": "Aggregated Usage",
            "classification": TIER_OPEN,
            "personal_information": False,
            "direct_identifier": False,
            "enhancement": None,
        },
    ],
}


def governance_vocabulary() -> GovernanceVocabulary:
    """The fictional vocabulary, loaded the way a caller's file is loaded."""
    return GovernanceVocabulary.from_json(FICTIONAL_VOCABULARY_JSON)


# Resolved once, so tests can pin an expected `ProtectionClass` without re-deriving it
# from the code under test.
FICTIONAL_VOCABULARY: dict[str, ProtectionClass] = {
    code: klass
    for code in ("METERKEY", "RESIDENT", "OUTAGENOTE", "USAGEAGG")
    if (klass := governance_vocabulary().get(code)) is not None
}


# =============================================================================
# THE FICTIONAL GLOSSARY
# =============================================================================

# ids are opaque strings with a made-up prefix. They are what `governanceId` promotes, so
# they must be recognisable in a response and must look like nothing real.
GLOSSARY: tuple[DictionaryEntry, ...] = (
    DictionaryEntry(
        id="LWP-0001",
        business_name="Resident Full Name",
        logical_name="resident_full_name",
        definition="The name of the person responsible for the water account.",
        data_type=DataType.STRING,
        protection_level=ProtectionLevel.PII,
        governance_code="RESIDENT",
        domain="CUSTOMER",
    ),
    DictionaryEntry(
        id="LWP-0002",
        business_name="Meter Access Key",
        logical_name="meter_access_key",
        definition="Secret used by a field technician to open a smart water meter.",
        data_type=DataType.STRING,
        protection_level=ProtectionLevel.RESTRICTED,
        governance_code="METERKEY",
        domain="METERING",
    ),
    DictionaryEntry(
        id="LWP-0003",
        business_name="Monthly Usage Litres",
        logical_name="monthly_usage_litres",
        definition="Total water drawn on the meter during one billing month.",
        data_type=DataType.LONG,
        protection_level=ProtectionLevel.INTERNAL,
        governance_code="USAGEAGG",
        domain="METERING",
    ),
    DictionaryEntry(
        id="LWP-0004",
        business_name="Outage Note",
        logical_name="outage_note",
        definition="Free text a dispatcher wrote about a supply interruption.",
        data_type=DataType.STRING,
        protection_level=ProtectionLevel.CONFIDENTIAL,
        governance_code="OUTAGENOTE",
        domain="OPERATIONS",
    ),
    DictionaryEntry(
        # Two jobs. Non-ASCII, because the response renderer promises a pure-ASCII body
        # and an accented business name has to survive as an escape rather than as raw
        # bytes. And NO governance code, because "this entry has no protection class"
        # must serialise as an explicit null rather than as a missing key.
        id="LWP-0005",
        business_name="Tariff Band (Étage)",
        logical_name="tariff_band",
        definition="Price band applied to the account. No protection class.",
        data_type=DataType.STRING,
        protection_level=ProtectionLevel.PUBLIC,
        governance_code=None,
        domain="BILLING",
    ),
)

# The class each glossary entry confers, resolved through the vocabulary above. Used to
# pin an expectation without asking the code under test what it thinks the answer is.
GOVERNANCE_BY_ENTRY: dict[str, ProtectionClass | None] = {
    entry.id: governance_vocabulary().get(entry.governance_code) for entry in GLOSSARY
}


def request_fields() -> list[dict[str, Any]]:
    """
    A request body's `fields`, in a stable order.

    One field per glossary entry, plus one that matches nothing. The per-entry coverage is
    deliberate: it makes every declared code reachable as a RANK-1 candidate, so a test
    can assert the whole vocabulary passes through rather than only the two or three codes
    that happen to surface. Ranks below the first are usually REJECT on this fixture, and
    `MatchResult` clears the class on a REJECT -- so a test that read whichever candidates
    came back would be asserting almost entirely about nulls.

    The paths are NOT in sorted order, and not in reverse order either, so a response that
    sorted its keys is distinguishable from one that kept the caller's order.
    """
    return [
        {
            "name": "resident_nm",
            "path": "account.resident_nm",
            "doc": "Name of the resident on the account",
            "type": "string",
        },
        {
            "name": "meter_key",
            "path": "meter.meter_key",
            "doc": "Technician access key for the meter",
            "type": "string",
        },
        {
            "name": "usage_litres",
            "path": "meter.usage_litres",
            "doc": "Water drawn this month",
            "type": "bigint",
        },
        {
            "name": "outage_note",
            "path": "ops.outage_note",
            "doc": "Dispatcher free text about a supply interruption",
            "type": "string",
        },
        {
            # The entry this matches carries NO protection class, so its `governance` must
            # be an explicit null for a reason other than having been rejected.
            "name": "tariff_band",
            "path": "billing.tariff_band",
            "doc": "Price band applied to the account",
            "type": "string",
        },
        {
            "name": "zzz_unmatchable",
            "path": "misc.zzz_unmatchable",
            "doc": "",
            "type": "",
        },
    ]


# =============================================================================
# A REAL MATCHER
# =============================================================================


def build_api_matcher(
    entries: tuple[DictionaryEntry, ...] = GLOSSARY,
    *,
    results_per_field: int = 5,
    config: MatchingConfig | None = None,
) -> NexusMatcher:
    """
    A real matcher over the fictional glossary, wired the way `from_config` wires one.

    Only the encoder is substituted, for the reasons `tests/properties/_support` sets out:
    determinism that does not depend on `PYTHONHASHSEED`, and not paying a 33 MB model
    load per test. Everything that this HTTP layer actually depends on -- the result key,
    the conservation count, the score breakdown, the decision, and the resolved
    `ProtectionClass` on every candidate -- is the shipped code.
    """
    matcher = NexusMatcher(
        embedding_provider=BagOfTokensProvider(),
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=DIMENSION)
        ),
        sparse_retriever=BM25Retriever(),
        config=config or MatchingConfig(results_per_field=results_per_field),
        governance=governance_vocabulary(),
    )
    matcher._index_dictionary(list(entries))
    return matcher


# =============================================================================
# A RESULT THE DOMAIN WOULD REFUSE TO BUILD
# =============================================================================


@dataclass(frozen=True)
class MalformedMatch:
    """
    A `MatchResult`-shaped object in a state the domain layer now forbids.

    `MatchResult.__post_init__` fills a blank `governance_id` from the entry and rejects a
    contradicting one, so a real result can no longer carry an empty id or a
    half-populated class. That is exactly why this exists: the HTTP boundary must refuse
    those states on its OWN account rather than inherit the guarantee, because the caller
    is promised a populated id by THIS response, not by an invariant one layer down that
    a future refactor could relax.

    A separate class rather than a patched `MatchResult` because that one is
    `frozen=True, slots=True` and cannot be coerced into an invalid state.
    """

    schema_field: SchemaField
    dictionary_entry: DictionaryEntry
    rank: int
    final_confidence: float
    score_breakdown: ScoreBreakdown
    decision: MatchDecision
    performance: PerformanceMetrics
    governance_id: str
    governance: object | None


# The five signals in weight order. Five DIFFERENT values, none equal to another's weight,
# so a component paired with the wrong weight cannot produce the same total by accident.
STAND_IN_SIGNALS: tuple[float, float, float, float, float] = (0.9012, 0.5, 0.4211, 0.8, 0.25)

# Hand-computed from STAND_IN_SIGNALS and the shipped weights 0.70/0.05/0.05/0.05/0.15:
#   0.7*0.9012 + 0.05*0.5 + 0.05*0.4211 + 0.05*0.8 + 0.15*0.25
#   = 0.630840 + 0.025000 + 0.021055 + 0.040000 + 0.037500 = 0.754395
# Written out rather than computed, so this file states an ABSOLUTE expectation. A fixture
# that derives its expected value from the code under test proves nothing (H-004).
STAND_IN_CONFIDENCE = 0.754395


def governed_match(
    field: SchemaField,
    entry: DictionaryEntry,
    rank: int = 1,
    decision: MatchDecision = MatchDecision.REVIEW,
) -> MatchResult:
    """
    A REAL `MatchResult` with hand-chosen scores, carrying the class the entry's code
    implies.

    Real, not a stand-in: the governance fields exist on the domain model now, so building
    the actual object means a rename or a validation change in the domain layer breaks
    these tests loudly instead of being mirrored by a look-alike. Only the numbers are
    fabricated, so that `explain` can be asserted against a hand-computed constant.

    `decision` defaults to REVIEW rather than REJECT on purpose -- `MatchResult` clears
    `governance` for a REJECT, so a REJECT fixture would silently test the null path while
    claiming to test the populated one.
    """
    sem, lex, edit, type_, domain = STAND_IN_SIGNALS
    return MatchResult(
        schema_field=field,
        dictionary_entry=entry,
        rank=rank,
        final_confidence=STAND_IN_CONFIDENCE,
        score_breakdown=ScoreBreakdown(
            fused_retrieval_score=sem,
            lexical_score=lex,
            edit_distance_score=edit,
            type_compatibility_score=type_,
            domain_score=domain,
            absolute_cosine=0.7657,
        ),
        decision=decision,
        performance=PerformanceMetrics(latency_ms=1.0),
        governance=GOVERNANCE_BY_ENTRY[entry.id],
    )


class FakeMatcher:
    """
    A matcher whose `_match_fields` this test controls.

    `_config` is present because that is where the endpoint reads `results_per_field` and
    the scoring weights; a stub without it would send every test down the
    weights-unavailable fallback instead of the real path.
    """

    def __init__(
        self,
        entries: tuple[DictionaryEntry, ...] = GLOSSARY,
        *,
        config: MatchingConfig | None = None,
        delay_seconds: float = 0.0,
        raises: Exception | None = None,
        drop_last_field: bool = False,
        empty_paths: tuple[str, ...] = (),
        mangle_keys: bool = False,
        serve_wrong_field: bool = False,
    ) -> None:
        self._entries = entries
        self._config = config or MatchingConfig()
        self._delay_seconds = delay_seconds
        self._raises = raises
        self._drop_last_field = drop_last_field
        self._empty_paths = frozenset(empty_paths)
        # The two ways a result set can be the right SIZE and still be wrong. They are
        # separate switches because the endpoint checks them separately, and a fixture
        # that could only produce both at once would let one of the two checks be removed
        # without any test noticing.
        self._mangle_keys = mangle_keys
        self._serve_wrong_field = serve_wrong_field
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def _match_fields(self, fields: list[SchemaField]) -> dict[str, tuple[MatchResult, ...]]:
        self.calls += 1
        self.started.set()
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        # A gate for tests that need work to still be in flight while they assert on
        # admission. `wait()` with no timeout would hang a suite forever if a test forgot
        # to release it, and a hanging test is exactly what this endpoint exists to
        # prevent, so it is bounded.
        self.release.wait(timeout=30.0)
        if self._raises is not None:
            raise self._raises

        served = fields[:-1] if self._drop_last_field else fields
        results: dict[str, tuple[MatchResult, ...]] = {}
        for position, requested in enumerate(served):
            key = requested.source_metadata.get("flattened_name", requested.full_path)
            if self._mangle_keys:
                # Right count, wrong address: the caller cannot look up their own column.
                key = f"{key}.derived"
            # Right count and right address, results computed for somebody else's column
            # -- so this field would inherit another column's governance.
            field = served[(position + 1) % len(served)] if self._serve_wrong_field else requested
            if key in self._empty_paths:
                # A field nothing matched. It must still reach the caller, with an empty
                # list -- that is the whole conservation law.
                results[key] = ()
                continue
            results[key] = tuple(
                governed_match(field, entry, rank=rank)
                for rank, entry in enumerate(self._entries[:3], 1)
            )
        return results
