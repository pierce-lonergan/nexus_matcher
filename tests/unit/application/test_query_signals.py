"""
tests.unit.application.test_query_signals | Layer: TEST
Tests: QuerySignals, NexusMatcher._request_expander/_build_query_text/_calculate_domain_score
Target: application/use_cases/match_schema.py

The per-request query-side signal channel (AR-6). Three shipped signals travel through it
-- an abbreviation overlay, a parent-record name, a domain prior -- but the channel is the
thing being tested: a deployment must be able to send context this library has never heard
of without being refused, and a caller who sends nothing must get the previous release's
answers.

The load-bearing properties, and what each one is the negation of:

  ABSENCE       no signals in, nothing different out. Proven by comparison against the
                unextended call rather than against a recorded expectation, so it cannot
                pass by both sides drifting together.
  NO MUTATION   an overlay is merged for one call and leaves nothing behind. The matcher
                is shared across concurrent requests; a leaked row would change a LATER
                request's answer, which is the one class of defect no accuracy measurement
                can attribute.
  TOLERANCE     an unknown key is carried and ignored. `extra="forbid"` is the right
                answer to a typo and the wrong answer to an extension, and this is where
                the difference is pinned.

What is NOT here: whether the overlay HELPS. That is a measurement, not an assertion, and
it belongs on the full corpus with a paired test -- see the report accompanying this
change and `docs/guides/governed_abbreviations.md`.
"""

from __future__ import annotations

import threading

import pytest

from nexus_matcher.application.use_cases.match_schema import (
    EMPTY_QUERY_SIGNALS,
    INTERPRETED_SIGNAL_NAMES,
    QUERY_SIGNALS_METADATA_KEY,
    MatchingConfig,
    NexusMatcher,
    QuerySignals,
)
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.services.abbreviation import (
    AbbreviationDictionary,
    AbbreviationExpander,
)
from nexus_matcher.shared.types.base import DataType, DocumentId
from tests.properties._support import build_matcher

# A governed catalog: exact lookup, no guessing. Every key is a short form a caller's
# approved-abbreviation list would plausibly hold, and none of them is in the bundled
# generic list, so an expansion here can only have come from the overlay.
CATALOG = {"psgr": "passenger", "brth": "berth", "mnfst": "manifest"}


def _field(
    name: str = "psgr_nm",
    path: str = "booking.psgr_nm",
    *,
    doc: str = "",
    signals: dict[str, object] | None = None,
) -> SchemaField:
    parent, _, _leaf = path.rpartition(".")
    metadata: dict[str, object] = {"flattened_name": path}
    if signals is not None:
        metadata[QUERY_SIGNALS_METADATA_KEY] = signals
    return SchemaField(
        name=name,
        data_type=DataType.STRING,
        full_path=path,
        parent_path=parent,
        description=doc,
        source_metadata=metadata,
    )


def _entry(entry_id: str, business_name: str, definition: str, domain: str = "") -> DictionaryEntry:
    return DictionaryEntry(
        id=DocumentId(entry_id),
        business_name=business_name,
        logical_name=business_name.lower().replace(" ", "_"),
        definition=definition,
        data_type=DataType.STRING,
        domain=domain,
    )


# =============================================================================
# THE CHANNEL ITSELF
# =============================================================================


class TestParsing:
    def test_nothing_in_is_the_shared_empty_object(self):
        # Identity, not equality: the no-signal path must not build anything, and a test
        # that can see that is a test that notices when it starts to.
        assert QuerySignals.coerce(None) is EMPTY_QUERY_SIGNALS
        assert QuerySignals.coerce({}) is EMPTY_QUERY_SIGNALS
        assert EMPTY_QUERY_SIGNALS.is_empty

    def test_canonical_names(self):
        signals = QuerySignals.from_mapping(
            {"abbreviations": CATALOG, "entity": "Booking", "domain": "Transport"}
        )
        assert signals.abbreviations == CATALOG
        assert signals.entity == "Booking"
        assert signals.domain == "Transport"
        assert not signals.is_empty

    @pytest.mark.parametrize(
        ("wire_name", "attribute"),
        [
            ("abbreviation_overlay", "abbreviations"),
            ("parent_record", "entity"),
            ("domain_prior", "domain"),
            ("namespace", "domain"),
        ],
    )
    def test_the_aliases_reach_the_same_signal(self, wire_name: str, attribute: str):
        value = CATALOG if attribute == "abbreviations" else "Bookings"
        assert getattr(QuerySignals.from_mapping({wire_name: value}), attribute) == value

    def test_the_canonical_name_outranks_its_aliases(self):
        signals = QuerySignals.from_mapping({"namespace": "com.x.a", "domain": "b"})
        assert signals.domain == "b"

    def test_domain_prior_outranks_namespace(self):
        signals = QuerySignals.from_mapping({"namespace": "com.x.a", "domain_prior": "b"})
        assert signals.domain == "b"

    def test_an_unknown_key_is_carried_not_refused(self):
        signals = QuerySignals.from_mapping({"protection_hint": "x", "entity": "Booking"})
        assert signals.carried == ("protection_hint",)
        assert signals.entity == "Booking"

    def test_a_request_of_only_unknown_keys_still_reads_as_empty(self):
        # `carried` cannot change an answer, so a request carrying only ignored keys must
        # take the untouched path -- which is what `is_empty` decides.
        signals = QuerySignals.from_mapping({"protection_hint": "x"})
        assert signals.is_empty
        assert signals.carried == ("protection_hint",)

    @pytest.mark.parametrize(
        "value", [None, "not a map", 7, ["a", "b"], {"ok": None}, {None: "x"}, {"": "x"}]
    )
    def test_a_malformed_overlay_is_dropped_never_raised(self, value: object):
        signals = QuerySignals.from_mapping({"abbreviations": value})
        assert signals.abbreviations == {}
        # Reported as not acted on, which is the honest answer to "did my signal apply?".
        assert "abbreviations" in signals.carried

    def test_a_partly_malformed_overlay_keeps_its_good_rows(self):
        # One null expansion out of a live feed must cost that row, not the catalog.
        signals = QuerySignals.from_mapping(
            {"abbreviations": {"psgr": "passenger", "brth": None, "": "x"}}
        )
        assert signals.abbreviations == {"psgr": "passenger"}

    @pytest.mark.parametrize("value", [None, ["a"], {"a": 1}, True])
    def test_a_malformed_scalar_signal_is_dropped_never_raised(self, value: object):
        assert QuerySignals.from_mapping({"entity": value}).entity == ""

    def test_whitespace_around_a_scalar_is_stripped(self):
        assert QuerySignals.from_mapping({"entity": "  Booking  "}).entity == "Booking"

    def test_merged_over_wins_key_by_key(self):
        request = QuerySignals.from_mapping({"abbreviations": CATALOG, "domain": "Transport"})
        field = QuerySignals.from_mapping({"entity": "Manifest"})
        merged = field.merged_over(request)
        assert merged.entity == "Manifest"
        assert merged.domain == "Transport"
        assert merged.abbreviations == CATALOG

    def test_a_field_can_override_the_requests_domain(self):
        request = QuerySignals.from_mapping({"domain": "Transport"})
        merged = QuerySignals.from_mapping({"domain": "Billing"}).merged_over(request)
        assert merged.domain == "Billing"

    def test_merging_nothing_returns_the_base_object(self):
        request = QuerySignals.from_mapping({"domain": "Transport"})
        assert EMPTY_QUERY_SIGNALS.merged_over(request) is request

    def test_the_interpreted_names_are_published(self):
        assert {"abbreviations", "entity", "domain"} <= INTERPRETED_SIGNAL_NAMES
        assert "protection_hint" not in INTERPRETED_SIGNAL_NAMES


# =============================================================================
# THE ABBREVIATION OVERLAY
# =============================================================================


def _matcher_with(catalog: dict[str, str] | None = None, *, expand: bool = False) -> NexusMatcher:
    matcher = build_matcher([_entry("e1", "Passenger Name", "The name of a passenger")])
    matcher._config = MatchingConfig(expand_query_abbreviations=expand)
    if catalog is not None:
        matcher._abbreviation_expander = AbbreviationExpander(
            AbbreviationDictionary.from_dict(catalog)
        )
    return matcher


class TestAbbreviationOverlay:
    def test_an_overlay_expands_the_query_for_one_request(self):
        matcher = _matcher_with()
        signals = QuerySignals.from_mapping({"abbreviations": CATALOG})
        expander, expand = matcher._request_expander(signals)
        assert expand is True
        query = matcher._build_query_text(_field(), expander=expander, expand=expand)
        assert "passenger" in query

    def test_without_the_overlay_the_same_field_is_unexpanded(self):
        matcher = _matcher_with()
        assert matcher._build_query_text(_field()) == "booking psgr nm"

    def test_the_overlay_does_not_survive_into_the_next_call(self):
        matcher = _matcher_with()
        matcher._match_fields([_field()], signals={"abbreviations": CATALOG})
        assert matcher._build_query_text(_field()) == "booking psgr nm"

    def test_the_matchers_own_expander_is_never_replaced(self):
        matcher = _matcher_with()
        before = matcher._abbreviation_expander
        matcher._match_fields([_field()], signals={"abbreviations": CATALOG})
        assert matcher._abbreviation_expander is before
        assert before.expand("psgr").expanded == "psgr"

    def test_with_the_flag_off_the_overlay_is_the_only_asserting_catalog(self):
        """
        The configured expander is NOT merged in when `expand_query_abbreviations` is off.

        That flag being off is the deployment saying it does not vouch for its configured
        catalog as a query-side source -- the shipped default carries the bundled generic
        list, measured at -1.60 P@1 (exact McNemar p=0.099) on the committed corpus.
        Merging a catalog the operator declined into one they just supplied would import
        that wrong-rate into every overlay request.
        """
        matcher = _matcher_with({"brth": "birth"}, expand=False)
        expander, expand = matcher._request_expander(
            QuerySignals.from_mapping({"abbreviations": CATALOG})
        )
        assert expand is True
        assert expander.expand("brth").expanded == "berth"

    def test_with_the_flag_on_both_catalogs_apply_and_the_overlay_wins(self):
        matcher = _matcher_with({"brth": "birth", "acct": "account"}, expand=True)
        expander, _expand = matcher._request_expander(
            QuerySignals.from_mapping({"abbreviations": CATALOG})
        )
        assert expander.expand("brth").expanded == "berth"
        assert expander.expand("acct").expanded == "account"

    def test_no_overlay_returns_the_matchers_own_expander_unwrapped(self):
        matcher = _matcher_with(expand=True)
        expander, expand = matcher._request_expander(EMPTY_QUERY_SIGNALS)
        assert expander is matcher._abbreviation_expander
        assert expand is True

    def test_two_concurrent_requests_do_not_see_each_others_catalog(self):
        matcher = _matcher_with()
        errors: list[str] = []
        barrier = threading.Barrier(2)

        def run(short: str, long: str, foreign: str) -> None:
            try:
                barrier.wait(timeout=10.0)
                for _ in range(60):
                    results = matcher._match_fields(
                        [_field(name=short, path=f"booking.{short}")],
                        signals={"abbreviations": {short: long}},
                    )
                    assert results
                    expander, expand = matcher._request_expander(
                        QuerySignals.from_mapping({"abbreviations": {short: long}})
                    )
                    query = matcher._build_query_text(
                        _field(name=short, path=f"booking.{short}"),
                        expander=expander,
                        expand=expand,
                    )
                    if long not in query:
                        errors.append(f"{short} lost its own expansion")
                        return
                    if foreign in query:
                        errors.append(f"{short} saw the other request's expansion")
                        return
            except Exception as exc:
                errors.append(f"{short} raised {type(exc).__name__}: {exc}")

        threads = [
            threading.Thread(target=run, args=("psgr", "passenger", "berth")),
            threading.Thread(target=run, args=("brth", "berth", "passenger")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60.0)
        assert errors == []


# =============================================================================
# THE PARENT-RECORD SIGNAL
# =============================================================================


class TestEntitySignal:
    def test_a_supplied_entity_becomes_the_leading_parent_level(self):
        matcher = _matcher_with()
        field = _field(name="legal_name", path="legal_name")
        assert matcher._build_query_text(field) == "legal name"
        assert matcher._build_query_text(field, entity="BookingPassenger") == (
            "booking passenger legal name"
        )

    def test_it_sits_in_front_of_a_path_that_already_has_parents(self):
        matcher = _matcher_with()
        field = _field(name="legal_name", path="detail.legal_name")
        assert matcher._build_query_text(field) == "detail legal name"
        assert matcher._build_query_text(field, entity="Booking") == "booking, detail legal name"

    def test_an_entity_the_path_already_leads_with_is_not_repeated(self):
        # Two parent levels, so the duplicate would be VISIBLE if it happened: without the
        # check this reads "booking passenger, booking passenger, detail legal name".
        matcher = _matcher_with()
        field = _field(name="legal_name", path="booking_passenger.detail.legal_name")
        assert matcher._build_query_text(field, entity="BookingPassenger") == (
            "booking passenger, detail legal name"
        )
        assert matcher._build_query_text(field, entity="BookingPassenger") == (
            matcher._build_query_text(field)
        )

    def test_a_bare_column_whose_name_repeats_the_entity_still_gets_it(self):
        # The dedup asks "does the path already lead with this entity as a PARENT". A path
        # with no parent leads with nothing, so dropping the entity here would discard the
        # caller's context because the leaf happened to repeat it -- a narrow re-entry of
        # the level-wise dedup `EnrichmentConfig` records as measured and removed.
        matcher = _matcher_with()
        field = _field(name="account", path="account")
        assert matcher._build_query_text(field, entity="Account") == "account account"
        assert matcher._build_query_text(field) == "account"

    def test_it_is_expanded_like_every_other_parent_level(self):
        # The entity rides in the path, so per-level expansion reaches it. That is the
        # whole reason it is injected there rather than pasted onto the enriched string.
        matcher = _matcher_with()
        expander, expand = matcher._request_expander(
            QuerySignals.from_mapping({"abbreviations": CATALOG})
        )
        query = matcher._build_query_text(
            _field(name="legal_name", path="legal_name"),
            expander=expander,
            expand=expand,
            entity="mnfst",
        )
        assert query == "manifest legal name"

    def test_the_field_the_scorer_sees_is_the_callers_own_object(self):
        # The entity copy is local to query building. `_project_results` compares
        # `MatchResult.schema_field` by identity, and rewriting the path for scoring would
        # also silently change domain inference, which reads `parent_path`.
        matcher = _matcher_with()
        field = _field(name="legal_name", path="legal_name")
        results = matcher._match_fields([field], signals={"entity": "Booking"})
        assert all(m.schema_field is field for m in next(iter(results.values())))

    def test_a_field_level_entity_overrides_the_requests(self):
        matcher = _matcher_with()
        field = _field(name="legal_name", path="legal_name", signals={"entity": "Manifest"})
        signals = matcher._field_signals(field, QuerySignals.from_mapping({"entity": "Booking"}))
        assert signals.entity == "Manifest"


# =============================================================================
# THE DOMAIN PRIOR
# =============================================================================

TRANSPORT = _entry("t1", "Traveller Reference", "A reference for a traveller", domain="Transport")
BILLING = _entry("b1", "Traveller Reference", "A reference for a traveller", domain="Billing")
UNDECLARED = _entry("u1", "Traveller Reference", "A reference for a traveller")


class TestDomainPrior:
    def test_a_prior_containing_the_entrys_domain_scores_a_direct_hit(self):
        matcher = _matcher_with()
        assert matcher._calculate_domain_score(_field(), TRANSPORT, domain_prior="Transport") == 1.0

    def test_a_dotted_namespace_containing_it_also_hits(self):
        matcher = _matcher_with()
        score = matcher._calculate_domain_score(
            _field(), TRANSPORT, domain_prior="com.example.transport"
        )
        assert score == 1.0

    def test_a_prior_that_is_only_half_the_domain_name_does_not_hit(self):
        # Containment is one-directional on purpose: half a domain name is not the domain.
        matcher = _matcher_with()
        wide = _entry("w1", "x", "y", domain="Customer Account")
        assert matcher._calculate_domain_score(_field(), wide, domain_prior="customer") < 1.0

    def test_a_prior_that_does_not_match_stays_at_or_below_neutral(self):
        matcher = _matcher_with()
        assert matcher._calculate_domain_score(_field(), BILLING, domain_prior="Transport") <= 0.5

    def test_an_entry_with_no_declared_domain_is_neutral(self):
        matcher = _matcher_with()
        assert (
            matcher._calculate_domain_score(_field(), UNDECLARED, domain_prior="Transport") == 0.5
        )

    def test_no_prior_leaves_the_shipped_derivation_alone(self):
        matcher = _matcher_with()
        without = matcher._calculate_domain_score(_field(), TRANSPORT)
        assert without == matcher._calculate_domain_score(_field(), TRANSPORT, domain_prior="")

    def test_it_breaks_a_tie_the_retrieval_score_cannot(self):
        """
        Two entries whose text is identical and whose DOMAIN is not -- the condition the
        spec calls normal at glossary scale. Retrieval cannot separate them; the caller's
        namespace can, and does, through the existing `domain_score` component rather than
        through a second ranking mechanism bolted beside it.
        """
        matcher = build_matcher([BILLING, TRANSPORT], sparse=False)
        field = _field(name="traveller_reference", path="journey.traveller_reference")

        neutral = matcher._match_fields([field])
        ranked = [m.dictionary_entry.id for m in next(iter(neutral.values()))]
        gap = abs(
            next(iter(neutral.values()))[0].final_confidence
            - next(iter(neutral.values()))[1].final_confidence
        )
        assert gap < 0.02, f"fixture is not a near-tie: gap {gap}"

        for prior, expected in (("Transport", TRANSPORT.id), ("Billing", BILLING.id)):
            results = matcher._match_fields([field], signals={"domain": prior})
            top = next(iter(results.values()))[0].dictionary_entry.id
            assert top == expected, f"prior {prior!r} did not win: ranked {ranked}"


# =============================================================================
# ABSENCE
# =============================================================================

CORPUS = [
    _entry("e1", "Passenger Name", "The legal name of a passenger", domain="Transport"),
    _entry("e2", "Berth Number", "The berth allocated to a passenger", domain="Transport"),
    _entry("e3", "Manifest Reference", "A reference to a sailing manifest", domain="Operations"),
    _entry("e4", "Fare Amount", "The fare charged for a journey", domain="Billing"),
]

FIELDS = [
    _field(name="psgr_nm", path="booking.psgr_nm", doc="Name of the passenger"),
    _field(name="brth_no", path="booking.brth_no", doc="Allocated berth"),
    _field(name="mnfst_ref", path="sailing.mnfst_ref", doc=""),
    _field(name="fare_amt", path="billing.fare_amt", doc="What was charged"),
    _field(name="zzz_unmatchable", path="misc.zzz_unmatchable", doc=""),
]


def _fingerprint(results: dict[str, tuple]) -> list[tuple]:
    return [
        (
            key,
            tuple(
                (m.rank, m.dictionary_entry.id, m.final_confidence, m.decision.value)
                for m in matches
            ),
        )
        for key, matches in results.items()
    ]


class TestAbsenceChangesNothing:
    """
    Paired over the same fields and the same corpus, comparing the extended call against
    the call that existed before it. Every recognised way of saying "nothing" has to land
    on the same answers -- and the reference side is the shipped code path, not a recorded
    expectation, so this cannot pass by both sides drifting together.
    """

    @pytest.mark.parametrize(
        "signals", [None, {}, {"abbreviations": {}}, {"entity": ""}, {"unknown_to_this_library": 1}]
    )
    def test_results_are_identical_to_the_unextended_call(self, signals: object):
        matcher = build_matcher(CORPUS)
        reference = _fingerprint(matcher._match_fields(FIELDS))
        got = _fingerprint(matcher._match_fields(FIELDS, signals=signals))
        discordant = [(a, b) for a, b in zip(reference, got, strict=True) if a != b]
        assert discordant == [], f"{len(discordant)} of {len(reference)} fields differ"

    def test_a_field_carrying_no_signal_key_is_the_object_it_always_was(self):
        matcher = build_matcher(CORPUS)
        plain = _field(name="psgr_nm", path="booking.psgr_nm")
        assert QUERY_SIGNALS_METADATA_KEY not in plain.source_metadata
        assert matcher._field_signals(plain, EMPTY_QUERY_SIGNALS) is EMPTY_QUERY_SIGNALS
