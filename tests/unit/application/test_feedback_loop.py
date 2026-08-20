"""
tests.unit.application.test_feedback_loop | Layer: TEST
Tests: read_feedback_trail, verdict_from_record, ApprovedPairBypass
Target: application/feedback_loop.py

The reference consumer of the reviewer-verdict trail (AR-7) -- reading it, folding it, and
the two decisions that make it either a precision feature or a way to be confidently wrong
at scale.

THE THREE THINGS THIS FILE EXISTS TO PIN

  1. WHAT THE KEY IS. `TestTheKeyCarriesTheParent` runs the repeated-leaf construction from
     `benchmarks/synthetic` -- one leaf name emitted under many parents, each occurrence
     governed by a different term -- approves ONE occurrence, and asserts the others are
     untouched. The fixture is checked for non-vacuity first: if the generator ever stops
     repeating leaf names, the test would pass while proving nothing, and that is the
     failure mode this whole directory is built to refuse.

  2. WHAT INVALIDATES A PAIR. Not just deletion. A term that is RE-CLASSIFIED has a
     byte-identical `content_hash` -- the hash deliberately excludes governance -- and a
     reviewer approving a field against a term is approving the class it will inherit. So
     the reclassification case is tested separately from the redefinition case, because a
     binding built on the content hash alone passes one and fails the other.

  3. WHAT THE PRE-WIDENING SHAPE COSTS. `wasCorrect: true` reads back exactly.
     `wasCorrect: false` does not and cannot: the reviewer chose something, and whether it
     had been proposed is gone. That asymmetry is WC-11 measured rather than argued, and
     `BypassReport.ambiguous` is where a deployment reads its own size of it.

NOT HERE: whether bypassing IMPROVES anything. Precision on a seen pair is 100% by
construction because a human supplied it and the matcher was not asked. That is a
tautology; asserting it would be measuring the fixture.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from nexus_matcher.application.feedback_loop import (
    ApprovedPairBypass,
    read_feedback_trail,
    verdict_from_record,
)
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.ports import MappingEntryLookup, ReviewVerdict
from nexus_matcher.presentation.api.feedback import _RECORD_KEYS
from nexus_matcher.shared.types.base import DataType, DocumentId

# =============================================================================
# FIXTURES
# =============================================================================


def _entry(entry_id: str, name: str, code: str | None = "PC-3") -> DictionaryEntry:
    return DictionaryEntry(
        id=DocumentId(entry_id),
        business_name=name,
        logical_name=name.lower().replace(" ", "_"),
        definition=f"The governed element recording the {name.lower()}.",
        data_type=DataType.STRING,
        governance_code=code,
    )


GLOSSARY = {
    "GBF-0001": _entry("GBF-0001", "Resident Full Name"),
    "GBF-0002": _entry("GBF-0002", "Account Opened Date", code=None),
    "GBF-0003": _entry("GBF-0003", "Meter Serial Number"),
}


def _lookup(entries: dict[str, DictionaryEntry] | None = None) -> MappingEntryLookup:
    return MappingEntryLookup(GLOSSARY if entries is None else entries)


def _field(key: str) -> SchemaField:
    parent, _, leaf = key.rpartition(".")
    return SchemaField(
        name=leaf,
        data_type=DataType.STRING,
        full_path=key,
        parent_path=parent,
    )


def _record(**overrides: object) -> dict[str, object]:
    """A stored line, in the recorder's own key order."""
    base: dict[str, object] = {
        "ts": "2026-08-10T09:15:00Z",
        "receivedAt": "2026-08-10T09:15:02.100000+00:00",
        "reviewer": "steward-a",
        "field": "account.resident_nm",
        "doc": "",
        "chosenGovernanceId": "GBF-0001",
        "suggestedGovernanceId": "GBF-0003",
        "wasCorrect": False,
        "verdict": "MANUAL_OVERRIDE",
    }
    base.update(overrides)
    return base


# =============================================================================
# READING THE TRAIL
# =============================================================================


class TestReadingOneRecord:
    def test_the_reader_and_the_recorder_agree_on_every_key_it_needs(self):
        """
        The trail is a FILE FORMAT shared across two layers and, in a rolling upgrade,
        across two builds. This reader names its keys as its own constants rather than
        importing the recorder's tuple, so that a rename over there cannot silently make
        this reader stop finding a key. That decision is only safe if something compares
        them, and this is that something.
        """
        from nexus_matcher.application import feedback_loop

        needed = {
            feedback_loop._TRAIL_FIELD,
            feedback_loop._TRAIL_CHOSEN,
            feedback_loop._TRAIL_SUGGESTED,
            feedback_loop._TRAIL_WAS_CORRECT,
            feedback_loop._TRAIL_VERDICT,
            feedback_loop._TRAIL_REVIEWER,
            feedback_loop._TRAIL_RECEIVED_AT,
        }
        assert needed <= set(_RECORD_KEYS), (
            f"the reader looks for {sorted(needed - set(_RECORD_KEYS))}, which the "
            f"recorder does not write. One of the two was renamed without the other."
        )

    def test_a_widened_record_reads_exactly(self):
        verdict = verdict_from_record(_record())
        assert verdict.verdict is ReviewVerdict.MANUAL_OVERRIDE
        assert verdict.chosen_entry_id == "GBF-0001"
        assert verdict.suggested_entry_id == "GBF-0003"
        assert verdict.field_key == "account.resident_nm"
        assert verdict.reviewer == "steward-a"

    def test_ordering_uses_the_servers_stamp_and_not_the_clients(self):
        """
        `ts` is the reviewer's own clock, stored verbatim and never parsed. Ordering by it
        would let one workstation with a wrong time zone overwrite somebody else's later
        decision, and ordering is what decides whose verdict is in force.
        """
        verdict = verdict_from_record(_record())
        assert verdict.recorded_at == "2026-08-10T09:15:02.100000+00:00"
        assert verdict.recorded_at != _record()["ts"]

    def test_a_rejection_carries_no_chosen_term_however_the_wire_spelled_it(self):
        """
        The pre-widening shape REQUIRES `chosenGovernanceId`, so a reviewer recording "the
        glossary does not govern this at all" has to name a term anyway. The reader drops
        it: a consumer that took that id as an approval would bypass to the term the
        reviewer had just refused, which is the one way this feature can be worse than
        having no feature.
        """
        verdict = verdict_from_record(_record(verdict="REJECTED", chosenGovernanceId="GBF-0001"))
        assert verdict.verdict is ReviewVerdict.REJECTED
        assert verdict.chosen_entry_id == ""


class TestReadingAPreWideningRecord:
    def test_a_true_boolean_reads_back_exactly(self):
        legacy = _record(wasCorrect=True)
        legacy.pop("verdict")
        assert verdict_from_record(legacy).verdict is ReviewVerdict.APPROVED

    def test_a_false_boolean_cannot_be_read_back_and_says_so(self):
        """
        WC-11's loss, in one assertion. `false` means the reviewer chose something other
        than the top suggestion. Whether that something was rank 3 or was never proposed at
        all -- a RANKING failure or a RECALL failure, which need opposite fixes -- is not
        recoverable from anything stored. It reads as UNSPECIFIED and is counted, rather
        than being guessed into MANUAL_OVERRIDE or into REJECTED.
        """
        legacy = _record(wasCorrect=False)
        legacy.pop("verdict")
        verdict = verdict_from_record(legacy)

        assert verdict.verdict is ReviewVerdict.UNSPECIFIED
        assert verdict.chosen_entry_id == "GBF-0001", (
            "the reviewer's own choice IS on the record and stays usable; what is lost is "
            "only whether the matcher had proposed it"
        )

    def test_a_record_that_says_nothing_at_all_is_refused(self):
        empty = _record()
        empty.pop("verdict")
        empty.pop("wasCorrect")
        with pytest.raises(ValueError, match="nothing in it says what the reviewer decided"):
            verdict_from_record(empty)

    def test_a_verdict_this_build_does_not_know_is_refused_by_name(self):
        """
        A newer server's fourth value must not be applied as whichever of ours is closest.
        The message names what this build does know, because the operator holding the file
        is the one who has to decide whether to upgrade or to filter.
        """
        with pytest.raises(ValueError, match="MANUAL_OVERRIDE"):
            verdict_from_record(_record(verdict="DEFERRED"))


class TestReadingAWholeTrail:
    def test_blank_lines_are_skipped_and_order_is_preserved(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        path.write_text(
            json.dumps(_record(field="a", verdict="APPROVED", wasCorrect=True))
            + "\n\n"
            + json.dumps(_record(field="b"))
            + "\n",
            encoding="ascii",
        )
        verdicts = read_feedback_trail(path)
        assert [v.field_key for v in verdicts] == ["a", "b"]

    def test_a_malformed_line_names_its_line_number(self, tmp_path):
        """
        "Something in this file is malformed" is not actionable to an operator holding five
        thousand lines, and skipping the line silently is worse: the bypass would be built
        from a subset nobody can name.
        """
        path = tmp_path / "feedback.jsonl"
        path.write_text(
            json.dumps(_record(field="a")) + "\n" + "{not json" + "\n", encoding="ascii"
        )
        with pytest.raises(ValueError, match="line 2"):
            read_feedback_trail(path)

    def test_a_defective_record_names_its_line_number_too(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        broken = _record(field="")
        path.write_text(json.dumps(broken) + "\n", encoding="ascii")
        with pytest.raises(ValueError, match="line 1"):
            read_feedback_trail(path)

    def test_lines_may_come_from_anywhere_not_only_a_path(self):
        verdicts = read_feedback_trail([json.dumps(_record(field="a"))])
        assert len(verdicts) == 1


# =============================================================================
# FOLDING A TRAIL INTO STANDING VERDICTS
# =============================================================================


class TestFolding:
    def test_the_later_verdict_on_one_field_wins(self):
        bypass = ApprovedPairBypass(
            read_feedback_trail(
                [
                    json.dumps(_record(field="f", chosenGovernanceId="GBF-0001")),
                    json.dumps(_record(field="f", chosenGovernanceId="GBF-0003")),
                ]
            )
        )
        bypass.bind(_lookup())
        assert bypass.approved_pair(_field("f")).entry.id == "GBF-0003"

    def test_a_rejection_revokes_an_earlier_approval(self):
        """
        A reviewer who goes back and says "actually, nothing governs this" must be able to
        switch a bypass OFF. A consumer that only ever accumulated approvals would make the
        first approval permanent, which is a bypass nobody can correct.
        """
        bypass = ApprovedPairBypass(
            read_feedback_trail(
                [
                    json.dumps(_record(field="f", verdict="APPROVED", wasCorrect=True)),
                    json.dumps(_record(field="f", verdict="REJECTED", chosenGovernanceId="")),
                ]
            )
        )
        bypass.bind(_lookup())

        assert bypass.approved_pair(_field("f")) is None
        assert bypass.bypass_report().revoked == 1
        assert bypass.bypass_report().standing == 0

    def test_a_rejection_for_a_field_nobody_approved_revokes_nothing(self):
        bypass = ApprovedPairBypass(
            read_feedback_trail(
                [json.dumps(_record(field="f", verdict="REJECTED", chosenGovernanceId=""))]
            )
        )
        bypass.bind(_lookup())
        assert bypass.bypass_report().revoked == 0
        assert bypass.bypass_report().verdicts == 0

    def test_an_unbound_consumer_answers_nothing(self):
        """
        Inert until `bind`. A consumer that answered from remembered ids before it had seen
        the loaded dictionary would be answering about a glossary it has never read.
        """
        bypass = ApprovedPairBypass(read_feedback_trail([json.dumps(_record(field="f"))]))
        assert bypass.approved_pair(_field("f")) is None


# =============================================================================
# WHAT INVALIDATES A PAIR
# =============================================================================


class TestInvalidation:
    @staticmethod
    def _bound() -> ApprovedPairBypass:
        bypass = ApprovedPairBypass(
            read_feedback_trail([json.dumps(_record(field="f", chosenGovernanceId="GBF-0001"))])
        )
        bypass.bind(_lookup())
        assert bypass.approved_pair(_field("f")) is not None
        return bypass

    def test_an_unchanged_glossary_leaves_the_pair_standing(self):
        bypass = self._bound()
        bypass.bind(_lookup())
        assert bypass.approved_pair(_field("f")) is not None
        assert bypass.bypass_report().invalidated == 0

    def test_a_redefinition_drops_the_pair(self):
        bypass = self._bound()
        moved = dict(GLOSSARY)
        moved["GBF-0001"] = _entry("GBF-0001", "Resident Full Name")
        moved["GBF-0001"] = DictionaryEntry(
            id=DocumentId("GBF-0001"),
            business_name="Resident Full Name",
            logical_name="resident_full_nm",
            definition="Now means the legal account holder, not the occupant.",
            data_type=DataType.STRING,
            governance_code="PC-3",
        )
        bypass.bind(_lookup(moved))

        assert bypass.approved_pair(_field("f")) is None
        assert bypass.bypass_report().invalidated == 1

    def test_a_reclassification_drops_the_pair_although_the_definition_is_unchanged(self):
        """
        THE CASE A CONTENT-HASH BINDING MISSES.

        Nothing a reader would call "the term" has changed -- same name, same definition,
        same type -- but the protection class the field would INHERIT has. That is what the
        reviewer was approving, so the approval is stale, and the assertion on
        `content_hash` below makes the premise explicit instead of leaving it to the reader.
        """
        bypass = self._bound()
        reclassified = dict(GLOSSARY)
        reclassified["GBF-0001"] = _entry("GBF-0001", "Resident Full Name", code="PC-9")

        assert reclassified["GBF-0001"].content_hash == GLOSSARY["GBF-0001"].content_hash

        bypass.bind(_lookup(reclassified))
        assert bypass.approved_pair(_field("f")) is None
        assert bypass.bypass_report().invalidated == 1

    def test_a_deleted_term_drops_the_pair_and_is_counted_separately(self):
        """
        Separate from `invalidated` because it needs a different person. A missing term is
        either a deletion or the wrong glossary loaded; a moved binding is a term somebody
        edited.

        A pair that HAD bound counts as `retired` rather than `unresolved`: it is gone for
        good (see `test_a_dropped_pair_does_not_come_back_when_the_term_does`), and
        `unresolved` is a state recomputed from the pairs still held, so it cannot go on
        describing one that is no longer held. `unresolved` keeps the pairs that have never
        bound and are still being retried.
        """
        bypass = self._bound()
        without = {k: v for k, v in GLOSSARY.items() if k != "GBF-0001"}
        bypass.bind(_lookup(without))

        assert bypass.approved_pair(_field("f")) is None
        assert bypass.bypass_report().retired == 1
        assert bypass.bypass_report().unresolved == 0
        assert bypass.bypass_report().invalidated == 0

    def test_a_dropped_pair_does_not_come_back_when_the_term_does(self):
        """
        What a reviewer approved was the term as it was. A term that comes back may not be
        the term that left, and a glossary that flapped must not resurrect an approval
        nobody re-gave.
        """
        bypass = self._bound()
        bypass.bind(_lookup({k: v for k, v in GLOSSARY.items() if k != "GBF-0001"}))
        bypass.bind(_lookup())
        assert bypass.approved_pair(_field("f")) is None

    def test_a_pair_that_has_never_bound_is_retried_rather_than_retired(self):
        """
        The asymmetry the test above must not be over-read into.

        A pair that has NEVER resolved has never been accepted against anything, so there
        is no "term as it was" to protect. Retiring it would mean that starting a server
        against the wrong glossary once destroys a whole review history for the lifetime of
        the process -- a failure with no symptom except a bypass that quietly does nothing.
        """
        bypass = ApprovedPairBypass(
            read_feedback_trail([json.dumps(_record(field="f", chosenGovernanceId="GBF-0001"))])
        )
        bypass.bind(_lookup({k: v for k, v in GLOSSARY.items() if k != "GBF-0001"}))
        assert bypass.approved_pair(_field("f")) is None
        assert bypass.bypass_report().unresolved == 1

        bypass.bind(_lookup())
        assert bypass.approved_pair(_field("f")) is not None
        assert bypass.bypass_report().unresolved == 0

    def test_an_invalidated_pair_is_never_retried_even_if_the_term_reverts(self):
        """
        `invalidated` is an EVENT, not a state. A term that was re-defined and then reverted
        is still a term a human has not looked at since it moved.
        """
        bypass = self._bound()
        moved = dict(GLOSSARY)
        moved["GBF-0001"] = _entry("GBF-0001", "Resident Full Name", code="PC-9")
        bypass.bind(_lookup(moved))
        assert bypass.approved_pair(_field("f")) is None

        bypass.bind(_lookup())
        assert bypass.approved_pair(_field("f")) is None
        assert bypass.bypass_report().invalidated == 1

    def test_the_report_reconciles_against_a_fixed_denominator(self):
        """
        `verdicts` is fixed at construction, so a shortfall is visible. A denominator that
        shrank to match the numerator would report "all standing" for a consumer that had
        quietly dropped half its pairs.
        """
        bypass = ApprovedPairBypass(
            read_feedback_trail(
                [
                    json.dumps(_record(field="a", chosenGovernanceId="GBF-0001")),
                    json.dumps(_record(field="b", chosenGovernanceId="GBF-0002")),
                    json.dumps(_record(field="c", chosenGovernanceId="GBF-9999")),
                ]
            )
        )
        bypass.bind(_lookup())
        report = bypass.bypass_report()

        assert report.verdicts == 3
        assert _reconciles(report)
        assert report.unresolved == 1
        assert report.standing == 2

        # And it keeps reconciling after a glossary change retires one of the two.
        moved = dict(GLOSSARY)
        moved["GBF-0001"] = _entry("GBF-0001", "Resident Full Name", code="PC-9")
        bypass.bind(_lookup(moved))
        after = bypass.bypass_report()
        assert after.verdicts == 3
        assert _reconciles(after)
        assert (after.standing, after.unresolved, after.retired, after.invalidated) == (1, 1, 0, 1)

    def test_the_ambiguous_count_is_the_size_of_the_pre_widening_loss(self):
        legacy = _record(field="a", wasCorrect=False)
        legacy.pop("verdict")
        bypass = ApprovedPairBypass(
            read_feedback_trail(
                [
                    json.dumps(legacy),
                    json.dumps(_record(field="b", verdict="APPROVED", wasCorrect=True)),
                ]
            )
        )
        bypass.bind(_lookup())

        assert bypass.bypass_report().standing == 2
        assert bypass.bypass_report().ambiguous == 1


# =============================================================================
# THE CONSERVATION IDENTITY, THROUGH EVERY BIND
# =============================================================================


def _reconciles(report) -> bool:
    """The identity `BypassReport` claims, in one place so no test can quote a subset."""
    total = report.standing + report.unresolved + report.retired + report.invalidated
    return total == report.verdicts


class TestTheConservationIdentityHoldsAtEveryMoment:
    """
    The identity is claimed "at every moment", so it is checked at every moment.

    A ONE-BIND ASSERTION CANNOT SEE THIS. The break needs three binds and a pair that HAS
    bound: a pair whose term is deleted after it bound is retired out of `_standing`, so
    the next bind cannot recount it, and a `unresolved` that is recomputed from `_standing`
    reports zero for a pair whose review no longer applies and never will again. A test
    driven with a NEVER-bound pair -- which is retried and stays in `_standing` -- passes
    while the invariant is false, which is exactly what happened.

    `retired` is what closes it, and it is an EVENT count for the same reason `invalidated`
    is: the pair is gone for good, and a number recomputed from what is still held cannot
    describe something that is no longer held.
    """

    @staticmethod
    def _two_bound() -> ApprovedPairBypass:
        return ApprovedPairBypass(
            read_feedback_trail(
                [
                    json.dumps(_record(field="a", chosenGovernanceId="GBF-0001")),
                    json.dumps(_record(field="b", chosenGovernanceId="GBF-0002")),
                ]
            )
        )

    def test_a_deletion_after_a_bind_stays_counted_on_every_later_bind(self):
        """
        THE DEFECT, driven directly. Bind 1 with both terms, bind 2 without `b`'s, then
        bind twice more. The operator's only view of "does my review history still apply?"
        must keep reconciling against a denominator of 2 rather than silently becoming a
        view of 1, because the whole point of the fixed denominator is that a shortfall is
        visible instead of hidden by a numerator that shrank to match.
        """
        bypass = self._two_bound()
        without_b = {k: v for k, v in GLOSSARY.items() if k != "GBF-0002"}

        seen = []
        for glossary in (GLOSSARY, without_b, GLOSSARY, GLOSSARY):
            bypass.bind(_lookup(glossary))
            report = bypass.bypass_report()
            assert report.verdicts == 2
            assert _reconciles(report), (
                f"standing={report.standing} unresolved={report.unresolved} "
                f"retired={report.retired} invalidated={report.invalidated} does not sum "
                f"to verdicts={report.verdicts}: a verdict has vanished from the only "
                f"view an operator has of whether their review history still applies"
            )
            seen.append((report.standing, report.unresolved, report.retired, report.invalidated))

        assert seen == [(2, 0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0), (1, 0, 1, 0)]

    def test_the_restored_term_does_not_resurrect_the_pair_or_the_count(self):
        """
        NON-VACUITY for the run above: bind 3 puts `GBF-0002` back, so `retired` staying at
        1 has to be a decision rather than an artifact of the term still being absent.
        """
        bypass = self._two_bound()
        bypass.bind(_lookup())
        bypass.bind(_lookup({k: v for k, v in GLOSSARY.items() if k != "GBF-0002"}))
        bypass.bind(_lookup())

        assert bypass.approved_pair(_field("b")) is None
        assert bypass.bypass_report().retired == 1

    def test_the_four_terminal_states_are_counted_once_each_and_never_twice(self):
        """
        All four outcomes in one consumer, so the identity is exercised where the four
        numbers can actually disagree rather than where three of them are zero.

        `a` stands, `b` binds and is then deleted (retired), `c` has never bound and is
        retried (unresolved), `d` binds and is then re-classified (invalidated). A pair
        counted in two buckets at once oversums; one counted in none undersums. Only the
        exact tuple passes.
        """
        bypass = ApprovedPairBypass(
            read_feedback_trail(
                [
                    json.dumps(_record(field="a", chosenGovernanceId="GBF-0001")),
                    json.dumps(_record(field="b", chosenGovernanceId="GBF-0002")),
                    json.dumps(_record(field="c", chosenGovernanceId="GBF-9999")),
                    json.dumps(_record(field="d", chosenGovernanceId="GBF-0003")),
                ]
            )
        )
        bypass.bind(_lookup())
        assert bypass.bypass_report().standing == 3

        changed = {k: v for k, v in GLOSSARY.items() if k != "GBF-0002"}
        changed["GBF-0003"] = _entry("GBF-0003", "Meter Serial Number", code="PC-9")
        bypass.bind(_lookup(changed))

        report = bypass.bypass_report()
        assert report.verdicts == 4
        assert (report.standing, report.unresolved, report.retired, report.invalidated) == (
            1,
            1,
            1,
            1,
        )
        assert _reconciles(report)

        # And it still reconciles once the missing term finally arrives: `c` resolves and
        # moves out of `unresolved` -- a STATE -- while `b` and `d` stay counted, because
        # `retired` and `invalidated` are EVENTS that no later load undoes.
        arrived = dict(GLOSSARY)
        arrived["GBF-9999"] = _entry("GBF-9999", "Supply Point Reference")
        bypass.bind(_lookup(arrived))
        settled = bypass.bypass_report()
        assert (settled.standing, settled.unresolved, settled.retired, settled.invalidated) == (
            2,
            0,
            1,
            1,
        )
        assert _reconciles(settled)

    def test_a_pair_that_never_bound_is_unresolved_and_not_retired(self):
        """
        The two deletion-shaped counts must not be one count wearing two names. A pair that
        has never bound is a STATE that a correct glossary fixes; a pair that bound and
        then went missing is an EVENT that no later load undoes. Reporting the first as
        `retired` would tell an operator a review is gone for good when loading the right
        glossary would bring it back.
        """
        bypass = ApprovedPairBypass(
            read_feedback_trail([json.dumps(_record(field="f", chosenGovernanceId="GBF-9999"))])
        )
        bypass.bind(_lookup())
        bypass.bind(_lookup())

        report = bypass.bypass_report()
        assert (report.unresolved, report.retired) == (1, 0)
        assert _reconciles(report)


# =============================================================================
# THE KEY -- THE REPEATED-LEAF HAZARD
# =============================================================================


class TestTheKeyCarriesTheParent:
    """
    One leaf name, many parents, a different governed term under each. The direct test of
    key composition, run against the generator that builds that shape on purpose.
    """

    @staticmethod
    def _repeated_leaf_fields() -> tuple[list[SchemaField], dict[str, str]]:
        from benchmarks.synthetic.pack import PackSpec, SyntheticPack
        from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import (
            FlattenedAvroParser,
        )

        pack = SyntheticPack.generate(PackSpec(rows=1000, feedback_events=10, schema_scale=0.05))
        schema = next(s for s in pack.schemas if s.name == "nested-repeated")
        answers = {t.flattened_name: t.correct_ids[0] for t in schema.truth if t.correct_ids}
        parsed = FlattenedAvroParser().parse(schema.flattened)
        return list(parsed.unwrap().fields), answers

    def test_the_fixture_really_does_repeat_a_leaf_under_different_answers(self):
        """
        NON-VACUITY. Without this the test below could pass on a schema where every leaf
        name is unique, proving nothing at all about key composition.
        """
        fields, answers = self._repeated_leaf_fields()
        by_leaf: dict[str, set[str]] = {}
        for field in fields:
            key = field.source_metadata["flattened_name"]
            leaf = str(field.source_metadata["leafName"])
            by_leaf.setdefault(leaf, set()).add(answers[key])

        widest = max(by_leaf.values(), key=len)
        assert len(widest) >= 3, (
            "no leaf name in this fixture is governed by three different terms, so the "
            "repeated-leaf hazard is not present and the test below measures nothing"
        )
        assert Counter(str(f.name) for f in fields).most_common(1)[0][1] > 1, (
            "the flattened parser no longer produces repeated SchemaField.name values"
        )

    def test_approving_one_occurrence_does_not_answer_the_others(self):
        """
        THE DEFECT THIS PREVENTS is not an exception and not a miss. A bypass keyed on the
        leaf would answer all N occurrences with the term a reviewer approved for one of
        them, and N-1 of those are confidently wrong classifications with no symptom except
        a count nobody has reason to check.
        """
        fields, answers = self._repeated_leaf_fields()
        by_leaf: dict[str, list[SchemaField]] = {}
        for field in fields:
            by_leaf.setdefault(str(field.source_metadata["leafName"]), []).append(field)
        siblings = max(by_leaf.values(), key=len)

        approved, *others = siblings
        approved_key = str(approved.source_metadata["flattened_name"])
        approved_id = answers[approved_key]

        entries = {
            entry_id: _entry(entry_id, f"Term {entry_id}")
            for entry_id in {answers[str(f.source_metadata["flattened_name"])] for f in siblings}
        }
        bypass = ApprovedPairBypass(
            read_feedback_trail(
                [
                    json.dumps(
                        _record(
                            field=approved_key,
                            chosenGovernanceId=approved_id,
                            verdict="APPROVED",
                            wasCorrect=True,
                        )
                    )
                ]
            )
        )
        bypass.bind(MappingEntryLookup(entries))

        assert bypass.approved_pair(approved).entry.id == approved_id
        leaked = [
            str(f.source_metadata["flattened_name"])
            for f in others
            if bypass.approved_pair(f) is not None
        ]
        assert not leaked, (
            f"{len(leaked)} sibling column(s) sharing the leaf name were answered from a "
            f"verdict given about a different parent: {leaked[:3]}"
        )

    def test_a_second_verdict_on_a_sibling_answers_only_that_sibling(self):
        """
        The other half: the key must not be so wide that two genuinely different columns
        can never both be approved. Two siblings, two verdicts, two different answers.
        """
        fields, answers = self._repeated_leaf_fields()
        by_leaf: dict[str, list[SchemaField]] = {}
        for field in fields:
            by_leaf.setdefault(str(field.source_metadata["leafName"]), []).append(field)
        siblings = max(by_leaf.values(), key=len)

        keys = [str(f.source_metadata["flattened_name"]) for f in siblings]
        pair = [(k, answers[k]) for k in keys]
        distinct = {}
        for key, entry_id in pair:
            distinct.setdefault(entry_id, key)
        first_id, first_key = next(iter(distinct.items()))
        second_id, second_key = list(distinct.items())[1]

        entries = {i: _entry(i, f"Term {i}") for i in (first_id, second_id)}
        bypass = ApprovedPairBypass(
            read_feedback_trail(
                [
                    json.dumps(
                        _record(field=k, chosenGovernanceId=i, verdict="APPROVED", wasCorrect=True)
                    )
                    for k, i in ((first_key, first_id), (second_key, second_id))
                ]
            )
        )
        bypass.bind(MappingEntryLookup(entries))

        by_key = {str(f.source_metadata["flattened_name"]): f for f in siblings}
        assert bypass.approved_pair(by_key[first_key]).entry.id == first_id
        assert bypass.approved_pair(by_key[second_key]).entry.id == second_id
