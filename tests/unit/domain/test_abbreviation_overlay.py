"""
tests.unit.domain.test_abbreviation_overlay | Layer: TEST
Tests: AbbreviationDictionary.merged_with, AbbreviationExpander.with_overlay
Target: domain/services/abbreviation.py

The per-request abbreviation overlay's whole reason to exist is that the caller's
approved-abbreviation list is SOURCED LIVE and changes between calls. A startup-time
catalog cannot express that, so the merge has to be per-call -- which means the shared
expander must come out of it byte-for-byte unchanged, on every call, from every thread.

These tests are about that property, not about accuracy. Whether an overlay HELPS is a
measurement (see tests/unit/application/test_query_signals.py and the note in
`docs/guides/governed_abbreviations.md`); whether it leaks is a correctness bug that no
benchmark would ever show, because a leaked row makes the NEXT request better or worse
by an amount nobody attributes to the previous one.
"""

from __future__ import annotations

import threading

import pytest

from nexus_matcher.domain.services.abbreviation import (
    DEFAULT_ABBREVIATIONS,
    AbbreviationDictionary,
    AbbreviationExpander,
)

BASE = {"txn": "transaction", "amt": "amount"}
OVERLAY = {"psgr": "passenger", "brth": "berth"}


def _expander(mapping: dict[str, str] = BASE) -> AbbreviationExpander:
    return AbbreviationExpander(AbbreviationDictionary.from_dict(mapping))


class TestMergedWith:
    def test_overlay_rows_are_present(self):
        merged = AbbreviationDictionary.from_dict(BASE).merged_with(OVERLAY)
        assert merged.lookup("psgr") == "passenger"

    def test_base_rows_survive(self):
        merged = AbbreviationDictionary.from_dict(BASE).merged_with(OVERLAY)
        assert merged.lookup("txn") == "transaction"

    def test_overlay_wins_on_collision(self):
        merged = AbbreviationDictionary.from_dict(BASE).merged_with({"amt": "amortisation"})
        assert merged.lookup("amt") == "amortisation"

    def test_the_base_dictionary_is_not_mutated(self):
        base = AbbreviationDictionary.from_dict(BASE)
        base.merged_with(OVERLAY)
        assert base.lookup("psgr") is None
        assert base.size == len(BASE)

    def test_keys_are_normalised_like_every_other_row(self):
        merged = AbbreviationDictionary.from_dict(BASE).merged_with({"  PSGR  ": "passenger"})
        assert merged.lookup("psgr") == "passenger"

    def test_an_invalid_overlay_row_is_skipped_not_raised(self):
        # Same admission rule as `from_dict`: a caller's live feed is not a place to
        # raise from, and one malformed row must not cost the other 7,839.
        merged = AbbreviationDictionary.from_dict(BASE).merged_with(
            {"psgr": "passenger", "": "nothing", "nul": ""}
        )
        assert merged.lookup("psgr") == "passenger"
        assert merged.size == len(BASE) + 1

    def test_an_empty_overlay_changes_nothing(self):
        merged = AbbreviationDictionary.from_dict(BASE).merged_with({})
        assert merged.size == len(BASE)


class TestWithOverlay:
    def test_expands_an_overlay_row(self):
        assert _expander().with_overlay(OVERLAY).expand("psgr_nm").expanded == "passenger_nm"

    def test_still_expands_a_base_row(self):
        assert _expander().with_overlay(OVERLAY).expand("txn_x").expanded == "transaction_x"

    def test_the_shared_expander_is_untouched(self):
        shared = _expander()
        shared.with_overlay(OVERLAY)
        assert shared.expand("psgr_nm").expanded == "psgr_nm"

    def test_two_overlays_do_not_see_each_other(self):
        shared = _expander()
        a = shared.with_overlay({"psgr": "passenger"})
        b = shared.with_overlay({"brth": "berth"})
        assert a.expand("brth").expanded == "brth"
        assert b.expand("psgr").expanded == "psgr"

    def test_an_empty_overlay_returns_the_same_object(self):
        # Identity, not merely equality: the no-signal path must not pay a dictionary
        # copy per request, and must not be able to diverge from the shipped expander.
        shared = _expander()
        assert shared.with_overlay({}) is shared
        assert shared.with_overlay(None) is shared

    def test_the_default_singleton_does_not_absorb_an_overlay(self):
        AbbreviationExpander.reset_default()
        try:
            default = AbbreviationExpander.default()
            default.with_overlay({"psgr": "passenger"})
            assert AbbreviationExpander.default().expand("psgr").expanded == "psgr"
            assert AbbreviationExpander.default() is default
        finally:
            AbbreviationExpander.reset_default()

    def test_overlay_can_correct_a_wrong_shipped_row(self):
        # The bundled generic list asserts "st" -> "state", which is wrong inside an
        # address. A live catalog outranking it is the whole point of the channel.
        AbbreviationExpander.reset_default()
        try:
            assert DEFAULT_ABBREVIATIONS["st"] == "state"
            corrected = AbbreviationExpander.default().with_overlay({"st": "street"})
            assert corrected.expand("st_addr").expanded == "street_address"
        finally:
            AbbreviationExpander.reset_default()

    @pytest.mark.parametrize("threads", [8])
    def test_concurrent_overlays_never_bleed(self, threads: int):
        """
        The matcher is shared across concurrent requests. If `with_overlay` mutated
        anything shared, this is the shape that would catch it: every thread asserts on
        a short form only IT supplied.
        """
        shared = _expander()
        errors: list[str] = []
        barrier = threading.Barrier(threads)

        def run(i: int) -> None:
            # An exception inside a thread is invisible to the assertions below unless it
            # is recorded, and a silently-dead worker would make this test vacuous.
            try:
                mine = f"z{i}z"
                local = shared.with_overlay({mine: f"expansion{i}"})
                barrier.wait(timeout=10.0)
                for _ in range(200):
                    if local.expand(mine).expanded != f"expansion{i}":
                        errors.append(f"thread {i} lost its own row")
                        return
                    for j in range(threads):
                        if j != i and local.expand(f"z{j}z").expanded != f"z{j}z":
                            errors.append(f"thread {i} saw thread {j}'s row")
                            return
            except Exception as exc:
                errors.append(f"thread {i} raised {type(exc).__name__}: {exc}")

        workers = [threading.Thread(target=run, args=(i,)) for i in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=30.0)

        assert errors == []
        assert shared.expand("z0z").expanded == "z0z"
