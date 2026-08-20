"""
tests.unit.presentation.api.test_governance_id_opacity | Layer: TEST
`governanceId` is an OPAQUE STRING. It is never a number, and nothing here treats it as one.

## Relationships
# TESTS → presentation/api/matching :: _governance_id, _candidate_payload
# TESTS → presentation/api/lookup   :: _entry_payload, LookupRequest.ids, the missing list
# TESTS → presentation/api/introspect :: the retrieval diagnostic's candidate ids

## The definition this file pins

An enterprise adoption specification listed "what IS `governanceId`?" as a documentation
BLOCKER, and the maintainer's answer was that on their deployment it is *just a number from
1 to 10000000*. That answer describes THEIR id space. It is not what this library
promises, and the gap between the two is the whole reason this file exists.

What the library promises is narrower and stronger: **`governanceId` is the dictionary
entry's own id, carried as an opaque string.** It is `type: string` in the published
schema, `String` in the Java client, and it is echoed back byte for byte. The library
never parses it, never widens it, never narrows it, and never compares two of them for
anything except equality.

Those two statements are compatible only while the second one holds. A deployment whose
ids happen to be numerals is EXACTLY the deployment where a consumer is most tempted to
parse them, and every one of the failures below is silent:

  * `0000123` is not `123`. A glossary that zero-pads its keys has two spellings of the
    same digits and only one of them is an id. A consumer that normalises by parsing
    resolves the wrong row, or none.
  * `9007199254740993` is not `9007199254740992`, but `float64` cannot tell them apart --
    and JSON in a language with no integer type (JavaScript, Lua, jq without `--raw`, a
    good deal of Excel) parses every number as `float64`. Two distinct governance rows
    collapse onto one, and the column silently inherits the wrong protection class. 2^53+1
    is the classic demonstration of this and it is used below on purpose.
  * An id compared or sorted as a number reorders a response whose order IS its contract.

So this file pins opacity in three ways, and each of them is a different failure:

  BYTES        the id crosses the wire as a JSON STRING, padding intact -- checked against
               the raw response body, not against a parsed dict, because `json.loads`
               would hide a numeric emission behind a Python `int` that compares equal to
               nothing the test asserts.
  RESOLUTION   only the exact string resolves. The unpadded form, and the float64 image of
               a large id, land in `missing`. There is no numeric equivalence anywhere.
  ORDER        permuting which id sits on which entry changes NOTHING except the ids. If
               anything in the library sorted, compared or ranged over an id, this is the
               test that would notice.

## Where the ids come from

They are the fictional Lumenport glossary from `_support.py` with numeric ids substituted,
so nothing new is invented here and the vocabulary stays the one that file already
documents as made up. The id VALUES are chosen for what they demonstrate:

    1                   the bottom of the range the maintainer described
    0000123             zero-padded, and `123` is deliberately NOT an id
    10000000            the top of the range the maintainer described
    9007199254740993    2^53 + 1
    9007199254740992    2^53, a DIFFERENT entry -- float64 merges these two
    007                 short zero-padding, and `7` is deliberately NOT an id
"""

from __future__ import annotations

import dataclasses
import json
import re

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api import matching
from nexus_matcher.presentation.api.app import create_app
from tests.unit.presentation.api._support import (
    GLOSSARY,
    build_api_matcher,
    request_fields,
)

# =============================================================================
# THE NUMERIC-ID GLOSSARY
# =============================================================================

# One id per Lumenport entry, in the fixture's order. Strings, always: an id is what the
# caller's glossary column held, and this library has no opinion about its shape.
NUMERIC_IDS: tuple[str, ...] = (
    "1",
    "0000123",
    "10000000",
    "9007199254740993",
    "007",
)

# 2^53 and 2^53+1. Two DISTINCT entries whose ids are indistinguishable as `float64`, which
# is the state a consumer parsing ids as JSON numbers arrives at without being told.
BIG = "9007199254740993"
BIG_FLOAT_TWIN = "9007199254740992"

NUMERIC_GLOSSARY = tuple(
    dataclasses.replace(entry, id=new_id)
    for entry, new_id in zip(GLOSSARY, NUMERIC_IDS, strict=True)
)

# A sixth entry carrying the float64 twin of the largest id, so the two are REAL and
# DIFFERENT glossary rows rather than one row and one miss. A consumer that round-trips ids
# through a float64 does not merely fail to resolve one of these -- it resolves the other,
# and inherits a protection class from a row it never asked for. That is the failure worth
# building a fixture for.
FLOAT_TWIN_GLOSSARY = (
    *NUMERIC_GLOSSARY,
    dataclasses.replace(
        GLOSSARY[2],
        id=BIG_FLOAT_TWIN,
        business_name="Hydrant Flow Litres",
        logical_name="hydrant_flow_litres",
        definition="Water drawn through a hydrant during a scheduled flow test.",
    ),
)

# The id on each entry, keyed by business name, so a test can name the row it expects
# without restating the fixture.
NAME_BY_ID = {entry.id: entry.business_name for entry in FLOAT_TWIN_GLOSSARY}


# =============================================================================
# THE WIRE-LEVEL ASSERTION
# =============================================================================

# `"governanceId":` followed by anything that is not the opening quote of a string. A
# response that emitted the id as a JSON number, a float, or null matches this; a correct
# one cannot.
#
# Applied to the RAW BODY. `response.json()` would parse `"governanceId": 123` into a
# Python `int` and `"governanceId": 0000123` into a syntax error or 123 depending on the
# parser, and neither outcome is what a Java, JavaScript or jq consumer sees. The bytes are
# what those consumers get, so the bytes are what is asserted.
UNQUOTED_ID = re.compile(rb'"governanceId"\s*:\s*[^"]')


def unquoted_ids(body: bytes) -> list[str]:
    """Every place this body spelled an id as something other than a JSON string."""
    return [match.group(0).decode("ascii", "replace") for match in UNQUOTED_ID.finditer(body)]


def quoted(entry_id: str) -> bytes:
    """The exact bytes a correct response spells this id as."""
    return b'"governanceId":"' + entry_id.encode("ascii") + b'"'


# =============================================================================
# CLIENTS
# =============================================================================


def client_for(matcher: object, **kwargs: object) -> TestClient:
    app = create_app(configure_logs=False, matcher=matcher, environ={}, **kwargs)
    return TestClient(app)


@pytest.fixture
def numeric_client():
    """A real matcher over the glossary whose ids are numerals."""
    with client_for(build_api_matcher(entries=FLOAT_TWIN_GLOSSARY)) as client:
        yield client


def lookup(client: TestClient, *ids: str):
    response = client.post("/api/v1/lookup", json={"ids": list(ids)})
    assert response.status_code == 200, response.text
    return response


def match(client: TestClient, **body: object):
    payload: dict[str, object] = {"fields": request_fields(), "top_k": 5}
    payload.update(body)
    response = client.post("/api/v1/match", json=payload)
    assert response.status_code == 200, response.text
    return response


# =============================================================================
# BYTES -- THE ID IS A JSON STRING, PADDING INTACT
# =============================================================================


class TestTheIdCrossesTheWireAsAString:
    """Asserted on the raw body, because a parsed dict cannot show a coercion."""

    @pytest.mark.parametrize("entry_id", sorted(NAME_BY_ID))
    def test_every_id_is_quoted_and_exact_on_lookup(self, numeric_client, entry_id):
        response = lookup(numeric_client, entry_id)

        assert quoted(entry_id) in response.content, (
            f"the lookup body does not contain the exact bytes "
            f"{quoted(entry_id)!r}. The id is the caller's own key and it is echoed, "
            f"not re-rendered: body was {response.content[:400]!r}"
        )
        assert not unquoted_ids(response.content)
        assert response.json()["results"][entry_id]["governanceId"] == entry_id

    @pytest.mark.parametrize("entry_id", sorted(NAME_BY_ID))
    def test_every_id_is_quoted_and_exact_on_match(self, numeric_client, entry_id):
        response = match(numeric_client)

        assert quoted(entry_id) in response.content, (
            f"no candidate anywhere in the match body carries the exact bytes "
            f"{quoted(entry_id)!r}, so either the id was re-rendered or this fixture no "
            f"longer surfaces every entry -- both make this file weaker and neither is "
            f"something to fix by relaxing the assertion"
        )
        assert not unquoted_ids(response.content)

    def test_the_zero_padding_survives_and_the_unpadded_spelling_never_appears(
        self, numeric_client
    ):
        """
        The narrow claim, stated on its own because it is the one an id-normalising
        consumer breaks first.
        """
        response = lookup(numeric_client, "0000123", "007")

        assert b'"governanceId":"0000123"' in response.content
        assert b'"governanceId":"007"' in response.content
        assert b'"governanceId":"123"' not in response.content
        assert b'"governanceId":"7"' not in response.content

    def test_the_retrieval_diagnostic_spells_ids_the_same_way(self, numeric_client):
        """
        The third surface that carries an id, and the one nobody remembers. A diagnostic
        that numericised the id would put a DIFFERENT id in the operator's ticket from the
        one the match response carried, which is a worse failure than either alone.
        """
        response = numeric_client.post(
            "/api/v1/diag/retrieval",
            json={
                "field": {
                    "name": "resident_nm",
                    "path": "account.resident_nm",
                    "doc": "Name of the resident on the account",
                    "type": "string",
                }
            },
        )

        assert response.status_code == 200, response.text
        assert b'"governanceId"' in response.content, (
            "the retrieval diagnostic no longer carries candidate ids, so this test is "
            "watching nothing"
        )
        assert not unquoted_ids(response.content)

    def test_the_published_schema_types_the_id_as_a_bare_string(self, numeric_client):
        """
        What a generated client binds. `type: string` and NOTHING ELSE -- a `format`, a
        `pattern` or a numeric bound would each be this library asserting a shape for
        somebody else's key column, and `format: int64` in particular is how a generated
        client ends up with a `long`.
        """
        schemas = numeric_client.get("/openapi.json").json()["components"]["schemas"]

        carriers = {
            name: schema["properties"]["governanceId"]
            for name, schema in schemas.items()
            if "governanceId" in schema.get("properties", {})
        }
        assert {"MatchCandidateView", "LookupEntryView"} <= set(carriers), (
            f"the two schemas this contract is stated on are no longer both published; "
            f"found {sorted(carriers)}"
        )

        for name, published in sorted(carriers.items()):
            assert published.get("type") == "string", (
                f"{name}.governanceId is published as {published.get('type')!r}. A "
                f"generated client binds that type, and a numeric one loses a zero-padded "
                f"id on the first parse."
            )
            constraints = {
                "format",
                "pattern",
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
                "minLength",
                "maxLength",
            } & set(published)
            assert not constraints, (
                f"{name}.governanceId now publishes {sorted(constraints)}. The id is the "
                f"caller's own key and this library does not get to constrain its shape; "
                f"a `format` is what turns a String into a long in a generated client."
            )

    def test_the_request_side_declares_ids_as_strings_too(self, numeric_client):
        """The other half: what a client is told to SEND is a string as well."""
        schemas = numeric_client.get("/openapi.json").json()["components"]["schemas"]
        ids = schemas["LookupRequest"]["properties"]["ids"]

        assert ids["type"] == "array"
        assert ids["items"] == {"type": "string"}, (
            f"LookupRequest.ids items are published as {ids['items']}; a numeric item type "
            f"would tell every generated client to parse its own ids"
        )


# =============================================================================
# RESOLUTION -- NO NUMERIC EQUIVALENCE, ANYWHERE
# =============================================================================


class TestOnlyTheExactStringResolves:
    """Equality on the string, and nothing else. Absence is an answer, not a near miss."""

    def test_the_unpadded_form_does_not_resolve_the_padded_entry(self, numeric_client):
        """
        THE CENTRAL CASE. Both spellings are sent in one request, so "the padded one
        resolved" and "the unpadded one missed" are two facts about ONE response and
        cannot be an artefact of two different requests.
        """
        body = lookup(numeric_client, "0000123", "123").json()

        assert body["results"]["0000123"]["governanceId"] == "0000123"
        assert body["results"]["123"] is None, (
            "`123` resolved. The library is treating the id as a number somewhere, and "
            "every zero-padded glossary now has two spellings of one key."
        )
        assert body["missing"] == ["123"]

    def test_it_holds_on_the_single_id_route_as_well(self, numeric_client):
        """
        Two routes, one resolver. A GET that normalised its path segment while the POST
        did not would be the same defect reachable from half the client code.
        """
        padded = numeric_client.get("/api/v1/lookup/0000123")
        unpadded = numeric_client.get("/api/v1/lookup/123")

        assert padded.status_code == 200 and unpadded.status_code == 200
        assert padded.json()["results"]["0000123"]["governanceId"] == "0000123"
        assert unpadded.json()["results"]["123"] is None
        assert unpadded.json()["missing"] == ["123"]

    @pytest.mark.parametrize("absent", ["123", "7", "1.0", "01", "1e7", "10000000.0"])
    def test_an_arithmetic_image_of_a_real_id_is_a_miss(self, numeric_client, absent):
        """
        Every spelling a consumer's number parser could emit for an id that IS in this
        glossary. Each one must miss: a hit would mean the library normalised, and a
        library that normalises has silently chosen one deployment's id convention.
        """
        assert absent not in NAME_BY_ID, "this fixture now contains the id it calls absent"

        body = lookup(numeric_client, absent).json()

        assert body["results"][absent] is None
        assert body["missing"] == [absent]

    def test_two_ids_a_float64_would_merge_stay_two_entries(self, numeric_client):
        """
        2^53 and 2^53+1, which `float64` cannot tell apart, on two DIFFERENT glossary rows.

        The first assertion is the hazard itself, stated in Python so that this test says
        out loud why the two values were chosen rather than leaving it in a comment. The
        rest is the library declining to have it: each id resolves to its own row, and a
        consumer that parsed ids as JSON numbers would have inherited the wrong row's
        protection class without any error anywhere.
        """
        assert float(BIG) == float(BIG_FLOAT_TWIN), (
            "the two ids this test is built on are no longer float64-equal, so it has "
            "stopped demonstrating the failure it names"
        )
        assert int(float(BIG)) == int(BIG_FLOAT_TWIN)
        assert BIG != BIG_FLOAT_TWIN

        body = lookup(numeric_client, BIG, BIG_FLOAT_TWIN).json()

        assert body["missing"] == []
        assert body["results"][BIG]["businessName"] == NAME_BY_ID[BIG]
        assert body["results"][BIG_FLOAT_TWIN]["businessName"] == NAME_BY_ID[BIG_FLOAT_TWIN]
        assert (
            body["results"][BIG]["businessName"] != body["results"][BIG_FLOAT_TWIN]["businessName"]
        ), (
            "the two float64-equal ids resolved to the same entry. A consumer parsing ids "
            "as JSON numbers -- which is every consumer in a language with no integer "
            "type -- would inherit one row's protection class onto the other's column."
        )

    @pytest.mark.parametrize("entry_id", sorted(NAME_BY_ID))
    def test_the_whole_stated_range_behaves_identically(self, numeric_client, entry_id):
        """
        1, 10000000, and 2^53+1 answered by the same code path with the same guarantees.
        The maintainer's range is a fact about one deployment; the behaviour must not
        change anywhere inside it, or at the point where a consumer's parser gives up.
        """
        body = lookup(numeric_client, entry_id).json()

        assert body["missing"] == []
        assert body["results"][entry_id]["governanceId"] == entry_id
        assert body["results"][entry_id]["businessName"] == NAME_BY_ID[entry_id]

    def test_an_id_sent_as_a_json_number_is_refused_rather_than_coerced(self, numeric_client):
        """
        The request side of the same promise, and the one that protects the consumer this
        whole file is about.

        A client whose language has no integer type will eventually send `ids: [123]`. The
        dangerous answer is a 200 that quietly stringifies it, because then the padded and
        unpadded spellings differ only in whether the client remembered to quote -- and the
        one that resolves is whichever the caller typed. A 422 naming the exact element is
        the answer that sends them back to their own serialiser.
        """
        response = numeric_client.post("/api/v1/lookup", json={"ids": [123]})

        assert response.status_code == 422, response.text
        violations = response.json()["error"]["details"]["violations"]
        assert any(v["location"][:2] == ["body", "ids"] for v in violations), violations
        assert any("string" in v["message"].lower() for v in violations), violations


# =============================================================================
# ORDER -- NOTHING SORTS, COMPARES OR RANGES OVER THE ID
# =============================================================================


class TestNothingRanksTheId:
    """
    Equality is the only operation. Any other one would show up as a reordering.
    """

    def test_lookup_answers_in_the_order_sent_and_not_in_id_order(self, numeric_client):
        """
        Sent in an order that is neither ascending nor descending, numerically or
        lexicographically, so a response that sorted by ANY reading of the id is
        distinguishable from one that kept the caller's order.
        """
        sent = ["10000000", "1", BIG, "0000123", "123", "007"]
        assert sent != sorted(sent), "the probe order is now lexicographically sorted"
        assert sent != sorted(sent, reverse=True)

        body = lookup(numeric_client, *sent).json()

        assert list(body["results"]) == sent
        assert body["missing"] == ["123"], (
            "`missing` reports the ids that did not resolve in the order they were sent; "
            "a sorted or de-duplicated list would stop being addressable against the "
            "caller's own request"
        )

    def test_candidates_come_back_by_rank_and_the_ids_are_not_in_order(self, numeric_client):
        """
        Rank orders a shortlist. Asserting that the id sequence is NOT sorted is what
        makes the rank assertion non-vacuous: on a fixture whose ids happened to ascend
        with relevance, "ordered by rank" and "ordered by id" are the same sequence and
        neither claim is being tested.
        """
        body = match(numeric_client).json()

        unsorted_somewhere = False
        for path, candidates in body["results"].items():
            assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1)), path
            ids = [c["governanceId"] for c in candidates]
            if ids != sorted(ids) and ids != sorted(ids, reverse=True):
                unsorted_somewhere = True

        assert unsorted_somewhere, (
            "every field's candidates came back in id order. Either something is now "
            "sorting by id, or this fixture has stopped being able to tell rank order "
            "from id order -- and until somebody decides which, the rank assertion above "
            "proves nothing."
        )

    def test_permuting_which_id_sits_on_which_entry_changes_only_the_ids(self):
        """
        THE GENERAL FORM, and the one that would catch a comparison this file never
        thought of.

        Two matchers over the same five glossary rows, differing ONLY in which numeric id
        is attached to which row. Retrieval sees the business name and the definition; the
        id is a label the pipeline carries and never reads. So every field's candidate
        list -- the names, the ranks, the confidences, the decisions, the verdicts -- must
        be identical between the two, and the id sequence must be the permutation.

        If anything anywhere sorted candidates by id, broke a tie on it, bucketed by a
        numeric range or compared two ids for order, this is where it surfaces. A test
        that enumerated the operations instead would only ever check the ones somebody
        remembered.
        """
        rotated_ids = (*NUMERIC_IDS[1:], *NUMERIC_IDS[:1])
        rotated = tuple(
            dataclasses.replace(entry, id=new_id)
            for entry, new_id in zip(NUMERIC_GLOSSARY, rotated_ids, strict=True)
        )
        assert [e.id for e in rotated] != [e.id for e in NUMERIC_GLOSSARY]
        assert {e.business_name for e in rotated} == {e.business_name for e in NUMERIC_GLOSSARY}

        with client_for(build_api_matcher(entries=NUMERIC_GLOSSARY)) as client:
            original = match(client).json()
        with client_for(build_api_matcher(entries=rotated)) as client:
            permuted = match(client).json()

        assert original["fieldDecisions"] == permuted["fieldDecisions"]
        assert list(original["results"]) == list(permuted["results"])

        moved = 0
        for path, candidates in original["results"].items():
            others = permuted["results"][path]
            assert [c["businessName"] for c in candidates] == [c["businessName"] for c in others], (
                f"{path} came back in a different ENTRY order once the ids were permuted, "
                f"so something in the pipeline is ordering by the id"
            )
            assert [c["rank"] for c in candidates] == [c["rank"] for c in others], path
            assert [c["confidence"] for c in candidates] == [c["confidence"] for c in others], path
            assert [c["decision"] for c in candidates] == [c["decision"] for c in others], path
            if [c["governanceId"] for c in candidates] != [c["governanceId"] for c in others]:
                moved += 1

        assert moved, (
            "permuting the ids changed no id in the response, so the two runs were not "
            "actually different and nothing above was tested"
        )


# =============================================================================
# VACUITY -- THIS FILE CAN SEE A COERCION
# =============================================================================


class TestTheseAssertionsAreNotVacuous:
    """
    Every check above reports a coercion as an ABSENCE -- a byte pattern that is not there,
    a regex that does not match, an id that did not resolve. A test shaped like that goes
    green forever the moment it stops looking at the right bytes, which is the failure this
    repository has been bitten by often enough to keep a control for.

    So the coercion is performed on purpose, against the shipped renderer, and the
    assertions are required to notice it.
    """

    def test_a_numericised_id_is_caught_on_the_match_plane(self, numeric_client, monkeypatch):
        """`_governance_id` is the one function the candidate renderer asks for the id."""
        monkeypatch.setattr(
            matching,
            "_governance_id",
            lambda result: int(result.dictionary_entry.id),
        )

        response = match(numeric_client)

        assert unquoted_ids(response.content), (
            "the id was rendered as a Python int and the raw-body scan still found "
            "nothing wrong, so UNQUOTED_ID has stopped matching what it claims to match"
        )
        assert quoted("0000123") not in response.content
        # And this is what the consumer would have been handed instead: the padding gone.
        assert b'"governanceId":123' in response.content

    def test_a_numericised_id_is_caught_on_the_lookup_plane(self, numeric_client, monkeypatch):
        """
        Same coercion, other plane. `_entry_payload` builds the id inline, so the whole
        payload function is wrapped rather than a helper patched -- which also proves the
        two planes are checked separately and one cannot cover for the other.
        """
        from nexus_matcher.presentation.api import lookup as lookup_module

        original = lookup_module._entry_payload

        def numericised(entry, vocabulary):
            payload = dict(original(entry, vocabulary))
            payload["governanceId"] = int(payload["governanceId"])
            return payload

        monkeypatch.setattr(lookup_module, "_entry_payload", numericised)

        response = lookup(numeric_client, "0000123")

        assert unquoted_ids(response.content), (
            "the lookup plane emitted a numeric id and the raw-body scan found nothing"
        )
        assert json.loads(response.content)["results"]["0000123"]["governanceId"] == 123
