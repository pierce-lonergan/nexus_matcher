"""
tests.unit.presentation.api.test_metadata_plane | Layer: TEST
The pass-through metadata plane, from the loader to the wire and back.

## Relationships
# TESTS → presentation/api/matching :: `sourceMetadata` on every candidate
# TESTS → presentation/api/lookup   :: the same object on every looked-up entry
# TESTS → presentation/api/schemas  :: the published shape a Java client generates from
# TESTS → application/ingest        :: the half that fills the plane, driven for real here

## Why this file exists

A deployment's glossary carries columns this library has no opinion about. The loader
already carried them onto the entry and into the index and the round trip was proven
byte-exact THAT far -- and then it stopped, because no response emitted them. The
capability existed everywhere except where a client could see it, which from a client's
side is indistinguishable from it not existing.

So this file drives the WHOLE chain: a real CSV through the real loader, into a real
matcher, through the real FastAPI app, and back out of `response.content` -- not a
hand-built entry handed to a renderer. The interesting failures live in the joins.

## The five properties, and why each is here

  ROUND TRIP   what the loader put in is what the response emits, byte for byte. The
               fixture value is deliberately awkward -- leading and trailing spaces, a
               non-breaking space, an em-dash, accents, a double quote, a tab and a
               newline -- because every one of those is something a layer might have
               "helpfully" trimmed, folded or re-encoded.
  ESCAPING     the body is ASCII-only by design, so a non-ASCII value travels as a JSON
               \\uXXXX escape. That is a representation of the character, not a
               substitution for it, and the test proves the distinction by parsing the
               body back and comparing to the original Python string.
  BOUNDEDNESS  the loader TRUNCATES an over-cap plane and records how many keys it
               dropped. Until that marker reached a response it was visible only to a
               library caller, so an HTTP consumer could not tell a whole map from a
               trimmed one -- a bound nobody can observe is not a bound they can act on.
  ISOLATION    nothing in this library reads the plane. Two dictionaries that differ ONLY
               in these values produce responses that are byte-identical everywhere else.
               This is the rule that lets the plane be carried at all.
  ALIGNMENT    a matched entry and a looked-up entry carry the SAME object for one id.

## The vocabulary here is INVENTED

The glossaries below describe Lumenport Water & Power, the municipal utility that does not
exist and that `_support.py` documents at length. The pass-through column names
(`steward_ref`, `ops_note`, `review_tag`) are invented for this file too, and that is the
point of the plane: this library never enumerates them, so a test that pinned real ones
would be pinning something the library is not allowed to know.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.application import ingest
from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.lookup import LookupResponseView
from nexus_matcher.presentation.api.matching import (
    _MAX_METADATA_DEPTH,
    _TRUNCATION_MARKER,
    _source_metadata_payload,
)
from nexus_matcher.presentation.api.schemas import MatchResponseView
from tests.unit.presentation.api._support import (
    GLOSSARY,
    build_api_matcher,
    governance_vocabulary,
)

# The awkward value, and every character in it is load-bearing.
#
#   two leading and two trailing spaces  a `.strip()` anywhere in the chain
#   U+00A0 NO-BREAK SPACE                whitespace folding, and a non-ASCII escape
#   U+2014 EM DASH                       a second, different, non-ASCII escape
#   U+00E9 U+00E0                        accents, which a re-encoding step mangles
#   " and /                              characters JSON and URLs have opinions about
#
# A deliberate echo of the value the LOADER half was proven byte-exact against, so both
# halves of the round trip are measured with one ruler.
#
# Built with `chr()` rather than written as literal characters, because a non-breaking
# space and an ordinary space are indistinguishable in a source file: spelled as itself,
# this fixture would stop exercising the escape path the first time an editor, a paste or
# a formatter swapped one for the other -- silently, and while still passing.
NBSP = chr(0x00A0)  # NO-BREAK SPACE
EM_DASH = chr(0x2014)  # EM DASH
ACCENTED = chr(0x00E9) + chr(0x00E0)  # E WITH ACUTE, A WITH GRAVE
AWKWARD = f'  Jetty{NBSP}Ref "A/B"  {EM_DASH} {ACCENTED} 001  '

# The other way a string gets mangled: characters JSON must escape with a BACKSLASH rather
# than with \\uXXXX. A tab, a newline and a literal backslash all round-trip or none do.
ESCAPED = 'a "quoted"\tvalue\\with a\nnewline'

# The pass-through columns of the fictional glossary below. Order matters: the plane must
# come back in the order the loader wrote it, so a response that sorted its keys is
# distinguishable from one that kept them.
PASS_THROUGH = ("steward_ref", "ops_note", "review_tag")

# The keys of the published object, in the order the contract fixes them. A literal, not
# something read back off the response: an expectation derived from the code under test is
# an identity, and an identity holds just as well when both sides are wrong.
PLANE_KEYS = ("values", "droppedKeyCount", "renderedKeys")

FIELDS = [
    {
        "name": "resident_nm",
        "path": "account.resident_nm",
        "doc": "Name of the resident on the account",
        "type": "string",
    },
    {
        "name": "usage_litres",
        "path": "meter.usage_litres",
        "doc": "Water drawn this month",
        "type": "bigint",
    },
]


# =============================================================================
# FIXTURES -- a real glossary file, loaded by the real loader
# =============================================================================


def write_glossary(directory, **overrides: str):
    """
    A two-row Lumenport glossary on disk, with three pass-through columns.

    A real file rather than an iterable of dicts, because the round trip this file is about
    starts at a reader: `csv` is what decides whether a quoted newline survives and whether
    a leading space is data, and skipping it would test the loader's arithmetic while
    assuming away the part that actually mangles strings.

    Carries `protection_class` and `classification` so the loader also writes its OWN
    reserved keys into the plane -- which is the case that proves the response tells the
    deployment's columns from the library's.
    """
    header = [
        "id",
        "business_name",
        "logical_name",
        "definition",
        "data_type",
        "domain",
        "protection_class",
        "classification",
        *PASS_THROUGH,
    ]
    rows = [
        [
            "LWP-9001",
            "Resident Full Name",
            "resident_full_name",
            "The name of the person responsible for the water account.",
            "string",
            "CUSTOMER",
            "RESIDENT",
            "LUMENPORT_GUARDED",
            overrides.get("steward_ref", AWKWARD),
            overrides.get("ops_note", ESCAPED),
            overrides.get("review_tag", "x"),
        ],
        [
            "LWP-9002",
            "Monthly Usage Litres",
            "monthly_usage_litres",
            "Total water drawn on the meter during one billing month.",
            "long",
            "METERING",
            "USAGEAGG",
            "LUMENPORT_OPEN",
            overrides.get("steward_ref", "Metering Desk"),
            overrides.get("ops_note", "none"),
            overrides.get("review_tag", "y"),
        ],
    ]
    path = directory / "glossary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def load(path, **kwargs: Any):
    """The glossary through the shipped loader, with the fictional vocabulary."""
    return tuple(ingest.load_entries(path, governance=governance_vocabulary(), **kwargs))


def client_for(entries) -> TestClient:
    """A client over an app whose only configuration is a matcher over `entries`."""
    app = create_app(configure_logs=False, matcher=build_api_matcher(entries=entries), environ={})
    return TestClient(app)


@pytest.fixture
def loaded_client(tmp_path):
    """Loader -> index -> HTTP, with nothing faked in between."""
    with client_for(load(write_glossary(tmp_path))) as client:
        yield client


def match(client: TestClient, **body: Any):
    """POST /match and return the raw response, failing loudly on a non-200."""
    payload: dict[str, Any] = {"fields": FIELDS, "top_k": 3}
    payload.update(body)
    response = client.post("/api/v1/match", json=payload)
    assert response.status_code == 200, response.text
    return response


def plane_of(client: TestClient, path: str = "account.resident_nm") -> dict[str, Any]:
    """The rank-1 candidate's plane for one field."""
    return match(client).json()["results"][path][0]["sourceMetadata"]


# =============================================================================
# ROUND TRIP
# =============================================================================


class TestWhatWentInComesBack:
    """AR-1's third rule, measured across the whole chain rather than at one seam."""

    def test_the_awkward_value_survives_loader_to_response_unchanged(self, loaded_client):
        """
        The single assertion this file exists for.

        Compared against the module-level constant, NOT against what the loader produced:
        deriving the expectation from the code under test would hold just as well if both
        halves trimmed the value identically.
        """
        values = plane_of(loaded_client)["values"]

        assert values["steward_ref"] == AWKWARD

    @pytest.mark.parametrize(
        ("what", "check"),
        [
            ("leading whitespace", lambda v: v.startswith("  ")),
            ("trailing whitespace", lambda v: v.endswith("  ")),
            ("the non-breaking space", lambda v: NBSP in v),
            ("the em-dash", lambda v: EM_DASH in v),
            ("the accents", lambda v: ACCENTED in v),
            ("the quotes", lambda v: '"A/B"' in v),
        ],
    )
    def test_no_part_of_it_is_normalised_away(self, loaded_client, what, check):
        """
        Named one property at a time, so a failure says WHICH transformation happened.
        A single equality would report "not equal" for a trim and for a re-encode alike.
        """
        value = plane_of(loaded_client)["values"]["steward_ref"]

        assert check(value), f"{what} did not survive the round trip: {value!r}"

    def test_backslash_escaped_characters_round_trip_too(self, loaded_client):
        """A tab, a newline and a literal backslash: JSON escapes these differently."""
        assert plane_of(loaded_client)["values"]["ops_note"] == ESCAPED

    def test_keys_keep_the_order_the_loader_wrote_them_in(self, loaded_client):
        """
        Not sorted, and not re-grouped. `PASS_THROUGH` is deliberately not alphabetical
        (`steward_ref`, `ops_note`, `review_tag`), so a renderer that sorted its keys
        fails here rather than passing by coincidence.
        """
        assert tuple(plane_of(loaded_client)["values"]) == PASS_THROUGH

    def test_the_object_carries_exactly_its_declared_members_in_order(self, loaded_client):
        assert tuple(plane_of(loaded_client)) == PLANE_KEYS


class TestTheEscapeIsAnEscape:
    """The body is ASCII-only, so this is how a non-ASCII value gets home."""

    def test_the_body_is_pure_ascii_and_the_value_is_not(self, loaded_client):
        body = match(loaded_client).content

        # Raises rather than asserts if it is not ASCII, which is the stronger statement.
        body.decode("ascii")
        assert not AWKWARD.isascii(), "the fixture stopped exercising the escape path"

    @pytest.mark.parametrize("code_point", ["u00a0", "u2014", "u00e9", "u00e0"])
    def test_the_non_ascii_characters_appear_as_json_escapes_on_the_wire(
        self, loaded_client, code_point
    ):
        """The literal six characters a JSON escape is, spelled without writing one."""
        escape = (chr(92) + code_point).encode("ascii")

        assert escape in match(loaded_client).content

    def test_parsing_the_escapes_back_gives_the_original_characters(self, loaded_client):
        """
        THE POINT OF THE ESCAPE. `\\u00a0` is a REPRESENTATION of the character, not a
        replacement for it, so any conformant parser -- Python's here, a Java client's
        there -- hands back the string the loader read. Asserted against the raw bytes
        rather than through `response.json()` so nothing between here and the socket is
        assumed.
        """
        parsed = json.loads(match(loaded_client).content.decode("ascii"))
        candidate = parsed["results"]["account.resident_nm"][0]

        assert candidate["sourceMetadata"]["values"]["steward_ref"] == AWKWARD

    def test_two_identical_requests_still_produce_identical_bytes(self, loaded_client):
        """Determinism is a promise about the whole body, and the plane is now in it."""
        assert match(loaded_client).content == match(loaded_client).content


# =============================================================================
# WHAT IS AND IS NOT THE DEPLOYMENT'S
# =============================================================================


class TestTheLoadersOwnKeysAreNotTheDeploymentsColumns:
    """
    The loader writes four keys of its own into the same map. Three do not travel, and the
    fourth travels as a declared member rather than as a caller column.
    """

    def test_values_is_exactly_the_declared_pass_through_columns(self, loaded_client, tmp_path):
        """
        Both directions in one assertion: nothing of the deployment's is missing, and
        nothing of the library's has leaked in. The loader really does put its own keys in
        this plane on this fixture -- the second half of the test proves the first half is
        not passing vacuously.
        """
        values = plane_of(loaded_client)["values"]

        assert values == {"steward_ref": AWKWARD, "ops_note": ESCAPED, "review_tag": "x"}

        raw = load(write_glossary(tmp_path))[0].source_metadata
        assert set(raw) - set(values), "the loader wrote no reserved keys; this proves nothing"

    @pytest.mark.parametrize("reserved", sorted(ingest.METADATA_RESERVED_KEYS))
    def test_no_reserved_key_is_published_as_a_deployment_column(self, loaded_client, reserved):
        """
        `governance_code_raw` is the one that matters most: it is the only place a code
        token the caller's own vocabulary REFUSED survives, and a body whose reader is
        deciding how to protect a column must not carry an undefined code beside
        `governance.code`.
        """
        assert reserved not in plane_of(loaded_client)["values"]

    def test_the_truncation_marker_this_layer_lifts_is_the_loaders_own(self):
        """
        The one name this layer restates. Pinned so that renaming it in the loader is a red
        test here, rather than a marker that silently stops being surfaced and starts being
        emitted as though it were one of the deployment's columns.
        """
        assert _TRUNCATION_MARKER in ingest.METADATA_RESERVED_KEYS


# =============================================================================
# BOUNDEDNESS
# =============================================================================


class TestTheBoundIsVisibleToAConsumer:
    """
    AR-1's fifth rule. The loader bounds the plane and marks what it dropped; until that
    marker reached a response it was visible only to a library caller, so an HTTP consumer
    could not tell a complete map from a trimmed one.
    """

    def test_a_whole_plane_reports_nothing_dropped(self, loaded_client):
        plane = plane_of(loaded_client)

        assert plane["droppedKeyCount"] == 0
        assert len(plane["values"]) == len(PASS_THROUGH)

    def test_a_trimmed_plane_says_how_many_keys_it_lost(self, tmp_path):
        """
        Driven through the real cap rather than by hand-building a marked entry: the number
        a consumer reads has to be the number the LOADER decided, and a fixture that wrote
        the marker itself would agree with itself.
        """
        with client_for(load(write_glossary(tmp_path), metadata_max_bytes=96)) as client:
            plane = plane_of(client)

        assert plane["droppedKeyCount"] == 2
        # A bounded SUBSEQUENCE, in the original order -- not a re-sorted remnant.
        assert tuple(plane["values"]) == ("review_tag",)

    def test_the_marker_does_not_also_appear_among_the_values(self, tmp_path):
        """It is lifted out, not copied out: a count is a declared member, not a column."""
        with client_for(load(write_glossary(tmp_path), metadata_max_bytes=96)) as client:
            plane = plane_of(client)

        assert _TRUNCATION_MARKER not in plane["values"]

    def test_lifting_the_cap_is_visible_as_a_wider_plane(self, tmp_path):
        """
        The knob a deployment actually turns, checked end to end. Same file, same request,
        `metadata_max_bytes=None`: every column comes back and nothing is reported dropped.
        """
        with client_for(load(write_glossary(tmp_path), metadata_max_bytes=None)) as client:
            plane = plane_of(client)

        assert plane["droppedKeyCount"] == 0
        assert tuple(plane["values"]) == PASS_THROUGH


# =============================================================================
# ISOLATION -- the rule that makes carrying it safe
# =============================================================================


def strip_planes(body: dict[str, Any]) -> dict[str, Any]:
    """The response with every `sourceMetadata` removed, and nothing else touched."""
    stripped = json.loads(json.dumps(body))
    for candidates in stripped["results"].values():
        for candidate in candidates:
            candidate.pop("sourceMetadata")
    return stripped


class TestNothingInThisLibraryReadsIt:
    """
    AR-1's FIRST rule, and the one that makes the other four affordable. A library that
    scored, filtered or classified on a key in this map would be one enterprise's matcher
    with a generic name; the next deployment's columns mean something else entirely.
    """

    def test_two_dictionaries_differing_only_in_the_plane_agree_everywhere_else(self, tmp_path):
        """
        The strongest form of the claim available at this boundary: same fields, same
        entries, planes that share not one key or value, and the two responses are BYTE
        IDENTICAL once the planes are removed. That covers every confidence, every rank,
        every decision, every absoluteScore, the field verdicts and the scoring contract in
        one comparison, and it would fail for any of them.
        """
        original = load(write_glossary(tmp_path))
        rewritten = tuple(
            dataclasses.replace(
                entry,
                source_metadata={
                    "utterly_different_key": f"nothing like the other one {index}",
                    "second_key": index * 1000,
                },
            )
            for index, entry in enumerate(original)
        )

        with client_for(original) as one, client_for(rewritten) as two:
            first, second = match(one).json(), match(two).json()

        assert strip_planes(first) == strip_planes(second)
        # And the planes really did differ, so the equality above is not two copies of one
        # thing agreeing with itself.
        one_plane = first["results"]["account.resident_nm"][0]["sourceMetadata"]
        other_plane = second["results"]["account.resident_nm"][0]["sourceMetadata"]
        assert one_plane != other_plane

    def test_the_field_verdicts_are_untouched_by_the_plane(self, tmp_path):
        """
        Stated separately from the byte comparison above because `fieldDecisions` is the
        value a consumer writes into a per-column decision, and "governance may not depend
        on a key in this map" is exactly a claim about it.
        """
        original = load(write_glossary(tmp_path))
        blanked = tuple(dataclasses.replace(entry, source_metadata={}) for entry in original)

        with client_for(original) as one, client_for(blanked) as two:
            assert match(one).json()["fieldDecisions"] == match(two).json()["fieldDecisions"]


# =============================================================================
# ALIGNMENT -- one glossary row, one answer
# =============================================================================


class TestTheTwoPlanesAgree:
    """WC-10's "the same enrichment surface as a match candidate", now including the plane."""

    def test_a_looked_up_entry_carries_the_identical_object(self, loaded_client):
        candidate = match(loaded_client).json()["results"]["account.resident_nm"][0]
        looked_up = loaded_client.post("/api/v1/lookup", json={"ids": [candidate["governanceId"]]})
        assert looked_up.status_code == 200, looked_up.text

        entry = looked_up.json()["results"][candidate["governanceId"]]

        assert entry["sourceMetadata"] == candidate["sourceMetadata"]

    def test_and_it_is_identical_as_rendered_bytes_not_merely_as_data(self, loaded_client):
        """
        Compared as the JSON fragment each route actually sent. Two equal dicts can be
        rendered with different key orders, and key order is this service's contract.
        """
        candidate = match(loaded_client).json()["results"]["account.resident_nm"][0]
        entry_id = candidate["governanceId"]
        single = loaded_client.get(f"/api/v1/lookup/{entry_id}")
        assert single.status_code == 200, single.text

        rendered = json.dumps(candidate["sourceMetadata"], ensure_ascii=True, separators=(",", ":"))

        assert rendered.encode("ascii") in single.content
        assert rendered.encode("ascii") in match(loaded_client).content

    def test_a_miss_still_carries_no_entry_at_all(self, loaded_client):
        """A not-found id is `null`, not an entry with an empty plane."""
        body = loaded_client.post("/api/v1/lookup", json={"ids": ["NOT-A-TERM"]}).json()

        assert body["results"]["NOT-A-TERM"] is None


# =============================================================================
# VALUES JSON CANNOT CARRY
# =============================================================================


class TestAnUnrenderableValueIsDeclaredRatherThanFatal:
    """
    A spreadsheet or a database gives back real objects, not text. This renderer runs with
    no `default=` hook and `allow_nan=False`, so before this was handled one date column in
    one glossary would have made EVERY match against that dictionary a 500 -- an outage
    caused by a value nobody scores on.
    """

    @staticmethod
    def entries_with(plane: dict[str, Any]):
        """The fictional glossary's first entry, carrying `plane`."""
        return (dataclasses.replace(GLOSSARY[0], source_metadata=plane),)

    @staticmethod
    def rendered_keys_for(plane: dict[str, Any]) -> list[str]:
        """Which keys of `plane` the renderer had to turn into text."""
        return _source_metadata_payload(dataclasses.replace(GLOSSARY[0], source_metadata=plane))[
            "renderedKeys"
        ]

    def test_a_date_cell_is_rendered_as_text_and_named(self):
        with client_for(self.entries_with({"reviewed_on": date(2026, 1, 4), "tag": "keep"})) as c:
            plane = plane_of(c)

        assert plane["values"] == {"reviewed_on": "2026-01-04", "tag": "keep"}
        assert plane["renderedKeys"] == ["reviewed_on"]

    def test_a_non_finite_number_does_not_take_the_response_down(self):
        """`allow_nan=False` is deliberate -- bare `NaN` is not JSON -- so it is rendered."""
        with client_for(self.entries_with({"ratio": math.nan})) as client:
            plane = plane_of(client)

        assert plane["renderedKeys"] == ["ratio"]
        assert plane["values"]["ratio"] == "nan"

    def test_arrays_and_objects_pass_through_natively(self):
        """
        A JSON or Parquet glossary can hold these in one cell and JSON can carry them, so
        they are NOT rendered: `renderedKeys` stays empty and the structure survives.
        """
        nested = {"tags": ["a", "b"], "nested": {"x": 1, "y": [True, None, 2.5]}}
        with client_for(self.entries_with(dict(nested))) as client:
            plane = plane_of(client)

        assert plane["values"] == nested
        assert plane["renderedKeys"] == []

    def test_a_cycle_is_rendered_rather_than_recursed_forever(self):
        """
        Not reachable from any shipped loader, and that is why the depth bound is here: a
        caller building entries in process can do it, and an unbounded walk on the event
        loop is a hang rather than an error.
        """
        cycle: dict[str, Any] = {}
        cycle["self"] = cycle

        payload = _source_metadata_payload(dataclasses.replace(GLOSSARY[0], source_metadata=cycle))

        assert payload["renderedKeys"] == ["self"]
        assert isinstance(payload["values"]["self"], str)

    def test_a_structure_deeper_than_the_bound_is_rendered(self):
        """
        The bound itself, pinned: one level under it survives, one level over does not.

        Counted in CONTAINERS, not in values, and that is the bound's real shape: the walk
        only recurses through a list or an object, so a scalar sitting one step past the
        limit costs nothing and is passed through. `shallow` puts its innermost list at the
        last permitted depth; `deep` wraps it once more and that list is the one refused.
        """
        shallow: Any = "leaf"
        for _ in range(_MAX_METADATA_DEPTH):
            shallow = [shallow]
        deep = [shallow]

        assert self.rendered_keys_for({"k": shallow}) == []
        assert self.rendered_keys_for({"k": deep}) == ["k"]

    def test_a_non_string_key_is_rendered_as_text(self):
        """
        `json.dumps` would coerce it silently, and two keys can coerce to one -- a key
        quietly answering for another key. Not reachable from a header row; here because
        the renderer must not be the place that discovers it.
        """
        payload = _source_metadata_payload(
            dataclasses.replace(GLOSSARY[0], source_metadata={7: "seven"})
        )

        assert payload["values"] == {"7": "seven"}


# =============================================================================
# AN ENTRY WITH NO PLANE
# =============================================================================


class TestAnEmptyPlaneIsStillAnAnswer:
    """
    Present with an empty `values`, never absent and never null. "This entry carries no
    enrichment" and "this response dropped it" must not look alike to a client whose
    pipeline is about to write these columns into its own model.
    """

    def test_a_candidate_for_an_entry_with_no_columns_still_carries_the_object(self):
        with client_for(GLOSSARY) as client:
            plane = plane_of(client)

        assert plane == {"values": {}, "droppedKeyCount": 0, "renderedKeys": []}

    def test_and_so_does_a_looked_up_entry(self):
        with client_for(GLOSSARY) as client:
            body = client.post("/api/v1/lookup", json={"ids": ["LWP-0001"]}).json()

        assert body["results"]["LWP-0001"]["sourceMetadata"] == {
            "values": {},
            "droppedKeyCount": 0,
            "renderedKeys": [],
        }


# =============================================================================
# THE PUBLISHED SCHEMA
# =============================================================================


class TestTheSchemaAClientGeneratesFrom:
    """A Java client is generated from `/openapi.json`, so an undescribed key is a mystery."""

    @pytest.fixture
    def schemas(self):
        return create_app(configure_logs=False, environ={}).openapi()["components"]["schemas"]

    def test_both_planes_validate_against_their_published_models(self, loaded_client):
        """
        The handlers render dicts themselves, so the published model is documentation until
        something checks it against a real body. This is that check for the new member.
        """
        MatchResponseView.model_validate(match(loaded_client).json())
        LookupResponseView.model_validate(
            loaded_client.post("/api/v1/lookup", json={"ids": ["LWP-9001", "NOPE"]}).json()
        )

    @pytest.mark.parametrize("carrier", ["MatchCandidateView", "LookupEntryView"])
    def test_the_member_is_published_and_described_on_both_surfaces(self, schemas, carrier):
        published = schemas[carrier]["properties"]["sourceMetadata"]

        assert published.get("description"), f"{carrier}.sourceMetadata has no description"

    @pytest.mark.parametrize("member", PLANE_KEYS)
    def test_every_member_of_the_plane_object_is_described(self, schemas, member):
        published = schemas["SourceMetadataView"]["properties"][member]

        assert published.get("description"), f"SourceMetadataView.{member} has no description"

    def test_the_member_is_required_rather_than_optional(self, schemas):
        """Present on every candidate and every hit, so a client never branches on absence."""
        assert "sourceMetadata" in schemas["MatchCandidateView"]["required"]
        assert "sourceMetadata" in schemas["LookupEntryView"]["required"]

    def test_nothing_about_the_plane_is_published_as_a_closed_set(self, schemas):
        """
        THE TAXONOMY RULE. The keys come from a deployment's configuration and the values
        from its cells, so an enum anywhere in this object would hard-code one
        organisation's spreadsheet into the schema every client generates from -- and this
        library ships no taxonomy.

        Walks the schema for a `enum`/`const` KEYWORD rather than grepping the rendered
        JSON: the descriptions on this object talk about enumerating columns in prose, and
        a substring search would fail on the word while a genuinely closed set added under
        a nested `items` would slip past a shallow one.
        """
        closed: list[str] = []

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                for keyword in ("enum", "const"):
                    if keyword in node:
                        closed.append(f"{path}.{keyword}")
                for key, value in node.items():
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")

        walk(schemas["SourceMetadataView"], "SourceMetadataView")

        assert not closed, f"the plane publishes a closed set at {closed}"
