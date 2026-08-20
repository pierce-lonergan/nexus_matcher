"""
tests.unit.presentation.api.test_payload_headroom | Layer: TEST
NM-V2-02 WC-4: the caps, measured against a REAL payload shape rather than a synthetic one.

## Relationships
# TESTS → presentation/api/limits :: the derived body cap and the two field caps
# TESTS → presentation/api/schemas :: the per-field character ceiling
# TESTS → presentation/api/matching :: the 413 that carries a suggested chunk size

WC-4 asks three things of this service, and the third is the one that needed measuring:

  1. document which endpoint is intended for 200-field chunks;
  2. keep the 413 carrying a suggested chunk size;
  3. "confirm the derived body-size cap accommodates realistic field docs. A flattened
     path plus a full glossary-grade `doc` is not small; the per-field ceiling should be
     validated against a REAL payload, not a synthetic one."

`test_body_limit.py` already proves the cap admits the WORST body the declared bounds
allow -- every string at its `max_length`, every character four UTF-8 bytes. That is the
right property for a byte cap and it is not an answer to (3), because it says nothing about
whether the bounds themselves are generous enough for real data. A ceiling can be
internally consistent and still be too small.

## Where the numbers come from

Measured over HL7 FHIR R5 StructureDefinitions -- 4,598 real element definitions and 1,556
real flattened element paths, the closest public proxy this repository has to the production
input shape, and the one `benchmarks/datasets/build_fhir.py` exists to build precisely
because its paths are genuinely nested and its three texts are independently authored.
Reproduce with:

    python benchmarks/datasets/build_fhir.py          # writes data/benchmarks/fhir/
    # then, over dictionary.jsonl and queries.jsonl:
    #   max(len(r["description"])) -> the longest real glossary-grade doc
    #   max(len(r["field_path"]))  -> the longest real flattened path
    #   max(len(r["field_name"]))  -> the longest real leaf name

The corpus is generated rather than committed, so the measurements are pinned here as
constants with that recipe beside them. What is asserted against them is the SHIPPED caps,
which are committed -- so this test goes red when a cap is lowered under real data, which is
the direction WC-4 is worried about.

For the record, the same measurement run as a body rather than as maxima: the 200 LARGEST
real fields, each carrying its own element definition as `doc`, serialise to 90,639 bytes.
The constants below are stricter, because they put every field at the maximum simultaneously.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.limits import MatchServiceLimits
from nexus_matcher.presentation.api.schemas import MAX_FIELD_SPEC_CHARS
from tests.unit.presentation.api._support import build_api_matcher

# The chunk size the adopting pipeline sends, from NM-V2-02 WC-4. Not a cap in this
# service; the number this file exists to show is comfortably served.
PIPELINE_CHUNK_FIELDS = 200

# Measured maxima over the FHIR corpus. See the module docstring for the recipe.
REAL_MAX_DOC_CHARS = 1_099
REAL_MAX_PATH_CHARS = 65
REAL_MAX_NAME_CHARS = 71
# `boolean`, `CodeableConcept`, `dateTime`... the longest FHIR primitive-or-complex type
# name is well inside this, and a generous round number here only makes the test stricter.
REAL_MAX_TYPE_CHARS = 32


def _realistic_field(index: int) -> dict[str, str]:
    """
    One field at the largest size real data produced, for every string at once.

    Padded ASCII rather than the corpus text itself: the property under test is a length
    against a bound, and shipping 200 real HL7 definitions into a test file would make the
    fixture the thing that has to be maintained.
    """
    return {
        "name": ("account_identifier_" * 8)[:REAL_MAX_NAME_CHARS],
        "path": (f"originations.customer.account.{index}." + "nested." * 8)[:REAL_MAX_PATH_CHARS],
        "doc": ("The identifier of the account this application was booked under. " * 32)[
            :REAL_MAX_DOC_CHARS
        ],
        "type": ("CodeableConcept" + "X" * 32)[:REAL_MAX_TYPE_CHARS],
    }


@pytest.fixture
def client():
    with TestClient(create_app(configure_logs=False, matcher=build_api_matcher(), environ={})) as c:
        yield c


# =============================================================================
# THE PER-FIELD CEILING, AGAINST REAL FIELD DOCS
# =============================================================================


def test_the_per_field_ceiling_accommodates_the_largest_real_field():
    """
    WC-4's actual worry: a flattened path plus a full glossary-grade `doc` is not small.

    Asserted with the headroom named, so the failure message says how much of the ceiling
    real data uses rather than only that it fits.
    """
    largest_real = (
        REAL_MAX_NAME_CHARS + REAL_MAX_PATH_CHARS + REAL_MAX_DOC_CHARS + REAL_MAX_TYPE_CHARS
    )

    assert largest_real < MAX_FIELD_SPEC_CHARS, (
        f"the largest field measured in a real corpus is {largest_real} characters and the "
        f"per-field ceiling is {MAX_FIELD_SPEC_CHARS}"
    )
    assert largest_real * 4 <= MAX_FIELD_SPEC_CHARS, (
        f"only {MAX_FIELD_SPEC_CHARS / largest_real:.1f}x headroom over the largest real "
        f"field; a corpus with longer definitions than FHIR's would start being refused"
    )


def test_a_real_field_doc_is_well_inside_the_doc_bound():
    """
    The `doc` bound is the one that would bite first: it is the only string on a field that
    carries prose, and 8 KiB was chosen as "already generous for a column comment" rather
    than measured. It is: the longest of 4,598 real element definitions uses an eighth of it.
    """
    from nexus_matcher.presentation.api.schemas import FieldSpec

    bounds = [
        getattr(constraint, "max_length", None)
        for constraint in FieldSpec.model_fields["doc"].metadata
    ]
    doc_bound = next(bound for bound in bounds if isinstance(bound, int))
    assert doc_bound > REAL_MAX_DOC_CHARS
    assert doc_bound >= REAL_MAX_DOC_CHARS * 4


# =============================================================================
# THE BODY CAP, AGAINST A REAL 200-FIELD CHUNK
# =============================================================================


def test_the_derived_cap_accommodates_a_realistic_two_hundred_field_chunk():
    """
    The derived cap is a floor under the WORST body; this is what a real one costs against
    it. Every field at the largest size real data produced, 200 of them, which is the chunk
    the adopting pipeline sends.
    """
    fields = [_realistic_field(index) for index in range(PIPELINE_CHUNK_FIELDS)]
    body = json.dumps({"fields": fields, "top_k": 5, "explain": True}).encode("utf-8")
    cap = MatchServiceLimits().body_byte_cap

    assert len(body) < cap, f"a realistic {PIPELINE_CHUNK_FIELDS}-field chunk exceeds the cap"
    assert cap >= len(body) * 10, (
        f"a realistic chunk is {len(body)} bytes against a {cap}-byte cap, only "
        f"{cap / len(body):.1f}x -- the cap is no longer generous against real data"
    )


def test_a_realistic_chunk_is_accepted_end_to_end(client):
    """
    Measurement is not admission. The body-size middleware, the field cap and the parser all
    sit between the two, and each is capable of refusing something the arithmetic accepts.
    """
    fields = [_realistic_field(index) for index in range(PIPELINE_CHUNK_FIELDS)]
    response = client.post("/api/v1/match/batch", json={"fields": fields})

    assert response.status_code == 200, response.text
    assert len(response.json()["results"]) == PIPELINE_CHUNK_FIELDS


# =============================================================================
# WHICH ENDPOINT IS FOR 200-FIELD CHUNKS, AND WHAT THE 413 SAYS
# =============================================================================


def test_the_batch_route_is_the_one_that_takes_a_two_hundred_field_chunk():
    """
    WC-4 item 1, as arithmetic rather than as a doc sentence. `/match` caps at 100, so a
    200-field chunk MUST go to `/match/batch` -- and a client author reading only the
    smaller cap would conclude this service cannot take their chunk size at all.
    """
    limits = MatchServiceLimits()

    assert limits.max_fields < PIPELINE_CHUNK_FIELDS <= limits.max_batch_fields


@pytest.mark.parametrize(
    ("route", "cap_name"),
    [("/api/v1/match", "max_fields"), ("/api/v1/match/batch", "max_batch_fields")],
)
def test_the_413_carries_the_chunk_size_to_retry_with(client, route, cap_name):
    """
    WC-4 item 2. A 413 that says "too many" and not "how many" leaves a client halving
    blindly; the Java client's `PayloadTooLargeException.suggestedChunkSize()` reads exactly
    this member, so it is a contract and not a courtesy.
    """
    cap = getattr(MatchServiceLimits(), cap_name)
    fields = [_realistic_field(index) for index in range(cap + 1)]

    response = client.post(route, json={"fields": fields})

    assert response.status_code == 413, response.text
    details = response.json()["error"]["details"]
    assert details["limit"] == cap
    assert details["fields"] == cap + 1
    # And in the prose too, because the number is what an operator reading a log needs.
    assert str(cap) in response.json()["error"]["message"]


def test_the_lookup_413_carries_it_as_well():
    """
    The id cap is the batch field cap, so a client that learned to read `details.limit` on
    one plane reads it on the other.
    """
    limits = MatchServiceLimits(max_batch_fields=4)
    app = create_app(configure_logs=False, matcher=build_api_matcher(), limits=limits, environ={})
    with TestClient(app) as client:
        response = client.post("/api/v1/lookup", json={"ids": [f"MISS-{n}" for n in range(5)]})

    assert response.status_code == 413, response.text
    assert response.json()["error"]["details"]["limit"] == 4


# =============================================================================
# WC-3 -- the duplicate refusal has to NAME the duplicates
# =============================================================================


def test_the_duplicate_path_422_names_the_offending_paths(client):
    """
    NM-V2-02 WC-3: "v2 must make the 422 message name the offending duplicates, because the
    failure will otherwise appear as an inexplicable rejection of a large batch."

    A 200-field chunk with one repeated path is exactly that scenario, and what is asserted
    is that the two offenders are findable WITHOUT reading all 200 back: named in `details`
    for a program, and in `message` for whoever is reading the log.
    """
    fields = [_realistic_field(index) for index in range(PIPELINE_CHUNK_FIELDS)]
    fields[137]["path"] = fields[11]["path"]

    response = client.post("/api/v1/match/batch", json={"fields": fields})

    assert response.status_code == 422, response.text
    body = response.json()["error"]
    assert body["details"]["duplicate_paths"] == [fields[11]["path"]]
    assert fields[11]["path"] in body["message"]
