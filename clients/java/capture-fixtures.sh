#!/usr/bin/env bash
# Regenerate every file in src/test/resources/captured/ from RUNNING services.
#
#     ./clients/java/serve-fixtures.sh        # in one shell, and leave it running
#     ./clients/java/capture-fixtures.sh      # in another, from the repository root
#
# Why this script exists rather than a note saying "curl it":
#
# The captured bodies are the only thing the *Test classes decode, so they are the only thing
# standing between a hand-written DTO and a wire shape that has moved. That is worth nothing if
# a fixture is ever EDITED to make a test pass -- a body somebody typed tests what the author
# believed the contract was, which is exactly the belief the fixture was supposed to check. The
# defence against that is not discipline, it is being able to re-derive every byte on demand:
# run this, `git diff`, and either the service changed or nothing did.
#
# Each capture writes the response body VERBATIM. No formatting, no key sorting, no pretty
# printing -- the service promises byte-stable output and reformatting here would throw that
# promise away before the test ever saw it.
#
# Four fixtures are needed, one of which serve-fixtures.sh does not start:
#   8000  the example pack                     everything normal
#   8003  the pack + an absolute-score floor    NO_MATCH is reachable
# 8001 and 8002 are the 503 and 504 fixtures; nothing here is captured from them, because their
# behaviour is a timing property that only a live *IT can assert.
set -euo pipefail

BASE="${NEXUS_MATCHER_BASE_URL:-http://127.0.0.1:8000}"
FLOOR_BASE="${NEXUS_MATCHER_FLOOR_BASE_URL:-http://127.0.0.1:8003}"
OUT="clients/java/src/test/resources/captured"

if [[ ! -d "$OUT" ]]; then
    echo "run this from the repository root: $OUT not found" >&2
    exit 1
fi

require_service() {
    local base="$1"
    local hint="$2"
    if ! curl -fsS -m 5 "${base}/health/live" > /dev/null 2>&1; then
        echo "no service at ${base}. ${hint}" >&2
        exit 1
    fi
}

require_service "$BASE" "Start it with ./clients/java/serve-fixtures.sh from the repository root."
require_service "$FLOOR_BASE" \
    "Start it with: NEXUS_API_DICTIONARY=examples/governance/glossary.csv \
NEXUS_API_GOVERNANCE=examples/governance/protection_classes.json \
NEXUS_API_MATCHING_CONFIG=clients/java/fixture-absolute-floor.json \
.venv/Scripts/python.exe -m uvicorn nexus_matcher.presentation.api.app:create_app --factory \
--host 127.0.0.1 --port 8003   (serve-fixtures.sh starts it too)"

# `--fail` is deliberately NOT used: half of these captures ARE failure bodies, and the point of
# capturing them is that the client decodes an error envelope it did not write.
get() {
    local name="$1"
    local base="$2"
    local path="$3"
    curl -sS -m 60 -o "${OUT}/${name}" "${base}${path}"
    echo "  ${name}  <-  GET ${path}"
}

post() {
    local name="$1"
    local base="$2"
    local path="$3"
    local body="$4"
    curl -sS -m 120 -o "${OUT}/${name}" \
        -X POST "${base}${path}" \
        -H "Content-Type: application/json" \
        --data-binary "$body"
    echo "  ${name}  <-  POST ${path}"
}

echo "capturing from ${BASE} and ${FLOOR_BASE}"

# ---------------------------------------------------------------------------------------------
# The happy paths
# ---------------------------------------------------------------------------------------------

get health.json "$BASE" /health
get status.json "$BASE" /api/v1/status

# Two fields chosen for the two documented meanings of a null class, in one body:
# `booking.passenger.legal_name` matches an entry that carries a protection code (CONFERRED),
# and `sailing.route_code` matches the pack's one row with an empty code column (OPEN_TIER),
# whose runner-ups are REJECT and still carry theirs.
post match-response.json "$BASE" /api/v1/match '{
  "fields": [
    {"name": "legal_name", "path": "booking.passenger.legal_name",
     "doc": "Full legal name of the passenger as printed on the sailing manifest.",
     "type": "string"},
    {"name": "route_code", "path": "sailing.route_code",
     "doc": "Short code identifying a scheduled route between two terminals.",
     "type": "string"}
  ],
  "top_k": 3
}'

# `type` is omitted on purpose so the type signal scores its neutral 0.5 rather than 1.0, which
# keeps the explain block showing a spread of component scores instead of a column of ones.
post match-response-explain.json "$BASE" /api/v1/match '{
  "fields": [
    {"name": "terminal_name", "path": "published.terminal_name",
     "doc": "The public name of a Gravel Bay ferry terminal."}
  ],
  "top_k": 1,
  "explain": true
}'

post lookup-response.json "$BASE" /api/v1/lookup '{
  "ids": ["GBF-0001", "GBF-0028", "GBF-NOT-A-REAL-ID"]
}'

# A field the glossary genuinely does not describe, against the entry somebody might have
# expected it to find -- so `expected` shows a real rank rather than a null.
post retrieval-diagnostic.json "$BASE" /api/v1/diag/retrieval '{
  "field": {"name": "quasar_flux_index", "path": "telemetry.quasar_flux_index",
            "doc": "Interstellar quasar flux index sampled by the orbital radio array.",
            "type": "float"},
  "expected_governance_id": "GBF-0022",
  "top_k": 3
}'

post feedback-receipt.json "$BASE" /api/v1/feedback '{
  "field": "booking.passenger.legal_name",
  "doc": "Full legal name of the passenger.",
  "chosenGovernanceId": "GBF-0001",
  "wasCorrect": true,
  "reviewer": "capture",
  "ts": "2026-08-11T09:00:00Z"
}'

# ---------------------------------------------------------------------------------------------
# NO_MATCH, from the floor fixture
# ---------------------------------------------------------------------------------------------
#
# Two fields: one the glossary describes and one it does not. The second earns NO_MATCH and comes
# back WITH its candidates, one of which carries a populated protection class -- which is the
# whole reason this capture exists. A client that reads rank 1 instead of the field verdict
# classifies a column from an entry the server has just said describes nothing.
post match-response-no-match.json "$FLOOR_BASE" /api/v1/match '{
  "fields": [
    {"name": "legal_name", "path": "booking.passenger.legal_name",
     "doc": "Full legal name of the passenger as printed on the sailing manifest.",
     "type": "string"},
    {"name": "lifejacket_locker_inspection_due",
     "path": "vessel.safety.lifejacket_locker_inspection_due",
     "doc": "Date the lifejacket locker is next due for inspection.",
     "type": "date"}
  ],
  "top_k": 3
}'

# ---------------------------------------------------------------------------------------------
# The failure envelopes
# ---------------------------------------------------------------------------------------------

# 101 fields against a cap of 100. Built here rather than typed out, so the body stays right if
# the server's default cap moves and somebody re-runs this with a bigger number.
FIELDS_101="$(
    PYTHON=".venv/Scripts/python.exe"
    [[ -x "$PYTHON" ]] || PYTHON=".venv/bin/python"
    "$PYTHON" -c 'import json; print(json.dumps({"fields": [{"name": f"col_{i}"} for i in range(101)]}))'
)"
post error-413-field-cap.json "$BASE" /api/v1/match "$FIELDS_101"

post error-422-duplicate-paths.json "$BASE" /api/v1/match '{
  "fields": [{"name": "a", "path": "t.a"}, {"name": "b", "path": "t.a"}]
}'

post error-422-top-k-cap.json "$BASE" /api/v1/match '{
  "fields": [{"name": "a"}],
  "top_k": 50
}'

# The 422 two reviewers have already hit: pasting a row of the example pack's own fields.json
# into a request. Those are the pack's INPUT spellings, not this API's.
post error-422-unknown-field-key.json "$BASE" /api/v1/match '{
  "fields": [{"flattenedName": "booking.passenger.legal_name", "dataType": "string"}]
}'

echo
echo "done. Now: git diff --stat ${OUT}"
echo "A diff here is the service having changed. Read it before you accept it."
