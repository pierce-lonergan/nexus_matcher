#!/usr/bin/env bash
# Start the five services `mvn verify` needs. Run from the repository root.
#
#     ./clients/java/serve-fixtures.sh
#
# Five, not one, because five of the behaviours the client has to get right are properties
# of a server's CONFIGURATION rather than of a request, and none of them can be provoked
# against a correctly configured one:
#
#   8000  the example pack, loaded            everything normal
#   8001  no dictionary at all                every match is a real 503
#   8002  the pack with a 1 ms deadline       every match is a real 504
#   8003  the pack + an absolute-score floor  NO_MATCH is reachable
#   8004  the pack + a reviewer's verdict     provenance APPROVED_PAIR is reachable
#
# 8003 and 8004 are the two least obvious, and they are the same kind of fixture: a behaviour
# the client must read correctly that NO correctly-configured stock server produces.
#
# `fieldDecisions` can report NO_MATCH two ways: a field that came back with no candidates,
# and a field whose rank-1 absolute score does not clear a configured floor. The library SHIPS
# NO FLOOR and will not invent one, so on 8000 the second way can never fire and the first does
# not occur against a 30-entry glossary -- which means the whole verdict would go untested
# against a live service. 8003 configures a floor so that it does. See
# clients/java/fixture-absolute-floor.json for the number and why it is that number.
#
# `provenance` reports APPROVED_PAIR for a candidate a reviewer decided and matching skipped.
# The library SHIPS NO FEEDBACK CONSUMER -- `create_app()` builds none and `NexusMatcher()`
# takes `feedback_consumer=None`, which is a measured decision rather than an unfinished
# wire-up -- so on 8000 every candidate is RETRIEVAL and the other half of the vocabulary is
# unreachable. 8004 attaches the reference consumer to a throwaway server so it is not. See
# clients/java/fixture_approved_pair_app.py.
#
# Both are TEST FIXTURES, not recommended defaults.
#
# Ctrl-C stops all five. PIDs are also written to clients/java/.fixture-pids.
set -euo pipefail

if [[ ! -f examples/governance/glossary.csv ]]; then
    echo "run this from the repository root: examples/governance/glossary.csv not found" >&2
    exit 1
fi

PYTHON=".venv/Scripts/python.exe"
[[ -x "$PYTHON" ]] || PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "no virtualenv python found at .venv/" >&2
    exit 1
fi

MODULE="nexus_matcher.presentation.api.app:create_app"
# 8004's own factory, which wires the approved-pair consumer the shipped one deliberately
# does not. It lives beside this script and is reached by putting clients/java on the path,
# so nothing in the package has to know a fixture exists.
BYPASS_MODULE="fixture_approved_pair_app:create_app"
PIDS=()

start_fixture() {
    local port="$1"
    local module="$2"
    shift 2
    env PYTHONIOENCODING=utf-8 "$@" \
        "$PYTHON" -m uvicorn "$module" --factory --host 127.0.0.1 --port "$port" \
        > "clients/java/.fixture-${port}.log" 2>&1 &
    PIDS+=("$!")
    echo "started fixture on 127.0.0.1:${port} (pid $!, log clients/java/.fixture-${port}.log)"
}

cleanup() {
    echo
    echo "stopping fixtures"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

DICT="NEXUS_API_DICTIONARY=examples/governance/glossary.csv"
VOCAB="NEXUS_API_GOVERNANCE=examples/governance/protection_classes.json"
FLOOR="NEXUS_API_MATCHING_CONFIG=clients/java/fixture-absolute-floor.json"
TRAIL="NEXUS_FIXTURE_APPROVED_PAIRS=clients/java/fixture-approved-pairs.jsonl"

# The pack, served normally.
start_fixture 8000 "$MODULE" "$DICT" "$VOCAB" \
    NEXUS_API_FEEDBACK_PATH=examples/governance/out/feedback_http.jsonl

# No dictionary: every match answers 503 naming the setting, and so does feedback.
start_fixture 8001 "$MODULE"

# The pack with a deadline shorter than a match takes, so every match answers 504.
start_fixture 8002 "$MODULE" "$DICT" "$VOCAB" NEXUS_API_DEADLINE_SECONDS=0.001

# The pack with an absolute-score floor configured, so a field the glossary does not describe
# earns a NO_MATCH field decision -- with its candidates still attached, which is the half a
# client gets wrong.
start_fixture 8003 "$MODULE" "$DICT" "$VOCAB" "$FLOOR"

# The pack with one reviewer verdict standing, so `booking.passenger.legal_name` is answered
# by a human's decision and comes back as provenance APPROVED_PAIR: one candidate however
# large top_k is, no absoluteScore and no explain, at confidence 1.0. This factory refuses to
# start if the verdict does not resolve, so the port cannot come up quietly answering by
# retrieval -- which would make every capture taken from it say the opposite of what it says.
start_fixture 8004 "$BYPASS_MODULE" "$DICT" "$VOCAB" "$TRAIL" PYTHONPATH=clients/java

printf '%s\n' "${PIDS[@]}" > clients/java/.fixture-pids

echo
echo "Waiting for 8000, 8002, 8003 and 8004 to load their encoder..."
for port in 8000 8002 8003 8004; do
    for _ in $(seq 1 120); do
        if curl -fsS -m 2 "http://127.0.0.1:${port}/health/ready" > /dev/null 2>&1; then
            echo "  ${port} ready"
            break
        fi
        sleep 1
    done
done

echo
echo "Now, in another shell:  cd clients/java && mvn verify"
echo "Ctrl-C here stops all five."
wait
