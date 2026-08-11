#!/usr/bin/env bash
# Start the three services `mvn verify` needs. Run from the repository root.
#
#     ./clients/java/serve-fixtures.sh
#
# Three, not one, because three of the behaviours the client has to get right are properties
# of a server's CONFIGURATION rather than of a request, and none of them can be provoked
# against a correctly configured one:
#
#   8000  the example pack, loaded            everything normal
#   8001  no dictionary at all                every match is a real 503
#   8002  the pack with a 1 ms deadline       every match is a real 504
#
# Ctrl-C stops all three. PIDs are also written to clients/java/.fixture-pids.
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
PIDS=()

start_fixture() {
    local port="$1"
    shift
    env PYTHONIOENCODING=utf-8 "$@" \
        "$PYTHON" -m uvicorn "$MODULE" --factory --host 127.0.0.1 --port "$port" \
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

# The pack, served normally.
start_fixture 8000 "$DICT" "$VOCAB" \
    NEXUS_API_FEEDBACK_PATH=examples/governance/out/feedback_http.jsonl

# No dictionary: every match answers 503 naming the setting, and so does feedback.
start_fixture 8001

# The pack with a deadline shorter than a match takes, so every match answers 504.
start_fixture 8002 "$DICT" "$VOCAB" NEXUS_API_DEADLINE_SECONDS=0.001

printf '%s\n' "${PIDS[@]}" > clients/java/.fixture-pids

echo
echo "Waiting for 8000 and 8002 to load their encoder..."
for port in 8000 8002; do
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
echo "Ctrl-C here stops all three."
wait
