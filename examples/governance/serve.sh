#!/usr/bin/env bash
# Serve the Gravel Bay pack over HTTP. Run from the repository root.
#
# The three variables below are the whole wiring. Without NEXUS_API_GOVERNANCE the server
# starts, answers 200, and returns `"governance": null` on every field -- indistinguishable
# from a glossary that carries no classes at all, which is why the pack ships this script
# rather than a sentence telling you to set some environment variables.
set -euo pipefail

export NEXUS_API_DICTIONARY="examples/governance/glossary.csv"
export NEXUS_API_GOVERNANCE="examples/governance/protection_classes.json"
export NEXUS_API_FEEDBACK_PATH="examples/governance/out/feedback_http.jsonl"

exec nexus-matcher api --host 127.0.0.1 --port 8000
