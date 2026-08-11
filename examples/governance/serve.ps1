# Serve the Gravel Bay pack over HTTP. Run from the repository root.
#
# The three variables below are the whole wiring. Without NEXUS_API_GOVERNANCE the server
# starts, answers 200, and returns `"governance": null` on every field -- indistinguishable
# from a glossary that carries no classes at all, which is why the pack ships this script
# rather than a sentence telling you to set some environment variables.
$ErrorActionPreference = "Stop"

$env:NEXUS_API_DICTIONARY = "examples/governance/glossary.csv"
$env:NEXUS_API_GOVERNANCE = "examples/governance/protection_classes.json"
$env:NEXUS_API_FEEDBACK_PATH = "examples/governance/out/feedback_http.jsonl"

nexus-matcher api --host 127.0.0.1 --port 8000
