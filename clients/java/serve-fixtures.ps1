# Start the five services `mvn verify` needs. Run from the repository root.
#
#     .\clients\java\serve-fixtures.ps1
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
# 8003 and 8004 are the least obvious, and they are the same kind of fixture: a behaviour the
# client must read correctly that no correctly-configured stock server produces.
#
# A NO_MATCH field decision needs either a field with no candidates at all or a configured
# absolute-score floor that rank 1 fails; the library ships no floor and will not invent one.
# See clients/java/fixture-absolute-floor.json.
#
# A candidate with provenance APPROVED_PAIR needs a feedback consumer attached; the library
# ships none -- `create_app()` builds none and `NexusMatcher()` takes `feedback_consumer=None`
# -- so on every other port every candidate is RETRIEVAL. See
# clients/java/fixture_approved_pair_app.py, which starts a throwaway server with the
# reference consumer wired in and refuses to start if the verdict does not resolve.
#
# Both are test fixtures, not recommended defaults.
#
# Stop them with .\clients\java\stop-fixtures.ps1, or close the five windows.

$ErrorActionPreference = "Stop"

if (-not (Test-Path "examples/governance/glossary.csv")) {
    throw "run this from the repository root: examples/governance/glossary.csv not found"
}

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "no virtualenv at $python"
}

$module = "nexus_matcher.presentation.api.app:create_app"
# 8004's own factory, beside this script and reached by putting clients/java on the path, so
# nothing in the package has to know a fixture exists.
$bypassModule = "fixture_approved_pair_app:create_app"
$common = @{
    NEXUS_API_DICTIONARY = "examples/governance/glossary.csv"
    NEXUS_API_GOVERNANCE = "examples/governance/protection_classes.json"
}

function Start-Fixture([int]$Port, [hashtable]$Environment, [string]$Factory = $module) {
    $assignments = ($Environment.GetEnumerator() | ForEach-Object {
        "`$env:$($_.Key)='$($_.Value)'"
    }) -join "; "
    $command = "$assignments; `$env:PYTHONIOENCODING='utf-8'; " +
               "& '$python' -m uvicorn $Factory --factory --host 127.0.0.1 --port $Port"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $command | Out-Null
    Write-Host "started fixture on 127.0.0.1:$Port"
}

# The pack, served normally.
Start-Fixture 8000 ($common + @{
    NEXUS_API_FEEDBACK_PATH = "examples/governance/out/feedback_http.jsonl"
})

# No dictionary: every match answers 503 naming the setting, and so does feedback.
Start-Fixture 8001 @{}

# The pack with a deadline shorter than a match takes, so every match answers 504.
Start-Fixture 8002 ($common + @{ NEXUS_API_DEADLINE_SECONDS = "0.001" })

# The pack with an absolute-score floor, so a field the glossary does not describe earns a
# NO_MATCH field decision -- with its candidates still attached, which is the half a client
# gets wrong.
Start-Fixture 8003 ($common + @{
    NEXUS_API_MATCHING_CONFIG = "clients/java/fixture-absolute-floor.json"
})

# The pack with one reviewer verdict standing, so `booking.passenger.legal_name` is answered by
# a human's decision and comes back as provenance APPROVED_PAIR -- one candidate however large
# top_k is, no absoluteScore and no explain, at confidence 1.0.
Start-Fixture 8004 ($common + @{
    NEXUS_FIXTURE_APPROVED_PAIRS = "clients/java/fixture-approved-pairs.jsonl"
    PYTHONPATH = "clients/java"
}) $bypassModule

Write-Host ""
Write-Host "Wait for 8000, 8002, 8003 and 8004 to load their encoder (about 20 s each), then:"
Write-Host "  cd clients\java; mvn verify"
Write-Host ""
Write-Host "If the integration tests die with 'Unable to establish loopback connection',"
Write-Host "this host blocks AF_UNIX and the JDK's HTTP client cannot start. See README.md:"
Write-Host '  mvn verify "-Dnexus.matcher.itJvmArgs=-Djdk.net.unixdomain.tmpdir=C:\no-such-dir"'
