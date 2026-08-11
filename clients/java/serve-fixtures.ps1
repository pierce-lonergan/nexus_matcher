# Start the three services `mvn verify` needs. Run from the repository root.
#
#     .\clients\java\serve-fixtures.ps1
#
# Three, not one, because three of the behaviours the client has to get right are properties
# of a server's CONFIGURATION rather than of a request, and none of them can be provoked
# against a correctly configured one:
#
#   8000  the example pack, loaded            everything normal
#   8001  no dictionary at all                every match is a real 503
#   8002  the pack with a 1 ms deadline       every match is a real 504
#
# Stop them with .\clients\java\stop-fixtures.ps1, or close the three windows.

$ErrorActionPreference = "Stop"

if (-not (Test-Path "examples/governance/glossary.csv")) {
    throw "run this from the repository root: examples/governance/glossary.csv not found"
}

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "no virtualenv at $python"
}

$module = "nexus_matcher.presentation.api.app:create_app"
$common = @{
    NEXUS_API_DICTIONARY = "examples/governance/glossary.csv"
    NEXUS_API_GOVERNANCE = "examples/governance/protection_classes.json"
}

function Start-Fixture([int]$Port, [hashtable]$Environment) {
    $assignments = ($Environment.GetEnumerator() | ForEach-Object {
        "`$env:$($_.Key)='$($_.Value)'"
    }) -join "; "
    $command = "$assignments; `$env:PYTHONIOENCODING='utf-8'; " +
               "& '$python' -m uvicorn $module --factory --host 127.0.0.1 --port $Port"
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

Write-Host ""
Write-Host "Wait for 8000 and 8002 to load their encoder (about 20 s each), then:"
Write-Host "  cd clients\java; mvn verify"
Write-Host ""
Write-Host "If the integration tests die with 'Unable to establish loopback connection',"
Write-Host "this host blocks AF_UNIX and the JDK's HTTP client cannot start. See README.md:"
Write-Host '  mvn verify "-Dnexus.matcher.itJvmArgs=-Djdk.net.unixdomain.tmpdir=C:\no-such-dir"'
