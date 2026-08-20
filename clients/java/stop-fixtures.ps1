# Stop the integration-test fixture servers started by serve-fixtures.ps1.
#
#     .\clients\java\stop-fixtures.ps1
#
# Matches on the LISTENING PORT rather than on the process name, so it cannot take down an
# unrelated python.exe -- four of which is exactly what a developer running this repository is
# likely to have.

$ErrorActionPreference = "Stop"

foreach ($port in 8000, 8001, 8002, 8003) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Host "nothing listening on $port"
        continue
    }
    # $procId, not $pid: $PID is a read-only automatic variable in PowerShell and assigning to
    # it in a foreach is a hard error, not a warning.
    foreach ($procId in ($connections.OwningProcess | Select-Object -Unique)) {
        $process = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "stopping $($process.ProcessName) (pid $procId) on port $port"
            Stop-Process -Id $procId -Force -Confirm:$false
        }
    }
}
