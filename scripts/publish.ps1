# scripts/publish.ps1 - Build and publish to PyPI (Windows)
#
# Usage:
#   .\scripts\publish.ps1          # Publish to PyPI
#   .\scripts\publish.ps1 test     # Publish to TestPyPI  
#   .\scripts\publish.ps1 build    # Build only (no upload)

# The bash twin, scripts/publish.sh, records the four defects both scripts shared and why
# each presented as "nothing happened". The same fixes are applied here: resolve an
# interpreter that can actually import nexus_matcher, check tools rather than installing
# them, test before deleting anything, and NEVER treat an unanswerable prompt as a decline
# that exits 0.

param(
    [string]$Target = "pypi",
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red }

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $scriptPath "..")

# The interpreter. Whichever `python` is on PATH is not necessarily one that has this
# project installed, and every step below needs it importable.
$python = $null
$candidates = @(".venv\Scripts\python.exe", ".venvin\python")
if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += (Get-Command python).Source }
foreach ($candidate in $candidates) {
    if (-not (Test-Path $candidate)) { continue }
    & $candidate -c "import nexus_matcher" 2>$null
    if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
}
if (-not $python) {
    Write-Err "No interpreter found that can import nexus_matcher."
    Write-Err "The tests, the version string and the preflight all need it importable."
    Write-Err "    python -m venv .venv; .venv\Scripts\python.exe -m pip install -e '.[dev]'"
    exit 1
}
Write-Info "Using interpreter: $python"

# Tools are CHECKED, not installed. Upgrading pip in a system interpreter needs write
# access a normal user does not have, and it aborted this script on its first line.
$missing = @()
foreach ($tool in @("build", "twine", "pytest")) {
    & $python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$tool') else 1)"
    if ($LASTEXITCODE -ne 0) { $missing += $tool }
}
if ($missing.Count -gt 0) {
    Write-Err "Missing from ${python}: $($missing -join ', ')"
    Write-Err "    $python -m pip install $($missing -join ' ')"
    exit 1
}

# Tests run BEFORE anything is deleted, so a failure leaves the verified artifact intact.
Write-Info "Running unit tests..."
& $python -m pytest tests/unit -q --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Err "Tests failed. Nothing was built and dist/ is untouched."
    exit 1
}

Write-Info "Cleaning previous builds..."
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path build) { Remove-Item -Recurse -Force build }
Get-ChildItem -Filter "*.egg-info" | Remove-Item -Recurse -Force
Get-ChildItem -Path src -Filter "*.egg-info" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

# Build
Write-Info "Building package..."
& $python -m build

# Show built packages
Write-Info "Built packages:"
Get-ChildItem dist

# Check package
Write-Info "Checking package metadata..."
& $python -m twine check dist/*

# Get version
$version = & $python -c "import nexus_matcher; print(nexus_matcher.__version__)"
Write-Info "Package version: $version"

if ($Target -eq "build") {
    Write-Info "Build complete. Packages in dist/"
    exit 0
}

# ---------------------------------------------------------------------------
# Release preflight -- the gate that stands between this artifact and PyPI.
#
# `twine check` reads the metadata. It cannot see any of what shipped broken in
# 2.0.0: a console script installed without the dependencies it needs, a CLI that
# crashed on a legacy Windows codepage, an `__all__` entry that broke
# `from nexus_matcher import *` on a default install. Only .github/workflows/publish.yml
# ran release_preflight.py, so the CI path was gated and THIS path -- the one a human
# uses, under time pressure, when CI is red or slow -- was not.
#
# `--wheel` rather than letting the preflight build its own: the point is to check the
# file that is about to be uploaded. A preflight that builds a second wheel proves
# something about a file nobody publishes, which is the same class of mistake as the
# stale `dist/` artifact that a review read as evidence of a shipped release.
#
# $LASTEXITCODE is tested explicitly. $ErrorActionPreference = "Stop" does not stop this
# script on a native executable's non-zero exit, so without the test the preflight would
# print "NOT FIT TO PUBLISH" and the upload would proceed anyway -- a gate that warns.
# ---------------------------------------------------------------------------
Write-Info "Running release preflight on the wheel that will be uploaded..."
$wheels = @(Get-ChildItem -Path dist -Filter *.whl)
if ($wheels.Count -ne 1) {
    Write-Err "Expected exactly one wheel in dist/, found $($wheels.Count). Aborting publish."
    exit 1
}
& $python scripts/release_preflight.py --wheel $wheels[0].FullName
if ($LASTEXITCODE -ne 0) {
    Write-Err "Release preflight failed! Aborting publish."
    Write-Err "It exits non-zero on any failed check; do not upload past it."
    exit 1
}

Write-Host ""

if ($Target -eq "test") {
    Write-Warn "Publishing to TestPyPI"
    if ($Yes) {
        Write-Warn "Confirmation waived by -Yes"
        $confirm = "y"
    } elseif ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
        $confirm = Read-Host "Continue? [y/N]"
    } else {
        # Exit NON-ZERO. A publish that silently does nothing and reports success is
        # indistinguishable from one that worked.
        Write-Err "No terminal attached, so the confirmation prompt cannot be answered."
        Write-Err "Nothing was uploaded. The artifacts in dist/ are built and preflighted."
        Write-Err "    .\scripts\publish.ps1 -Yes"
        exit 1
    }
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Info "Aborted at the confirmation prompt. Nothing was uploaded."
        exit 1
    }
    
    Write-Info "Uploading to TestPyPI..."
    & $python -m twine upload --repository testpypi dist/*
    
    Write-Host ""
    Write-Info "Published to TestPyPI!"
    Write-Info "Install with: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nexus-matcher==$version"
}
else {
    Write-Warn "Publishing to PyPI (PRODUCTION)"
    if ($Yes) {
        Write-Warn "Confirmation waived by -Yes"
        $confirm = "y"
    } elseif ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
        $confirm = Read-Host "Are you sure? [y/N]"
    } else {
        # Exit NON-ZERO. A publish that silently does nothing and reports success is
        # indistinguishable from one that worked.
        Write-Err "No terminal attached, so the confirmation prompt cannot be answered."
        Write-Err "Nothing was uploaded. The artifacts in dist/ are built and preflighted."
        Write-Err "    .\scripts\publish.ps1 -Yes"
        exit 1
    }
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Info "Aborted at the confirmation prompt. Nothing was uploaded."
        exit 1
    }
    
    Write-Info "Uploading to PyPI..."
    & $python -m twine upload dist/*
    
    Write-Host ""
    Write-Info "Published to PyPI!"
    Write-Info "Install with: pip install nexus-matcher==$version"
}

Write-Host ""
Write-Info "Done! 🎉"
