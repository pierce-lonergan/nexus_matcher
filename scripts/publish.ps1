# scripts/publish.ps1 - Build and publish to PyPI (Windows)
#
# Usage:
#   .\scripts\publish.ps1          # Publish to PyPI
#   .\scripts\publish.ps1 test     # Publish to TestPyPI  
#   .\scripts\publish.ps1 build    # Build only (no upload)

param(
    [string]$Target = "pypi"
)

$ErrorActionPreference = "Stop"

function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red }

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $scriptPath "..")

# Check dependencies
Write-Info "Checking dependencies..."
python -m pip install --quiet --upgrade pip build twine

# Clean previous builds
Write-Info "Cleaning previous builds..."
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path build) { Remove-Item -Recurse -Force build }
Get-ChildItem -Filter "*.egg-info" | Remove-Item -Recurse -Force
Get-ChildItem -Path src -Filter "*.egg-info" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

# Run tests
Write-Info "Running tests..."
pytest tests/unit -q --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Err "Tests failed! Aborting publish."
    exit 1
}

# Build
Write-Info "Building package..."
python -m build

# Show built packages
Write-Info "Built packages:"
Get-ChildItem dist

# Check package
Write-Info "Checking package metadata..."
twine check dist/*

# Get version
$version = python -c "import nexus_matcher; print(nexus_matcher.__version__)"
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
python scripts/release_preflight.py --wheel $wheels[0].FullName
if ($LASTEXITCODE -ne 0) {
    Write-Err "Release preflight failed! Aborting publish."
    Write-Err "It exits non-zero on any failed check; do not upload past it."
    exit 1
}

Write-Host ""

if ($Target -eq "test") {
    Write-Warn "Publishing to TestPyPI"
    $confirm = Read-Host "Continue? [y/N]"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Info "Aborted."
        exit 0
    }
    
    Write-Info "Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    
    Write-Host ""
    Write-Info "Published to TestPyPI!"
    Write-Info "Install with: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nexus-matcher==$version"
}
else {
    Write-Warn "Publishing to PyPI (PRODUCTION)"
    $confirm = Read-Host "Are you sure? [y/N]"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Info "Aborted."
        exit 0
    }
    
    Write-Info "Uploading to PyPI..."
    twine upload dist/*
    
    Write-Host ""
    Write-Info "Published to PyPI!"
    Write-Info "Install with: pip install nexus-matcher==$version"
}

Write-Host ""
Write-Info "Done! 🎉"
