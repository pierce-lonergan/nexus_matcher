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
