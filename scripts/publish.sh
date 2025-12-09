#!/usr/bin/env bash
# scripts/publish.sh - Build and publish to PyPI
#
# Usage:
#   ./scripts/publish.sh          # Publish to PyPI
#   ./scripts/publish.sh test     # Publish to TestPyPI
#   ./scripts/publish.sh build    # Build only (no upload)

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Ensure we're in project root
cd "$(dirname "$0")/.."

# Parse arguments
TARGET="${1:-pypi}"

# Check for required tools
log_info "Checking dependencies..."
python -m pip install --quiet --upgrade pip build twine

# Clean previous builds
log_info "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Run tests first
log_info "Running tests..."
pytest tests/unit -q --tb=short || {
    log_error "Tests failed! Aborting publish."
    exit 1
}

# Build
log_info "Building package..."
python -m build

# Show what was built
log_info "Built packages:"
ls -la dist/

# Check package
log_info "Checking package metadata..."
twine check dist/*

# Extract version
VERSION=$(python -c "import nexus_matcher; print(nexus_matcher.__version__)")
log_info "Package version: $VERSION"

if [[ "$TARGET" == "build" ]]; then
    log_info "Build complete. Packages in dist/"
    exit 0
fi

# Confirm before publishing
echo ""
if [[ "$TARGET" == "test" ]]; then
    log_warn "Publishing to TestPyPI"
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted."
        exit 0
    fi
    
    log_info "Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    
    echo ""
    log_info "Published to TestPyPI!"
    log_info "Install with: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nexus-matcher==$VERSION"
else
    log_warn "Publishing to PyPI (PRODUCTION)"
    read -p "Are you sure? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted."
        exit 0
    fi
    
    log_info "Uploading to PyPI..."
    twine upload dist/*
    
    echo ""
    log_info "Published to PyPI!"
    log_info "Install with: pip install nexus-matcher==$VERSION"
fi

echo ""
log_info "Done! 🎉"
