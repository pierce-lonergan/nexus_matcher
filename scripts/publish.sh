#!/usr/bin/env bash
# scripts/publish.sh - Build and publish to PyPI
#
# Usage:
#   ./scripts/publish.sh              # publish to PyPI (asks for confirmation)
#   ./scripts/publish.sh --yes        # publish without asking -- required when there is no TTY
#   ./scripts/publish.sh test         # publish to TestPyPI
#   ./scripts/publish.sh build        # build and verify only, no upload
#
# ---------------------------------------------------------------------------
# Four things this script used to get wrong, all of which presented as "nothing
# happened", which is the worst way for a publish script to fail:
#
#   1. It ran `python`, whichever one was on PATH. On the machine this was written for
#      that is a system interpreter with no nexus_matcher installed, so `pytest tests/unit`
#      died loading conftest.py and `import nexus_matcher` for the version string could
#      never have worked. The venv is now found and verified before anything else runs.
#
#   2. It began with `pip install --upgrade pip build twine` into that interpreter.
#      Upgrading pip in a system Python needs write access this user does not have, so
#      under `set -e` the script exited at line 29 having printed one line. Tools are now
#      CHECKED, not installed -- a publish script is the wrong place to mutate an
#      interpreter.
#
#   3. `read -p "Are you sure?"` with no TTY returns EOF instantly, leaving $REPLY empty,
#      which failed the [Yy] test, printed "Aborted." and exited **0**. A publish that
#      silently does nothing and reports success is indistinguishable from one that
#      worked. Non-interactive now requires --yes and exits non-zero without it.
#
#   4. `rm -rf dist/` ran BEFORE the tests, so a failing test left no artifact at all --
#      including the verified one that was already there. Nothing is deleted until
#      everything that can fail has passed.
# ---------------------------------------------------------------------------

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

cd "$(dirname "$0")/.."

TARGET="pypi"
ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) ASSUME_YES=1 ;;
        test|build|pypi) TARGET="$arg" ;;
        *) log_error "unknown argument: $arg"; exit 2 ;;
    esac
done

# --- the interpreter -------------------------------------------------------
PYTHON=""
for candidate in ".venv/Scripts/python.exe" ".venv/bin/python" "$(command -v python3 || true)" "$(command -v python || true)"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if "$candidate" -c "import nexus_matcher" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    log_error "No interpreter found that can import nexus_matcher."
    log_error "The tests, the version string and the preflight all need it importable."
    log_error "Create the venv and install the project into it:"
    log_error "    python -m venv .venv && .venv/Scripts/python.exe -m pip install -e '.[dev]'"
    exit 1
fi
log_info "Using interpreter: $PYTHON"

# --- the tools -------------------------------------------------------------
MISSING=()
for tool in build twine pytest; do
    "$PYTHON" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$tool') else 1)" \
        || MISSING+=("$tool")
done
if (( ${#MISSING[@]} )); then
    log_error "Missing from $PYTHON: ${MISSING[*]}"
    log_error "Install them there, then re-run:"
    log_error "    $PYTHON -m pip install ${MISSING[*]}"
    exit 1
fi

# --- tests, BEFORE anything is deleted -------------------------------------
log_info "Running unit tests..."
if ! "$PYTHON" -m pytest tests/unit -q --tb=short; then
    log_error "Tests failed. Nothing was built and dist/ is untouched."
    exit 1
fi

log_info "Cleaning previous builds..."
rm -rf dist/ build/ ./*.egg-info src/*.egg-info

log_info "Building package..."
"$PYTHON" -m build

log_info "Built packages:"
ls -la dist/

log_info "Checking package metadata..."
"$PYTHON" -m twine check dist/*

VERSION=$("$PYTHON" -c "import nexus_matcher; print(nexus_matcher.__version__)")
log_info "Package version: $VERSION"

if [[ "$TARGET" == "build" ]]; then
    log_info "Build complete and metadata checked. Packages in dist/"
    log_info "Run the preflight before uploading:  $PYTHON scripts/release_preflight.py --wheel dist/*.whl"
    exit 0
fi

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
# ---------------------------------------------------------------------------
log_info "Running release preflight on the wheel that will be uploaded..."
shopt -s nullglob
WHEELS=(dist/*.whl)
shopt -u nullglob
if [[ ${#WHEELS[@]} -ne 1 ]]; then
    log_error "Expected exactly one wheel in dist/, found ${#WHEELS[@]}. Aborting publish."
    exit 1
fi
if ! "$PYTHON" scripts/release_preflight.py --wheel "${WHEELS[0]}"; then
    log_error "Release preflight failed! Aborting publish."
    log_error "It exits non-zero on any failed check; do not upload past it."
    exit 1
fi

# --- confirmation ----------------------------------------------------------
DESTINATION="PyPI (PRODUCTION)"
[[ "$TARGET" == "test" ]] && DESTINATION="TestPyPI"

echo ""
if (( ASSUME_YES )); then
    log_warn "Publishing $VERSION to $DESTINATION -- confirmation waived by --yes"
elif [[ -t 0 ]]; then
    log_warn "Publishing $VERSION to $DESTINATION"
    read -r -p "Are you sure? [y/N] " -n 1 REPLY
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted at the confirmation prompt. Nothing was uploaded."
        exit 1
    fi
else
    # Exit NON-ZERO. Silently succeeding here is how a publish "runs" and does nothing.
    log_error "No terminal attached, so the confirmation prompt cannot be answered."
    log_error "Nothing was uploaded. The artifacts in dist/ are built and preflighted."
    log_error "Re-run with --yes to publish without confirmation:"
    log_error "    bash scripts/publish.sh --yes"
    exit 1
fi

log_info "Uploading to $DESTINATION..."
if [[ "$TARGET" == "test" ]]; then
    "$PYTHON" -m twine upload --repository testpypi dist/*
    echo ""
    log_info "Published to TestPyPI!"
    log_info "Install with: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nexus-matcher==$VERSION"
else
    "$PYTHON" -m twine upload dist/*
    echo ""
    log_info "Published to PyPI!"
    log_info "Install with: pip install nexus-matcher==$VERSION"
fi

echo ""
log_info "Done!"
