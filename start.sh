#!/usr/bin/env bash
# =============================================================================
# NexusMatcher v2.0.0 — Comprehensive Startup Script
# =============================================================================
# This script:
#   1. Checks system prerequisites (Python, pip)
#   2. Creates a virtual environment
#   3. Installs dependencies from requirements.txt
#   4. Runs the test suite
#   5. Starts the system if tests pass
#
# Usage:
#   ./start.sh              # Full setup + test + start
#   ./start.sh --test-only  # Only run tests
#   ./start.sh --start-only # Only start server (skip tests)
#   ./start.sh --install    # Only install dependencies
#   ./start.sh --clean      # Remove venv and cache
#   ./start.sh --help       # Show help
#   ./start.sh --debug      # Enable debug mode
#
# Environment Variables:
#   NEXUS_HOST      - Server host (default: 0.0.0.0)
#   NEXUS_PORT      - Server port (default: 8000)
#   NEXUS_WORKERS   - Number of workers (default: 4)
#   NEXUS_LOG_LEVEL - Log level (default: info)
#   SKIP_TESTS      - Skip tests if set to "1" or "true"
# =============================================================================

# Error handling - don't exit immediately, handle errors gracefully
set -E  # Inherit ERR trap
trap 'handle_error $? $LINENO' ERR

handle_error() {
    local exit_code=$1
    local line_number=$2
    echo ""
    echo -e "${RED}[ERROR]${NC} Script failed at line $line_number with exit code $exit_code"
    echo ""
    echo "Press Enter to exit..."
    read -r
    exit $exit_code
}

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="nexus-matcher"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_MIN_VERSION="3.10"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

# Server defaults
HOST="${NEXUS_HOST:-0.0.0.0}"
PORT="${NEXUS_PORT:-8000}"
WORKERS="${NEXUS_WORKERS:-4}"
LOG_LEVEL="${NEXUS_LOG_LEVEL:-info}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                   ║"
    echo "║   ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗                    ║"
    echo "║   ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝                    ║"
    echo "║   ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗                    ║"
    echo "║   ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║                    ║"
    echo "║   ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║                    ║"
    echo "║   ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝                    ║"
    echo "║                                                                   ║"
    echo "║   ${BOLD}NexusMatcher v2.0.0${NC}${CYAN} — Enterprise Schema Matching             ║"
    echo "║                                                                   ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_step() {
    echo -e "\n${BOLD}${CYAN}═══ $1 ═══${NC}\n"
}

show_help() {
    echo "NexusMatcher Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h       Show this help message"
    echo "  --test-only      Only run tests (don't start server)"
    echo "  --start-only     Only start server (skip tests)"
    echo "  --install        Only install dependencies"
    echo "  --clean          Remove virtual environment and cache"
    echo "  --benchmark      Run benchmarks after tests"
    echo "  --dev            Install development dependencies"
    echo "  --full           Install all optional dependencies"
    echo "  --debug          Enable debug mode (verbose output)"
    echo ""
    echo "Environment Variables:"
    echo "  NEXUS_HOST       Server host (default: 0.0.0.0)"
    echo "  NEXUS_PORT       Server port (default: 8000)"
    echo "  NEXUS_WORKERS    Number of workers (default: 4)"
    echo "  NEXUS_LOG_LEVEL  Log level (default: info)"
    echo "  SKIP_TESTS       Skip tests if set to '1' or 'true'"
    echo ""
    echo "Examples:"
    echo "  $0                           # Full setup, test, and start"
    echo "  $0 --test-only               # Just run tests"
    echo "  $0 --debug                   # Run with debug output"
    echo "  NEXUS_PORT=9000 $0           # Start on port 9000"
    echo "  SKIP_TESTS=1 $0              # Skip tests, just start"
}

# =============================================================================
# PREREQUISITE CHECKS
# =============================================================================

check_python() {
    log_step "Checking Python Installation"
    
    PYTHON_CMD=""
    
    # On Windows (Git Bash), python3 often points to a broken Windows Store alias
    # So we test if the command actually WORKS, not just if it exists
    
    # Try python first (more reliable on Windows)
    if command -v python &> /dev/null; then
        if python -c "import sys; print(sys.version)" &> /dev/null; then
            PYTHON_CMD="python"
        fi
    fi
    
    # Try python3 if python didn't work
    if [[ -z "$PYTHON_CMD" ]] && command -v python3 &> /dev/null; then
        if python3 -c "import sys; print(sys.version)" &> /dev/null; then
            PYTHON_CMD="python3"
        fi
    fi
    
    # Try py (Windows Python Launcher)
    if [[ -z "$PYTHON_CMD" ]] && command -v py &> /dev/null; then
        if py -c "import sys; print(sys.version)" &> /dev/null; then
            PYTHON_CMD="py"
        fi
    fi
    
    if [[ -z "$PYTHON_CMD" ]]; then
        log_error "Python not found or not working!"
        log_info "Please install Python ${PYTHON_MIN_VERSION}+ from https://python.org"
        log_info "Make sure to check 'Add Python to PATH' during installation"
        return 1
    fi
    
    # Check version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
    PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")
    
    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10 ]]; then
        log_error "Python ${PYTHON_MIN_VERSION}+ required. Found: ${PYTHON_VERSION}"
        return 1
    fi
    
    log_success "Python ${PYTHON_VERSION} found (${PYTHON_CMD})"
}

check_pip() {
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        log_error "pip not found! Please install pip."
        exit 1
    fi
    log_success "pip is available"
}

# =============================================================================
# VIRTUAL ENVIRONMENT
# =============================================================================

setup_venv() {
    log_step "Setting Up Virtual Environment"
    
    if [[ -d "$VENV_DIR" ]]; then
        log_info "Virtual environment already exists at ${VENV_DIR}"
    else
        log_info "Creating virtual environment at ${VENV_DIR}..."
        if ! $PYTHON_CMD -m venv "$VENV_DIR"; then
            log_error "Failed to create virtual environment!"
            log_info "Try: $PYTHON_CMD -m pip install --user virtualenv"
            return 1
        fi
        log_success "Virtual environment created"
    fi
    
    # Activate virtual environment - handle both Unix and Windows (Git Bash)
    local ACTIVATE_SCRIPT=""
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"
    elif [[ -f "${VENV_DIR}/Scripts/activate" ]]; then
        # Windows Git Bash uses Scripts directory
        ACTIVATE_SCRIPT="${VENV_DIR}/Scripts/activate"
    fi
    
    if [[ -n "$ACTIVATE_SCRIPT" ]]; then
        log_info "Activating virtual environment..."
        source "$ACTIVATE_SCRIPT"
        log_success "Virtual environment activated"
    else
        log_error "Activation script not found!"
        log_info "Looked for:"
        log_info "  - ${VENV_DIR}/bin/activate"
        log_info "  - ${VENV_DIR}/Scripts/activate"
        return 1
    fi
    
    # Upgrade pip
    log_info "Upgrading pip..."
    if ! pip install --upgrade pip --quiet 2>/dev/null; then
        log_warning "pip upgrade failed, continuing anyway..."
    else
        log_success "pip upgraded"
    fi
}

# =============================================================================
# DEPENDENCY INSTALLATION
# =============================================================================

install_dependencies() {
    log_step "Installing Dependencies"
    
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        log_error "requirements.txt not found at ${REQUIREMENTS_FILE}"
        log_info "Current directory: $(pwd)"
        log_info "Script directory: ${SCRIPT_DIR}"
        ls -la "$SCRIPT_DIR" | head -20
        return 1
    fi
    
    log_info "Installing from requirements.txt..."
    if ! pip install -r "$REQUIREMENTS_FILE"; then
        log_error "Failed to install requirements!"
        return 1
    fi
    log_success "Core dependencies installed"
    
    # Install package in editable mode
    log_info "Installing NexusMatcher in editable mode..."
    if ! pip install -e "$SCRIPT_DIR"; then
        log_error "Failed to install NexusMatcher!"
        return 1
    fi
    log_success "NexusMatcher installed"
    
    # Optional: Install dev dependencies
    if [[ "$INSTALL_DEV" == "true" ]]; then
        log_info "Installing development dependencies..."
        pip install -e "${SCRIPT_DIR}[dev]" || log_warning "Dev dependencies partially failed"
        log_success "Development dependencies installed"
    fi
    
    # Optional: Install full dependencies
    if [[ "$INSTALL_FULL" == "true" ]]; then
        log_info "Installing all optional dependencies..."
        pip install -e "${SCRIPT_DIR}[full]" || log_warning "Full dependencies partially failed"
        log_success "All dependencies installed"
    fi
}

# =============================================================================
# TESTING
# =============================================================================

run_tests() {
    log_step "Running Test Suite"
    
    cd "$SCRIPT_DIR"
    
    log_info "Executing pytest..."
    echo ""
    
    # Run tests with coverage (but don't fail on coverage threshold for startup)
    if pytest tests/ -v --tb=short --no-cov; then
        echo ""
        log_success "All tests passed!"
        return 0
    else
        echo ""
        log_error "Some tests failed!"
        return 1
    fi
}

run_benchmarks() {
    log_step "Running Benchmarks"
    
    cd "$SCRIPT_DIR"
    
    if [[ -d "benchmarks" ]]; then
        log_info "Running SUITE-003 (ColBERT Reranking)..."
        python benchmarks/suite_003_colbert_reranking.py
        
        log_info "Running SUITE-004 (Cache Performance)..."
        python benchmarks/suite_004_cache_performance.py
        
        log_info "Running SUITE-005 (Incremental Updates)..."
        CORPUS_SIZE=10000 python benchmarks/suite_005_incremental_updates.py
        
        log_success "Benchmarks completed!"
    else
        log_warning "Benchmarks directory not found"
    fi
}

# =============================================================================
# SERVER STARTUP
# =============================================================================

start_server() {
    log_step "Starting NexusMatcher Server"
    
    cd "$SCRIPT_DIR"
    
    # On Windows (Git Bash/MSYS), multi-worker mode can be problematic
    # Detect Windows and adjust workers
    local ACTUAL_WORKERS="$WORKERS"
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ -n "$WINDIR" ]]; then
        if [[ "$WORKERS" -gt 1 ]]; then
            log_warning "Windows detected: Using single worker (multi-worker can cause issues)"
            ACTUAL_WORKERS=1
        fi
    fi
    
    log_info "Configuration:"
    echo "  Host:     ${HOST}"
    echo "  Port:     ${PORT}"
    echo "  Workers:  ${ACTUAL_WORKERS}"
    echo "  Log Level: ${LOG_LEVEL}"
    echo ""
    
    log_info "Starting uvicorn server..."
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  NexusMatcher API is starting...${NC}"
    echo -e "${GREEN}  API Documentation: http://${HOST}:${PORT}/docs${NC}"
    echo -e "${GREEN}  Health Check: http://${HOST}:${PORT}/health${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Start the server
    if [[ "$ACTUAL_WORKERS" -eq 1 ]]; then
        # Single worker mode (more compatible)
        exec uvicorn nexus_matcher.presentation.api.app:app \
            --host "$HOST" \
            --port "$PORT" \
            --log-level "$LOG_LEVEL"
    else
        # Multi-worker mode
        exec uvicorn nexus_matcher.presentation.api.app:app \
            --host "$HOST" \
            --port "$PORT" \
            --workers "$ACTUAL_WORKERS" \
            --log-level "$LOG_LEVEL"
    fi
}

# =============================================================================
# CLEANUP
# =============================================================================

clean() {
    log_step "Cleaning Up"
    
    if [[ -d "$VENV_DIR" ]]; then
        log_info "Removing virtual environment..."
        rm -rf "$VENV_DIR"
        log_success "Virtual environment removed"
    fi
    
    log_info "Removing Python cache..."
    find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$SCRIPT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "$SCRIPT_DIR" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find "$SCRIPT_DIR" -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find "$SCRIPT_DIR" -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    find "$SCRIPT_DIR" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    rm -f "$SCRIPT_DIR/coverage.xml" 2>/dev/null || true
    rm -f "$SCRIPT_DIR/.coverage" 2>/dev/null || true
    
    log_success "Cleanup complete"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    # Parse arguments
    TEST_ONLY=false
    START_ONLY=false
    INSTALL_ONLY=false
    DO_CLEAN=false
    RUN_BENCHMARKS=false
    INSTALL_DEV=false
    INSTALL_FULL=false
    DEBUG_MODE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --test-only)
                TEST_ONLY=true
                shift
                ;;
            --start-only)
                START_ONLY=true
                shift
                ;;
            --install)
                INSTALL_ONLY=true
                shift
                ;;
            --clean)
                DO_CLEAN=true
                shift
                ;;
            --benchmark)
                RUN_BENCHMARKS=true
                shift
                ;;
            --dev)
                INSTALL_DEV=true
                shift
                ;;
            --full)
                INSTALL_FULL=true
                shift
                ;;
            --debug)
                DEBUG_MODE=true
                set -x  # Enable bash debug mode
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Print banner
    print_banner
    
    # Debug info
    if [[ "$DEBUG_MODE" == "true" ]]; then
        log_info "Debug mode enabled"
        log_info "Script directory: $SCRIPT_DIR"
        log_info "Current directory: $(pwd)"
        log_info "Shell: $SHELL"
        log_info "Bash version: $BASH_VERSION"
        log_info "OS type: $OSTYPE"
        echo ""
        log_info "Testing Python commands..."
        echo -n "  python: "
        if python -c "import sys; print(sys.version.split()[0])" 2>/dev/null; then
            echo " ✓ works"
        else
            echo " ✗ not working"
        fi
        echo -n "  python3: "
        if python3 -c "import sys; print(sys.version.split()[0])" 2>/dev/null; then
            echo " ✓ works"
        else
            echo " ✗ not working (Windows Store alias?)"
        fi
        echo -n "  py: "
        if py -c "import sys; print(sys.version.split()[0])" 2>/dev/null; then
            echo " ✓ works"
        else
            echo " ✗ not found"
        fi
        echo ""
    fi
    
    # Handle clean
    if [[ "$DO_CLEAN" == "true" ]]; then
        clean
        echo ""
        echo "Press Enter to exit..."
        read -r
        exit 0
    fi
    
    # Prerequisites
    check_python
    check_pip
    
    # Setup
    setup_venv
    install_dependencies
    
    # Install only mode
    if [[ "$INSTALL_ONLY" == "true" ]]; then
        log_success "Installation complete!"
        echo ""
        echo "Press Enter to exit..."
        read -r
        exit 0
    fi
    
    # Run tests (unless skipped)
    SKIP_TESTS="${SKIP_TESTS:-false}"
    if [[ "$START_ONLY" != "true" && "$SKIP_TESTS" != "1" && "$SKIP_TESTS" != "true" ]]; then
        if ! run_tests; then
            log_error "Tests failed! Server will not start."
            log_info "Use --start-only or SKIP_TESTS=1 to skip tests"
            echo ""
            echo "Press Enter to exit..."
            read -r
            exit 1
        fi
    fi
    
    # Run benchmarks if requested
    if [[ "$RUN_BENCHMARKS" == "true" ]]; then
        run_benchmarks
    fi
    
    # Test only mode
    if [[ "$TEST_ONLY" == "true" ]]; then
        log_success "Test run complete!"
        echo ""
        echo "Press Enter to exit..."
        read -r
        exit 0
    fi
    
    # Start server
    start_server
}

# Run main function
main "$@"
