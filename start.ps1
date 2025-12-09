# =============================================================================
# NexusMatcher v2.0.0 — Windows Startup Script (PowerShell)
# =============================================================================
# This script:
#   1. Checks system prerequisites (Python, pip)
#   2. Creates a virtual environment
#   3. Installs dependencies from requirements.txt
#   4. Runs the test suite
#   5. Starts the system if tests pass
#
# Usage:
#   .\start.ps1              # Full setup + test + start
#   .\start.ps1 -TestOnly    # Only run tests
#   .\start.ps1 -StartOnly   # Only start server (skip tests)
#   .\start.ps1 -Install     # Only install dependencies
#   .\start.ps1 -Clean       # Remove venv and cache
#   .\start.ps1 -Help        # Show help
#
# Environment Variables:
#   $env:NEXUS_HOST      - Server host (default: 0.0.0.0)
#   $env:NEXUS_PORT      - Server port (default: 8000)
#   $env:NEXUS_WORKERS   - Number of workers (default: 4)
#   $env:NEXUS_LOG_LEVEL - Log level (default: info)
# =============================================================================

param(
    [switch]$Help,
    [switch]$TestOnly,
    [switch]$StartOnly,
    [switch]$Install,
    [switch]$Clean,
    [switch]$Benchmark,
    [switch]$Dev,
    [switch]$Full
)

# =============================================================================
# CONFIGURATION
# =============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectName = "nexus-matcher"
$VenvDir = Join-Path $ScriptDir ".venv"
$PythonMinVersion = "3.10"
$RequirementsFile = Join-Path $ScriptDir "requirements.txt"

# Server defaults
$Host_ = if ($env:NEXUS_HOST) { $env:NEXUS_HOST } else { "0.0.0.0" }
$Port = if ($env:NEXUS_PORT) { $env:NEXUS_PORT } else { "8000" }
$Workers = if ($env:NEXUS_WORKERS) { $env:NEXUS_WORKERS } else { "4" }
$LogLevel = if ($env:NEXUS_LOG_LEVEL) { $env:NEXUS_LOG_LEVEL } else { "info" }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function Write-Banner {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║                                                                   ║" -ForegroundColor Cyan
    Write-Host "║   ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗                    ║" -ForegroundColor Cyan
    Write-Host "║   ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝                    ║" -ForegroundColor Cyan
    Write-Host "║   ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗                    ║" -ForegroundColor Cyan
    Write-Host "║   ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║                    ║" -ForegroundColor Cyan
    Write-Host "║   ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║                    ║" -ForegroundColor Cyan
    Write-Host "║   ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝                    ║" -ForegroundColor Cyan
    Write-Host "║                                                                   ║" -ForegroundColor Cyan
    Write-Host "║   NexusMatcher v2.0.0 — Enterprise Schema Matching               ║" -ForegroundColor Cyan
    Write-Host "║                                                                   ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Info($message) {
    Write-Host "[INFO] " -ForegroundColor Blue -NoNewline
    Write-Host $message
}

function Write-Success($message) {
    Write-Host "[✓] " -ForegroundColor Green -NoNewline
    Write-Host $message
}

function Write-Warning($message) {
    Write-Host "[!] " -ForegroundColor Yellow -NoNewline
    Write-Host $message
}

function Write-Error_($message) {
    Write-Host "[✗] " -ForegroundColor Red -NoNewline
    Write-Host $message
}

function Write-Step($message) {
    Write-Host ""
    Write-Host "═══ $message ═══" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Help {
    Write-Host "NexusMatcher Startup Script (PowerShell)"
    Write-Host ""
    Write-Host "Usage: .\start.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Help        Show this help message"
    Write-Host "  -TestOnly    Only run tests (don't start server)"
    Write-Host "  -StartOnly   Only start server (skip tests)"
    Write-Host "  -Install     Only install dependencies"
    Write-Host "  -Clean       Remove virtual environment and cache"
    Write-Host "  -Benchmark   Run benchmarks after tests"
    Write-Host "  -Dev         Install development dependencies"
    Write-Host "  -Full        Install all optional dependencies"
    Write-Host ""
    Write-Host "Environment Variables:"
    Write-Host "  `$env:NEXUS_HOST       Server host (default: 0.0.0.0)"
    Write-Host "  `$env:NEXUS_PORT       Server port (default: 8000)"
    Write-Host "  `$env:NEXUS_WORKERS    Number of workers (default: 4)"
    Write-Host "  `$env:NEXUS_LOG_LEVEL  Log level (default: info)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\start.ps1                           # Full setup, test, and start"
    Write-Host "  .\start.ps1 -TestOnly                 # Just run tests"
    Write-Host "  `$env:NEXUS_PORT=9000; .\start.ps1    # Start on port 9000"
}

# =============================================================================
# PREREQUISITE CHECKS
# =============================================================================

function Test-Python {
    Write-Step "Checking Python Installation"
    
    # Try to find Python
    $pythonCmd = $null
    
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        $pythonCmd = "python"
    }
    elseif (Get-Command "python3" -ErrorAction SilentlyContinue) {
        $pythonCmd = "python3"
    }
    elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
        $pythonCmd = "py"
    }
    
    if (-not $pythonCmd) {
        Write-Error_ "Python not found! Please install Python $PythonMinVersion+"
        exit 1
    }
    
    # Check version
    $versionOutput = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $major = & $pythonCmd -c "import sys; print(sys.version_info.major)"
    $minor = & $pythonCmd -c "import sys; print(sys.version_info.minor)"
    
    if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 10)) {
        Write-Error_ "Python $PythonMinVersion+ required. Found: $versionOutput"
        exit 1
    }
    
    Write-Success "Python $versionOutput found ($pythonCmd)"
    return $pythonCmd
}

# =============================================================================
# VIRTUAL ENVIRONMENT
# =============================================================================

function Setup-Venv($pythonCmd) {
    Write-Step "Setting Up Virtual Environment"
    
    if (Test-Path $VenvDir) {
        Write-Info "Virtual environment already exists at $VenvDir"
    }
    else {
        Write-Info "Creating virtual environment..."
        & $pythonCmd -m venv $VenvDir
        Write-Success "Virtual environment created"
    }
    
    # Activate virtual environment
    $activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        Write-Success "Virtual environment activated"
    }
    else {
        Write-Error_ "Could not find activation script at $activateScript"
        exit 1
    }
    
    # Upgrade pip
    Write-Info "Upgrading pip..."
    & pip install --upgrade pip --quiet
    Write-Success "pip upgraded"
}

# =============================================================================
# DEPENDENCY INSTALLATION
# =============================================================================

function Install-Dependencies {
    Write-Step "Installing Dependencies"
    
    if (-not (Test-Path $RequirementsFile)) {
        Write-Error_ "requirements.txt not found at $RequirementsFile"
        exit 1
    }
    
    Write-Info "Installing from requirements.txt..."
    & pip install -r $RequirementsFile --quiet
    Write-Success "Core dependencies installed"
    
    # Install package in editable mode
    Write-Info "Installing NexusMatcher in editable mode..."
    & pip install -e $ScriptDir --quiet
    Write-Success "NexusMatcher installed"
    
    # Optional: Install dev dependencies
    if ($Dev) {
        Write-Info "Installing development dependencies..."
        & pip install -e "$ScriptDir[dev]" --quiet
        Write-Success "Development dependencies installed"
    }
    
    # Optional: Install full dependencies
    if ($Full) {
        Write-Info "Installing all optional dependencies..."
        & pip install -e "$ScriptDir[full]" --quiet
        Write-Success "All dependencies installed"
    }
}

# =============================================================================
# TESTING
# =============================================================================

function Invoke-Tests {
    Write-Step "Running Test Suite"
    
    Set-Location $ScriptDir
    
    Write-Info "Executing pytest..."
    Write-Host ""
    
    $result = & pytest tests/ -v --tb=short --no-cov
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Success "All tests passed!"
        return $true
    }
    else {
        Write-Host ""
        Write-Error_ "Some tests failed!"
        return $false
    }
}

function Invoke-Benchmarks {
    Write-Step "Running Benchmarks"
    
    Set-Location $ScriptDir
    
    $benchmarksDir = Join-Path $ScriptDir "benchmarks"
    if (Test-Path $benchmarksDir) {
        Write-Info "Running SUITE-003 (ColBERT Reranking)..."
        & python (Join-Path $benchmarksDir "suite_003_colbert_reranking.py")
        
        Write-Info "Running SUITE-004 (Cache Performance)..."
        & python (Join-Path $benchmarksDir "suite_004_cache_performance.py")
        
        Write-Info "Running SUITE-005 (Incremental Updates)..."
        $env:CORPUS_SIZE = "10000"
        & python (Join-Path $benchmarksDir "suite_005_incremental_updates.py")
        
        Write-Success "Benchmarks completed!"
    }
    else {
        Write-Warning "Benchmarks directory not found"
    }
}

# =============================================================================
# SERVER STARTUP
# =============================================================================

function Start-Server {
    Write-Step "Starting NexusMatcher Server"
    
    Set-Location $ScriptDir
    
    Write-Info "Configuration:"
    Write-Host "  Host:     $Host_"
    Write-Host "  Port:     $Port"
    Write-Host "  Workers:  $Workers"
    Write-Host "  Log Level: $LogLevel"
    Write-Host ""
    
    Write-Info "Starting uvicorn server..."
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  NexusMatcher API is starting..." -ForegroundColor Green
    Write-Host "  API Documentation: http://${Host_}:${Port}/docs" -ForegroundColor Green
    Write-Host "  Health Check: http://${Host_}:${Port}/health" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    
    # Start the server
    & uvicorn nexus_matcher.presentation.api.app:app `
        --host $Host_ `
        --port $Port `
        --workers $Workers `
        --log-level $LogLevel
}

# =============================================================================
# CLEANUP
# =============================================================================

function Invoke-Clean {
    Write-Step "Cleaning Up"
    
    if (Test-Path $VenvDir) {
        Write-Info "Removing virtual environment..."
        Remove-Item -Recurse -Force $VenvDir
        Write-Success "Virtual environment removed"
    }
    
    Write-Info "Removing Python cache..."
    Get-ChildItem -Path $ScriptDir -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $ScriptDir -Recurse -File -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $ScriptDir -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $ScriptDir -Recurse -Directory -Filter ".mypy_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $ScriptDir -Recurse -Directory -Filter ".ruff_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $ScriptDir -Recurse -Directory -Filter "*.egg-info" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $ScriptDir "coverage.xml") -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $ScriptDir ".coverage") -Force -ErrorAction SilentlyContinue
    
    Write-Success "Cleanup complete"
}

# =============================================================================
# MAIN
# =============================================================================

function Main {
    # Show help if requested
    if ($Help) {
        Show-Help
        exit 0
    }
    
    # Print banner
    Write-Banner
    
    # Handle clean
    if ($Clean) {
        Invoke-Clean
        exit 0
    }
    
    # Prerequisites
    $pythonCmd = Test-Python
    
    # Setup
    Setup-Venv $pythonCmd
    Install-Dependencies
    
    # Install only mode
    if ($Install) {
        Write-Success "Installation complete!"
        exit 0
    }
    
    # Run tests (unless skipped)
    if (-not $StartOnly) {
        $testsPassed = Invoke-Tests
        if (-not $testsPassed) {
            Write-Error_ "Tests failed! Server will not start."
            Write-Info "Use -StartOnly to skip tests"
            exit 1
        }
    }
    
    # Run benchmarks if requested
    if ($Benchmark) {
        Invoke-Benchmarks
    }
    
    # Test only mode
    if ($TestOnly) {
        Write-Success "Test run complete!"
        exit 0
    }
    
    # Start server
    Start-Server
}

# Run main function
Main
