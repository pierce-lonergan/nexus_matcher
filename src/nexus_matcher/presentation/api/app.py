"""
nexus_matcher.presentation.api.app | Layer: PRESENTATION
FastAPI application factory: health, introspection, and the matching endpoints.

## Relationships
# DEPENDS_ON → application/use_cases/* :: use case orchestration
# DEPENDS_ON → infrastructure/config :: application configuration
# DEPENDS_ON → presentation/api/matching :: POST /api/v1/match[/batch]
# DEPENDS_ON → presentation/api/feedback :: POST /api/v1/feedback
# DEPENDS_ON → presentation/api/limits :: admission control and the deadline
# DEPENDS_ON → shared/logging :: structured logging
# DEPENDS_ON → shared/exceptions :: error handling

## Attributes
# Security: CORS, rate limiting, API key auth (optional)
# Performance: Async handlers, connection pooling
# Reliability: Health checks, graceful shutdown

## Configuration, and why it is environment-driven

Both documented ways to start this server -- `nexus-matcher api` and
`uvicorn nexus_matcher.presentation.api.app:create_app --factory` -- call `create_app()`
with NO arguments. An operator therefore has no place to pass a dictionary, a deadline or
a feedback path, so every one of those is also readable from the environment:

    NEXUS_API_DICTIONARY        dictionary file to load at startup (.xlsx / .csv)
    NEXUS_API_GOVERNANCE        the caller's controlled-vocabulary JSON file
    NEXUS_API_MATCHING_CONFIG   optional MatchingConfig JSON/TOML file
    NEXUS_API_FEEDBACK_PATH     append-only JSONL file for reviewer feedback
    NEXUS_API_DEADLINE_SECONDS  server-side deadline, default 25.0
    NEXUS_API_MAX_WORKERS       matching threads, default 4
    NEXUS_API_MAX_QUEUED        work admitted beyond the workers, default 32
    NEXUS_API_MAX_FIELDS        cap for /api/v1/match, default 100
    NEXUS_API_MAX_BATCH_FIELDS  cap for /api/v1/match/batch, default 250

The keyword arguments exist for embedders and for tests. Without the environment path the
new endpoints would be code with no way to reach them from either shipped entry point --
H-006's exact shape, and its most expensive instance shipped 2.5x faster and unreachable.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nexus_matcher.presentation.api.errors import validation_exception_handler
from nexus_matcher.presentation.api.feedback import FeedbackRecorder, create_feedback_router
from nexus_matcher.presentation.api.limits import BoundedWorkPool, MatchServiceLimits
from nexus_matcher.presentation.api.matching import (
    DeterministicJSONResponse,
    MatcherHandle,
    MatchService,
    create_matching_router,
)
from nexus_matcher.shared.exceptions import (
    APIError,
    NexusMatcherError,
)
from nexus_matcher.shared.logging import (
    add_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
    log_performance,
    set_correlation_id,
)

# =============================================================================
# RESPONSE MODELS
# =============================================================================


# Helper function for timezone-aware UTC datetime
def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Health status: healthy, degraded, unhealthy")
    timestamp: datetime = Field(default_factory=_utc_now)
    version: str = Field(default="2.0.0")
    checks: dict[str, Any] = Field(default_factory=dict)


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool
    timestamp: datetime = Field(default_factory=_utc_now)
    components: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: dict[str, Any]


# =============================================================================
# APPLICATION STATE
# =============================================================================


class AppState:
    """Application state holder."""

    def __init__(self) -> None:
        self.startup_time: datetime = _utc_now()
        self.is_ready: bool = False
        self.components: dict[str, bool] = {}

    def set_component_status(self, name: str, healthy: bool) -> None:
        """Update component health status."""
        self.components[name] = healthy

    def check_ready(self) -> bool:
        """Check if all components are ready."""
        return all(self.components.values()) if self.components else False


# =============================================================================
# MIDDLEWARE
# =============================================================================


async def request_id_middleware(request: Request, call_next: Callable) -> Response:
    """Add request ID to all requests."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    set_correlation_id(request_id)
    add_request_context(
        path=request.url.path,
        method=request.method,
    )

    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        log_performance(
            operation="http_request",
            duration_ms=duration_ms,
            success=response.status_code < 400,
            status_code=response.status_code,
            path=request.url.path,
        )

        return response
    finally:
        clear_request_context()


# =============================================================================
# ERROR HANDLERS
# =============================================================================


async def nexus_exception_handler(request: Request, exc: NexusMatcherError) -> JSONResponse:
    """Handle NexusMatcher exceptions."""
    logger = get_logger("api.error")

    # Determine status code
    status_code = exc.status_code if isinstance(exc, APIError) else 500

    logger.error(
        "request_failed",
        error_code=exc.code.value,
        error_message=exc.message,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger = get_logger("api.error")

    logger.exception(
        "unexpected_error",
        error_type=type(exc).__name__,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "NEXUS-1000",
                "message": "An unexpected error occurred",
                "details": {},
            }
        },
    )


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================


def create_health_router(app_state: AppState):
    """Create health check router."""
    from fastapi import APIRouter

    router = APIRouter(tags=["Health"])

    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check",
        description="Basic health check endpoint",
    )
    async def health_check() -> HealthResponse:
        """
        Check application health.

        Returns basic health status. For detailed component status,
        use /health/ready.
        """
        all_healthy = all(app_state.components.values()) if app_state.components else True

        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            version="2.0.0",
            checks={
                "uptime_seconds": (_utc_now() - app_state.startup_time).total_seconds(),
            },
        )

    @router.get(
        "/health/live",
        summary="Liveness probe",
        description="Kubernetes liveness probe endpoint",
    )
    async def liveness() -> dict[str, str]:
        """
        Liveness probe for Kubernetes.

        Returns 200 if the application is running.
        """
        return {"status": "alive"}

    @router.get(
        "/health/ready",
        response_model=ReadinessResponse,
        summary="Readiness probe",
        description="Kubernetes readiness probe endpoint",
    )
    async def readiness() -> ReadinessResponse:
        """
        Readiness probe for Kubernetes.

        Returns 200 if the application is ready to serve traffic.
        Returns 503 if any critical component is not ready.
        """
        ready = app_state.check_ready()

        if not ready:
            raise HTTPException(
                status_code=503,
                detail="Service not ready",
            )

        return ReadinessResponse(
            ready=ready,
            components=app_state.components.copy(),
        )

    @router.get(
        "/health/startup",
        summary="Startup probe",
        description="Kubernetes startup probe endpoint",
    )
    async def startup_probe() -> dict[str, Any]:
        """
        Startup probe for Kubernetes.

        Returns 200 if initial startup is complete.
        """
        if not app_state.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Service starting up",
            )

        return {
            "status": "started",
            "startup_time": app_state.startup_time.isoformat(),
        }

    return router


# =============================================================================
# MATCHER BOOTSTRAP
# =============================================================================


def _load_configured_matcher(environ: Mapping[str, str]) -> object | None:
    """
    Build the matcher `NEXUS_API_DICTIONARY` names, or return None when none is named.

    Raises whatever the loader raises. The caller turns that into an unhealthy component
    and a 503 on the matching routes, which is the deterministic-degradation contract: a
    server that cannot classify anything must SAY so on every relevant surface, not start
    up looking healthy and fail one request at a time.

    ## Why `ingest.load_entries` and not `matcher.load_dictionary`

    Because `load_dictionary` LOSES THE GOVERNANCE CODE. It dispatches to the loader
    registry, and neither `CsvDictionaryLoader` nor `ExcelDictionaryLoader` reads a
    protection-code column -- verified 2026-08-10: a glossary row carrying
    `protection_class,GROWERID` comes back with `governance_code=None`. The governance
    columns are understood by `application.ingest`, which is a different reader.

    Routed through `load_dictionary`, this server starts cleanly, answers 200, returns the
    right entry with the right id -- and `"governance": null` on every field of every
    request, which a caller cannot distinguish from a glossary that genuinely carries no
    classes. That is NM-0005's shape at the top of the stack: not an error, not a warning,
    just a column that silently inherits nothing.

    `load_entries` reads the same file formats (`read_source` handles xlsx, csv and a
    database URL) and additionally validates every row against the vocabulary, so the
    honest wiring is one path, not two. `_index_dictionary` is the private half of
    `load_dictionary` that does the indexing without the re-reading; the coupling is the
    same one `matching.py` documents for `_match_fields`, and a public
    `NexusMatcher.index(entries)` would remove both.
    """
    dictionary = (environ.get("NEXUS_API_DICTIONARY") or "").strip()
    if not dictionary:
        return None

    # Deferred: importing the matching stack (and, through `from_config`, the bundled
    # encoder) at module scope would make merely importing this module load a model.
    from nexus_matcher.application.ingest import load_entries
    from nexus_matcher.application.use_cases.match_schema import NexusMatcher

    config_path = (environ.get("NEXUS_API_MATCHING_CONFIG") or "").strip() or None
    # The controlled vocabulary is CALLER-SUPPLIED -- this library ships no taxonomy --
    # and both shipped ways to start this server call `create_app()` with no arguments.
    # Without this variable the vocabulary the domain layer resolves every
    # `MatchResult.governance` through would be unreachable over HTTP.
    vocabulary = (environ.get("NEXUS_API_GOVERNANCE") or "").strip() or None

    if vocabulary is None:
        column = _unread_governance_column(dictionary)
        if column is not None:
            raise ValueError(
                f"{dictionary} has a protection-code column ({column!r}) and no "
                f"vocabulary is configured to interpret it, so every match would come "
                f"back with governance=null -- indistinguishable, to the caller, from a "
                f"glossary that carries no classes at all. Set NEXUS_API_GOVERNANCE to "
                f"the JSON file that declares those codes, or remove the column."
            )

    matcher = NexusMatcher.from_config(config_path, governance=vocabulary)
    # `governance_strict` left at its default: a glossary row whose stated tier
    # contradicts its own code is a DATA DEFECT, and loading it anyway would let a field
    # inherit a tier its own code disowns. Refusing at startup is the cheaper failure.
    matcher._index_dictionary(load_entries(dictionary, governance=vocabulary))
    return matcher


def _unread_governance_column(dictionary: str) -> str | None:
    """
    The protection-code column this glossary carries and nobody is configured to read.

    Closes the one silent path left in the wiring above, and it is a genuinely circular
    one: `load_entries` attaches a governance code only when a vocabulary is configured,
    and the matcher refuses codes it cannot resolve -- so with NO vocabulary there are no
    codes, nothing to refuse, and a server that starts perfectly and answers
    `"governance": null` for a glossary whose header plainly says `protection_class`.
    Neither layer below can see it: one has no vocabulary to check against, the other has
    no column to complain about.

    Costs one extra read of the glossary, and only in this case -- when a vocabulary IS
    configured the file is read once. That is the right trade at startup: the alternative
    is a server that classifies nothing and says so nowhere.

    The alias list comes from `domain.governance`, so this cannot drift from the columns
    the ingest path actually recognises. The normalisation is inlined because both
    `_norm_key` implementations are private to their own modules; it must stay
    lowercase-alphanumeric to match them.
    """
    from nexus_matcher.application.ingest import read_source
    from nexus_matcher.domain.governance import CODE_COLUMN_ALIASES

    _rows, header = read_source(dictionary)
    aliases = set(CODE_COLUMN_ALIASES)
    for column in header:
        if "".join(ch for ch in str(column).lower() if ch.isalnum()) in aliases:
            return str(column)
    return None


def _feedback_recorder(
    feedback_path: str | None, environ: Mapping[str, str]
) -> FeedbackRecorder | None:
    """The recorder for `feedback_path` or `NEXUS_API_FEEDBACK_PATH`, else None."""
    configured = feedback_path or (environ.get("NEXUS_API_FEEDBACK_PATH") or "").strip()
    return FeedbackRecorder(configured) if configured else None


def _bring_up_matcher(
    handle: MatcherHandle,
    app_state: AppState,
    environ: Mapping[str, str],
    logger: Any,
) -> None:
    """
    Resolve the matcher at startup and report it through readiness.

    The `matcher` component is registered ONLY when this deployment asked for one. A
    health-and-introspection deployment -- which is all this service was before the
    matching endpoints existed -- must keep reporting ready exactly as it did; a component
    that defaulted to False would turn every existing `/health/ready` into a 503 on
    upgrade, which is a breaking change wearing a feature's clothes.

    A configured-but-unloadable dictionary is the opposite case and gets the opposite
    treatment: the component goes False, readiness goes 503, and the matching routes
    answer 503 with the loader's own message. Starting green and failing one request at a
    time is how a bad deploy gets through a rollout gate.
    """
    if handle.is_ready:
        # Injected by the caller; nothing to load, but it is real and must be reported.
        app_state.set_component_status("matcher", True)
        return

    try:
        loaded = _load_configured_matcher(environ)
    except Exception as exc:
        reason = f"loading NEXUS_API_DICTIONARY failed: {type(exc).__name__}: {exc}"
        logger.error("matcher_load_failed", error=str(exc))
        handle.record_failure(reason)
        app_state.set_component_status("matcher", False)
        return

    if loaded is not None:
        handle.bind(loaded)
        app_state.set_component_status("matcher", True)
        logger.info("matcher_ready", dictionary=environ.get("NEXUS_API_DICTIONARY"))


# =============================================================================
# APPLICATION FACTORY
# =============================================================================


def create_app(
    title: str = "NexusMatcher API",
    version: str = "2.0.0",
    configure_logs: bool = True,
    *,
    matcher: object | None = None,
    limits: MatchServiceLimits | None = None,
    feedback_path: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        title: API title
        version: API version
        configure_logs: Whether to configure logging
        matcher: An already-loaded matcher for the matching endpoints. When None, one is
            built at startup from `NEXUS_API_DICTIONARY` if that is set; when neither is
            given, the matching endpoints answer 503 naming the setting to change.
        limits: Deadline and admission limits. Defaults to `MatchServiceLimits.from_env()`.
        feedback_path: Append-only JSONL file for reviewer feedback. Falls back to
            `NEXUS_API_FEEDBACK_PATH`; without either, POST /api/v1/feedback answers 503.
        environ: Environment mapping to read, for tests. Defaults to `os.environ`.

    Returns:
        Configured FastAPI application
    """
    env: Mapping[str, str] = os.environ if environ is None else environ

    # Initialize logging
    if configure_logs:
        configure_logging(level="INFO", json_format=True)

    logger = get_logger("api")

    # Application state
    app_state = AppState()

    # Matching service. Constructed here rather than in the lifespan handler because the
    # ROUTES are registered at import time and must exist whether or not a dictionary is
    # ever loaded -- an endpoint that only sometimes exists gives a 404 that means two
    # different things. `MatcherHandle` carries the "not loaded, and here is why" state.
    service_limits = limits if limits is not None else MatchServiceLimits.from_env(env)
    matcher_handle = MatcherHandle()
    work_pool = BoundedWorkPool(
        max_workers=service_limits.max_workers,
        max_queued=service_limits.max_queued,
    )
    if matcher is not None:
        matcher_handle.bind(matcher)
    recorder = _feedback_recorder(feedback_path, env)

    # Lifespan context manager
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Application lifespan handler."""
        logger.info("application_starting", version=version)

        # Initialize components
        try:
            # Mark core components as ready
            # In production, these would check actual connectivity
            app_state.set_component_status("api", True)
            app_state.set_component_status("config", True)

            # Try to initialize optional components
            try:
                # Vector store check would go here
                app_state.set_component_status("vector_store", True)
            except Exception as e:
                logger.warning("vector_store_unavailable", error=str(e))
                app_state.set_component_status("vector_store", False)

            try:
                # Cache check would go here
                app_state.set_component_status("cache", True)
            except Exception as e:
                logger.warning("cache_unavailable", error=str(e))
                app_state.set_component_status("cache", False)

            _bring_up_matcher(matcher_handle, app_state, env, logger)

            app_state.is_ready = True
            logger.info("application_ready", components=app_state.components)

            yield

        finally:
            logger.info("application_shutting_down")
            app_state.is_ready = False
            # Stop admitting work and drop anything not started. Without this the pool's
            # threads outlive the application object, and a test suite that builds many
            # apps accumulates them.
            work_pool.shutdown()

    # Create app
    app = FastAPI(
        title=title,
        version=version,
        description="""
NexusMatcher API - Enterprise Semantic Schema Matching

Automatically map schema fields to data dictionary entries using
multi-stage semantic search with dense + sparse retrieval,
neural reranking, and confidence scoring.

## Features

- **Schema Matching**: Match Avro, JSON Schema, SQL DDL, or CSV schemas
- **Dictionary Management**: Load and sync data dictionaries
- **Batch Processing**: Process multiple schemas efficiently
- **Flexible Deployment**: Use as library, API, or CLI

## Authentication

API key authentication can be enabled via the `NEXUS_API_KEY` environment variable.
Pass the key in the `X-API-Key` header.
        """,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add middleware
    app.middleware("http")(request_id_middleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(NexusMatcherError, nexus_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    # Body validation, rendered into the same `{"error": {...}}` envelope as everything
    # else. FastAPI's default is `{"detail": [...]}`, and one service answering with two
    # error shapes is what makes a client library branch on the shape of the error before
    # it can read the error.
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Health endpoints
    app.include_router(create_health_router(app_state))

    # Matching endpoints -- the reason this API exists beyond health and introspection.
    app.include_router(
        create_matching_router(
            MatchService(matcher_handle, service_limits, work_pool),
            service_limits,
        )
    )
    app.include_router(create_feedback_router(recorder, DeterministicJSONResponse))

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "nexus-matcher",
            "version": version,
            "docs": "/docs",
        }

    return app


# =============================================================================
# DEVELOPMENT SERVER
# =============================================================================


def run_dev_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
) -> None:
    """
    Run development server.

    Args:
        host: Bind host
        port: Bind port
        reload: Enable auto-reload
    """
    import uvicorn

    uvicorn.run(
        "nexus_matcher.presentation.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


# =============================================================================
# MODULE-LEVEL APP INSTANCE
# =============================================================================

# Create default app instance for uvicorn (used by: uvicorn app:app)
app = create_app()


if __name__ == "__main__":
    run_dev_server()
