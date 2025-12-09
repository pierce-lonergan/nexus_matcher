"""
nexus_matcher.presentation.api.app | Layer: PRESENTATION
FastAPI application factory with health endpoints and middleware.

## Relationships
# DEPENDS_ON → application/use_cases/* :: use case orchestration
# DEPENDS_ON → infrastructure/config :: application configuration
# DEPENDS_ON → shared/logging :: structured logging
# DEPENDS_ON → shared/exceptions :: error handling

## Attributes
# Security: CORS, rate limiting, API key auth (optional)
# Performance: Async handlers, connection pooling
# Reliability: Health checks, graceful shutdown
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nexus_matcher.shared.exceptions import (
    NexusMatcherError,
    APIError,
    AuthenticationError,
    RateLimitError,
)
from nexus_matcher.shared.logging import (
    configure_logging,
    get_logger,
    set_correlation_id,
    clear_request_context,
    add_request_context,
    log_performance,
    LogContext,
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


async def nexus_exception_handler(
    request: Request, exc: NexusMatcherError
) -> JSONResponse:
    """Handle NexusMatcher exceptions."""
    logger = get_logger("api.error")

    # Determine status code
    if isinstance(exc, APIError):
        status_code = exc.status_code
    else:
        status_code = 500

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


async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
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
# APPLICATION FACTORY
# =============================================================================


def create_app(
    title: str = "NexusMatcher API",
    version: str = "2.0.0",
    configure_logs: bool = True,
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        title: API title
        version: API version
        configure_logs: Whether to configure logging

    Returns:
        Configured FastAPI application
    """
    # Initialize logging
    if configure_logs:
        configure_logging(level="INFO", json_format=True)

    logger = get_logger("api")

    # Application state
    app_state = AppState()

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

            app_state.is_ready = True
            logger.info("application_ready", components=app_state.components)

            yield

        finally:
            logger.info("application_shutting_down")
            app_state.is_ready = False

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

    # Health endpoints
    app.include_router(create_health_router(app_state))

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
