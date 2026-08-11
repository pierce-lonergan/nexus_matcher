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
# Security: UNAUTHENTICATED by design -- no API key, no rate limiting; CORS closed by default
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
    NEXUS_API_MAX_BODY_BYTES    request-body cap; derived from the field caps by default,
                                and REFUSED at startup if set below what they admit
    NEXUS_API_CORS_ORIGINS      comma-separated browser origins; empty (default) = no CORS
    NEXUS_API_CORS_ALLOW_CREDENTIALS   send cookies cross-origin, default false
    NEXUS_API_MATCHING_OPTIONAL match need not be ready for /health/ready, default false

The keyword arguments exist for embedders and for tests. Without the environment path the
new endpoints would be code with no way to reach them from either shipped entry point --
H-006's exact shape, and its most expensive instance shipped 2.5x faster and unreachable.

## What this service does NOT do

It authenticates nobody. There is no API key, no OAuth and no rate limiting, and the
published description says so in as many words -- because for two releases it said the
opposite, offering `X-API-Key` and a `NEXUS_API_KEY` variable that no code has ever read,
in the `/openapi.json` a Java client is generated from. Deleting the sentence is half the
fix; stating the position is the half that stops it being invented again.
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

# `StarletteHTTPException` is the class the ROUTER raises for 404 and 405. FastAPI's own
# `HTTPException` is a SUBCLASS of it and Starlette resolves handlers by walking the raised
# type's MRO, so registering the subclass would catch neither -- which is how both statuses
# kept answering in FastAPI's `{"detail": ...}` envelope.
#
# Taken from FastAPI's re-export rather than `from starlette.exceptions import ...`
# because `tests/packaging/test_extras_graph` requires every third-party module `src/`
# imports to name an extra that installs it, and starlette is a hard dependency OF fastapi
# that no extra lists; `body_limit.py` avoids the same gate by spelling out the ASGI types.
# The re-export is not in an `__all__`, hence the ignore. If it is ever dropped this fails
# loudly at import; the 404 assertion in `test_error_envelope` covers the quiet direction.
from fastapi.exceptions import (  # type: ignore[attr-defined]
    RequestValidationError,
    StarletteHTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nexus_matcher.presentation.api.body_limit import BodySizeLimitMiddleware
from nexus_matcher.presentation.api.errors import (
    http_exception_handler,
    validation_exception_handler,
)
from nexus_matcher.presentation.api.feedback import FeedbackRecorder, create_feedback_router
from nexus_matcher.presentation.api.limits import BoundedWorkPool, MatchServiceLimits
from nexus_matcher.presentation.api.matching import (
    DeterministicJSONResponse,
    MatcherHandle,
    MatchService,
    create_matching_router,
)
from nexus_matcher.presentation.api.schemas import ErrorResponse
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
    service_version,
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
    version: str = Field(default_factory=service_version)
    checks: dict[str, Any] = Field(default_factory=dict)


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool
    timestamp: datetime = Field(default_factory=_utc_now)
    components: dict[str, bool] = Field(default_factory=dict)


# =============================================================================
# APPLICATION STATE
# =============================================================================


class AppState:
    """Application state holder."""

    def __init__(self) -> None:
        self.startup_time: datetime = _utc_now()
        self.is_ready: bool = False
        self.components: dict[str, bool] = {}
        self.gating: set[str] = set()

    def set_component_status(
        self, name: str, healthy: bool, *, gates_readiness: bool = True
    ) -> None:
        """
        Record a component's health, and whether readiness depends on it.

        `gates_readiness=False` reports a component without letting it decide the rollout
        gate. It exists for exactly one caller -- a deployment that set
        `NEXUS_API_MATCHING_OPTIONAL` and whose configured dictionary then failed to load
        -- so that "you asked for a dictionary and it broke" can be visible in
        `/health/ready` without failing a gate the operator deliberately opened. Gating is
        the default because the alternative default is a component that reports red while
        the service reports ready.
        """
        self.components[name] = healthy
        if gates_readiness:
            self.gating.add(name)
        else:
            self.gating.discard(name)

    def check_ready(self) -> bool:
        """
        Whether every component that gates readiness is healthy.

        `all(())` is True, so aggregating over an EMPTY set would make a process that
        registered nothing look ready -- which is how `/health/ready` answered 200 with no
        `matcher` key while every match answered 503. Nothing registered means the
        lifespan has not run, so the answer is no.
        """
        if not self.gating:
            return False
        return all(self.components[name] for name in self.gating)


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


# Two different conditions, and their bodies differ, so their descriptions do too --
# `MatcherUnavailableError`'s docstring makes the same distinction and warns against
# conflating them. Readiness carries the component map; the startup probe has none to
# carry, because nothing has registered yet.
_NOT_READY = "Not ready to serve. `details.components` names the component that is red."
_STILL_STARTING = "Startup has not completed. No component has reported in yet."


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
            version=service_version(),
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
        # The 503 is the answer this route exists to give, and it was published nowhere:
        # a client generated from the spec saw a readiness endpoint that could only
        # succeed. It renders through `http_exception_handler` into the same
        # `{"error": {...}}` envelope as every other failure, so it is the same DTO.
        responses={503: {"model": ErrorResponse, "description": _NOT_READY}},
    )
    async def readiness() -> ReadinessResponse:
        """
        Readiness probe for Kubernetes.

        Returns 200 if the application is ready to serve traffic.
        Returns 503 if any critical component is not ready.

        The 503 carries the whole component map, because the 200 does and an operator
        needs it far more when the answer is no. The previous body was the string
        "Service not ready" and nothing else, so the one deployment shape this endpoint
        exists to catch -- a dictionary that failed to load -- arrived at the operator as
        a status code with no subject.
        """
        ready = app_state.check_ready()

        if not ready:
            red = sorted(name for name in app_state.gating if not app_state.components[name])
            raise HTTPException(
                status_code=503,
                detail={
                    "message": (
                        f"The service is not ready: {', '.join(red)} not healthy."
                        if red
                        else "The service is not ready: startup has not registered any "
                        "component yet."
                    ),
                    "components": app_state.components.copy(),
                },
            )

        return ReadinessResponse(
            ready=ready,
            components=app_state.components.copy(),
        )

    @router.get(
        "/health/startup",
        summary="Startup probe",
        description="Kubernetes startup probe endpoint",
        responses={503: {"model": ErrorResponse, "description": _STILL_STARTING}},
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

    # Loaded BEFORE the matcher is built, because `from_config` builds the bundled encoder
    # and a glossary this deployment cannot interpret is a startup failure either way. The
    # guard that used to stand here failed before any model was loaded; keeping the load
    # first keeps that, and it is now the only read of the file rather than the second one.
    #
    # `governance_strict` left at its default: a glossary row whose stated tier contradicts
    # its own code is a DATA DEFECT, and loading it anyway would let a field inherit a tier
    # its own code disowns. Refusing at startup is the cheaper failure.
    try:
        entries = load_entries(dictionary, governance=vocabulary)
    except ValueError as exc:
        operator_message = _operator_grade_refusal(exc, vocabulary)
        if operator_message is None:
            raise
        raise ValueError(operator_message) from exc

    matcher = NexusMatcher.from_config(config_path, governance=vocabulary)
    matcher._index_dictionary(entries)
    return matcher


def _operator_grade_refusal(exc: ValueError, vocabulary: str | None) -> str | None:
    """
    `load_entries`'s refusal with the variable name added, or None for any other refusal.

    A glossary carrying protection codes nobody is configured to read is a circular
    silence -- no vocabulary means no codes, no codes means nothing to refuse, and a server
    that starts perfectly and answers `"governance": null` for a file whose header plainly
    says `protection_class`. This module used to catch it by reading the whole glossary a
    second time just to look at its header. `load_entries` catches it now, off a `mapping`
    it has already built, so the read is gone: measured on a 30,000-row, 4.23 MB glossary,
    195 ms of startup became 145 ms and 19.9 MB of traced allocation was not made. It was
    paid whether or not the column was there -- finding nothing costs the same full read as
    finding something -- so the deployment it taxed hardest was the one with no governance
    at all, which is also the one that gains nothing from the check.

    What the library cannot say is `NEXUS_API_GOVERNANCE`. It does not know it is being
    called by an HTTP server, so its advice is `governance=` and `governance_strict=False`
    -- neither of which an operator has a call site to pass. That is why the message here
    was the better one and why `test_operator_configuration` pins it, so the library's text
    is kept verbatim and the deployment-level sentence is appended to it rather than
    replacing it.

    The phrase match is the seam, and it is deliberately narrow. `load_entries` also raises
    ValueError for a glossary with no business-name column, and dressing THAT up with a
    vocabulary variable would send an operator to the wrong setting; anything unrecognised
    is re-raised exactly as it came.
    """
    if vocabulary is not None or "protection-code column" not in str(exc):
        return None
    return (
        f"{exc}\n"
        f"Over HTTP there is no call site to pass either of those at: set "
        f"NEXUS_API_GOVERNANCE to that JSON file, or remove the column from the glossary. "
        f"This server does not offer the governance_strict=False escape hatch, because a "
        f"deployment that classifies nothing and says so nowhere is the outcome this "
        f"refusal exists to prevent."
    )


def _feedback_recorder(
    feedback_path: str | None, environ: Mapping[str, str]
) -> FeedbackRecorder | None:
    """The recorder for `feedback_path` or `NEXUS_API_FEEDBACK_PATH`, else None."""
    configured = feedback_path or (environ.get("NEXUS_API_FEEDBACK_PATH") or "").strip()
    return FeedbackRecorder(configured) if configured else None


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})


def _env_flag(environ: Mapping[str, str], name: str) -> bool:
    """
    A boolean environment variable, or a refusal to guess what the operator meant.

    Same standard as `MatchServiceLimits.from_env`, and for the same reason: both flags
    read through here decide whether a gate is open. `MATCHING_OPTIONAL=yes-please` read
    as False gives an operator 503s they believe they switched off, and read as True opens
    a rollout gate they believe they closed. A value nobody can act on correctly is one
    the process should refuse to start with.
    """
    raw = (environ.get(name) or "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(
        f"{name}={environ.get(name)!r} is not a boolean. Use one of "
        f"{sorted(_TRUE)} or {sorted(_FALSE - {''})}."
    )


def _bring_up_matcher(
    handle: MatcherHandle,
    app_state: AppState,
    environ: Mapping[str, str],
    logger: Any,
    *,
    optional: bool,
) -> None:
    """
    Resolve the matcher at startup and report it through readiness.

    The `matcher` component is registered whether or not a dictionary was configured. It
    used to be registered only on success, and since `check_ready()` is `all()` over what
    is registered, a deployment whose `NEXUS_API_DICTIONARY` was absent, empty, or
    misspelled registered nothing and answered `/health/ready` 200 -- while every `POST
    /api/v1/match` answered 503. A *broken* dictionary went red correctly; a *missing* one
    did not, and missing is what a Helm value that fails to resolve produces. The matching
    router is included unconditionally, so a process advertising an endpoint that 503s
    every request is not ready under any honest definition.

    `NEXUS_API_MATCHING_OPTIONAL` keeps the health-and-introspection deployment -- all
    this service was before the matching endpoints existed -- supported, at the price of
    saying so. It is inverted from the obvious `NEXUS_API_REQUIRE_MATCHER` deliberately: a
    knob whose default is the unsafe value protects only the deployments whose operator
    remembered to set it, which are not the misconfigured ones. It is parsed in
    `create_app` rather than here so an unreadable value fails the process rather than one
    lifespan callback.
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
        # Reported even under the opt-out: the operator named a dictionary and it did not
        # load, which is a misconfiguration whichever gate it is allowed to fail.
        app_state.set_component_status("matcher", False, gates_readiness=not optional)
        return

    if loaded is not None:
        handle.bind(loaded)
        app_state.set_component_status("matcher", True)
        logger.info("matcher_ready", dictionary=environ.get("NEXUS_API_DICTIONARY"))
        return

    if optional:
        # Nothing was asked for and nothing is missing. Registering a red component for a
        # capability this deployment declared out of scope would report a fault that does
        # not exist.
        logger.info("matching_disabled", reason="NEXUS_API_MATCHING_OPTIONAL")
        return

    app_state.set_component_status("matcher", False)


# =============================================================================
# CROSS-ORIGIN ACCESS
# =============================================================================


def _cors_options(environ: Mapping[str, str]) -> dict[str, Any] | None:
    """
    The `CORSMiddleware` keywords for this deployment, or None to mount no CORS at all.

    Shipped as `allow_origins=["*"]` with `allow_credentials=True` behind a `# Configure
    in production` comment, which is not a configuration -- it is a note asking somebody
    to edit the source of an installed package. Measured: a preflight from any origin came
    back with that origin reflected and `access-control-allow-credentials: true`, so any
    page anywhere could read the response of an endpoint that asks for no credentials, and
    could `POST /api/v1/feedback`, which fsyncs a reviewer verdict into the audit trail.

    Default is closed and the middleware is not added at all -- not `allow_origins=[]`,
    which mounts a middleware to answer nothing. The Java pipeline this endpoint was built
    for is server-to-server and never sends an `Origin`, so closed by default costs the
    driving adopter nothing.

    Comma-separated rather than the JSON-list syntax `DEPLOYMENT.md` used to document,
    because parsing two spellings of one setting is how an operator ends up with a value
    that silently means nothing. Both doc lines were corrected to match.
    """
    origins = [
        origin.strip()
        for origin in (environ.get("NEXUS_API_CORS_ORIGINS") or "").split(",")
        if origin.strip()
    ]
    if not origins:
        return None

    credentials = _env_flag(environ, "NEXUS_API_CORS_ALLOW_CREDENTIALS")
    if credentials and "*" in origins:
        raise ValueError(
            "NEXUS_API_CORS_ALLOW_CREDENTIALS=true with NEXUS_API_CORS_ORIGINS=* is "
            "refused. Starlette reflects the requesting origin rather than sending '*' "
            "when credentials are enabled, so 'allow every origin' silently becomes "
            "'allow THIS origin, with cookies' -- a policy that cannot be read off the "
            "settings that produced it. Name the origins, or turn credentials off."
        )

    return {
        "allow_origins": origins,
        "allow_credentials": credentials,
        # The verbs this service serves. `["*"]` told a browser that DELETE was fine on
        # every route, which is an advertisement for surface that does not exist.
        "allow_methods": ["GET", "OPTIONS", "POST"],
        "allow_headers": ["*"],
    }


# =============================================================================
# APPLICATION FACTORY
# =============================================================================


def create_app(
    title: str = "NexusMatcher API",
    version: str | None = None,
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
        version: API version. Defaults to the running package version -- it was a literal
            "2.0.0" here and in three other places, which drifted silently and made the
            service report a release that had been deleted from PyPI.
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

    # Resolved once, here, so the OpenAPI `info.version`, the `/health` payload and every
    # log record all name the same build. Callers may still pass one explicitly.
    version = version if version is not None else service_version()

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
    # Both read here rather than where they are used, so a value no operator could have
    # meant stops the process at construction -- the point at which uvicorn reports it --
    # instead of inside a lifespan callback or on the first cross-origin request.
    matching_optional = _env_flag(env, "NEXUS_API_MATCHING_OPTIONAL")
    cors_options = _cors_options(env)
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

            # `vector_store` and `cache` used to be reported here as well, each set True
            # inside a `try:` whose body was a comment saying the check would go here --
            # so no failure could reach the `except:` and neither could ever be False.
            # A component that reports True unconditionally is not a check, it is a claim,
            # and this map is read by rollout gates. They come back when something
            # actually probes Qdrant and Redis.
            _bring_up_matcher(matcher_handle, app_state, env, logger, optional=matching_optional)

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

**This service ships unauthenticated.** It declares no security scheme, implements no
API-key or OAuth check, and reads no credential from the environment. Deploy it behind
your own gateway and authenticate there.

An earlier revision of this description offered an API-key header and named an
environment variable that enabled it. No code implemented either; a request carrying no
header returned 200 and a real protection class. That is retracted, and the header and
variable are not repeated here so that nobody sends them believing they do something.

Cross-origin access is refused unless the operator names the permitted origins in
`NEXUS_API_CORS_ORIGINS`; excess load is shed with 503 by the bounded work pool.
        """,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add middleware
    app.middleware("http")(request_id_middleware)

    # CORS, if and only if this operator named the origins that may use a browser.
    if cors_options is not None:
        app.add_middleware(CORSMiddleware, **cors_options)

    # Exception handlers
    app.add_exception_handler(NexusMatcherError, nexus_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    # 404, 405 and the health 503s. These are raised by Starlette and by the health
    # router, not by `errors.py`, and they were the half of the two-envelope defect that
    # no choice of exception class could fix -- a client could not read this service's
    # errors without first branching on their shape. See `errors.http_exception_handler`
    # for why `Allow` survives the round trip.
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
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

    # The byte cap on the request body, added LAST so it is the outermost middleware.
    #
    # Registration order is reversed by Starlette, so this wraps everything above it and
    # gets `receive` before `BaseHTTPMiddleware` -- which never hands a middleware
    # `receive` at all -- has a chance to stream the body through. What the cap protects is
    # MEMORY: an oversized body is never buffered or parsed, and 8 concurrent 198 MiB
    # requests move RSS by ~1 MiB instead of 3.5 GiB.
    #
    # It is NOT true that no byte past the cap is read. A body within twice the cap is read
    # and discarded first, because refusing while the client is still writing closes the
    # socket with bytes unread, and that RST destroys the 413 the caller needs in order to
    # know to re-chunk. Draining discards, so this costs bandwidth and up to 2 s, never
    # memory. `body_limit.py` carries the measurements and the reason a PARTIAL drain is
    # worse than none.
    #
    # `body_byte_cap` rather than a literal, so an operator who raises
    # NEXUS_API_MAX_BATCH_FIELDS does not inherit a stale cap and a 413 naming the wrong
    # limit.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=service_limits.body_byte_cap)

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
