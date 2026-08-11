"""
nexus_matcher.shared.logging | Layer: SHARED
Structured logging infrastructure with JSON output, correlation IDs, and PII redaction.

## Relationships
# USED_BY    → all modules :: logging
# DEPENDS_ON → structlog :: structured logging
# DEPENDS_ON → infrastructure/config :: logging configuration

## Attributes
# Security: PII redaction enabled by default
# Performance: Async logging for high throughput
# Reliability: Fallback to stdlib logging if structlog unavailable
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import structlog
from structlog.types import EventDict, Processor, WrappedLogger

# =============================================================================
# CONTEXT VARIABLES
# =============================================================================

# Correlation ID for request tracing
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Additional context that can be added per-request.
# The default is None rather than {} on purpose: a mutable default is a single
# dict shared by every context that never called set(), so any in-place mutation
# would leak context between requests. Readers below normalise None to {}.
request_context_var: ContextVar[dict[str, Any] | None] = ContextVar("request_context", default=None)


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    cid = correlation_id_var.get()
    if cid is None:
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(cid)


def add_request_context(**kwargs: Any) -> None:
    """Add key-value pairs to request context."""
    ctx = dict(request_context_var.get() or {})
    ctx.update(kwargs)
    request_context_var.set(ctx)


def clear_request_context() -> None:
    """Clear request context."""
    request_context_var.set({})


# =============================================================================
# PII PATTERNS
# =============================================================================

# Patterns for common PII
PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL_REDACTED]"),
    # Phone numbers (various formats)
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
    # SSN
    (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[SSN_REDACTED]"),
    # Credit card numbers (basic pattern)
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[CC_REDACTED]"),
    # IP addresses
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_REDACTED]"),
]

# Keys that should have their values redacted
SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "api-key",
        "authorization",
        "auth",
        "credential",
        "private_key",
        "ssn",
        "social_security",
        "credit_card",
        "card_number",
    }
)


def redact_pii(value: str) -> str:
    """Redact PII from a string value."""
    result = value
    for pattern, replacement in PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def redact_dict(d: dict[str, Any], redact: bool = True) -> dict[str, Any]:
    """Recursively redact PII from dictionary values."""
    if not redact:
        return d

    result = {}
    for key, value in d.items():
        key_lower = key.lower()

        # Check if key is sensitive
        if key_lower in SENSITIVE_KEYS:
            result[key] = "[REDACTED]"
        elif isinstance(value, str):
            result[key] = redact_pii(value)
        elif isinstance(value, dict):
            result[key] = redact_dict(value, redact)
        elif isinstance(value, (list, tuple)):
            result[key] = [
                redact_dict(v, redact)
                if isinstance(v, dict)
                else redact_pii(v)
                if isinstance(v, str)
                else v
                for v in value
            ]
        else:
            result[key] = value

    return result


# =============================================================================
# STRUCTLOG PROCESSORS
# =============================================================================


def add_correlation_id(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add correlation ID to log events."""
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict


def add_request_context_processor(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add request context to log events."""
    ctx = request_context_var.get()
    if ctx:
        event_dict.update(ctx)
    return event_dict


def add_timestamp(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO timestamp to log events."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return event_dict


@lru_cache(maxsize=1)
def service_version() -> str:
    """The running package version, from the one place that defines it.

    `pyproject.toml` sets `dynamic = ["version"]` with `[tool.hatch.version] path =
    src/nexus_matcher/__init__.py`, so `__version__` IS the version the wheel is built
    with. Anything else spelling it out is a second copy that drifts -- and did: this
    line and three in the API said "2.0.0" while the package was at 2.1.0, so every log
    record and the `/health` payload identified the service as a release that had by then
    been deleted from PyPI.

    Imported inside the function, and cached, because this module is low-level enough to
    be imported while `nexus_matcher/__init__` is still executing; deferring to first log
    call resolves it well after that finishes.
    """
    from nexus_matcher import __version__

    return __version__


def add_service_info(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add service metadata to log events."""
    event_dict["service"] = "nexus-matcher"
    event_dict["version"] = service_version()
    return event_dict


class PIIRedactor:
    """Processor that redacts PII from log events."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        if not self.enabled:
            return event_dict
        return redact_dict(dict(event_dict), redact=True)


def drop_color_message_key(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Drop color_message key used by stdlib integration."""
    event_dict.pop("color_message", None)
    return event_dict


# =============================================================================
# LOGGER SETUP
# =============================================================================


def get_processors(
    json_format: bool = True,
    redact_pii: bool = True,
) -> list[Processor]:
    """Get list of structlog processors."""
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_timestamp,
        add_correlation_id,
        add_request_context_processor,
        add_service_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if redact_pii:
        processors.append(PIIRedactor(enabled=True))

    if json_format:
        processors.extend(
            [
                drop_color_message_key,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ]
        )
    else:
        processors.extend(
            [
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        )

    return processors


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    redact_pii: bool = True,
    log_file: str | None = None,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Output logs as JSON
        redact_pii: Enable PII redaction
        log_file: Optional file path for log output
    """
    # Configure stdlib logging
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        handlers=handlers,
        force=True,
    )

    # Configure structlog
    structlog.configure(
        processors=get_processors(json_format=json_format, redact_pii=redact_pii),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


@lru_cache(maxsize=128)
def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger.

    Args:
        name: Logger name (defaults to caller module)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def log_with_context(
    level: str,
    message: str,
    **context: Any,
) -> None:
    """
    Log a message with additional context.

    Args:
        level: Log level
        message: Log message
        **context: Additional context key-values
    """
    logger = get_logger()
    log_method = getattr(logger, level.lower())
    log_method(message, **context)


def log_exception(
    message: str,
    exc: Exception,
    **context: Any,
) -> None:
    """
    Log an exception with traceback.

    Args:
        message: Log message
        exc: Exception to log
        **context: Additional context
    """
    logger = get_logger()
    logger.exception(message, exc_info=exc, **context)


def log_performance(
    operation: str,
    duration_ms: float,
    success: bool = True,
    **context: Any,
) -> None:
    """
    Log a performance metric.

    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        success: Whether operation succeeded
        **context: Additional context
    """
    logger = get_logger("performance")
    logger.info(
        "operation_complete",
        operation=operation,
        duration_ms=duration_ms,
        success=success,
        **context,
    )


# =============================================================================
# CONTEXT MANAGER
# =============================================================================


class LogContext:
    """
    Context manager for request-scoped logging context.

    Example:
        with LogContext(user_id="123", request_id="abc"):
            logger.info("Processing request")
            # All logs will include user_id and request_id
    """

    def __init__(self, correlation_id: str | None = None, **context: Any) -> None:
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self.context = context
        self._token = None

    def __enter__(self) -> LogContext:
        set_correlation_id(self.correlation_id)
        add_request_context(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        clear_request_context()


# =============================================================================
# INITIALIZATION
# =============================================================================


def init_logging_from_config() -> None:
    """Initialize logging from configuration."""
    try:
        from nexus_matcher.infrastructure.config.settings import get_config

        config = get_config()
        configure_logging(
            level=config.logging.level,
            json_format=config.logging.format == "json",
            redact_pii=config.logging.redact_pii,
            log_file=str(config.logging.file_path) if config.logging.file_path else None,
        )
    except ImportError:
        # Config not available, use defaults
        configure_logging()
