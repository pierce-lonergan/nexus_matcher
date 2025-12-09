"""
nexus_matcher.shared.metrics | Layer: SHARED
Metrics and observability infrastructure for monitoring NexusMatcher.

## Relationships
# DEPENDS_ON → shared/plugins :: HookPoint, HookContext
# USED_BY    → application/use_cases/* :: metric collection
# USED_BY    → presentation/api :: metrics endpoint
# USED_BY    → infrastructure/adapters/* :: adapter metrics

## Attributes
# Security: No PII in metrics, sanitized labels
# Performance: Low overhead counters, sampling for histograms
# Reliability: Graceful degradation if backend unavailable
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generator, Protocol, runtime_checkable


# =============================================================================
# METRIC TYPES
# =============================================================================


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = auto()      # Monotonically increasing value
    GAUGE = auto()        # Value that can go up or down
    HISTOGRAM = auto()    # Distribution of values
    SUMMARY = auto()      # Similar to histogram with quantiles


@dataclass
class MetricValue:
    """A single metric value with labels."""

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HistogramBucket:
    """A histogram bucket."""

    le: float  # Less than or equal
    count: int


@dataclass
class HistogramValue:
    """Histogram metric value."""

    name: str
    buckets: list[HistogramBucket]
    sum: float
    count: int
    labels: dict[str, str] = field(default_factory=dict)


# =============================================================================
# METRICS BACKEND PROTOCOL
# =============================================================================


@runtime_checkable
class MetricsBackend(Protocol):
    """Protocol for metrics backends (Prometheus, StatsD, etc.)."""

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter."""
        ...

    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value."""
        ...

    def histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram observation."""
        ...

    def flush(self) -> None:
        """Flush any buffered metrics."""
        ...


# =============================================================================
# IN-MEMORY METRICS (Default)
# =============================================================================


class InMemoryMetrics:
    """
    In-memory metrics backend for development and testing.

    Stores all metrics in memory with full history.
    Useful for testing and when no external metrics system is available.

    Example:
        ```python
        metrics = InMemoryMetrics()
        metrics.counter("requests_total", labels={"endpoint": "/match"})
        metrics.histogram("request_duration_seconds", 0.123)

        # Get all metrics
        for metric in metrics.get_all():
            print(f"{metric.name}: {metric.value}")
        ```
    """

    def __init__(self, max_history: int = 10000) -> None:
        self._counters: dict[str, dict[tuple, float]] = defaultdict(dict)
        self._gauges: dict[str, dict[tuple, float]] = defaultdict(dict)
        self._histograms: dict[str, dict[tuple, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._max_history = max_history

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter."""
        key = self._labels_to_key(labels)
        current = self._counters[name].get(key, 0.0)
        self._counters[name][key] = current + value

    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value."""
        key = self._labels_to_key(labels)
        self._gauges[name][key] = value

    def histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram observation."""
        key = self._labels_to_key(labels)
        values = self._histograms[name][key]
        values.append(value)

        # Trim if too many values
        if len(values) > self._max_history:
            self._histograms[name][key] = values[-self._max_history:]

    def flush(self) -> None:
        """No-op for in-memory backend."""
        pass

    def _labels_to_key(self, labels: dict[str, str] | None) -> tuple:
        """Convert labels dict to hashable key."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def _key_to_labels(self, key: tuple) -> dict[str, str]:
        """Convert key back to labels dict."""
        return dict(key)

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_counter(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float:
        """Get counter value."""
        key = self._labels_to_key(labels)
        return self._counters.get(name, {}).get(key, 0.0)

    def get_gauge(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float | None:
        """Get gauge value."""
        key = self._labels_to_key(labels)
        return self._gauges.get(name, {}).get(key)

    def get_histogram_stats(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> dict[str, float] | None:
        """Get histogram statistics."""
        key = self._labels_to_key(labels)
        values = self._histograms.get(name, {}).get(key)

        if not values:
            return None

        sorted_values = sorted(values)
        count = len(sorted_values)

        return {
            "count": count,
            "sum": sum(sorted_values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(sorted_values) / count,
            "p50": sorted_values[count // 2],
            "p90": sorted_values[int(count * 0.9)],
            "p99": sorted_values[int(count * 0.99)] if count >= 100 else sorted_values[-1],
        }

    def get_all(self) -> list[MetricValue]:
        """Get all metrics as MetricValue objects."""
        result = []

        for name, label_values in self._counters.items():
            for key, value in label_values.items():
                result.append(MetricValue(
                    name=name,
                    value=value,
                    labels=self._key_to_labels(key),
                    metric_type=MetricType.COUNTER,
                ))

        for name, label_values in self._gauges.items():
            for key, value in label_values.items():
                result.append(MetricValue(
                    name=name,
                    value=value,
                    labels=self._key_to_labels(key),
                    metric_type=MetricType.GAUGE,
                ))

        return result

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# =============================================================================
# PROMETHEUS METRICS (Optional)
# =============================================================================


class PrometheusMetrics:
    """
    Prometheus metrics backend.

    Requires: pip install prometheus-client

    Example:
        ```python
        from nexus_matcher.shared.metrics import PrometheusMetrics

        metrics = PrometheusMetrics(prefix="nexus_matcher")
        metrics.counter("requests_total", labels={"endpoint": "/match"})

        # Expose via /metrics endpoint
        from prometheus_client import generate_latest
        app.get("/metrics")(lambda: generate_latest())
        ```
    """

    def __init__(self, prefix: str = "nexus_matcher") -> None:
        try:
            from prometheus_client import Counter, Gauge, Histogram
        except ImportError:
            raise ImportError(
                "prometheus-client is required. "
                "Install with: pip install prometheus-client"
            )

        self._prefix = prefix
        self._counters: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}

        # Default buckets for latency histograms
        self._default_buckets = (
            0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0
        )

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter."""
        from prometheus_client import Counter

        full_name = f"{self._prefix}_{name}"
        label_names = tuple(sorted(labels.keys())) if labels else ()

        if full_name not in self._counters:
            self._counters[full_name] = Counter(
                full_name,
                f"Counter: {name}",
                label_names,
            )

        counter = self._counters[full_name]
        if labels:
            counter.labels(**labels).inc(value)
        else:
            counter.inc(value)

    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value."""
        from prometheus_client import Gauge

        full_name = f"{self._prefix}_{name}"
        label_names = tuple(sorted(labels.keys())) if labels else ()

        if full_name not in self._gauges:
            self._gauges[full_name] = Gauge(
                full_name,
                f"Gauge: {name}",
                label_names,
            )

        gauge = self._gauges[full_name]
        if labels:
            gauge.labels(**labels).set(value)
        else:
            gauge.set(value)

    def histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram observation."""
        from prometheus_client import Histogram

        full_name = f"{self._prefix}_{name}"
        label_names = tuple(sorted(labels.keys())) if labels else ()

        if full_name not in self._histograms:
            self._histograms[full_name] = Histogram(
                full_name,
                f"Histogram: {name}",
                label_names,
                buckets=self._default_buckets,
            )

        histogram = self._histograms[full_name]
        if labels:
            histogram.labels(**labels).observe(value)
        else:
            histogram.observe(value)

    def flush(self) -> None:
        """No-op - Prometheus scrapes metrics."""
        pass


# =============================================================================
# METRICS COLLECTOR
# =============================================================================


class MetricsCollector:
    """
    Central metrics collector that routes to configured backend.

    Provides:
    - Pre-defined metrics for NexusMatcher operations
    - Context managers for timing operations
    - Decorators for automatic metric collection

    Example:
        ```python
        from nexus_matcher.shared.metrics import get_metrics

        metrics = get_metrics()

        # Count operations
        metrics.count_match_request(schema_name="customer.avsc")

        # Time operations
        with metrics.time_operation("embedding"):
            embeddings = embed(texts)

        # Or use decorator
        @metrics.timed("match_field")
        def match_field(field):
            ...
        ```
    """

    def __init__(self, backend: MetricsBackend | None = None) -> None:
        self._backend = backend or InMemoryMetrics()

    @property
    def backend(self) -> MetricsBackend:
        """Get the metrics backend."""
        return self._backend

    # -------------------------------------------------------------------------
    # NexusMatcher-Specific Metrics
    # -------------------------------------------------------------------------

    def count_match_request(
        self,
        schema_name: str | None = None,
        status: str = "success",
    ) -> None:
        """Count a schema match request."""
        labels = {"status": status}
        if schema_name:
            labels["schema"] = schema_name
        self._backend.counter("match_requests_total", labels=labels)

    def count_field_match(
        self,
        decision: str,
        confidence_bucket: str | None = None,
    ) -> None:
        """Count a field match result."""
        labels = {"decision": decision}
        if confidence_bucket:
            labels["confidence"] = confidence_bucket
        self._backend.counter("field_matches_total", labels=labels)

    def record_match_latency(
        self,
        latency_seconds: float,
        operation: str = "total",
    ) -> None:
        """Record matching latency."""
        self._backend.histogram(
            "match_latency_seconds",
            latency_seconds,
            labels={"operation": operation},
        )

    def set_dictionary_size(self, size: int) -> None:
        """Set current dictionary size."""
        self._backend.gauge("dictionary_entries", float(size))

    def set_cache_size(self, cache_type: str, size: int) -> None:
        """Set cache size."""
        self._backend.gauge(
            "cache_entries",
            float(size),
            labels={"cache": cache_type},
        )

    def count_cache_hit(self, cache_type: str, hit: bool) -> None:
        """Count cache hit/miss."""
        self._backend.counter(
            "cache_requests_total",
            labels={"cache": cache_type, "result": "hit" if hit else "miss"},
        )

    def count_embedding_request(self, batch_size: int) -> None:
        """Count embedding generation."""
        self._backend.counter("embedding_requests_total")
        self._backend.counter("embeddings_generated_total", float(batch_size))

    def count_retrieval_candidates(self, stage: str, count: int) -> None:
        """Record retrieval candidates at each stage."""
        self._backend.histogram(
            "retrieval_candidates",
            float(count),
            labels={"stage": stage},
        )

    def count_error(self, error_type: str, component: str) -> None:
        """Count errors by type."""
        self._backend.counter(
            "errors_total",
            labels={"type": error_type, "component": component},
        )

    # -------------------------------------------------------------------------
    # Generic Methods
    # -------------------------------------------------------------------------

    @contextmanager
    def time_operation(
        self,
        operation: str,
        labels: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """
        Context manager to time an operation.

        Example:
            with metrics.time_operation("embedding"):
                result = embed(texts)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            all_labels = {"operation": operation}
            if labels:
                all_labels.update(labels)
            self._backend.histogram("operation_duration_seconds", duration, all_labels)

    def timed(
        self,
        operation: str,
    ) -> Callable:
        """
        Decorator to time a function.

        Example:
            @metrics.timed("match_field")
            def match_field(field):
                ...
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.time_operation(operation):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# =============================================================================
# GLOBAL METRICS INSTANCE
# =============================================================================

_global_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def configure_metrics(backend: MetricsBackend) -> MetricsCollector:
    """Configure the global metrics collector with a specific backend."""
    global _global_metrics
    _global_metrics = MetricsCollector(backend)
    return _global_metrics


def reset_metrics() -> None:
    """Reset the global metrics collector (for testing)."""
    global _global_metrics
    _global_metrics = None


# =============================================================================
# OPENTELEMETRY INTEGRATION (Optional)
# =============================================================================


def create_opentelemetry_backend(
    service_name: str = "nexus-matcher",
    endpoint: str | None = None,
) -> MetricsBackend:
    """
    Create an OpenTelemetry metrics backend.

    Requires: pip install opentelemetry-api opentelemetry-sdk

    Args:
        service_name: Service name for telemetry
        endpoint: OTLP endpoint (uses env var if not set)

    Returns:
        Configured metrics backend
    """
    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        raise ImportError(
            "OpenTelemetry is required. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk"
        )

    # This is a simplified implementation
    # Full implementation would include OTLP exporter configuration

    resource = Resource.create({"service.name": service_name})
    provider = MeterProvider(resource=resource)
    metrics.set_meter_provider(provider)

    # Return an adapter that uses OTEL
    # For now, fall back to in-memory
    return InMemoryMetrics()
