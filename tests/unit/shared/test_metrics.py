"""
tests.unit.shared.test_metrics | Layer: TEST
Unit tests for metrics and observability.

## Relationships
# TESTS → shared/metrics :: MetricsCollector, InMemoryMetrics
"""

import time

import pytest

from nexus_matcher.shared.metrics import (
    InMemoryMetrics,
    MetricType,
    MetricsCollector,
    get_metrics,
    reset_metrics,
    configure_metrics,
)


class TestInMemoryMetrics:
    """Test InMemoryMetrics backend."""

    @pytest.fixture
    def metrics(self):
        """Create fresh metrics backend."""
        return InMemoryMetrics()

    def test_counter_increment(self, metrics):
        """Test counter increments."""
        metrics.counter("requests_total")
        metrics.counter("requests_total")
        metrics.counter("requests_total", 3.0)

        assert metrics.get_counter("requests_total") == 5.0

    def test_counter_with_labels(self, metrics):
        """Test counter with labels."""
        metrics.counter("requests_total", labels={"method": "GET"})
        metrics.counter("requests_total", labels={"method": "POST"})
        metrics.counter("requests_total", labels={"method": "GET"})

        assert metrics.get_counter("requests_total", {"method": "GET"}) == 2.0
        assert metrics.get_counter("requests_total", {"method": "POST"}) == 1.0

    def test_gauge_set(self, metrics):
        """Test gauge setting."""
        metrics.gauge("temperature", 25.5)
        assert metrics.get_gauge("temperature") == 25.5

        metrics.gauge("temperature", 30.0)
        assert metrics.get_gauge("temperature") == 30.0

    def test_gauge_with_labels(self, metrics):
        """Test gauge with labels."""
        metrics.gauge("cache_size", 100, labels={"cache": "L1"})
        metrics.gauge("cache_size", 1000, labels={"cache": "L2"})

        assert metrics.get_gauge("cache_size", {"cache": "L1"}) == 100
        assert metrics.get_gauge("cache_size", {"cache": "L2"}) == 1000

    def test_histogram_observations(self, metrics):
        """Test histogram observations."""
        for value in [0.1, 0.2, 0.3, 0.4, 0.5]:
            metrics.histogram("latency", value)

        stats = metrics.get_histogram_stats("latency")
        assert stats is not None
        assert stats["count"] == 5
        assert stats["sum"] == pytest.approx(1.5)
        assert stats["min"] == 0.1
        assert stats["max"] == 0.5
        assert stats["avg"] == pytest.approx(0.3)

    def test_histogram_with_labels(self, metrics):
        """Test histogram with labels."""
        metrics.histogram("latency", 0.1, labels={"endpoint": "/match"})
        metrics.histogram("latency", 0.2, labels={"endpoint": "/match"})
        metrics.histogram("latency", 0.5, labels={"endpoint": "/sync"})

        match_stats = metrics.get_histogram_stats("latency", {"endpoint": "/match"})
        sync_stats = metrics.get_histogram_stats("latency", {"endpoint": "/sync"})

        assert match_stats["count"] == 2
        assert sync_stats["count"] == 1

    def test_get_all_metrics(self, metrics):
        """Test getting all metrics."""
        metrics.counter("requests_total")
        metrics.gauge("active_connections", 5)

        all_metrics = metrics.get_all()
        assert len(all_metrics) == 2

        names = {m.name for m in all_metrics}
        assert "requests_total" in names
        assert "active_connections" in names

    def test_reset(self, metrics):
        """Test metrics reset."""
        metrics.counter("requests_total")
        metrics.gauge("temperature", 25)
        metrics.histogram("latency", 0.1)

        metrics.reset()

        assert metrics.get_counter("requests_total") == 0.0
        assert metrics.get_gauge("temperature") is None
        assert metrics.get_histogram_stats("latency") is None


class TestMetricsCollector:
    """Test MetricsCollector high-level API."""

    @pytest.fixture
    def collector(self):
        """Create fresh collector with in-memory backend."""
        backend = InMemoryMetrics()
        return MetricsCollector(backend)

    def test_count_match_request(self, collector):
        """Test match request counting."""
        collector.count_match_request(schema_name="test.avsc", status="success")
        collector.count_match_request(schema_name="test.avsc", status="error")

        backend = collector.backend
        assert isinstance(backend, InMemoryMetrics)
        assert backend.get_counter(
            "match_requests_total",
            {"schema": "test.avsc", "status": "success"}
        ) == 1.0

    def test_record_match_latency(self, collector):
        """Test match latency recording."""
        collector.record_match_latency(0.150, operation="embedding")
        collector.record_match_latency(0.200, operation="retrieval")

        backend = collector.backend
        assert isinstance(backend, InMemoryMetrics)

        stats = backend.get_histogram_stats(
            "match_latency_seconds",
            {"operation": "embedding"}
        )
        assert stats is not None
        assert stats["count"] == 1

    def test_set_dictionary_size(self, collector):
        """Test dictionary size gauge."""
        collector.set_dictionary_size(5000)

        backend = collector.backend
        assert isinstance(backend, InMemoryMetrics)
        assert backend.get_gauge("dictionary_entries") == 5000.0

    def test_count_cache_hit(self, collector):
        """Test cache hit/miss counting."""
        collector.count_cache_hit("L1", hit=True)
        collector.count_cache_hit("L1", hit=True)
        collector.count_cache_hit("L1", hit=False)

        backend = collector.backend
        assert isinstance(backend, InMemoryMetrics)
        assert backend.get_counter(
            "cache_requests_total",
            {"cache": "L1", "result": "hit"}
        ) == 2.0
        assert backend.get_counter(
            "cache_requests_total",
            {"cache": "L1", "result": "miss"}
        ) == 1.0

    def test_time_operation_context_manager(self, collector):
        """Test time_operation context manager."""
        with collector.time_operation("test_op"):
            time.sleep(0.01)  # 10ms

        backend = collector.backend
        assert isinstance(backend, InMemoryMetrics)

        stats = backend.get_histogram_stats(
            "operation_duration_seconds",
            {"operation": "test_op"}
        )
        assert stats is not None
        assert stats["count"] == 1
        assert stats["min"] >= 0.01  # At least 10ms

    def test_timed_decorator(self, collector):
        """Test timed decorator."""
        @collector.timed("decorated_op")
        def slow_function():
            time.sleep(0.01)
            return "result"

        result = slow_function()
        assert result == "result"

        backend = collector.backend
        assert isinstance(backend, InMemoryMetrics)

        stats = backend.get_histogram_stats(
            "operation_duration_seconds",
            {"operation": "decorated_op"}
        )
        assert stats is not None
        assert stats["count"] == 1


class TestGlobalMetrics:
    """Test global metrics functions."""

    def teardown_method(self):
        """Reset global metrics after each test."""
        reset_metrics()

    def test_get_metrics_singleton(self):
        """Test global metrics is singleton."""
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_configure_metrics(self):
        """Test configuring global metrics."""
        backend = InMemoryMetrics()
        collector = configure_metrics(backend)

        assert get_metrics() is collector
        assert collector.backend is backend

    def test_reset_metrics(self):
        """Test metrics reset."""
        m1 = get_metrics()
        reset_metrics()
        m2 = get_metrics()
        assert m1 is not m2
