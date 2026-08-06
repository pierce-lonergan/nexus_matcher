"""
tests.unit.application.test_batch_match | Layer: TEST
Unit tests for batch processing use case.

## Relationships
# TESTS → application/use_cases/batch_match :: BatchProcessor, BatchConfig
"""

from nexus_matcher.application.use_cases.batch_match import (
    BatchConfig,
    BatchProgress,
    BatchResult,
)


class TestBatchConfig:
    """Test BatchConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BatchConfig()

        assert config.max_workers == 4
        assert config.chunk_size == 10
        assert config.fail_fast is False
        assert config.max_errors == 100
        assert config.progress_callback is None
        assert config.checkpoint_dir is None
        assert config.checkpoint_interval == 10

    def test_custom_values(self):
        """Test custom configuration."""

        def callback(p):
            return None

        config = BatchConfig(
            max_workers=8,
            chunk_size=50,
            fail_fast=True,
            progress_callback=callback,
        )

        assert config.max_workers == 8
        assert config.chunk_size == 50
        assert config.fail_fast is True
        assert config.progress_callback is callback


class TestBatchProgress:
    """Test BatchProgress dataclass."""

    def test_percent_complete(self):
        """Test percent complete calculation."""
        progress = BatchProgress(
            total=100,
            completed=25,
            failed=5,
            current_schema="test.avsc",
            elapsed_seconds=10.0,
            estimated_remaining_seconds=30.0,
        )

        assert progress.percent_complete == 25.0

    def test_percent_complete_zero_total(self):
        """Test percent complete with zero total."""
        progress = BatchProgress(
            total=0,
            completed=0,
            failed=0,
            current_schema=None,
            elapsed_seconds=0.0,
            estimated_remaining_seconds=None,
        )

        assert progress.percent_complete == 0.0

    def test_success_rate(self):
        """Test success rate calculation."""
        progress = BatchProgress(
            total=100,
            completed=80,
            failed=20,
            current_schema=None,
            elapsed_seconds=10.0,
            estimated_remaining_seconds=None,
        )

        assert progress.success_rate == 0.8

    def test_success_rate_no_processed(self):
        """Test success rate with nothing processed."""
        progress = BatchProgress(
            total=100,
            completed=0,
            failed=0,
            current_schema="starting.avsc",
            elapsed_seconds=0.1,
            estimated_remaining_seconds=None,
        )

        assert progress.success_rate == 1.0  # Default to 100%


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = BatchResult()

        assert result.sessions == []
        assert result.errors == []
        assert result.total_schemas == 0
        assert result.successful == 0
        assert result.failed == 0
        assert result.total_duration_ms == 0.0
        assert result.avg_duration_ms == 0.0

    def test_success_rate(self):
        """Test success rate property."""
        result = BatchResult(
            total_schemas=100,
            successful=90,
            failed=10,
        )

        assert result.success_rate == 0.9

    def test_success_rate_zero_schemas(self):
        """Test success rate with zero schemas."""
        result = BatchResult(
            total_schemas=0,
            successful=0,
            failed=0,
        )

        assert result.success_rate == 0.0

    def test_errors_accumulation(self):
        """Test error accumulation."""
        result = BatchResult()
        result.errors.append(("schema1.avsc", "Parse error"))
        result.errors.append(("schema2.avsc", "Validation error"))

        assert len(result.errors) == 2
        assert result.errors[0] == ("schema1.avsc", "Parse error")
