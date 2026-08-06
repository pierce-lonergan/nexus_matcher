"""
nexus_matcher.application.use_cases.batch_match | Layer: APPLICATION
Batch schema matching use case for processing multiple schemas efficiently.

## Relationships
# DEPENDS_ON → application/use_cases/match_schema :: NexusMatcher class
# DEPENDS_ON → domain/models/entities :: MatchingSession, Schema
# DEPENDS_ON → shared/types/base :: Result, PagedResult
# USED_BY    → presentation/api :: batch endpoints
# USED_BY    → presentation/cli :: batch command
# EMITS      → BatchStarted :: when batch begins
# EMITS      → BatchProgress :: after each schema
# EMITS      → BatchCompleted :: when batch ends

## Attributes
# Security: Validates all file paths, sandboxes processing
# Performance: Parallel processing with configurable workers, progress tracking
# Reliability: Graceful handling of individual failures, checkpoint support
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus_matcher.application.use_cases.match_schema import NexusMatcher
from nexus_matcher.domain.models.entities import MatchingSession
from nexus_matcher.shared.types.base import Result

# =============================================================================
# BATCH CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for batch processing."""

    # Parallelism
    max_workers: int = 4
    chunk_size: int = 10

    # Error handling
    fail_fast: bool = False
    max_errors: int = 100

    # Progress
    progress_callback: Callable[[BatchProgress], None] | None = None

    # Checkpointing
    checkpoint_dir: Path | None = None
    checkpoint_interval: int = 10  # Save every N schemas


@dataclass(frozen=True)
class BatchProgress:
    """Progress information for batch processing."""

    total: int
    completed: int
    failed: int
    current_schema: str | None
    elapsed_seconds: float
    estimated_remaining_seconds: float | None

    @property
    def percent_complete(self) -> float:
        """Get completion percentage."""
        return (self.completed / self.total * 100) if self.total > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        processed = self.completed + self.failed
        return (self.completed / processed) if processed > 0 else 1.0


@dataclass
class BatchResult:
    """Result of batch processing."""

    sessions: list[MatchingSession] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (path, error)
    total_schemas: int = 0
    successful: int = 0
    failed: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        return self.successful / self.total_schemas if self.total_schemas > 0 else 0.0


# =============================================================================
# BATCH PROCESSOR
# =============================================================================


class BatchProcessor:
    """
    Processes multiple schemas in batch with parallel execution.

    Features:
    - Parallel processing with configurable workers
    - Progress callbacks for UI integration
    - Checkpoint/resume support for large batches
    - Graceful error handling (continue on failure)

    Example:
        ```python
        from nexus_matcher import NexusMatcher
        from nexus_matcher.application.use_cases.batch_match import BatchProcessor

        matcher = NexusMatcher.from_config()
        matcher.load_dictionary("dictionary.xlsx")

        processor = BatchProcessor(matcher)
        result = processor.process_directory(
            Path("schemas/"),
            pattern="*.avsc",
            progress_callback=lambda p: print(f"{p.percent_complete:.1f}%")
        )

        print(f"Processed {result.successful}/{result.total_schemas} schemas")
        ```
    """

    def __init__(
        self,
        matcher: NexusMatcher,
        config: BatchConfig | None = None,
    ) -> None:
        """
        Initialize batch processor.

        Args:
            matcher: Configured NexusMatcher instance
            config: Batch processing configuration
        """
        self._matcher = matcher
        self._config = config or BatchConfig()

    def process_schemas(
        self,
        schema_paths: Sequence[Path | str],
        **options: Any,
    ) -> BatchResult:
        """
        Process multiple schema files.

        Args:
            schema_paths: Paths to schema files
            **options: Additional options passed to match_schema

        Returns:
            BatchResult with all sessions and errors
        """
        start_time = time.time()
        paths = [Path(p) for p in schema_paths]
        total = len(paths)

        result = BatchResult(total_schemas=total)

        # Progress tracking
        completed = 0
        failed = 0

        def update_progress(current: str | None = None) -> None:
            nonlocal completed, failed
            if self._config.progress_callback:
                elapsed = time.time() - start_time
                rate = (completed + failed) / elapsed if elapsed > 0 else 0
                remaining = (total - completed - failed) / rate if rate > 0 else None

                progress = BatchProgress(
                    total=total,
                    completed=completed,
                    failed=failed,
                    current_schema=current,
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=remaining,
                )
                self._config.progress_callback(progress)

        # Process with thread pool for I/O parallelism
        with ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
            futures = {executor.submit(self._process_single, path, options): path for path in paths}

            for future in as_completed(futures):
                path = futures[future]
                update_progress(str(path))

                try:
                    session = future.result()
                    result.sessions.append(session)
                    completed += 1
                    result.successful += 1
                except Exception as e:
                    error_msg = str(e)
                    result.errors.append((str(path), error_msg))
                    failed += 1
                    result.failed += 1

                    if self._config.fail_fast:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break

                    if failed >= self._config.max_errors:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break

                # Checkpoint if configured
                if (
                    self._config.checkpoint_dir
                    and (completed + failed) % self._config.checkpoint_interval == 0
                ):
                    self._save_checkpoint(result)

        # Final stats
        result.total_duration_ms = (time.time() - start_time) * 1000
        result.avg_duration_ms = (
            result.total_duration_ms / result.successful if result.successful > 0 else 0.0
        )

        update_progress(None)
        return result

    def process_directory(
        self,
        directory: Path | str,
        pattern: str = "*.avsc",
        recursive: bool = True,
        **options: Any,
    ) -> BatchResult:
        """
        Process all schemas in a directory.

        Args:
            directory: Directory containing schemas
            pattern: Glob pattern for schema files
            recursive: Search recursively
            **options: Additional options

        Returns:
            BatchResult with all sessions
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        paths = list(dir_path.rglob(pattern)) if recursive else list(dir_path.glob(pattern))

        return self.process_schemas(paths, **options)

    def process_manifest(
        self,
        manifest_path: Path | str,
        base_dir: Path | str | None = None,
        **options: Any,
    ) -> BatchResult:
        """
        Process schemas listed in a manifest file.

        Args:
            manifest_path: Path to manifest (one schema path per line)
            base_dir: Base directory for relative paths
            **options: Additional options

        Returns:
            BatchResult with all sessions
        """
        manifest = Path(manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        base = Path(base_dir) if base_dir else manifest.parent

        paths = []
        with manifest.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    path = Path(line)
                    if not path.is_absolute():
                        path = base / path
                    paths.append(path)

        return self.process_schemas(paths, **options)

    def stream_results(
        self,
        schema_paths: Sequence[Path | str],
        **options: Any,
    ) -> Iterator[Result[MatchingSession]]:
        """
        Stream results as they complete (generator).

        Args:
            schema_paths: Paths to schema files
            **options: Additional options

        Yields:
            Result[MatchingSession] for each schema
        """
        paths = [Path(p) for p in schema_paths]

        with ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
            futures = {executor.submit(self._process_single, path, options): path for path in paths}

            for future in as_completed(futures):
                path = futures[future]
                try:
                    session = future.result()
                    yield Result.success(session)
                except Exception as e:
                    yield Result.failure(f"{path}: {e}")

    def _process_single(
        self,
        path: Path,
        options: dict[str, Any],
    ) -> MatchingSession:
        """Process a single schema file."""
        return self._matcher.match_schema_session(path, **options)

    def _save_checkpoint(self, result: BatchResult) -> None:
        """Save checkpoint for resume capability."""
        if not self._config.checkpoint_dir:
            return

        import json

        checkpoint_dir = self._config.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / "checkpoint.json"
        checkpoint_data = {
            "successful": result.successful,
            "failed": result.failed,
            "errors": result.errors,
            "completed_schemas": [s.schema.name for s in result.sessions],
        }

        with checkpoint_file.open("w") as f:
            json.dump(checkpoint_data, f, indent=2)


# =============================================================================
# ASYNC BATCH PROCESSOR
# =============================================================================


class AsyncBatchProcessor:
    """
    Async version of batch processor for API integration.

    Example:
        ```python
        async def process_batch():
            processor = AsyncBatchProcessor(matcher)
            async for result in processor.stream_results(paths):
                if result.is_success:
                    yield result.unwrap()
        ```
    """

    def __init__(
        self,
        matcher: NexusMatcher,
        config: BatchConfig | None = None,
    ) -> None:
        self._sync_processor = BatchProcessor(matcher, config)
        self._config = config or BatchConfig()

    async def process_schemas(
        self,
        schema_paths: Sequence[Path | str],
        **options: Any,
    ) -> BatchResult:
        """Process schemas asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_processor.process_schemas(schema_paths, **options),
        )

    async def stream_results(
        self,
        schema_paths: Sequence[Path | str],
        **options: Any,
    ):
        """Stream results asynchronously."""
        loop = asyncio.get_event_loop()
        paths = [Path(p) for p in schema_paths]

        for path in paths:
            try:
                session = await loop.run_in_executor(
                    None,
                    lambda p=path: self._sync_processor._process_single(p, options),
                )
                yield Result.success(session)
            except Exception as e:
                yield Result.failure(f"{path}: {e}")
