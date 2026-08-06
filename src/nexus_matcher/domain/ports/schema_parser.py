"""
nexus_matcher.domain.ports.schema_parser | Layer: DOMAIN
Port interface for schema parsing - supports multiple schema formats.

## Relationships
# USED_BY    → application/use_cases/match_schema :: parsing schemas before matching
# USED_BY    → infrastructure/adapters/schema_parsers/* :: implementations
# DEPENDS_ON → domain/models/entities :: Schema, SchemaField models

## Attributes
# Security: Parsers should validate input to prevent injection attacks
# Performance: Parsing should be O(n) where n is schema size
# Reliability: Should handle malformed schemas gracefully with clear errors

## Extension Points
# Plugin: nexus_matcher.schema_parsers entry point
# Register: SchemaParserRegistry.register()
"""

from __future__ import annotations

# `json` must be imported at module scope, not inside parse()'s
# `if isinstance(content, str)` branch. It was function-local there, so when a
# dict was passed and _parse_content raised, evaluating the
# `except json.JSONDecodeError` clause hit UnboundLocalError -- which escaped
# parse() entirely instead of producing Result.failure. Any parser raising a
# validation error on dict input crashed its caller rather than returning a
# failed Result.
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from nexus_matcher.domain.models.entities import Schema
from nexus_matcher.shared.types.base import Result

# =============================================================================
# SCHEMA PARSER PROTOCOL
# =============================================================================


@runtime_checkable
class SchemaParser(Protocol):
    """
    Protocol for schema parsers.

    Implementations parse schema definitions from various formats
    (Avro, JSON Schema, SQL DDL, CSV headers, etc.) into the unified
    Schema domain model.

    Example usage:
        parser = AvroSchemaParser()
        result = parser.parse_file(Path("schema.avsc"))
        if result.is_success:
            schema = result.unwrap()
            for field in schema.fields:
                print(f"{field.full_path}: {field.data_type}")
    """

    @property
    def format_name(self) -> str:
        """
        Get the format name this parser handles.

        Returns:
            Format identifier (e.g., "avro", "json_schema", "sql_ddl")
        """
        ...

    @property
    def file_extensions(self) -> frozenset[str]:
        """
        Get supported file extensions.

        Returns:
            Set of extensions (e.g., {".avsc", ".avro"})
        """
        ...

    def parse(self, content: str | dict[str, Any]) -> Result[Schema]:
        """
        Parse schema from string or dictionary content.

        Args:
            content: Schema content as string or parsed dict

        Returns:
            Result containing Schema on success, error message on failure
        """
        ...

    def parse_file(self, path: Path) -> Result[Schema]:
        """
        Parse schema from a file.

        Args:
            path: Path to schema file

        Returns:
            Result containing Schema on success, error message on failure
        """
        ...

    def can_parse(self, content: str | dict[str, Any]) -> bool:
        """
        Check if this parser can handle the given content.

        Args:
            content: Content to check

        Returns:
            True if this parser can handle the content
        """
        ...


# =============================================================================
# BASE IMPLEMENTATION
# =============================================================================


class BaseSchemaParser(ABC):
    """
    Abstract base class for schema parsers with common functionality.

    Provides:
    - File reading with encoding handling
    - Common validation
    - Error wrapping
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Format identifier."""
        ...

    @property
    @abstractmethod
    def file_extensions(self) -> frozenset[str]:
        """Supported file extensions."""
        ...

    @abstractmethod
    def _parse_content(self, content: dict[str, Any]) -> Schema:
        """
        Internal parsing implementation.

        Args:
            content: Parsed schema content as dictionary

        Returns:
            Parsed Schema

        Raises:
            ValueError: If schema is invalid
        """
        ...

    def parse(self, content: str | dict[str, Any]) -> Result[Schema]:
        """Parse schema from string or dict."""
        try:
            parsed = json.loads(content) if isinstance(content, str) else content

            schema = self._parse_content(parsed)
            return Result.success(schema)

        except json.JSONDecodeError as e:
            return Result.failure(f"Invalid JSON: {e}", "PARSE_ERROR")
        except ValueError as e:
            return Result.failure(str(e), "VALIDATION_ERROR")
        except Exception as e:
            return Result.failure(f"Unexpected error: {e}", "UNKNOWN_ERROR")

    def parse_file(self, path: Path) -> Result[Schema]:
        """Parse schema from file."""
        if not path.exists():
            return Result.failure(f"File not found: {path}", "FILE_NOT_FOUND")

        if path.suffix.lower() not in self.file_extensions:
            return Result.failure(
                f"Unsupported extension {path.suffix}, expected one of {self.file_extensions}",
                "UNSUPPORTED_FORMAT",
            )

        try:
            content = path.read_text(encoding="utf-8")
            return self.parse(content)
        except UnicodeDecodeError:
            return Result.failure(f"File is not valid UTF-8: {path}", "ENCODING_ERROR")
        except Exception as e:
            return Result.failure(f"Error reading file: {e}", "IO_ERROR")

    def can_parse(self, content: str | dict[str, Any]) -> bool:
        """Check if content can be parsed."""
        result = self.parse(content)
        return result.is_success


# =============================================================================
# PARSER REGISTRY
# =============================================================================


class SchemaParserRegistry:
    """
    Registry for schema parser implementations.

    Supports:
    - Manual registration
    - Entry point discovery
    - Format-based lookup
    - Extension-based lookup

    Example:
        registry = SchemaParserRegistry()
        registry.register(AvroSchemaParser())

        # By format
        parser = registry.get_parser("avro")

        # By file extension
        parser = registry.get_parser_for_file(Path("schema.avsc"))

        # Auto-detect
        parser = registry.detect_parser(schema_content)
    """

    def __init__(self) -> None:
        self._parsers: dict[str, SchemaParser] = {}
        self._extension_map: dict[str, str] = {}

    def register(self, parser: SchemaParser) -> None:
        """
        Register a schema parser.

        Args:
            parser: Parser instance to register
        """
        self._parsers[parser.format_name] = parser
        for ext in parser.file_extensions:
            self._extension_map[ext.lower()] = parser.format_name

    def get_parser(self, format_name: str) -> SchemaParser | None:
        """
        Get parser by format name.

        Args:
            format_name: Format identifier

        Returns:
            Parser instance or None if not found
        """
        return self._parsers.get(format_name)

    def get_parser_for_file(self, path: Path) -> SchemaParser | None:
        """
        Get parser by file extension.

        Args:
            path: File path

        Returns:
            Parser instance or None if extension not supported
        """
        format_name = self._extension_map.get(path.suffix.lower())
        if format_name:
            return self._parsers.get(format_name)
        return None

    def detect_parser(self, content: str | dict[str, Any]) -> SchemaParser | None:
        """
        Auto-detect parser for content.

        Args:
            content: Schema content

        Returns:
            First parser that can handle the content, or None
        """
        for parser in self._parsers.values():
            if parser.can_parse(content):
                return parser
        return None

    def list_formats(self) -> list[str]:
        """Get list of registered format names."""
        return list(self._parsers.keys())

    def list_extensions(self) -> list[str]:
        """Get list of supported extensions."""
        return list(self._extension_map.keys())

    def load_entry_points(self) -> None:
        """
        Load parsers from entry points.

        Entry point group: nexus_matcher.schema_parsers
        """
        try:
            from importlib.metadata import entry_points

            eps = entry_points(group="nexus_matcher.schema_parsers")
            for ep in eps:
                try:
                    parser_class = ep.load()
                    parser = parser_class()
                    self.register(parser)
                except Exception as e:
                    import logging

                    logging.warning(f"Failed to load parser {ep.name}: {e}")
        except ImportError:
            pass  # Entry points not available
