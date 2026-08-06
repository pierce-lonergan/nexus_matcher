"""
nexus_matcher.shared.exceptions | Layer: SHARED
Custom exception hierarchy for NexusMatcher.

## Relationships
# USED_BY    → all modules :: error handling
# USED_BY    → presentation/api :: HTTP error mapping

## Attributes
# Security: Exceptions should not leak sensitive data in messages
# Reliability: All exceptions include error codes for programmatic handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# ERROR CODES
# =============================================================================


class ErrorCode(str, Enum):
    """Standardized error codes for programmatic handling."""

    # General errors (1xxx)
    UNKNOWN = "NEXUS-1000"
    CONFIGURATION = "NEXUS-1001"
    INITIALIZATION = "NEXUS-1002"
    VALIDATION = "NEXUS-1003"

    # Schema parsing errors (2xxx)
    SCHEMA_PARSE = "NEXUS-2000"
    SCHEMA_UNSUPPORTED_FORMAT = "NEXUS-2001"
    SCHEMA_INVALID_STRUCTURE = "NEXUS-2002"
    SCHEMA_FILE_NOT_FOUND = "NEXUS-2003"

    # Dictionary errors (3xxx)
    DICTIONARY_LOAD = "NEXUS-3000"
    DICTIONARY_NOT_FOUND = "NEXUS-3001"
    DICTIONARY_INVALID_FORMAT = "NEXUS-3002"
    DICTIONARY_SYNC = "NEXUS-3003"

    # Embedding errors (4xxx)
    EMBEDDING_GENERATION = "NEXUS-4000"
    EMBEDDING_MODEL_LOAD = "NEXUS-4001"
    EMBEDDING_DIMENSION_MISMATCH = "NEXUS-4002"

    # Vector store errors (5xxx)
    VECTOR_STORE = "NEXUS-5000"
    VECTOR_STORE_CONNECTION = "NEXUS-5001"
    VECTOR_STORE_COLLECTION = "NEXUS-5002"
    VECTOR_STORE_SEARCH = "NEXUS-5003"
    VECTOR_STORE_INDEX = "NEXUS-5004"

    # Matching errors (6xxx)
    MATCHING = "NEXUS-6000"
    MATCHING_NO_RESULTS = "NEXUS-6001"
    MATCHING_TIMEOUT = "NEXUS-6002"
    MATCHING_INSUFFICIENT_DATA = "NEXUS-6003"

    # Cache errors (7xxx)
    CACHE = "NEXUS-7000"
    CACHE_CONNECTION = "NEXUS-7001"
    CACHE_SERIALIZATION = "NEXUS-7002"

    # API errors (8xxx)
    API = "NEXUS-8000"
    API_AUTHENTICATION = "NEXUS-8001"
    API_AUTHORIZATION = "NEXUS-8002"
    API_RATE_LIMIT = "NEXUS-8003"
    API_INVALID_REQUEST = "NEXUS-8004"


# =============================================================================
# BASE EXCEPTION
# =============================================================================


@dataclass
class NexusMatcherError(Exception):
    """
    Base exception for all NexusMatcher errors.

    Attributes:
        message: Human-readable error message
        code: Standardized error code
        details: Additional error context
        cause: Original exception if wrapping
    """

    message: str
    code: ErrorCode = ErrorCode.UNKNOWN
    details: dict[str, Any] = field(default_factory=dict)
    cause: Exception | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": self.details,
            }
        }


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================


@dataclass
class ConfigurationError(NexusMatcherError):
    """Error in configuration loading or validation."""

    code: ErrorCode = ErrorCode.CONFIGURATION


@dataclass
class InitializationError(NexusMatcherError):
    """Error during component initialization."""

    code: ErrorCode = ErrorCode.INITIALIZATION


@dataclass
class ValidationError(NexusMatcherError):
    """Input validation error."""

    code: ErrorCode = ErrorCode.VALIDATION
    field: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.field:
            self.details["field"] = self.field


# =============================================================================
# SCHEMA ERRORS
# =============================================================================


@dataclass
class SchemaError(NexusMatcherError):
    """Base class for schema-related errors."""

    code: ErrorCode = ErrorCode.SCHEMA_PARSE


@dataclass
class SchemaParseError(SchemaError):
    """Error parsing schema content."""

    code: ErrorCode = ErrorCode.SCHEMA_PARSE
    line: int | None = None
    column: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.line is not None:
            self.details["line"] = self.line
        if self.column is not None:
            self.details["column"] = self.column


@dataclass
class UnsupportedSchemaFormatError(SchemaError):
    """Schema format is not supported."""

    code: ErrorCode = ErrorCode.SCHEMA_UNSUPPORTED_FORMAT
    format: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.format:
            self.details["format"] = self.format


@dataclass
class SchemaFileNotFoundError(SchemaError):
    """Schema file does not exist."""

    code: ErrorCode = ErrorCode.SCHEMA_FILE_NOT_FOUND
    path: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.path:
            self.details["path"] = self.path


# =============================================================================
# DICTIONARY ERRORS
# =============================================================================


@dataclass
class DictionaryError(NexusMatcherError):
    """Base class for dictionary-related errors."""

    code: ErrorCode = ErrorCode.DICTIONARY_LOAD


@dataclass
class DictionaryLoadError(DictionaryError):
    """Error loading dictionary from source."""

    code: ErrorCode = ErrorCode.DICTIONARY_LOAD
    source: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source:
            self.details["source"] = self.source


@dataclass
class DictionaryNotFoundError(DictionaryError):
    """Dictionary not found or not loaded."""

    code: ErrorCode = ErrorCode.DICTIONARY_NOT_FOUND


@dataclass
class DictionarySyncError(DictionaryError):
    """Error synchronizing dictionary to vector store."""

    code: ErrorCode = ErrorCode.DICTIONARY_SYNC


# =============================================================================
# EMBEDDING ERRORS
# =============================================================================


@dataclass
class EmbeddingError(NexusMatcherError):
    """Base class for embedding-related errors."""

    code: ErrorCode = ErrorCode.EMBEDDING_GENERATION


@dataclass
class EmbeddingModelError(EmbeddingError):
    """Error loading embedding model."""

    code: ErrorCode = ErrorCode.EMBEDDING_MODEL_LOAD
    model_name: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.model_name:
            self.details["model_name"] = self.model_name


@dataclass
class EmbeddingDimensionError(EmbeddingError):
    """Embedding dimension mismatch."""

    code: ErrorCode = ErrorCode.EMBEDDING_DIMENSION_MISMATCH
    expected: int = 0
    actual: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.details["expected_dimension"] = self.expected
        self.details["actual_dimension"] = self.actual


# =============================================================================
# VECTOR STORE ERRORS
# =============================================================================


@dataclass
class VectorStoreError(NexusMatcherError):
    """Base class for vector store errors."""

    code: ErrorCode = ErrorCode.VECTOR_STORE


@dataclass
class VectorStoreConnectionError(VectorStoreError):
    """Cannot connect to vector store."""

    code: ErrorCode = ErrorCode.VECTOR_STORE_CONNECTION
    host: str = ""
    port: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.host:
            self.details["host"] = self.host
        if self.port:
            self.details["port"] = self.port


@dataclass
class VectorStoreCollectionError(VectorStoreError):
    """Collection operation failed."""

    code: ErrorCode = ErrorCode.VECTOR_STORE_COLLECTION
    collection: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.collection:
            self.details["collection"] = self.collection


@dataclass
class VectorSearchError(VectorStoreError):
    """Vector search failed."""

    code: ErrorCode = ErrorCode.VECTOR_STORE_SEARCH


# =============================================================================
# MATCHING ERRORS
# =============================================================================


@dataclass
class MatchingError(NexusMatcherError):
    """Base class for matching errors."""

    code: ErrorCode = ErrorCode.MATCHING


@dataclass
class NoMatchResultsError(MatchingError):
    """No match results found."""

    code: ErrorCode = ErrorCode.MATCHING_NO_RESULTS
    field_path: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.field_path:
            self.details["field_path"] = self.field_path


@dataclass
class MatchingTimeoutError(MatchingError):
    """Matching operation timed out."""

    code: ErrorCode = ErrorCode.MATCHING_TIMEOUT
    timeout_seconds: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.timeout_seconds:
            self.details["timeout_seconds"] = self.timeout_seconds


# =============================================================================
# CACHE ERRORS
# =============================================================================


@dataclass
class CacheError(NexusMatcherError):
    """Base class for cache errors."""

    code: ErrorCode = ErrorCode.CACHE


@dataclass
class CacheConnectionError(CacheError):
    """Cannot connect to cache backend."""

    code: ErrorCode = ErrorCode.CACHE_CONNECTION


# =============================================================================
# API ERRORS
# =============================================================================


@dataclass
class APIError(NexusMatcherError):
    """Base class for API errors."""

    code: ErrorCode = ErrorCode.API
    status_code: int = 500

    def __post_init__(self) -> None:
        super().__post_init__()
        self.details["status_code"] = self.status_code


@dataclass
class AuthenticationError(APIError):
    """Authentication failed."""

    code: ErrorCode = ErrorCode.API_AUTHENTICATION
    status_code: int = 401


@dataclass
class AuthorizationError(APIError):
    """Authorization failed."""

    code: ErrorCode = ErrorCode.API_AUTHORIZATION
    status_code: int = 403


@dataclass
class RateLimitError(APIError):
    """Rate limit exceeded."""

    code: ErrorCode = ErrorCode.API_RATE_LIMIT
    status_code: int = 429
    retry_after: int = 60

    def __post_init__(self) -> None:
        super().__post_init__()
        self.details["retry_after"] = self.retry_after


@dataclass
class InvalidRequestError(APIError):
    """Invalid request format or content."""

    code: ErrorCode = ErrorCode.API_INVALID_REQUEST
    status_code: int = 400
