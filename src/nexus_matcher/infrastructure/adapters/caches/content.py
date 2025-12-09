"""
nexus_matcher.infrastructure.adapters.caches.content | Layer: INFRASTRUCTURE
Semantic content cache using BLAKE3 hashing.

## Relationships
# USES       → infrastructure/adapters/caches/memory :: L1LRUCache for storage
# USED_BY    → application/services :: Embedding computation caching
# DEPENDS_ON → blake3 :: Content hashing (8-10x faster than SHA-256)

## Attributes
# Security: Hash-based, no raw content stored as keys
# Performance: BLAKE3 hashing (8-10x faster than SHA-256)
# Reliability: Content normalization for consistent cache hits

## Research Reference
# README_RESEARCH_3.md, Lines 21-37
# Target: 50-70% cost reduction, 40-60% cache hit rate
# "Cache embeddings by content hash using BLAKE3 to avoid recomputing identical queries"
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any, Callable, TypeVar

import numpy as np

from nexus_matcher.domain.ports.cache import CacheConfig, CacheStats
from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _get_blake3():
    """Lazy import of blake3."""
    try:
        import blake3
        return blake3
    except ImportError:
        logger.warning("blake3 not installed, falling back to hashlib")
        return None


class ContentHasher:
    """
    BLAKE3-based content hasher for semantic deduplication.
    
    BLAKE3 is 8-10x faster than SHA-256 while maintaining cryptographic security.
    Supports content normalization for consistent hashing of semantically equivalent text.
    
    Example:
        hasher = ContentHasher(normalize=True)
        hash1 = hasher.hash("Customer Email")
        hash2 = hasher.hash("customer  email")  # Same hash if lowercase=True
    """
    
    def __init__(
        self,
        normalize: bool = False,
        lowercase: bool = False,
    ) -> None:
        """
        Initialize content hasher.
        
        Args:
            normalize: Normalize whitespace (collapse multiple spaces, trim)
            lowercase: Convert to lowercase before hashing
        """
        self._normalize = normalize
        self._lowercase = lowercase
        self._blake3 = _get_blake3()
    
    @property
    def algorithm(self) -> str:
        """Get hashing algorithm name."""
        return "blake3" if self._blake3 else "sha256"
    
    def _preprocess(self, content: str) -> str:
        """Preprocess content before hashing."""
        if self._normalize:
            # Collapse whitespace
            content = re.sub(r'\s+', ' ', content.strip())
        
        if self._lowercase:
            content = content.lower()
        
        return content
    
    def hash(self, content: str) -> str:
        """
        Hash content using BLAKE3.
        
        Args:
            content: Text content to hash
            
        Returns:
            64-character hex string (256 bits)
        """
        processed = self._preprocess(content)
        encoded = processed.encode('utf-8')
        
        if self._blake3:
            return self._blake3.blake3(encoded).hexdigest()
        else:
            # Fallback to SHA-256
            import hashlib
            return hashlib.sha256(encoded).hexdigest()
    
    def hash_batch(self, contents: list[str]) -> list[str]:
        """Hash multiple contents."""
        return [self.hash(content) for content in contents]


class SemanticContentCache:
    """
    Semantic content cache for embedding deduplication.
    
    Caches embeddings by content hash to avoid recomputing identical queries.
    Uses BLAKE3 for fast hashing and L1LRUCache for storage.
    
    Research targets (README_RESEARCH_3.md):
    - 50-70% cost reduction for embedding computation
    - 40-60% cache hit rate for typical workloads
    
    Example:
        cache = SemanticContentCache(max_size=10000)
        
        def compute_embedding(text):
            return model.encode(text)
        
        # First call - computes and caches
        embedding1 = cache.get_or_compute("customer email", compute_embedding)
        
        # Second call - returns cached (no computation)
        embedding2 = cache.get_or_compute("customer email", compute_embedding)
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl: timedelta | None = timedelta(hours=1),
        normalize: bool = True,
        lowercase: bool = False,
    ) -> None:
        """
        Initialize semantic content cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl: Time-to-live for cached entries
            normalize: Normalize whitespace before hashing
            lowercase: Convert to lowercase before hashing
        """
        self._hasher = ContentHasher(normalize=normalize, lowercase=lowercase)
        
        config = CacheConfig(max_size=max_size, ttl=ttl)
        self._cache = L1LRUCache(config)
        
        logger.info(
            f"Initialized SemanticContentCache with max_size={max_size}, "
            f"normalize={normalize}, lowercase={lowercase}"
        )
    
    @property
    def cache_type(self) -> str:
        """Get cache type identifier."""
        return "semantic_content"
    
    def _get_hash(self, content: str) -> str:
        """Get content hash."""
        return self._hasher.hash(content)
    
    def get_by_content(self, content: str) -> np.ndarray | None:
        """
        Get cached embedding by content.
        
        Args:
            content: Text content to look up
            
        Returns:
            Cached embedding or None if not found
        """
        content_hash = self._get_hash(content)
        return self._cache.get(content_hash)
    
    def set_by_content(
        self,
        content: str,
        embedding: np.ndarray,
        ttl: timedelta | None = None,
    ) -> bool:
        """
        Cache embedding by content.
        
        Args:
            content: Text content (used to generate hash key)
            embedding: Embedding vector to cache
            ttl: Optional TTL override
            
        Returns:
            True if successfully cached
        """
        content_hash = self._get_hash(content)
        return self._cache.set(content_hash, embedding, ttl)
    
    def get_or_compute(
        self,
        content: str,
        compute_fn: Callable[[str], np.ndarray],
    ) -> np.ndarray:
        """
        Get cached embedding or compute and cache.
        
        This is the primary interface for cost-saving embedding lookups.
        
        Args:
            content: Text content
            compute_fn: Function to compute embedding if not cached
            
        Returns:
            Embedding vector (from cache or freshly computed)
        """
        # Try cache first
        cached = self.get_by_content(content)
        if cached is not None:
            return cached
        
        # Compute and cache
        embedding = compute_fn(content)
        self.set_by_content(content, embedding)
        return embedding
    
    def batch_get_or_compute(
        self,
        contents: list[str],
        compute_fn: Callable[[list[str]], list[np.ndarray]],
    ) -> list[np.ndarray]:
        """
        Batch get or compute embeddings.
        
        Efficiently handles batches by only computing uncached contents.
        
        Args:
            contents: List of text contents
            compute_fn: Function that takes list of texts and returns list of embeddings
            
        Returns:
            List of embeddings (same order as contents)
        """
        results: dict[int, np.ndarray] = {}
        to_compute: list[tuple[int, str]] = []
        
        # Check cache for each content
        for i, content in enumerate(contents):
            cached = self.get_by_content(content)
            if cached is not None:
                results[i] = cached
            else:
                to_compute.append((i, content))
        
        # Compute uncached in batch
        if to_compute:
            indices, texts = zip(*to_compute)
            computed = compute_fn(list(texts))
            
            for idx, text, embedding in zip(indices, texts, computed):
                self.set_by_content(text, embedding)
                results[idx] = embedding
        
        # Return in original order
        return [results[i] for i in range(len(contents))]
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()
    
    def clear(self) -> int:
        """Clear all cached entries."""
        return self._cache.clear()
    
    def contains(self, content: str) -> bool:
        """Check if content is cached."""
        content_hash = self._get_hash(content)
        return self._cache.exists(content_hash)
