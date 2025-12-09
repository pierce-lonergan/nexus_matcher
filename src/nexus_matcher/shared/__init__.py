"""
nexus_matcher.shared | Layer: SHARED
Cross-cutting concerns and utilities used across all layers.

Contains:
- types/: Common type definitions, enums, and base classes
- utils/: Utility functions and helpers
- exceptions/: Custom exception hierarchy
- events/: Event definitions for pub/sub
- container: Dependency injection container
"""

from nexus_matcher.shared.container import (
    Container,
    ContainerBuilder,
    Lifecycle,
    Provider,
    get_container,
    reset_container,
    set_container,
)

__all__ = [
    # Container
    "Container",
    "ContainerBuilder",
    "Lifecycle",
    "Provider",
    "get_container",
    "set_container",
    "reset_container",
]
