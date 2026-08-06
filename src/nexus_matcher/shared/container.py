"""
nexus_matcher.shared.container | Layer: SHARED
Dependency injection container for managing component lifecycle.

## Relationships
# USED_BY    → application/* :: obtaining dependencies
# USED_BY    → presentation/* :: wiring up the application
# DEPENDS_ON → domain/ports/* :: port interfaces for type hints

## Attributes
# Security: Container should not expose internal state
# Performance: Singleton instances cached for efficiency
# Reliability: Thread-safe registration and resolution

## Design Pattern
Provider pattern with lazy initialization and lifecycle management.
Supports:
- Singleton: One instance per container
- Transient: New instance per request
- Scoped: One instance per scope (e.g., per request)
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


# =============================================================================
# LIFECYCLE
# =============================================================================


class Lifecycle(str, Enum):
    """Component lifecycle management."""

    SINGLETON = "singleton"  # One instance for container lifetime
    TRANSIENT = "transient"  # New instance every time
    SCOPED = "scoped"  # One instance per scope


# =============================================================================
# PROVIDER PROTOCOL
# =============================================================================


class Provider(ABC, Generic[T]):
    """
    Abstract provider for dependency resolution.

    Providers encapsulate the logic for creating and managing
    instances of a particular type.
    """

    @abstractmethod
    def get(self, container: Container) -> T:
        """
        Get an instance of the provided type.

        Args:
            container: Container for resolving nested dependencies

        Returns:
            Instance of type T
        """
        ...

    @property
    @abstractmethod
    def lifecycle(self) -> Lifecycle:
        """Get the lifecycle of this provider."""
        ...


# =============================================================================
# PROVIDER IMPLEMENTATIONS
# =============================================================================


class SingletonProvider(Provider[T]):
    """Provider that returns the same instance every time."""

    def __init__(self, factory: Callable[[Container], T]) -> None:
        self._factory = factory
        self._instance: T | None = None
        self._lock = threading.Lock()

    @property
    def lifecycle(self) -> Lifecycle:
        return Lifecycle.SINGLETON

    def get(self, container: Container) -> T:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory(container)
        return self._instance


class TransientProvider(Provider[T]):
    """Provider that creates a new instance every time."""

    def __init__(self, factory: Callable[[Container], T]) -> None:
        self._factory = factory

    @property
    def lifecycle(self) -> Lifecycle:
        return Lifecycle.TRANSIENT

    def get(self, container: Container) -> T:
        return self._factory(container)


class InstanceProvider(Provider[T]):
    """Provider that returns a pre-created instance."""

    def __init__(self, instance: T) -> None:
        self._instance = instance

    @property
    def lifecycle(self) -> Lifecycle:
        return Lifecycle.SINGLETON

    def get(self, container: Container) -> T:
        return self._instance


class FactoryProvider(Provider[T]):
    """Provider that uses a factory function with custom args."""

    def __init__(
        self,
        factory: Callable[..., T],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        lifecycle: Lifecycle = Lifecycle.TRANSIENT,
    ) -> None:
        self._factory = factory
        self._args = args
        self._kwargs = kwargs or {}
        self._lifecycle = lifecycle
        self._instance: T | None = None
        self._lock = threading.Lock()

    @property
    def lifecycle(self) -> Lifecycle:
        return self._lifecycle

    def get(self, container: Container) -> T:
        if self._lifecycle == Lifecycle.SINGLETON:
            if self._instance is None:
                with self._lock:
                    if self._instance is None:
                        self._instance = self._factory(*self._args, **self._kwargs)
            return self._instance
        return self._factory(*self._args, **self._kwargs)


# =============================================================================
# CONTAINER
# =============================================================================


class Container:
    """
    Dependency injection container.

    Manages registration and resolution of dependencies with
    support for different lifecycles.

    Example usage:
        container = Container()

        # Register with factory
        container.register(
            EmbeddingProvider,
            lambda c: SentenceTransformersProvider("BAAI/bge-base-en-v1.5"),
            lifecycle=Lifecycle.SINGLETON,
        )

        # Register instance
        container.register_instance(Config, my_config)

        # Resolve
        embedder = container.resolve(EmbeddingProvider)

        # With type inference
        embedder: EmbeddingProvider = container[EmbeddingProvider]
    """

    def __init__(self, parent: Container | None = None) -> None:
        """
        Initialize container.

        Args:
            parent: Optional parent container for hierarchical resolution
        """
        self._providers: dict[type, Provider] = {}
        self._parent = parent
        self._lock = threading.Lock()

    def register(
        self,
        interface: type[T],
        factory: Callable[[Container], T],
        lifecycle: Lifecycle = Lifecycle.SINGLETON,
    ) -> Container:
        """
        Register a type with a factory function.

        Args:
            interface: Type/interface to register
            factory: Factory function that receives container
            lifecycle: Component lifecycle

        Returns:
            Self for chaining
        """
        with self._lock:
            if lifecycle == Lifecycle.SINGLETON:
                self._providers[interface] = SingletonProvider(factory)
            else:
                self._providers[interface] = TransientProvider(factory)
        return self

    def register_instance(
        self,
        interface: type[T],
        instance: T,
    ) -> Container:
        """
        Register a pre-created instance.

        Args:
            interface: Type/interface to register
            instance: Instance to return

        Returns:
            Self for chaining
        """
        with self._lock:
            self._providers[interface] = InstanceProvider(instance)
        return self

    def register_factory(
        self,
        interface: type[T],
        factory: Callable[..., T],
        *args: Any,
        lifecycle: Lifecycle = Lifecycle.TRANSIENT,
        **kwargs: Any,
    ) -> Container:
        """
        Register with a factory and custom arguments.

        Args:
            interface: Type/interface to register
            factory: Factory callable
            *args: Positional arguments for factory
            lifecycle: Component lifecycle
            **kwargs: Keyword arguments for factory

        Returns:
            Self for chaining
        """
        with self._lock:
            self._providers[interface] = FactoryProvider(factory, args, kwargs, lifecycle)
        return self

    def register_provider(
        self,
        interface: type[T],
        provider: Provider[T],
    ) -> Container:
        """
        Register a custom provider.

        Args:
            interface: Type/interface to register
            provider: Provider instance

        Returns:
            Self for chaining
        """
        with self._lock:
            self._providers[interface] = provider
        return self

    def resolve(self, interface: type[T]) -> T:
        """
        Resolve a registered type.

        Args:
            interface: Type/interface to resolve

        Returns:
            Instance of the registered type

        Raises:
            KeyError: If type is not registered
        """
        provider = self._providers.get(interface)

        if provider is not None:
            return provider.get(self)

        if self._parent is not None:
            return self._parent.resolve(interface)

        raise KeyError(f"No provider registered for {interface.__name__}")

    def try_resolve(self, interface: type[T]) -> T | None:
        """
        Try to resolve a type, returning None if not found.

        Args:
            interface: Type/interface to resolve

        Returns:
            Instance or None if not registered
        """
        try:
            return self.resolve(interface)
        except KeyError:
            return None

    def is_registered(self, interface: type) -> bool:
        """Check if a type is registered."""
        if interface in self._providers:
            return True
        if self._parent is not None:
            return self._parent.is_registered(interface)
        return False

    def create_scope(self) -> Container:
        """
        Create a child container for scoped resolution.

        Returns:
            New container with this container as parent
        """
        return Container(parent=self)

    def __getitem__(self, interface: type[T]) -> T:
        """Allow dict-style access: container[MyType]."""
        return self.resolve(interface)

    def __contains__(self, interface: type) -> bool:
        """Allow 'in' operator: MyType in container."""
        return self.is_registered(interface)


# =============================================================================
# CONTAINER BUILDER
# =============================================================================


class ContainerBuilder:
    """
    Fluent builder for configuring containers.

    Example:
        container = (
            ContainerBuilder()
            .with_config(my_config)
            .with_embedding_provider("sentence_transformers", model="BAAI/bge-base-en-v1.5")
            .with_vector_store("qdrant", path="./data")
            .build()
        )
    """

    def __init__(self) -> None:
        self._container = Container()
        self._registrations: list[Callable[[Container], None]] = []

    def register(
        self,
        interface: type[T],
        factory: Callable[[Container], T],
        lifecycle: Lifecycle = Lifecycle.SINGLETON,
    ) -> ContainerBuilder:
        """Register a type."""
        self._container.register(interface, factory, lifecycle)
        return self

    def register_instance(
        self,
        interface: type[T],
        instance: T,
    ) -> ContainerBuilder:
        """Register an instance."""
        self._container.register_instance(interface, instance)
        return self

    def add_module(
        self,
        module: Callable[[Container], None],
    ) -> ContainerBuilder:
        """
        Add a configuration module.

        Modules are callables that register multiple related services.

        Args:
            module: Callable that registers services on a container
        """
        self._registrations.append(module)
        return self

    def build(self) -> Container:
        """
        Build the container.

        Returns:
            Configured container
        """
        # Apply all module registrations
        for registration in self._registrations:
            registration(self._container)

        return self._container


# =============================================================================
# GLOBAL CONTAINER (Optional convenience)
# =============================================================================

# Global container instance (use with caution)
_global_container: Container | None = None
_global_lock = threading.Lock()


def get_container() -> Container:
    """
    Get the global container instance.

    Creates a new container if none exists.

    Returns:
        Global container
    """
    global _global_container
    if _global_container is None:
        with _global_lock:
            if _global_container is None:
                _global_container = Container()
    return _global_container


def set_container(container: Container) -> None:
    """
    Set the global container instance.

    Args:
        container: Container to use globally
    """
    global _global_container
    with _global_lock:
        _global_container = container


def reset_container() -> None:
    """Reset the global container."""
    global _global_container
    with _global_lock:
        _global_container = None
