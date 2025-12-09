"""
nexus_matcher.shared.plugins | Layer: SHARED
Plugin interface and hook system for extending NexusMatcher.

## Relationships
# DEPENDS_ON → domain/models/entities :: domain models
# DEPENDS_ON → domain/ports/* :: port interfaces
# USED_BY    → application/use_cases/match_schema :: hook invocation
# USED_BY    → external :: plugin registration

## Attributes
# Security: Sandboxed plugin execution, validated inputs
# Performance: Lazy loading, optional async hooks
# Reliability: Plugin isolation, graceful degradation on failure
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar

from nexus_matcher.domain.models.entities import (
    DictionaryEntry,
    MatchResult,
    MatchingSession,
    Schema,
    SchemaField,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# HOOK TYPES
# =============================================================================


class HookPoint(Enum):
    """Available hook points in the matching pipeline."""

    # Lifecycle hooks
    MATCHER_INITIALIZED = auto()
    DICTIONARY_LOADED = auto()
    DICTIONARY_UPDATED = auto()

    # Schema processing hooks
    SCHEMA_PARSED = auto()
    SCHEMA_VALIDATION = auto()

    # Field matching hooks
    PRE_FIELD_MATCH = auto()
    POST_RETRIEVAL = auto()
    POST_RERANK = auto()
    POST_SCORING = auto()
    POST_FIELD_MATCH = auto()

    # Session hooks
    SESSION_STARTED = auto()
    SESSION_COMPLETED = auto()

    # Result hooks
    RESULT_FILTERED = auto()
    RESULT_FORMATTED = auto()

    # Error hooks
    ON_ERROR = auto()


@dataclass
class HookContext:
    """Context passed to hook handlers."""

    hook_point: HookPoint
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Common data accessors
    @property
    def schema(self) -> Schema | None:
        return self.data.get("schema")

    @property
    def field(self) -> SchemaField | None:
        return self.data.get("field")

    @property
    def results(self) -> list[MatchResult] | None:
        return self.data.get("results")

    @property
    def session(self) -> MatchingSession | None:
        return self.data.get("session")


@dataclass
class HookResult:
    """Result from a hook handler."""

    modified_data: dict[str, Any] | None = None
    skip_remaining: bool = False
    error: str | None = None

    @classmethod
    def passthrough(cls) -> HookResult:
        """Return unchanged data."""
        return cls()

    @classmethod
    def modify(cls, **data: Any) -> HookResult:
        """Return modified data."""
        return cls(modified_data=data)

    @classmethod
    def stop(cls) -> HookResult:
        """Stop processing remaining hooks."""
        return cls(skip_remaining=True)

    @classmethod
    def fail(cls, error: str) -> HookResult:
        """Indicate hook failure."""
        return cls(error=error)


# Type for hook handlers
HookHandler = Callable[[HookContext], HookResult]


# =============================================================================
# PLUGIN BASE CLASS
# =============================================================================


class Plugin(ABC):
    """
    Base class for NexusMatcher plugins.

    Plugins can:
    - Register hook handlers for pipeline customization
    - Provide custom adapters (parsers, loaders, embedders)
    - Add CLI commands
    - Extend API endpoints

    Example:
        ```python
        class MyPlugin(Plugin):
            name = "my-plugin"
            version = "1.0.0"

            def initialize(self, registry: PluginRegistry) -> None:
                registry.register_hook(
                    HookPoint.POST_FIELD_MATCH,
                    self.enhance_results,
                    priority=100,
                )

            def enhance_results(self, context: HookContext) -> HookResult:
                results = context.results
                # Modify results...
                return HookResult.modify(results=enhanced_results)
        ```
    """

    # Plugin metadata (override in subclass)
    name: str = "base-plugin"
    version: str = "0.0.0"
    description: str = ""
    author: str = ""

    @abstractmethod
    def initialize(self, registry: PluginRegistry) -> None:
        """
        Initialize the plugin with the registry.

        Args:
            registry: Plugin registry for registering hooks and adapters
        """
        pass

    def shutdown(self) -> None:
        """Clean up plugin resources."""
        pass


# =============================================================================
# PLUGIN REGISTRY
# =============================================================================


@dataclass
class RegisteredHook:
    """A registered hook handler."""

    handler: HookHandler
    priority: int
    plugin_name: str
    enabled: bool = True


class PluginRegistry:
    """
    Central registry for plugins and hooks.

    Manages:
    - Plugin lifecycle
    - Hook registration and invocation
    - Adapter registration
    - Plugin discovery via entry points

    Example:
        ```python
        registry = PluginRegistry()

        # Discover and load plugins
        registry.discover_plugins()

        # Or register manually
        registry.register_plugin(MyPlugin())

        # Invoke hooks during processing
        context = HookContext(
            hook_point=HookPoint.POST_FIELD_MATCH,
            data={"results": results}
        )
        modified = registry.invoke_hooks(context)
        ```
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._hooks: dict[HookPoint, list[RegisteredHook]] = {
            hp: [] for hp in HookPoint
        }
        self._adapters: dict[str, dict[str, Any]] = {
            "schema_parsers": {},
            "dictionary_loaders": {},
            "embedding_providers": {},
            "vector_stores": {},
            "rerankers": {},
        }

    # -------------------------------------------------------------------------
    # Plugin Management
    # -------------------------------------------------------------------------

    def register_plugin(self, plugin: Plugin) -> None:
        """
        Register and initialize a plugin.

        Args:
            plugin: Plugin instance to register
        """
        if plugin.name in self._plugins:
            logger.warning(f"Plugin '{plugin.name}' already registered, replacing")

        self._plugins[plugin.name] = plugin
        plugin.initialize(self)
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")

    def unregister_plugin(self, name: str) -> None:
        """
        Unregister and shut down a plugin.

        Args:
            name: Plugin name to unregister
        """
        if name not in self._plugins:
            return

        plugin = self._plugins[name]
        plugin.shutdown()

        # Remove plugin's hooks
        for hooks in self._hooks.values():
            hooks[:] = [h for h in hooks if h.plugin_name != name]

        del self._plugins[name]
        logger.info(f"Unregistered plugin: {name}")

    def get_plugin(self, name: str) -> Plugin | None:
        """Get a registered plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> list[tuple[str, str, str]]:
        """List all registered plugins (name, version, description)."""
        return [
            (p.name, p.version, p.description)
            for p in self._plugins.values()
        ]

    def discover_plugins(self, entry_point: str = "nexus_matcher.plugins") -> int:
        """
        Discover and load plugins from entry points.

        Args:
            entry_point: Entry point group name

        Returns:
            Number of plugins loaded
        """
        loaded = 0

        try:
            eps = importlib.metadata.entry_points(group=entry_point)
        except TypeError:
            # Python < 3.10 compatibility
            eps = importlib.metadata.entry_points().get(entry_point, [])

        for ep in eps:
            try:
                plugin_class = ep.load()
                plugin = plugin_class()
                self.register_plugin(plugin)
                loaded += 1
            except Exception as e:
                logger.error(f"Failed to load plugin '{ep.name}': {e}")

        return loaded

    # -------------------------------------------------------------------------
    # Hook Management
    # -------------------------------------------------------------------------

    def register_hook(
        self,
        hook_point: HookPoint,
        handler: HookHandler,
        priority: int = 50,
        plugin_name: str = "anonymous",
    ) -> None:
        """
        Register a hook handler.

        Args:
            hook_point: Where to invoke the hook
            handler: Callable to handle the hook
            priority: Order of execution (lower = earlier, default 50)
            plugin_name: Name of registering plugin
        """
        registered = RegisteredHook(
            handler=handler,
            priority=priority,
            plugin_name=plugin_name,
        )

        self._hooks[hook_point].append(registered)
        # Sort by priority
        self._hooks[hook_point].sort(key=lambda h: h.priority)

        logger.debug(
            f"Registered hook: {plugin_name} -> {hook_point.name} "
            f"(priority={priority})"
        )

    def unregister_hook(
        self,
        hook_point: HookPoint,
        handler: HookHandler,
    ) -> bool:
        """
        Unregister a specific hook handler.

        Returns:
            True if handler was found and removed
        """
        hooks = self._hooks[hook_point]
        original_len = len(hooks)
        self._hooks[hook_point] = [h for h in hooks if h.handler != handler]
        return len(self._hooks[hook_point]) < original_len

    def invoke_hooks(self, context: HookContext) -> HookContext:
        """
        Invoke all hooks for a hook point.

        Args:
            context: Context with hook point and data

        Returns:
            Modified context (or original if no modifications)
        """
        hooks = self._hooks.get(context.hook_point, [])

        for registered in hooks:
            if not registered.enabled:
                continue

            try:
                result = registered.handler(context)

                if result.error:
                    logger.warning(
                        f"Hook error ({registered.plugin_name}): {result.error}"
                    )
                    continue

                if result.modified_data:
                    context.data.update(result.modified_data)

                if result.skip_remaining:
                    break

            except Exception as e:
                logger.error(
                    f"Hook exception ({registered.plugin_name} -> "
                    f"{context.hook_point.name}): {e}"
                )

        return context

    def has_hooks(self, hook_point: HookPoint) -> bool:
        """Check if any hooks are registered for a point."""
        return bool(self._hooks.get(hook_point))

    # -------------------------------------------------------------------------
    # Adapter Registration
    # -------------------------------------------------------------------------

    def register_adapter(
        self,
        category: str,
        name: str,
        adapter: Any,
    ) -> None:
        """
        Register a custom adapter.

        Args:
            category: Adapter category (schema_parsers, dictionary_loaders, etc.)
            name: Adapter name
            adapter: Adapter instance or factory
        """
        if category not in self._adapters:
            raise ValueError(f"Unknown adapter category: {category}")

        self._adapters[category][name] = adapter
        logger.info(f"Registered adapter: {category}/{name}")

    def get_adapter(self, category: str, name: str) -> Any | None:
        """Get a registered adapter."""
        return self._adapters.get(category, {}).get(name)

    def list_adapters(self, category: str) -> list[str]:
        """List adapters in a category."""
        return list(self._adapters.get(category, {}).keys())


# =============================================================================
# BUILT-IN HOOKS
# =============================================================================


def logging_hook(context: HookContext) -> HookResult:
    """Built-in hook that logs pipeline events."""
    logger.debug(
        f"Hook: {context.hook_point.name}",
        extra={"hook_data": context.data},
    )
    return HookResult.passthrough()


def validation_hook(context: HookContext) -> HookResult:
    """Built-in hook that validates data at each stage."""
    if context.hook_point == HookPoint.SCHEMA_VALIDATION:
        schema = context.schema
        if schema and not schema.fields:
            return HookResult.fail("Schema has no fields")

    return HookResult.passthrough()


# =============================================================================
# GLOBAL REGISTRY
# =============================================================================

# Singleton registry for global access
_global_registry: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def reset_plugin_registry() -> None:
    """Reset the global plugin registry (for testing)."""
    global _global_registry
    if _global_registry:
        for plugin_name in list(_global_registry._plugins.keys()):
            _global_registry.unregister_plugin(plugin_name)
    _global_registry = None
