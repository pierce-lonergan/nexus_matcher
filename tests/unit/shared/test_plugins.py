"""
tests.unit.shared.test_plugins | Layer: TEST
Unit tests for plugin system.

## Relationships
# TESTS → shared/plugins :: Plugin, PluginRegistry, HookPoint
"""

import pytest

from nexus_matcher.shared.plugins import (
    HookContext,
    HookPoint,
    HookResult,
    Plugin,
    PluginRegistry,
    get_plugin_registry,
    reset_plugin_registry,
)


class TestHookResult:
    """Test HookResult factory methods."""

    def test_passthrough(self):
        """Test passthrough result."""
        result = HookResult.passthrough()
        assert result.modified_data is None
        assert result.skip_remaining is False
        assert result.error is None

    def test_modify(self):
        """Test modify result."""
        result = HookResult.modify(foo="bar", count=42)
        assert result.modified_data == {"foo": "bar", "count": 42}
        assert result.skip_remaining is False

    def test_stop(self):
        """Test stop result."""
        result = HookResult.stop()
        assert result.skip_remaining is True

    def test_fail(self):
        """Test fail result."""
        result = HookResult.fail("Something went wrong")
        assert result.error == "Something went wrong"


class TestHookContext:
    """Test HookContext data access."""

    def test_schema_accessor(self):
        """Test schema property."""
        context = HookContext(
            hook_point=HookPoint.SCHEMA_PARSED,
            data={"schema": "mock_schema"},
        )
        assert context.schema == "mock_schema"

    def test_field_accessor(self):
        """Test field property."""
        context = HookContext(
            hook_point=HookPoint.PRE_FIELD_MATCH,
            data={"field": "mock_field"},
        )
        assert context.field == "mock_field"

    def test_results_accessor(self):
        """Test results property."""
        context = HookContext(
            hook_point=HookPoint.POST_FIELD_MATCH,
            data={"results": ["result1", "result2"]},
        )
        assert context.results == ["result1", "result2"]


class SamplePlugin(Plugin):
    """Sample plugin for testing."""

    name = "sample-plugin"
    version = "1.0.0"
    description = "A sample plugin"

    def __init__(self):
        self.initialized = False
        self.shutdown_called = False
        self.hook_calls = []

    def initialize(self, registry: PluginRegistry) -> None:
        self.initialized = True
        registry.register_hook(
            HookPoint.SCHEMA_PARSED,
            self.on_schema_parsed,
            priority=50,
            plugin_name=self.name,
        )

    def shutdown(self) -> None:
        self.shutdown_called = True

    def on_schema_parsed(self, context: HookContext) -> HookResult:
        self.hook_calls.append(context.hook_point)
        return HookResult.passthrough()


class TestPluginRegistry:
    """Test PluginRegistry."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry."""
        return PluginRegistry()

    def test_register_plugin(self, registry):
        """Test plugin registration."""
        plugin = SamplePlugin()
        registry.register_plugin(plugin)

        assert plugin.initialized
        assert registry.get_plugin("sample-plugin") is plugin
        assert ("sample-plugin", "1.0.0", "A sample plugin") in registry.list_plugins()

    def test_unregister_plugin(self, registry):
        """Test plugin unregistration."""
        plugin = SamplePlugin()
        registry.register_plugin(plugin)
        registry.unregister_plugin("sample-plugin")

        assert plugin.shutdown_called
        assert registry.get_plugin("sample-plugin") is None

    def test_register_hook(self, registry):
        """Test hook registration."""

        def handler(ctx):
            return HookResult.passthrough()

        registry.register_hook(
            HookPoint.SCHEMA_PARSED,
            handler,
            priority=25,
            plugin_name="test",
        )

        assert registry.has_hooks(HookPoint.SCHEMA_PARSED)
        assert not registry.has_hooks(HookPoint.SESSION_COMPLETED)

    def test_invoke_hooks(self, registry):
        """Test hook invocation."""
        call_log = []

        def handler1(ctx):
            call_log.append("handler1")
            return HookResult.passthrough()

        def handler2(ctx):
            call_log.append("handler2")
            return HookResult.modify(enhanced=True)

        registry.register_hook(HookPoint.SCHEMA_PARSED, handler1, priority=10)
        registry.register_hook(HookPoint.SCHEMA_PARSED, handler2, priority=20)

        context = HookContext(
            hook_point=HookPoint.SCHEMA_PARSED,
            data={"schema": "test"},
        )

        result = registry.invoke_hooks(context)

        assert call_log == ["handler1", "handler2"]
        assert result.data.get("enhanced") is True

    def test_invoke_hooks_with_stop(self, registry):
        """Test hook invocation stops when requested."""
        call_log = []

        def handler1(ctx):
            call_log.append("handler1")
            return HookResult.stop()

        def handler2(ctx):
            call_log.append("handler2")
            return HookResult.passthrough()

        registry.register_hook(HookPoint.SCHEMA_PARSED, handler1, priority=10)
        registry.register_hook(HookPoint.SCHEMA_PARSED, handler2, priority=20)

        context = HookContext(hook_point=HookPoint.SCHEMA_PARSED)
        registry.invoke_hooks(context)

        assert call_log == ["handler1"]  # handler2 should not be called

    def test_unregister_hook(self, registry):
        """Test hook unregistration."""

        def handler(ctx):
            return HookResult.passthrough()

        registry.register_hook(HookPoint.SCHEMA_PARSED, handler)

        assert registry.has_hooks(HookPoint.SCHEMA_PARSED)

        registry.unregister_hook(HookPoint.SCHEMA_PARSED, handler)
        # Hook list exists but is empty - has_hooks checks for non-empty
        assert not registry.has_hooks(HookPoint.SCHEMA_PARSED)

    def test_register_adapter(self, registry):
        """Test adapter registration."""
        mock_parser = object()
        registry.register_adapter("schema_parsers", "custom", mock_parser)

        assert registry.get_adapter("schema_parsers", "custom") is mock_parser
        assert "custom" in registry.list_adapters("schema_parsers")

    def test_register_adapter_unknown_category(self, registry):
        """Test adapter registration with unknown category."""
        with pytest.raises(ValueError, match="Unknown adapter category"):
            registry.register_adapter("unknown_category", "test", object())


class TestGlobalRegistry:
    """Test global registry functions."""

    def teardown_method(self):
        """Reset global registry after each test."""
        reset_plugin_registry()

    def test_get_plugin_registry_singleton(self):
        """Test global registry is singleton."""
        reg1 = get_plugin_registry()
        reg2 = get_plugin_registry()
        assert reg1 is reg2

    def test_reset_plugin_registry(self):
        """Test registry reset."""
        reg1 = get_plugin_registry()
        reset_plugin_registry()
        reg2 = get_plugin_registry()
        assert reg1 is not reg2
