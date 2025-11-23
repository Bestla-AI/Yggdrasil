"""Final tests to achieve maximum coverage."""

from typing import Tuple

import pytest

from bestla.yggdrasil import Context, Toolkit, tool
from bestla.yggdrasil.dynamic_types import (
    DynamicArray,
    DynamicConstraints,
    DynamicNested,
    DynamicPattern,
    DynamicType,
)


class TestDynamicTypeBaseClass:
    """Test DynamicType base class."""

    def test_dynamic_type_base_not_implemented(self):
        """Test DynamicType base class raises NotImplementedError."""

        # Create instance of base class
        dt = DynamicType("test_key")

        assert dt.context_key == "test_key"

        # generate_schema should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            dt.generate_schema(Context())


class TestDynamicArrayEdgeCases:
    """Test DynamicArray edge cases."""

    def test_dynamic_array_with_non_list(self):
        """Test DynamicArray when value is not a list."""
        context = Context()
        context.set("items", "just_a_string")

        schema = DynamicArray["items"].generate_schema(context)

        # Should return basic array schema
        assert schema["type"] == "array"
        # No items constraint when value is not a list


class TestDynamicNestedEdgeCases:
    """Test DynamicNested edge cases."""

    def test_dynamic_nested_with_list(self):
        """Test DynamicNested when nested value is list."""
        context = Context()
        context.set("project", {"statuses": ["todo", "in_progress", "done"]})

        schema = DynamicNested["project.statuses"].generate_schema(context)

        # Should create enum from nested list
        assert schema["type"] == "string"
        assert "enum" in schema
        assert schema["enum"] == ["todo", "in_progress", "done"]

    def test_dynamic_nested_with_non_list(self):
        """Test DynamicNested when nested value is not list."""
        context = Context()
        context.set("project", {"status": "active"})

        schema = DynamicNested["project.status"].generate_schema(context)

        # Should return basic string schema
        assert schema == {"type": "string"}


class TestDynamicPatternEdgeCases:
    """Test DynamicPattern edge cases."""

    def test_dynamic_pattern_with_string_value(self):
        """Test DynamicPattern when context value is string pattern."""
        context = Context()
        context.set("issue_pattern", "^[A-Z]+-[0-9]+$")

        schema = DynamicPattern["issue_pattern"].generate_schema(context)

        assert schema["type"] == "string"
        assert schema["pattern"] == "^[A-Z]+-[0-9]+$"

    def test_dynamic_pattern_with_none(self):
        """Test DynamicPattern when context value is None."""
        context = Context()
        context.set("pattern", None)

        schema = DynamicPattern["pattern"].generate_schema(context)

        # Should return basic string schema
        assert schema == {"type": "string"}

    def test_dynamic_pattern_missing_key(self):
        """Test DynamicPattern when key doesn't exist."""
        context = Context()

        schema = DynamicPattern["missing"].generate_schema(context)

        # Should return basic string schema
        assert schema == {"type": "string"}


class TestDynamicConstraintsEdgeCases:
    """Test DynamicConstraints edge cases."""

    def test_dynamic_constraints_with_dict_value(self):
        """Test DynamicConstraints when value is dict (not nested in 'schema' key)."""
        context = Context()
        # Direct schema without nesting
        context.set(
            "direct_schema",
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        )

        schema = DynamicConstraints["direct_schema"].generate_schema(context)

        assert schema["type"] == "object"
        assert "properties" in schema

    def test_dynamic_constraints_with_none(self):
        """Test DynamicConstraints when value is None."""
        context = Context()
        context.set("schema", None)

        schema = DynamicConstraints["schema"].generate_schema(context)

        # Returns empty dict when value is None
        assert schema == {}


class TestToolkitExecuteParallelEdgeCases:
    """Test additional execute_parallel edge cases."""

    def test_execute_parallel_empty_list(self):
        """Test execute_parallel with empty tool calls list."""
        toolkit = Toolkit()

        results = toolkit.execute_parallel([])

        assert results == []

    def test_execute_parallel_single_tool(self):
        """Test execute_parallel with single tool."""
        toolkit = Toolkit()

        @tool()
        def single() -> Tuple[str, dict]:
            return "result", {"key": "value"}

        toolkit.register_tool(single)

        results = toolkit.execute_parallel([{"name": "single", "arguments": {}}])

        assert len(results) == 1
        assert results[0]["success"]


class TestToolkitCopyWithFilters:
    """Test toolkit copy with custom filters."""

    def test_copy_includes_filters(self):
        """Test that toolkit.copy() includes custom filters."""
        toolkit = Toolkit()

        # Register custom filter
        toolkit.register_filter("custom", lambda x: x)

        # Copy toolkit
        copy = toolkit.copy()

        # Should include filter
        assert "custom" in copy.filters


class TestAgentToolkitPrefixEdgeCases:
    """Test agent toolkit prefix handling."""

    def test_agent_with_empty_independent_toolkit(self):
        """Test agent with no independent tools."""
        from unittest.mock import Mock

        from bestla.yggdrasil import Agent

        agent = Agent(provider=Mock(), model="gpt-4")

        # Should have independent toolkit even if empty
        assert agent.independent_toolkit is not None
        assert len(agent.independent_toolkit.tools) == 0

    def test_agent_toolkit_prefixes_updated(self):
        """Test toolkit_prefixes dict is updated when adding toolkit."""
        from unittest.mock import Mock

        from bestla.yggdrasil import Agent

        agent = Agent(provider=Mock(), model="gpt-4")

        toolkit = Toolkit()
        toolkit.add_tool("tool1", lambda: ("r", {}))
        toolkit.add_tool("tool2", lambda: ("r", {}))

        agent.add_toolkit("myprefix", toolkit)

        # Should have prefixes for both tools
        assert "myprefix::tool1" in agent.toolkit_prefixes
        assert "myprefix::tool2" in agent.toolkit_prefixes
        assert agent.toolkit_prefixes["myprefix::tool1"] == "myprefix"
        assert agent.toolkit_prefixes["myprefix::tool2"] == "myprefix"


class TestContextEdgeCases:
    """Test additional context edge cases."""

    def test_context_get_with_integer_key(self):
        """Test context.get with integer key."""
        context = Context()

        # Set with integer key
        context.set(123, "value")

        # Get with integer key
        assert context.get(123) == "value"

    def test_context_nested_with_non_string_key(self):
        """Test nested access only works with string keys containing dots."""
        context = Context()

        # Set with integer key (no dots, so no nested access)
        context.set(123, {"nested": "value"})

        # Integer keys don't support nested access
        result = context.get(123)
        assert result == {"nested": "value"}


class TestToolDecoratorEdgeCases:
    """Test tool decorator edge cases."""

    def test_tool_with_all_metadata(self):
        """Test tool with all possible metadata."""

        @tool(
            required_context=["ctx1", "ctx2"],
            required_states=["state1", "state2"],
            forbidden_states=["banned"],
            enables_states=["enabled1", "enabled2"],
            disables_states=["disabled1"],
        )
        def complex_tool() -> Tuple[str, dict]:
            """Complex tool with all metadata."""
            return "result", {}

        # Verify all metadata is stored
        assert complex_tool.required_context == ["ctx1", "ctx2"]
        # States are converted to sets internally
        assert set(complex_tool.required_states) == {"state1", "state2"}
        assert set(complex_tool.forbidden_states) == {"banned"}
        assert set(complex_tool.enables_states) == {"enabled1", "enabled2"}
        assert set(complex_tool.disables_states) == {"disabled1"}

    def test_tool_has_name_attribute(self):
        """Test tool has name attribute."""

        @tool()
        def my_tool(param: str) -> Tuple[str, dict]:
            """This is a docstring."""
            return param, {}

        # Tool object has .name attribute (not __name__)
        assert my_tool.name == "my_tool"
        assert my_tool.description == "This is a docstring."
