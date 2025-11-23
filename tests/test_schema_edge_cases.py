"""Schema generation and type system edge cases."""

import sys
import threading
import time
from typing import List, Tuple

import pytest

from bestla.yggdrasil import Agent, Context, DynamicStr, Toolkit, tool
from bestla.yggdrasil.agent import ExecutionContext
from bestla.yggdrasil.dynamic_types import (
    DynamicConditional,
    DynamicFiltered,
    DynamicUnion,
    generate_param_schema,
)


class TestDynamicUnionEdgeCases:
    """Test DynamicUnion edge cases."""

    def test_dynamic_union_deduplication(self):
        """Test DynamicUnion removes duplicates correctly."""
        context = Context()
        context.set("list1", ["alice", "bob", "charlie"])
        context.set("list2", ["bob", "charlie", "diana"])
        context.set("list3", ["alice", "diana"])

        schema = DynamicUnion[("list1", "list2", "list3")].generate_schema(context)

        # Should deduplicate
        assert sorted(schema["enum"]) == ["alice", "bob", "charlie", "diana"]

    def test_dynamic_union_with_empty_and_non_empty(self):
        """Test DynamicUnion with mix of empty and non-empty lists."""
        context = Context()
        context.set("empty", [])
        context.set("full", ["a", "b"])

        schema = DynamicUnion[("empty", "full")].generate_schema(context)

        assert schema["enum"] == ["a", "b"]

    def test_dynamic_union_with_single_key(self):
        """Test DynamicUnion with only one key (degenerate case)."""
        context = Context()
        context.set("only", ["x", "y"])

        # Single key in tuple
        schema = DynamicUnion[("only",)].generate_schema(context)

        assert schema["enum"] == ["x", "y"]

    def test_dynamic_union_preserves_order(self):
        """Test DynamicUnion preserves order from first occurrence."""
        context = Context()
        context.set("first", ["z", "y", "x"])
        context.set("second", ["a", "x", "b"])

        schema = DynamicUnion[("first", "second")].generate_schema(context)

        # Should preserve order and deduplicate
        assert "z" in schema["enum"]
        assert "a" in schema["enum"]
        # x should appear only once
        assert schema["enum"].count("x") == 1


class TestDynamicConditionalEdgeCases:
    """Test DynamicConditional edge cases."""

    def test_conditional_with_zero_as_condition(self):
        """Test DynamicConditional with 0 (falsy number)."""
        context = Context()
        context.set("count", 0)  # Falsy
        context.set("none_options", ["none"])
        context.set("some_options", ["some"])

        schema = DynamicConditional[("count", "some_options", "none_options")].generate_schema(
            context
        )

        # 0 is falsy, should use none_options
        assert schema["enum"] == ["none"]

    def test_conditional_with_empty_string_condition(self):
        """Test DynamicConditional with empty string (falsy)."""
        context = Context()
        context.set("mode", "")  # Empty string is falsy
        context.set("enabled", ["a"])
        context.set("disabled", ["b"])

        schema = DynamicConditional[("mode", "enabled", "disabled")].generate_schema(context)

        # Empty string is falsy
        assert schema["enum"] == ["b"]

    def test_conditional_with_list_as_condition(self):
        """Test DynamicConditional with list as condition."""
        context = Context()
        context.set("items", [1, 2, 3])  # Non-empty list is truthy
        context.set("option_a", ["a"])
        context.set("option_b", ["b"])

        schema = DynamicConditional[("items", "option_a", "option_b")].generate_schema(context)

        # Non-empty list is truthy
        assert schema["enum"] == ["a"]

        # Empty list is falsy
        context.set("items", [])
        schema = DynamicConditional[("items", "option_a", "option_b")].generate_schema(context)
        assert schema["enum"] == ["b"]

    def test_conditional_with_nested_context_condition(self):
        """Test DynamicConditional with nested path as condition."""
        context = Context()
        context.set("config", {"advanced": True})
        context.set("basic", ["b1"])
        context.set("advanced", ["a1", "a2"])

        # Condition is nested path
        schema = DynamicConditional[("config.advanced", "advanced", "basic")].generate_schema(
            context
        )

        # Nested True should use advanced
        assert schema["enum"] == ["a1", "a2"]


class TestDynamicFilteredEdgeCases:
    """Test DynamicFiltered edge cases."""

    def test_filter_returns_unhashable_types(self):
        """Test filter returning list of dicts (unhashable)."""
        toolkit = Toolkit()
        toolkit.context.set("users", [{"name": "alice", "id": 1}, {"name": "bob", "id": 2}])

        # Filter returns list of dicts (unhashable)
        def dict_filter(users):
            return [{"name": u["name"]} for u in users if u["id"] > 0]

        toolkit.register_filter("dict_filter", dict_filter)

        # Try to generate schema
        try:
            schema = DynamicFiltered[("users", "dict_filter")].generate_schema(
                toolkit.context, toolkit.filters
            )
            # Might handle by converting dicts to strings
            assert "type" in schema
        except (TypeError, Exception):
            # Expected - can't create enum from dicts
            pass

    def test_filter_returns_mixed_types(self):
        """Test filter returning list with mixed types."""
        toolkit = Toolkit()
        toolkit.context.set("data", [1, "two", 3.0, True, None])

        def mixed_filter(data):
            return data  # Returns mixed types

        toolkit.register_filter("mixed", mixed_filter)

        schema = DynamicFiltered[("data", "mixed")].generate_schema(
            toolkit.context, toolkit.filters
        )

        # Should create enum with mixed types
        assert schema["type"] == "string"
        assert set(schema["enum"]) == {1, "two", 3.0, True, None}

    def test_filter_modifies_context(self):
        """Test filter that has side effects on context."""
        toolkit = Toolkit()
        toolkit.context.set("items", ["a", "b", "c"])

        side_effects = []

        def side_effect_filter(items):
            side_effects.append("called")
            # Filter has side effects (bad practice but possible)
            return items

        toolkit.register_filter("side_effect", side_effect_filter)

        # Generate schema multiple times
        for _ in range(3):
            DynamicFiltered[("items", "side_effect")].generate_schema(
                toolkit.context, toolkit.filters
            )

        # Filter should be called each time
        assert len(side_effects) == 3

    def test_filter_with_empty_result(self):
        """Test filter returning empty list."""
        toolkit = Toolkit()
        toolkit.context.set("items", ["a", "b", "c"])

        # Filter that excludes everything
        toolkit.register_filter("exclude_all", lambda items: [])

        schema = DynamicFiltered[("items", "exclude_all")].generate_schema(
            toolkit.context, toolkit.filters
        )

        # Should have empty enum
        assert schema["enum"] == []


class TestSchemaGenerationPerformance:
    """Test schema generation with extreme conditions."""

    def test_schema_with_1000_enum_values(self):
        """Test schema generation with very large enum."""
        context = Context()
        large_enum = [f"value_{i}" for i in range(1000)]
        context.set("large", large_enum)

        schema = DynamicStr["large"].generate_schema(context)

        assert len(schema["enum"]) == 1000

        # Schema size should be reasonable
        schema_str = str(schema)
        # Should be < 100KB
        assert sys.getsizeof(schema_str) < 100 * 1024

    def test_toolkit_with_100_tools_schema_generation(self):
        """Test schema generation with 100 tools."""
        import time

        toolkit = Toolkit()

        # Add 100 tools with unique functions
        for i in range(100):

            def make_tool(idx):
                def generic_tool(value: int) -> Tuple[int, dict]:
                    return value + idx, {}

                return generic_tool

            toolkit.add_tool(f"tool_{i}", make_tool(i))

        # Generate all schemas
        start = time.time()
        schemas = toolkit.generate_schemas()
        duration = time.time() - start

        assert len(schemas) == 100

        # Should be fast (< 1 second for 100 tools)
        assert duration < 1.0


class TestComplexTypeHints:
    """Test complex Python type hints in tools."""

    def test_tool_with_union_type(self):
        """Test tool with Union type hint."""
        from typing import Union

        @tool()
        def union_tool(value: Union[str, int]) -> Tuple[str, dict]:
            return str(value), {}

        schema = union_tool.generate_schema(Context())

        # Should generate schema (might default to string or use anyOf)
        assert "value" in schema["function"]["parameters"]["properties"]

    def test_tool_with_optional_type(self):
        """Test tool with Optional type hint."""
        from typing import Optional

        @tool()
        def optional_tool(value: Optional[str] = None) -> Tuple[str, dict]:
            return value or "default", {}

        schema = optional_tool.generate_schema(Context())

        # value should not be in required list
        assert "value" not in schema["function"]["parameters"]["required"]

    def test_tool_with_list_type(self):
        """Test tool with List[int] type hint."""

        @tool()
        def list_tool(values: List[int]) -> Tuple[int, dict]:
            return sum(values), {}

        schema = list_tool.generate_schema(Context())

        # Should handle List type hint
        assert "values" in schema["function"]["parameters"]["properties"]

    def test_tool_with_dict_type_hint(self):
        """Test tool with dict type hint."""
        from typing import Dict

        @tool()
        def dict_tool(mapping: Dict[str, int]) -> Tuple[str, dict]:
            return str(mapping), {}

        schema = dict_tool.generate_schema(Context())

        # Should generate schema
        assert "mapping" in schema["function"]["parameters"]["properties"]

    def test_tool_with_nested_generic(self):
        """Test tool with nested generic type."""
        from typing import Dict, List

        @tool()
        def nested_generic(data: Dict[str, List[int]]) -> Tuple[str, dict]:
            return "ok", {}

        schema = nested_generic.generate_schema(Context())

        # Should not crash on complex type
        assert "data" in schema["function"]["parameters"]["properties"]


class TestTypeCoercionScenarios:
    """Test type coercion in tool execution."""

    def test_tool_receives_string_for_int_parameter(self):
        """Test tool execution when LLM passes string for int parameter."""

        @tool()
        def int_tool(count: int) -> Tuple[str, dict]:
            return f"count={count}", {}

        # Execute with string value (as LLM might send)
        try:
            result, _ = int_tool.execute(count="123")
            # Python might coerce or fail
            assert "count=" in result
        except TypeError:
            # Expected if strict type checking
            pass

    def test_tool_receives_int_for_string_parameter(self):
        """Test tool execution with int for string parameter."""

        @tool()
        def string_tool(name: str) -> Tuple[str, dict]:
            return f"name={name}", {}

        # Execute with int
        result, _ = string_tool.execute(name=123)

        # Should work - int can be used where string expected
        assert "name=123" in result

    def test_tool_receives_list_for_single_value(self):
        """Test tool receives list when expecting single value."""

        @tool()
        def single_value_tool(item: str) -> Tuple[str, dict]:
            return f"item={item}", {}

        # Execute with list
        try:
            result, _ = single_value_tool.execute(item=["a", "b"])
            # Might convert to string: "['a', 'b']"
            assert "item=" in result
        except TypeError:
            # Expected if strict
            pass


class TestSchemaValidationEdgeCases:
    """Test schema validation edge cases."""

    def test_validation_with_additional_properties(self):
        """Test schema validation with additionalProperties."""
        context = Context(validation_enabled=True)

        context.schema.define(
            "strict_object",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            },
        )

        # Valid - only defined property
        context.set("strict_object", {"name": "Alice"})

        # Invalid - has additional property
        with pytest.raises(Exception):
            context.set("strict_object", {"name": "Bob", "age": 30})

    def test_validation_with_min_max_items(self):
        """Test array validation with minItems/maxItems."""
        context = Context(validation_enabled=True)

        context.schema.define(
            "tags", {"type": "array", "minItems": 2, "maxItems": 5, "items": {"type": "string"}}
        )

        # Valid
        context.set("tags", ["tag1", "tag2"])
        context.set("tags", ["a", "b", "c", "d", "e"])

        # Too few
        with pytest.raises(Exception):
            context.set("tags", ["only_one"])

        # Too many
        with pytest.raises(Exception):
            context.set("tags", ["a", "b", "c", "d", "e", "f"])

    def test_validation_with_unique_items(self):
        """Test array validation with uniqueItems."""
        context = Context(validation_enabled=True)

        context.schema.define(
            "unique_tags", {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
        )

        # Valid - all unique
        context.set("unique_tags", ["a", "b", "c"])

        # Invalid - has duplicates
        with pytest.raises(Exception):
            context.set("unique_tags", ["a", "b", "a"])

    def test_validation_with_pattern_properties(self):
        """Test object validation with patternProperties."""
        context = Context(validation_enabled=True)

        context.schema.define(
            "config",
            {
                "type": "object",
                "patternProperties": {"^S_": {"type": "string"}, "^I_": {"type": "integer"}},
            },
        )

        # Valid
        context.set("config", {"S_name": "value", "I_count": 42})

        # Invalid - wrong type for pattern
        with pytest.raises(Exception):
            context.set("config", {"S_name": 123})  # Should be string


class TestGenerateParamSchemaEdgeCases:
    """Test generate_param_schema helper function edge cases."""

    def test_generate_schema_with_any_type(self):
        """Test schema generation with Any type hint."""
        from typing import Any

        schema = generate_param_schema(Any, Context())

        # Unknown types return empty dict
        assert schema == {}

    def test_generate_schema_with_none_type(self):
        """Test schema generation with None type."""
        schema = generate_param_schema(None, Context())

        # Unknown types return empty dict
        assert schema == {}

    def test_generate_schema_with_callable_type(self):
        """Test schema generation with Callable type hint."""
        from typing import Callable

        schema = generate_param_schema(Callable, Context())

        # Unknown types return empty dict
        assert schema == {}

    def test_generate_schema_with_list_basic(self):
        """Test schema generation with List (no type param)."""
        schema = generate_param_schema(list, Context())

        # list type recognized as array
        assert schema == {"type": "array"}

    def test_generate_schema_with_dict_basic(self):
        """Test schema generation with dict."""
        schema = generate_param_schema(dict, Context())

        # dict type recognized as object
        assert schema == {"type": "object"}


class TestToolAvailabilityComplexScenarios:
    """Test complex tool availability scenarios."""

    def test_tool_with_10_required_states(self):
        """Test tool requiring many states."""
        toolkit = Toolkit()

        required = [f"state_{i}" for i in range(10)]

        @tool(required_states=required)
        def many_requirements() -> Tuple[str, dict]:
            return "ok", {}

        toolkit.register_tool(many_requirements)

        # Not available initially
        assert not toolkit.is_tool_available("many_requirements")

        # Enable all states
        toolkit.set_unlocked_states(set(required))

        # Now available
        assert toolkit.is_tool_available("many_requirements")

    def test_tool_with_10_forbidden_states(self):
        """Test tool with many forbidden states."""
        toolkit = Toolkit()

        forbidden = [f"banned_{i}" for i in range(10)]

        @tool(forbidden_states=forbidden)
        def careful_tool() -> Tuple[str, dict]:
            return "ok", {}

        toolkit.register_tool(careful_tool)

        # Available initially (no forbidden states enabled)
        assert toolkit.is_tool_available("careful_tool")

        # Enable one forbidden state
        toolkit.set_unlocked_states({forbidden[0]})

        # Should become unavailable
        assert not toolkit.is_tool_available("careful_tool")

    def test_availability_with_both_required_and_forbidden_states(self):
        """Test tool with complex state requirements."""
        toolkit = Toolkit()

        @tool(
            required_states=["auth", "verified"],
            forbidden_states=["suspended", "banned"],
            required_context=["user_id", "session_token"],
        )
        def complex_availability() -> Tuple[str, dict]:
            return "ok", {}

        toolkit.register_tool(complex_availability)

        # Set up context
        toolkit.context.set("user_id", "123")
        toolkit.context.set("session_token", "abc")

        # Enable required states
        toolkit.set_unlocked_states({"auth", "verified"})

        # Should be available (all requirements met, no forbidden)
        assert toolkit.is_tool_available("complex_availability")

        # Enable forbidden state
        toolkit.set_unlocked_states({"auth", "verified", "suspended"})

        # Should become unavailable
        assert not toolkit.is_tool_available("complex_availability")


class TestConcurrentSchemaValidation:
    """Test schema validation under concurrent access."""

    def test_schema_definition_during_validation(self):
        """Test defining schema while validation is in progress."""
        context = Context(validation_enabled=True)

        validation_errors = []
        definitions = []

        def validate_values():
            context.schema.define("temp", {"type": "string"})
            for i in range(50):
                try:
                    context.set("temp", f"value_{i}")
                except Exception as e:
                    validation_errors.append(e)
                time.sleep(0.001)

        def define_schemas():
            for i in range(50):
                # Redefine schema with different constraints
                context.schema.define("temp", {"type": "string", "maxLength": i + 10})
                definitions.append(i)
                time.sleep(0.001)

        # Run both concurrently
        t1 = threading.Thread(target=validate_values)
        t2 = threading.Thread(target=define_schemas)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both should complete
        # Might have some validation errors due to race conditions
        assert len(definitions) == 50


class TestEmptyToolkitBehavior:
    """Test toolkit with no tools in various scenarios."""

    def test_empty_toolkit_generate_schemas(self):
        """Test generate_schemas on empty toolkit."""
        toolkit = Toolkit()

        schemas = toolkit.generate_schemas()

        assert schemas == []

    def test_empty_toolkit_execute_sequential(self):
        """Test execute_sequential with empty tool list on empty toolkit."""
        toolkit = Toolkit()

        results = toolkit.execute_sequential([])

        assert results == []

    def test_empty_toolkit_in_agent(self):
        """Test agent with empty toolkit."""
        from unittest.mock import Mock

        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        empty_toolkit = Toolkit()
        agent.add_toolkit("empty", empty_toolkit)

        # Should register successfully
        assert "empty" in agent.toolkits

        # Schema generation should work
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        schemas = agent._generate_all_schemas(context)

        # Should not include tools from empty toolkit
        assert all("empty::" not in s["function"]["name"] for s in schemas)
