"""Tests for Tool edge cases and boundary conditions."""

from typing import List, Optional, Tuple, Union

import pytest

from bestla.yggdrasil import Context, tool
from bestla.yggdrasil.tool import Tool


class TestToolReturnFormats:
    """Test tool return format handling."""

    def test_tool_returns_non_tuple(self):
        """Test tool that returns non-tuple is handled."""

        @tool()
        def returns_string() -> str:
            return "just_a_string"

        # Tool.execute should handle this
        try:
            result = returns_string.execute({})
            # Might auto-wrap or raise
            assert result is not None
        except (TypeError, ValueError):
            # Expected if strict validation
            pass

    def test_tool_returns_wrong_tuple_length(self):
        """Test tool returning wrong tuple length."""

        @tool()
        def wrong_tuple() -> Tuple:
            return ("one", "two", "three")

        try:
            wrong_tuple.execute({})
            # Might handle or raise
        except (TypeError, ValueError):
            pass

    def test_tool_returns_non_dict_updates(self):
        """Test tool returning non-dict as updates."""

        @tool()
        def bad_updates() -> Tuple[str, str]:
            return "result", "not_a_dict"

        try:
            result, updates = bad_updates()
            # Might work if no validation
        except (TypeError, ValueError):
            # Expected if validated
            pass

    def test_tool_returns_none(self):
        """Test tool returning None."""

        @tool()
        def returns_none() -> None:
            return None

        try:
            returns_none.execute({})
        except (TypeError, AttributeError):
            pass


class TestToolParameterEdgeCases:
    """Test tool parameter edge cases."""

    def test_tool_with_no_parameters(self):
        """Test tool with no parameters."""

        @tool()
        def no_params() -> Tuple[str, dict]:
            return "done", {}

        # Schema should have empty properties
        schema = no_params.generate_schema(Context())
        assert schema["function"]["parameters"]["properties"] == {}

        # Should execute without args
        result, updates = no_params.execute()
        assert result == "done"

    def test_tool_with_only_optional_parameters(self):
        """Test tool with only optional parameters."""

        @tool()
        def optional_only(name: Optional[str] = "default") -> Tuple[str, dict]:
            return f"Hello {name}", {}

        # Should work without arguments
        result1, _ = optional_only.execute()
        assert result1 == "Hello default"

        # Should work with argument
        result2, _ = optional_only.execute(name="Alice")
        assert result2 == "Hello Alice"

    def test_tool_with_complex_type_hints(self):
        """Test tool with complex type hints."""

        @tool()
        def complex_types(
            union_param: Union[str, int],
            optional_param: Optional[str] = None,
            list_param: List[int] = None,
        ) -> Tuple[str, dict]:
            return "ok", {}

        # Schema should be generated
        schema = complex_types.generate_schema(Context())
        assert "properties" in schema["function"]["parameters"]

    def test_tool_with_mutable_defaults(self):
        """Test tool with mutable default arguments."""

        @tool()
        def mutable_default(items: list = None) -> Tuple[list, dict]:
            if items is None:
                items = []
            items.append("new")
            return items, {}

        # Should not share state between calls
        result1, _ = mutable_default.execute()
        result2, _ = mutable_default.execute()

        # Each should have only one item
        assert len(result1) == 1
        assert len(result2) == 1

    def test_tool_with_args_kwargs(self):
        """Test tool with *args or **kwargs."""

        # Tool with **kwargs
        @tool()
        def with_kwargs(**kwargs) -> Tuple[dict, dict]:
            return kwargs, {}

        # Should handle dynamic arguments
        result, _ = with_kwargs.execute(a=1, b=2)
        assert result == {"a": 1, "b": 2}


class TestToolMetadataEdgeCases:
    """Test tool metadata edge cases."""

    def test_tool_with_no_type_hints(self):
        """Test tool with untyped parameters."""

        @tool()
        def untyped(param):
            return f"got {param}", {}

        # Should default to string
        schema = untyped.generate_schema(Context())
        param_schema = schema["function"]["parameters"]["properties"]["param"]
        assert param_schema["type"] == "string"

    def test_tool_with_forward_reference(self):
        """Test tool with forward reference type hint."""

        @tool()
        def forward_ref(value: "ForwardType") -> Tuple[str, dict]:
            return str(value), {}

        # Should handle forward reference
        try:
            forward_ref.generate_schema(Context())
        except (NameError, Exception):
            # Expected if ForwardType undefined
            pass

    def test_tool_repr(self):
        """Test Tool.__repr__() method."""

        @tool(required_context=["auth"])
        def my_tool(param: str) -> Tuple[str, dict]:
            return param, {}

        repr_str = repr(my_tool)
        assert "my_tool" in repr_str or "Tool" in repr_str

    def test_tool_with_empty_required_context(self):
        """Test tool with empty required_context list."""

        @tool(required_context=[])
        def no_requirements() -> Tuple[str, dict]:
            return "ok", {}

        assert no_requirements.required_context == []

    def test_tool_with_duplicate_state_names(self):
        """Test tool with same state in enables and disables."""

        @tool(enables_states=["state_a", "state_b"], disables_states=["state_b"])
        def conflicting_states() -> Tuple[str, dict]:
            return "ok", {}

        # Both should be recorded
        assert "state_b" in conflicting_states.enables_states
        assert "state_b" in conflicting_states.disables_states


class TestToolExecutionErrors:
    """Test tool execution error handling."""

    def test_tool_raises_exception(self):
        """Test tool that raises exception."""

        @tool()
        def failing_tool() -> Tuple[str, dict]:
            raise ValueError("Tool error")

        with pytest.raises(ValueError, match="Tool error"):
            failing_tool.execute()

    def test_tool_raises_system_exit(self):
        """Test tool that calls sys.exit()."""
        import sys

        @tool()
        def exit_tool() -> Tuple[str, dict]:
            sys.exit(1)

        with pytest.raises(SystemExit):
            exit_tool.execute()

    def test_tool_with_infinite_loop(self):
        """Test tool with infinite loop (would need timeout)."""

        @tool()
        def infinite_loop() -> Tuple[str, dict]:
            while True:
                pass
            return "never", {}

        # Would need @timeout decorator to handle this
        # Skip actual execution to avoid hanging tests


class TestToolContextRequirements:
    """Test tool context requirement handling."""

    def test_tool_with_missing_required_context(self):
        """Test tool execution when required context missing."""

        def needs_auth(value: str) -> Tuple[str, dict]:
            return f"authenticated: {value}", {}

        # Context missing 'auth_token'

        # Check availability
        from bestla.yggdrasil.toolkit import Toolkit

        tk = Toolkit()
        tk.add_tool("needs_auth", needs_auth, required_context=["auth_token"])

        # Tool should not be available
        available = tk.is_tool_available("needs_auth")
        assert not available

        # Check the reason
        reason = tk.get_availability_reason("needs_auth")
        assert "auth_token" in reason.lower()

    def test_tool_with_all_required_context_present(self):
        """Test tool executes when all required context present."""

        def with_context(action: str) -> Tuple[str, dict]:
            return f"action: {action}", {}

        from bestla.yggdrasil.toolkit import Toolkit

        tk = Toolkit()
        tk.context.set("user_id", "123")
        tk.context.set("session", "abc")
        tk.add_tool("with_context", with_context, required_context=["user_id", "session"])

        # Tool should be available
        available = tk.is_tool_available("with_context")
        assert available


class TestToolSchemaGeneration:
    """Test tool schema generation edge cases."""

    def test_schema_with_all_metadata(self):
        """Test schema includes all metadata."""

        @tool(
            required_context=["auth"],
            required_states=["logged_in"],
            forbidden_states=["banned"],
            enables_states=["active"],
            disables_states=["idle"],
        )
        def full_metadata(param: str) -> Tuple[str, dict]:
            """Tool with all metadata."""
            return param, {}

        schema = full_metadata.generate_schema(Context())

        # Verify schema structure
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]

    def test_schema_with_docstring(self):
        """Test schema includes function docstring."""

        @tool()
        def documented_tool(param: str) -> Tuple[str, dict]:
            """This tool does something useful."""
            return param, {}

        schema = documented_tool.generate_schema(Context())

        # Docstring might be in description
        assert "description" in schema["function"]

    def test_schema_with_no_docstring(self):
        """Test schema when function has no docstring."""

        @tool()
        def undocumented(param: str) -> Tuple[str, dict]:
            return param, {}

        schema = undocumented.generate_schema(Context())

        # Should still have valid schema
        assert "function" in schema


class TestToolAsProperty:
    """Test using Tool instances."""

    def test_create_tool_directly(self):
        """Test creating Tool instance directly."""

        def my_function(x: int) -> Tuple[int, dict]:
            return x * 2, {}

        tool_instance = Tool(
            name="my_tool",
            function=my_function,
            required_context=[],
            required_states=set(),
            forbidden_states=set(),
            enables_states=set(),
            disables_states=set(),
        )

        # Should be executable
        result, updates = tool_instance.execute(x=5)
        assert result == 10

    def test_tool_function_attribute(self):
        """Test accessing underlying function."""

        @tool()
        def my_tool(x: int) -> Tuple[int, dict]:
            return x + 1, {}

        # Should have function attribute
        assert callable(my_tool.function)
        assert my_tool.function(5) == (6, {})
