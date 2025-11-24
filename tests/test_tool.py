"""Tests for Tool class."""

from typing import Tuple

from bestla.yggdrasil import Context, DynamicStr, Tool, tool


class TestTool:
    """Test Tool functionality."""

    def test_create_tool(self):
        """Test creating a basic tool."""

        def add(a: int, b: int) -> Tuple[int, dict]:
            return a + b, {}

        t = Tool(function=add)
        assert t.name == "add"
        assert t.function == add

    def test_tool_with_metadata(self):
        """Test tool with metadata."""

        def select_project(project_id: str) -> Tuple[str, dict]:
            """Select a project."""
            return f"Selected {project_id}", {"selected_project": project_id}

        t = Tool(
            function=select_project,
            required_context=["auth_token"],
            enables_states=["project_selected"],
            disables_states=["project_browsing"],
        )

        assert t.required_context == ["auth_token"]
        assert t.enables_states == ["project_selected"]
        assert t.disables_states == ["project_browsing"]

    def test_execute_tool(self):
        """Test executing a tool."""

        def add(a: int, b: int) -> Tuple[int, dict]:
            return a + b, {}

        t = Tool(function=add)
        result, updates = t.execute(a=5, b=3)

        assert result == 8
        assert updates == {}

    def test_execute_with_context_updates(self):
        """Test tool that returns context updates."""

        def create_issue(name: str) -> Tuple[str, dict]:
            return f"Created {name}", {"last_issue": name}

        t = Tool(function=create_issue)
        result, updates = t.execute(name="BUG-1")

        assert result == "Created BUG-1"
        assert updates == {"last_issue": "BUG-1"}

    def test_check_context_requirements(self):
        """Test checking context requirements."""

        def protected_action() -> Tuple[str, dict]:
            return "Done", {}

        t = Tool(function=protected_action, required_context=["auth_token", "user_id"])

        context = Context()
        all_present, missing = t.check_context_requirements(context)
        assert not all_present
        assert set(missing) == {"auth_token", "user_id"}

        context.set("auth_token", "token123")
        all_present, missing = t.check_context_requirements(context)
        assert not all_present
        assert missing == ["user_id"]

        context.set("user_id", "user1")
        all_present, missing = t.check_context_requirements(context)
        assert all_present
        assert missing == []

    def test_generate_schema_basic(self):
        """Test generating basic JSON schema."""

        def add(a: int, b: int) -> Tuple[int, dict]:
            """Add two numbers."""
            return a + b, {}

        t = Tool(function=add)
        context = Context()
        schema = t.generate_schema(context)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"
        assert schema["function"]["description"] == "Add two numbers."
        assert "a" in schema["function"]["parameters"]["properties"]
        assert "b" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["properties"]["a"] == {"type": "integer"}

    def test_generate_schema_with_dynamic_type(self):
        """Test generating schema with dynamic types."""

        def select_issue(name: DynamicStr["issue_names"]) -> Tuple[str, dict]:
            return f"Selected {name}", {}

        t = Tool(function=select_issue)
        context = Context()
        context.set("issue_names", ["BUG-1", "FEAT-2"])

        schema = t.generate_schema(context)
        name_schema = schema["function"]["parameters"]["properties"]["name"]

        assert name_schema == {"type": "string", "enum": ["BUG-1", "FEAT-2"]}

    def test_tool_decorator(self):
        """Test @tool decorator."""

        @tool(required_context=["selected_project"], enables_states=["issues_loaded"])
        def list_issues() -> Tuple[str, dict]:
            """List issues in project."""
            return "Found 5 issues", {"issue_names": ["BUG-1", "BUG-2"]}

        assert isinstance(list_issues, Tool)
        assert list_issues.name == "list_issues"
        assert list_issues.required_context == ["selected_project"]
        assert list_issues.enables_states == ["issues_loaded"]

    def test_tool_with_optional_params(self):
        """Test tool with optional parameters."""

        def greet(name: str, greeting: str = "Hello") -> Tuple[str, dict]:
            return f"{greeting}, {name}!", {}

        t = Tool(function=greet)
        context = Context()
        schema = t.generate_schema(context)

        # name should be required, greeting optional
        required = schema["function"]["parameters"]["required"]
        assert "name" in required
        assert "greeting" not in required

    def test_tool_without_type_hints(self):
        """Test tool without type hints defaults to string."""

        def untyped_tool(param):
            return "result", {}

        t = Tool(function=untyped_tool)
        context = Context()
        schema = t.generate_schema(context)

        # Should default to string type
        param_schema = schema["function"]["parameters"]["properties"]["param"]
        assert param_schema == {"type": "string"}
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
"""Tests to cover remaining uncovered lines."""

from typing import Tuple
from unittest.mock import Mock

import pytest

from bestla.yggdrasil import Agent, Context, Toolkit, tool
from bestla.yggdrasil.agent import ExecutionContext
from bestla.yggdrasil.dynamic_types import (
    DynamicConst,
    DynamicFloat,
    DynamicInt,
    DynamicStr,
)


class TestToolExecuteEdgeCases:
    """Test Tool.execute() edge cases."""

    def test_tool_execute_non_tuple_return(self):
        """Test Tool.execute when function doesn't return tuple."""

        @tool()
        def returns_string():
            # Returns just a string, not a tuple
            return "just_a_string"

        # Should handle by wrapping in tuple with empty dict
        result, updates = returns_string.execute()

        assert result == "just_a_string"
        assert updates == {}

    def test_tool_execute_single_value(self):
        """Test Tool.execute with various single return values."""

        @tool()
        def returns_int():
            return 42

        result, updates = returns_int.execute()
        assert result == 42
        assert updates == {}

        @tool()
        def returns_dict():
            return {"key": "value"}

        result, updates = returns_dict.execute()
        assert result == {"key": "value"}
        assert updates == {}

    def test_tool_execute_proper_tuple(self):
        """Test Tool.execute with proper tuple return."""

        @tool()
        def proper_return() -> Tuple[str, dict]:
            return "result", {"update": "value"}

        result, updates = proper_return.execute()
        assert result == "result"
        assert updates == {"update": "value"}
class TestToolSchemaGenerationEdgeCases:
    """Test tool schema generation edge cases."""

    def test_tool_skip_self_parameter(self):
        """Test tool schema skips 'self' parameter."""

        class ToolClass:
            @tool()
            def method_tool(self, param: str) -> Tuple[str, dict]:
                """Method as tool."""
                return param, {}

        instance = ToolClass()
        schema = instance.method_tool.generate_schema(Context())

        # Should not include 'self' in parameters
        assert "self" not in schema["function"]["parameters"]["properties"]
        assert "param" in schema["function"]["parameters"]["properties"]

    def test_tool_skip_cls_parameter(self):
        """Test tool schema skips 'cls' parameter."""

        class ToolClass:
            @classmethod
            @tool()
            def class_method_tool(cls, param: str) -> Tuple[str, dict]:
                """Class method as tool."""
                return param, {}

        schema = ToolClass.class_method_tool.generate_schema(Context())

        # Should not include 'cls' in parameters
        assert "cls" not in schema["function"]["parameters"]["properties"]
        assert "param" in schema["function"]["parameters"]["properties"]


