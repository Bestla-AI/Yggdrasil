"""Tests for Toolkit class."""

from typing import Tuple

import pytest

from bestla.yggdrasil import Context, DynamicStr, Toolkit, tool
from bestla.yggdrasil.exceptions import ToolkitPipelineError


class TestToolkit:
    """Test Toolkit functionality."""

    def test_create_toolkit(self):
        """Test creating a toolkit."""
        toolkit = Toolkit()
        assert toolkit is not None
        assert len(toolkit.tools) == 0
        assert isinstance(toolkit.context, Context)

    def test_add_tool(self):
        """Test adding a tool to toolkit."""
        toolkit = Toolkit()

        def test_func(x: int) -> Tuple[int, dict]:
            return x * 2, {}

        toolkit.add_tool("double", test_func)

        assert "double" in toolkit.tools
        assert toolkit.tools["double"].name == "double"

    def test_register_tool(self):
        """Test registering existing Tool."""
        toolkit = Toolkit()

        @tool()
        def my_tool() -> Tuple[str, dict]:
            return "result", {}

        toolkit.register_tool(my_tool)
        assert "my_tool" in toolkit.tools

    def test_set_available_tools(self):
        """Test setting initially available tools."""
        toolkit = Toolkit()

        def tool1() -> Tuple[str, dict]:
            return "1", {}

        def tool2() -> Tuple[str, dict]:
            return "2", {}

        toolkit.add_tool("tool1", tool1)
        toolkit.add_tool("tool2", tool2)
        toolkit.set_available_tools({"tool1"})

        assert toolkit.is_tool_available("tool1")
        assert not toolkit.is_tool_available("tool2")

    def test_tool_availability_with_context_requirements(self):
        """Test tool availability based on context requirements."""
        toolkit = Toolkit()

        def protected_tool() -> Tuple[str, dict]:
            return "protected", {}

        toolkit.add_tool(
            "protected",
            protected_tool,
            requires_context=["auth_token"]
        )
        toolkit.set_available_tools({"protected"})

        # Not available without context
        assert not toolkit.is_tool_available("protected")

        # Available with context
        toolkit.context.set("auth_token", "token123")
        assert toolkit.is_tool_available("protected")

    def test_unlock_mechanism(self):
        """Test unlocking tools."""
        toolkit = Toolkit()

        @tool(unlocks=["action2"])
        def action1() -> Tuple[str, dict]:
            return "action1 done", {}

        @tool()
        def action2() -> Tuple[str, dict]:
            return "action2 done", {}

        toolkit.register_tool(action1)
        toolkit.register_tool(action2)
        toolkit.set_available_tools({"action1"})

        # Initially action2 not available
        assert toolkit.is_tool_available("action1")
        assert not toolkit.is_tool_available("action2")

        # Execute action1
        _ = toolkit.execute_sequential([
            {"name": "action1", "arguments": {}}
        ])

        # Now action2 should be unlocked
        assert toolkit.is_tool_available("action2")

    def test_lock_mechanism(self):
        """Test locking tools."""
        toolkit = Toolkit()

        @tool(locks=["action1"])
        def logout() -> Tuple[str, dict]:
            return "logged out", {}

        @tool()
        def action1() -> Tuple[str, dict]:
            return "action1", {}

        toolkit.register_tool(logout)
        toolkit.register_tool(action1)
        toolkit.set_available_tools({"logout", "action1"})

        # Both available initially
        assert toolkit.is_tool_available("logout")
        assert toolkit.is_tool_available("action1")

        # Execute logout
        toolkit.execute_sequential([{"name": "logout", "arguments": {}}])

        # action1 should be locked
        assert not toolkit.is_tool_available("action1")

    def test_sequential_execution_with_context_updates(self):
        """Test sequential execution with immediate context updates."""
        toolkit = Toolkit()

        @tool(provides=["selected_project"], unlocks=["list_issues"])
        def select_project(project_id: str) -> Tuple[str, dict]:
            return f"Selected {project_id}", {"selected_project": project_id}

        @tool(requires=["selected_project"], provides=["issue_names"])
        def list_issues() -> Tuple[str, dict]:
            return "Found 3 issues", {"issue_names": ["BUG-1", "BUG-2", "FEAT-3"]}

        toolkit.register_tool(select_project)
        toolkit.register_tool(list_issues)
        toolkit.set_available_tools({"select_project"})

        # Execute sequence
        results = toolkit.execute_sequential([
            {"name": "select_project", "arguments": {"project_id": "proj-1"}},
            {"name": "list_issues", "arguments": {}}
        ])

        assert len(results) == 2
        assert results[0]["success"]
        assert results[1]["success"]

        # Context should be updated
        assert toolkit.context.get("selected_project") == "proj-1"
        assert toolkit.context.get("issue_names") == ["BUG-1", "BUG-2", "FEAT-3"]

    def test_pipeline_failure_stops_execution(self):
        """Test that pipeline stops on first failure."""
        toolkit = Toolkit()

        @tool()
        def step1() -> Tuple[str, dict]:
            return "step1 ok", {"step1": True}

        @tool()
        def step2() -> Tuple[str, dict]:
            raise ValueError("Step 2 failed")

        @tool()
        def step3() -> Tuple[str, dict]:
            return "step3 ok", {"step3": True}

        toolkit.register_tool(step1)
        toolkit.register_tool(step2)
        toolkit.register_tool(step3)
        toolkit.set_available_tools({"step1", "step2", "step3"})

        # Execute pipeline
        with pytest.raises(ToolkitPipelineError) as exc_info:
            toolkit.execute_sequential([
                {"name": "step1", "arguments": {}},
                {"name": "step2", "arguments": {}},
                {"name": "step3", "arguments": {}}
            ])

        # Check partial results
        error = exc_info.value
        assert len(error.partial_results) == 2
        assert error.partial_results[0]["success"]
        assert not error.partial_results[1]["success"]

        # step1's context updates should be applied
        assert toolkit.context.has("step1")
        # step3 should not have executed
        assert not toolkit.context.has("step3")

    def test_generate_schemas(self):
        """Test generating schemas for available tools."""
        toolkit = Toolkit()

        @tool()
        def available_tool(x: int) -> Tuple[int, dict]:
            """Available tool."""
            return x, {}

        @tool()
        def unavailable_tool(y: int) -> Tuple[int, dict]:
            """Unavailable tool."""
            return y, {}

        toolkit.register_tool(available_tool)
        toolkit.register_tool(unavailable_tool)
        toolkit.set_available_tools({"available_tool"})

        schemas = toolkit.generate_schemas()

        # Only available tool should have schema
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "available_tool"

    def test_generate_schemas_with_dynamic_types(self):
        """Test schema generation with dynamic types."""
        toolkit = Toolkit()
        toolkit.context.set("options", ["a", "b", "c"])

        @tool()
        def select_option(opt: DynamicStr["options"]) -> Tuple[str, dict]:
            return opt, {}

        toolkit.register_tool(select_option)
        toolkit.set_available_tools({"select_option"})

        schemas = toolkit.generate_schemas()
        opt_schema = schemas[0]["function"]["parameters"]["properties"]["opt"]

        assert opt_schema == {"type": "string", "enum": ["a", "b", "c"]}

    def test_custom_filter(self):
        """Test custom filter registration."""
        toolkit = Toolkit()
        toolkit.context.set("users", [
            {"name": "alice", "score": 10},
            {"name": "bob", "score": 5}
        ])

        # Register custom filter
        toolkit.register_filter(
            "high_score",
            lambda users: [u["name"] for u in users if u["score"] >= 10]
        )

        # Create tool with filtered type
        from bestla.yggdrasil.dynamic_types import DynamicFiltered

        @tool()
        def select_user(name: DynamicFiltered[("users", "high_score")]) -> Tuple[str, dict]:
            return name, {}

        toolkit.register_tool(select_user)
        toolkit.set_available_tools({"select_user"})

        schemas = toolkit.generate_schemas()
        name_schema = schemas[0]["function"]["parameters"]["properties"]["name"]

        # Should only have high-scoring users
        assert name_schema == {"type": "string", "enum": ["alice"]}

    def test_toolkit_copy(self):
        """Test deep copying toolkit."""
        toolkit = Toolkit()
        toolkit.context.set("key", "value")
        toolkit.add_tool("tool1", lambda: ("result", {}))
        toolkit.set_available_tools({"tool1"})

        # Copy toolkit
        copy = toolkit.copy()
        copy.context.set("key2", "value2")

        # Original unchanged
        assert not toolkit.context.has("key2")
        assert copy.context.has("key")
        assert copy.context.has("key2")

    def test_builtin_filters(self):
        """Test builtin filters."""
        toolkit = Toolkit()

        # Test active_only filter
        users = [
            {"name": "alice", "active": True},
            {"name": "bob", "active": False}
        ]

        filtered = toolkit.filters["active_only"](users)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "alice"

    def test_tool_not_found_error(self):
        """Test error when tool doesn't exist."""
        toolkit = Toolkit()

        with pytest.raises(ToolkitPipelineError) as exc_info:
            toolkit.execute_sequential([
                {"name": "nonexistent", "arguments": {}}
            ])

        assert "not found" in str(exc_info.value)

    def test_tool_not_available_error(self):
        """Test error when tool exists but not available."""
        toolkit = Toolkit()

        @tool(requires=["auth_token"])
        def protected() -> Tuple[str, dict]:
            return "protected", {}

        toolkit.register_tool(protected)
        toolkit.set_available_tools({"protected"})

        # Try to execute without auth_token
        with pytest.raises(ToolkitPipelineError) as exc_info:
            toolkit.execute_sequential([
                {"name": "protected", "arguments": {}}
            ])

        assert "not available" in str(exc_info.value)
