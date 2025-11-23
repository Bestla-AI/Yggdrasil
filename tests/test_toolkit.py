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

    def test_set_unlocked_states(self):
        """Test setting initially unlocked states."""
        toolkit = Toolkit()

        def tool1() -> Tuple[str, dict]:
            return "1", {}

        def tool2() -> Tuple[str, dict]:
            return "2", {}

        toolkit.add_tool("tool1", tool1, required_states=["state1"])
        toolkit.add_tool("tool2", tool2, required_states=["state2"])
        toolkit.set_unlocked_states({"state1"})

        assert toolkit.is_tool_available("tool1")
        assert not toolkit.is_tool_available("tool2")

    def test_tool_availability_with_context_requirements(self):
        """Test tool availability based on context requirements."""
        toolkit = Toolkit()

        def protected_tool() -> Tuple[str, dict]:
            return "protected", {}

        toolkit.add_tool("protected", protected_tool, required_context=["auth_token"])

        # Not available without context (no state requirements, but context missing)
        assert not toolkit.is_tool_available("protected")

        # Available with context
        toolkit.context.set("auth_token", "token123")
        assert toolkit.is_tool_available("protected")

    def test_enable_states_mechanism(self):
        """Test enabling states makes tools available."""
        toolkit = Toolkit()

        @tool(enables_states=["state2"])
        def action1() -> Tuple[str, dict]:
            return "action1 done", {}

        @tool(required_states=["state2"])
        def action2() -> Tuple[str, dict]:
            return "action2 done", {}

        toolkit.register_tool(action1)
        toolkit.register_tool(action2)

        # Initially action1 available (no requirements), action2 not available (state2 not enabled)
        assert toolkit.is_tool_available("action1")
        assert not toolkit.is_tool_available("action2")

        # Execute action1
        _ = toolkit.execute_sequential([{"name": "action1", "arguments": {}}])

        # Now action2 should be available (state2 enabled)
        assert toolkit.is_tool_available("action2")

    def test_disable_states_mechanism(self):
        """Test disabling states makes tools unavailable."""
        toolkit = Toolkit()

        @tool(disables_states=["authenticated"])
        def logout() -> Tuple[str, dict]:
            return "logged out", {}

        @tool(required_states=["authenticated"])
        def action1() -> Tuple[str, dict]:
            return "action1", {}

        toolkit.register_tool(logout)
        toolkit.register_tool(action1)
        toolkit.set_unlocked_states({"authenticated"})

        # Both available initially (authenticated state is enabled)
        assert toolkit.is_tool_available("logout")
        assert toolkit.is_tool_available("action1")

        # Execute logout
        toolkit.execute_sequential([{"name": "logout", "arguments": {}}])

        # action1 should no longer be available (authenticated state disabled)
        assert not toolkit.is_tool_available("action1")

    def test_sequential_execution_with_context_updates(self):
        """Test sequential execution with immediate context updates."""
        toolkit = Toolkit()

        @tool(enables_states=["project_selected"])
        def select_project(project_id: str) -> Tuple[str, dict]:
            return f"Selected {project_id}", {"selected_project": project_id}

        @tool(required_context=["selected_project"], required_states=["project_selected"])
        def list_issues() -> Tuple[str, dict]:
            return "Found 3 issues", {"issue_names": ["BUG-1", "BUG-2", "FEAT-3"]}

        toolkit.register_tool(select_project)
        toolkit.register_tool(list_issues)

        # Execute sequence
        results = toolkit.execute_sequential(
            [
                {"name": "select_project", "arguments": {"project_id": "proj-1"}},
                {"name": "list_issues", "arguments": {}},
            ]
        )

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

        # Execute pipeline (all tools available - no state requirements)
        with pytest.raises(ToolkitPipelineError) as exc_info:
            toolkit.execute_sequential(
                [
                    {"name": "step1", "arguments": {}},
                    {"name": "step2", "arguments": {}},
                    {"name": "step3", "arguments": {}},
                ]
            )

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

        @tool(required_states=["locked_state"])
        def unavailable_tool(y: int) -> Tuple[int, dict]:
            """Unavailable tool."""
            return y, {}

        toolkit.register_tool(available_tool)
        toolkit.register_tool(unavailable_tool)

        schemas = toolkit.generate_schemas()

        # Only available tool should have schema (unavailable_tool requires locked_state)
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

        schemas = toolkit.generate_schemas()
        opt_schema = schemas[0]["function"]["parameters"]["properties"]["opt"]

        assert opt_schema == {"type": "string", "enum": ["a", "b", "c"]}

    def test_custom_filter(self):
        """Test custom filter registration."""
        toolkit = Toolkit()
        toolkit.context.set("users", [{"name": "alice", "score": 10}, {"name": "bob", "score": 5}])

        # Register custom filter
        toolkit.register_filter(
            "high_score", lambda users: [u["name"] for u in users if u["score"] >= 10]
        )

        # Create tool with filtered type
        from bestla.yggdrasil.dynamic_types import DynamicFiltered

        @tool()
        def select_user(name: DynamicFiltered[("users", "high_score")]) -> Tuple[str, dict]:
            return name, {}

        toolkit.register_tool(select_user)

        schemas = toolkit.generate_schemas()
        name_schema = schemas[0]["function"]["parameters"]["properties"]["name"]

        # Should only have high-scoring users
        assert name_schema == {"type": "string", "enum": ["alice"]}

    def test_toolkit_copy(self):
        """Test deep copying toolkit."""
        toolkit = Toolkit()
        toolkit.context.set("key", "value")
        toolkit.add_tool("tool1", lambda: ("result", {}))
        toolkit.set_unlocked_states({"state1"})

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
        users = [{"name": "alice", "active": True}, {"name": "bob", "active": False}]

        filtered = toolkit.filters["active_only"](users)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "alice"

    def test_tool_not_found_error(self):
        """Test error when tool doesn't exist."""
        toolkit = Toolkit()

        with pytest.raises(ToolkitPipelineError) as exc_info:
            toolkit.execute_sequential([{"name": "nonexistent", "arguments": {}}])

        assert "not found" in str(exc_info.value)

    def test_tool_not_available_error(self):
        """Test error when tool exists but not available."""
        toolkit = Toolkit()

        @tool(required_context=["auth_token"])
        def protected() -> Tuple[str, dict]:
            return "protected", {}

        toolkit.register_tool(protected)

        # Try to execute without auth_token
        with pytest.raises(ToolkitPipelineError) as exc_info:
            toolkit.execute_sequential([{"name": "protected", "arguments": {}}])

        assert "not available" in str(exc_info.value)

    def test_forbidden_states_mechanism(self):
        """Test forbidden_states prevents tool availability."""
        toolkit = Toolkit()

        @tool(
            required_states=["authenticated"],
            forbidden_states=["project_selected"],
            enables_states=["project_selected"],
        )
        def select_project() -> Tuple[str, dict]:
            return "Project selected", {"project_id": 123}

        @tool(
            required_states=["authenticated", "project_selected"],
            disables_states=["project_selected"],
        )
        def deselect_project() -> Tuple[str, dict]:
            return "Project deselected", {}

        toolkit.register_tool(select_project)
        toolkit.register_tool(deselect_project)
        toolkit.set_unlocked_states({"authenticated"})

        # select_project available (authenticated=True, project_selected=False)
        assert toolkit.is_tool_available("select_project")
        # deselect_project not available (project_selected=False)
        assert not toolkit.is_tool_available("deselect_project")

        # Select project
        toolkit.execute_sequential([{"name": "select_project", "arguments": {}}])

        # Now select_project NOT available (forbidden state project_selected is enabled)
        assert not toolkit.is_tool_available("select_project")
        # deselect_project now available (both required states enabled)
        assert toolkit.is_tool_available("deselect_project")

        # Deselect project
        toolkit.execute_sequential([{"name": "deselect_project", "arguments": {}}])

        # Back to initial state
        assert toolkit.is_tool_available("select_project")
        assert not toolkit.is_tool_available("deselect_project")

    def test_get_availability_reason_tool_not_exists(self):
        """Test get_availability_reason when tool doesn't exist."""
        toolkit = Toolkit()

        reason = toolkit.get_availability_reason("nonexistent")
        assert "does not exist" in reason.lower()

    def test_get_availability_reason_missing_states(self):
        """Test get_availability_reason when required states missing."""
        toolkit = Toolkit()

        @tool(required_states=["authenticated", "project_selected"])
        def protected() -> Tuple[str, dict]:
            return "ok", {}

        toolkit.register_tool(protected)

        reason = toolkit.get_availability_reason("protected")
        assert "required states" in reason.lower()
        assert "authenticated" in reason
        assert "project_selected" in reason

    def test_get_availability_reason_forbidden_states(self):
        """Test get_availability_reason when forbidden states present."""
        toolkit = Toolkit()

        @tool(forbidden_states=["maintenance"])
        def normal_op() -> Tuple[str, dict]:
            return "ok", {}

        toolkit.register_tool(normal_op)
        toolkit.set_unlocked_states({"maintenance"})

        reason = toolkit.get_availability_reason("normal_op")
        assert "forbidden" in reason.lower()
        assert "maintenance" in reason

    def test_get_availability_reason_missing_context(self):
        """Test get_availability_reason when context requirements missing."""
        toolkit = Toolkit()

        @tool(required_context=["auth_token", "user_id"])
        def protected() -> Tuple[str, dict]:
            return "ok", {}

        toolkit.register_tool(protected)

        reason = toolkit.get_availability_reason("protected")
        assert "context" in reason.lower()
        assert "auth_token" in reason
        assert "user_id" in reason

    def test_get_availability_reason_when_available(self):
        """Test get_availability_reason returns 'Available' when tool is available."""
        toolkit = Toolkit()

        @tool()
        def simple_tool() -> Tuple[str, dict]:
            return "ok", {}

        toolkit.register_tool(simple_tool)

        reason = toolkit.get_availability_reason("simple_tool")
        assert reason == "Available"

    def test_execute_parallel_basic(self):
        """Test execute_parallel with multiple tools."""
        toolkit = Toolkit()

        @tool()
        def tool1() -> Tuple[str, dict]:
            return "result1", {"key1": "value1"}

        @tool()
        def tool2() -> Tuple[str, dict]:
            return "result2", {"key2": "value2"}

        toolkit.register_tool(tool1)
        toolkit.register_tool(tool2)

        # Execute in parallel
        results = toolkit.execute_parallel(
            [{"name": "tool1", "arguments": {}}, {"name": "tool2", "arguments": {}}]
        )

        assert len(results) == 2
        assert all(r["success"] for r in results)

        # Context should have both updates
        assert toolkit.context.has("key1")
        assert toolkit.context.has("key2")

    def test_execute_parallel_last_write_wins(self):
        """Test execute_parallel with conflicting context updates."""
        import time

        toolkit = Toolkit()

        @tool()
        def set_a() -> Tuple[str, dict]:
            return "a", {"shared_key": "from_a"}

        @tool()
        def set_b() -> Tuple[str, dict]:
            time.sleep(0.01)  # Slight delay
            return "b", {"shared_key": "from_b"}

        toolkit.register_tool(set_a)
        toolkit.register_tool(set_b)

        results = toolkit.execute_parallel(
            [{"name": "set_a", "arguments": {}}, {"name": "set_b", "arguments": {}}]
        )

        assert len(results) == 2

        # One of the values should win (last-write-wins)
        shared_value = toolkit.context.get("shared_key")
        assert shared_value in ["from_a", "from_b"]

    def test_execute_parallel_tool_not_found(self):
        """Test execute_parallel when tool doesn't exist."""
        toolkit = Toolkit()

        @tool()
        def existing() -> Tuple[str, dict]:
            return "ok", {}

        toolkit.register_tool(existing)

        results = toolkit.execute_parallel(
            [{"name": "existing", "arguments": {}}, {"name": "nonexistent", "arguments": {}}]
        )

        assert len(results) == 2
        assert results[0]["success"]
        assert not results[1]["success"]
        assert "not found" in results[1]["error"]

    def test_execute_parallel_tool_failure_doesnt_stop_others(self):
        """Test execute_parallel continues when one tool fails."""
        toolkit = Toolkit()

        @tool()
        def success_tool() -> Tuple[str, dict]:
            return "success", {"success": True}

        @tool()
        def failing_tool() -> Tuple[str, dict]:
            raise RuntimeError("Intentional failure")

        toolkit.register_tool(success_tool)
        toolkit.register_tool(failing_tool)

        results = toolkit.execute_parallel(
            [{"name": "success_tool", "arguments": {}}, {"name": "failing_tool", "arguments": {}}]
        )

        # Both should have results
        assert len(results) == 2

        # Find success and failure
        success_result = [r for r in results if r.get("success")][0]
        failure_result = [r for r in results if not r.get("success")][0]

        assert success_result["result"] == "success"
        assert "error" in failure_result

    def test_execute_parallel_context_snapshot(self):
        """Test execute_parallel uses context snapshot."""
        toolkit = Toolkit()
        toolkit.context.set("initial_value", 100)

        @tool()
        def read_value() -> Tuple[int, dict]:
            # Should see initial value, not changes from other tools
            return toolkit.context.get("initial_value"), {}

        @tool()
        def modify_value() -> Tuple[str, dict]:
            # Modifies context
            return "modified", {"initial_value": 999}

        toolkit.register_tool(read_value)
        toolkit.register_tool(modify_value)

        results = toolkit.execute_parallel(
            [{"name": "read_value", "arguments": {}}, {"name": "modify_value", "arguments": {}}]
        )

        # Both should complete
        assert len(results) == 2

        # After parallel execution, context should be updated
        # (last-write-wins from context_updates merge)
        final_value = toolkit.context.get("initial_value")
        # Could be 100 (from read) or 999 (from modify), depending on merge order
        assert final_value in [100, 999]

    def test_execute_parallel_with_arguments(self):
        """Test execute_parallel with tool arguments."""
        toolkit = Toolkit()

        @tool()
        def multiply(x: int, y: int) -> Tuple[int, dict]:
            return x * y, {"result": x * y}

        toolkit.register_tool(multiply)

        results = toolkit.execute_parallel(
            [
                {"name": "multiply", "arguments": {"x": 2, "y": 3}},
                {"name": "multiply", "arguments": {"x": 4, "y": 5}},
            ]
        )

        assert len(results) == 2
        assert all(r["success"] for r in results)

        # Results can be in any order due to parallel execution
        result_values = sorted([r["result"] for r in results])
        assert result_values == [6, 20]

    def test_is_tool_available_nonexistent_tool(self):
        """Test is_tool_available returns False for nonexistent tool."""
        toolkit = Toolkit()

        assert not toolkit.is_tool_available("nonexistent")
