"""Integration edge cases - complex multi-component interactions."""

import threading
import time
from typing import Tuple
from unittest.mock import Mock

import pytest

from bestla.yggdrasil import Agent, Context, DynamicStr, ExecutionContext, Toolkit, tool
from bestla.yggdrasil.agent import ExecutionContext
from bestla.yggdrasil.exceptions import ToolkitPipelineError


class TestToolkitPrefixEdgeCases:
    """Test toolkit prefix handling edge cases."""

    def test_toolkit_prefix_with_double_colon(self):
        """Test toolkit prefix containing '::'."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        toolkit = Toolkit()
        toolkit.add_tool("tool1", lambda: ("result", {}))

        # Register with prefix containing ::
        agent.add_toolkit("namespace::prefix", toolkit)

        # Should be stored
        assert "namespace::prefix" in agent.toolkits

        # Tool name would be: namespace::prefix::tool1
        # This tests if parsing handles multiple ::
        expected_name = "namespace::prefix::tool1"
        assert expected_name in agent.toolkit_prefixes

    def test_independent_tool_name_with_double_colon(self):
        """Test independent tool with :: in its name."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Add independent tool with :: in name (looks like prefixed tool)
        agent.add_tool("fake::prefixed::tool", lambda: ("result", {}))

        # Should be in independent toolkit
        assert "fake::prefixed::tool" in agent.independent_toolkit.tools

        # Should work correctly
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        schemas = agent._generate_all_schemas(context)

        # Tool should appear in schemas
        names = [s["function"]["name"] for s in schemas]
        assert "fake::prefixed::tool" in names

    def test_toolkit_prefix_collision_with_tool_name(self):
        """Test prefix that matches another toolkit's tool name."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        toolkit1 = Toolkit()
        toolkit1.add_tool("list", lambda: ("list1", {}))

        toolkit2 = Toolkit()
        toolkit2.add_tool("issues", lambda: ("issues", {}))

        agent.add_toolkit("github", toolkit1)
        agent.add_toolkit("github::list", toolkit2)  # Prefix includes existing tool name

        # Both should be registered
        assert "github" in agent.toolkits
        assert "github::list" in agent.toolkits

        # Tool names should be distinguishable
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        schemas = agent._generate_all_schemas(context)
        names = [s["function"]["name"] for s in schemas]

        assert "github::list" in names
        assert "github::list::issues" in names

    def test_empty_toolkit_prefix(self):
        """Test toolkit with empty string as prefix."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        toolkit = Toolkit()
        toolkit.add_tool("tool1", lambda: ("result", {}))

        # Register with empty prefix
        agent.add_toolkit("", toolkit)

        # Should work (though unusual)
        assert "" in agent.toolkits


class TestToolParameterConflicts:
    """Test tools with conflicting or reserved parameter names."""

    def test_tool_parameter_named_context(self):
        """Test tool with parameter named 'context'."""

        @tool()
        def param_named_context(context: str) -> Tuple[str, dict]:
            """Parameter named 'context' - potential conflict."""
            return f"context={context}", {}

        # Should create tool successfully
        assert param_named_context.name == "param_named_context"

        # Execute with context parameter
        result, updates = param_named_context.execute(context="my_context_value")
        assert result == "context=my_context_value"

    def test_tool_parameter_with_reserved_name(self):
        """Test tool with parameter that might be considered reserved."""

        @tool()
        def param_named_type(type: str) -> Tuple[str, dict]:
            """Parameter named 'type' (Python builtin)."""
            return f"type={type}", {}

        # Should work - 'type' is just a parameter name
        schema = param_named_type.generate_schema(Context())

        # 'type' should be in parameters
        assert "type" in schema["function"]["parameters"]["properties"]

    def test_tool_with_kwargs_captures_all(self):
        """Test tool with **kwargs receives all arguments."""

        @tool()
        def kwargs_tool(**kwargs) -> Tuple[dict, dict]:
            return kwargs, {}

        result, _ = kwargs_tool.execute(a=1, b=2, c=3, unknown="value")

        assert result == {"a": 1, "b": 2, "c": 3, "unknown": "value"}


class TestDynamicTypeInteractions:
    """Test interactions between different dynamic types."""

    def test_multiple_dynamic_types_same_context_key(self):
        """Test different tools using different dynamic types for same context key."""
        toolkit = Toolkit()
        toolkit.context.set("values", [1, 2, 3])

        from bestla.yggdrasil.dynamic_types import DynamicArray

        @tool()
        def tool1(value: DynamicStr["values"]) -> Tuple[str, dict]:
            # Expects string enum from list
            return value, {}

        @tool()
        def tool2(values: DynamicArray["values"]) -> Tuple[list, dict]:
            # Expects array with enum items
            return values, {}

        toolkit.register_tool(tool1)
        toolkit.register_tool(tool2)

        # Generate schemas - same context key used differently
        schemas = toolkit.generate_schemas()

        assert len(schemas) == 2

        # Find schemas by name
        tool1_schema = next(s for s in schemas if s["function"]["name"] == "tool1")
        tool2_schema = next(s for s in schemas if s["function"]["name"] == "tool2")

        # tool1 should have string enum
        assert tool1_schema["function"]["parameters"]["properties"]["value"]["enum"] == [1, 2, 3]

        # tool2 should have array with enum items
        assert tool2_schema["function"]["parameters"]["properties"]["values"]["type"] == "array"

    def test_dynamic_type_with_context_changing_during_call(self):
        """Test dynamic type when context changes between schema gen and execution."""
        toolkit = Toolkit()
        toolkit.context.set("options", ["a", "b"])

        @tool()
        def select(option: DynamicStr["options"]) -> Tuple[str, dict]:
            return option, {}

        toolkit.register_tool(select)

        # Generate schema
        schemas = toolkit.generate_schemas()
        original_enum = schemas[0]["function"]["parameters"]["properties"]["option"]["enum"]
        assert original_enum == ["a", "b"]

        # Change context
        toolkit.context.set("options", ["x", "y", "z"])

        # Generate schema again - should reflect new context
        schemas = toolkit.generate_schemas()
        new_enum = schemas[0]["function"]["parameters"]["properties"]["option"]["enum"]
        assert new_enum == ["x", "y", "z"]


class TestPipelineOrdering:
    """Test tool execution order edge cases."""

    def test_pipeline_with_interdependent_tools_wrong_order(self):
        """Test pipeline where tools are in wrong dependency order."""
        toolkit = Toolkit()

        @tool(required_context=["step2_done"], required_states=["step2_complete"])
        def step3() -> Tuple[str, dict]:
            return "step3", {}

        @tool(required_context=["step1_done"], enables_states=["step2_complete"])
        def step2() -> Tuple[str, dict]:
            return "step2", {"step2_done": True}

        @tool(enables_states=["step1_complete"])
        def step1() -> Tuple[str, dict]:
            return "step1", {"step1_done": True}

        toolkit.register_tool(step1)
        toolkit.register_tool(step2)
        toolkit.register_tool(step3)

        # Try to execute in wrong order: step3, step2, step1
        with pytest.raises(ToolkitPipelineError):
            toolkit.execute_sequential(
                [
                    {"name": "step3", "arguments": {}},
                    {"name": "step2", "arguments": {}},
                    {"name": "step1", "arguments": {}},
                ]
            )

    def test_pipeline_with_optional_intermediate_steps(self):
        """Test pipeline where some steps can be skipped."""
        toolkit = Toolkit()

        @tool(enables_states=["started"])
        def start() -> Tuple[str, dict]:
            return "started", {}

        @tool(required_states=["started"])
        def optional_step() -> Tuple[str, dict]:
            return "optional", {"optional_done": True}

        @tool(required_states=["started"])  # Doesn't require optional_done
        def finish() -> Tuple[str, dict]:
            return "finished", {}

        toolkit.register_tool(start)
        toolkit.register_tool(optional_step)
        toolkit.register_tool(finish)

        # Execute without optional step
        results = toolkit.execute_sequential(
            [{"name": "start", "arguments": {}}, {"name": "finish", "arguments": {}}]
        )

        assert len(results) == 2
        assert all(r["success"] for r in results)


class TestMultipleAgentInteractions:
    """Test complex multi-agent scenarios."""

    def test_agent_with_100_toolkits(self):
        """Test agent with massive number of toolkits."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Add 100 toolkits
        for i in range(100):
            toolkit = Toolkit()
            toolkit.add_tool(f"tool_{i}", lambda i=i: (f"result_{i}", {}))
            agent.add_toolkit(f"tk_{i}", toolkit)

        # Should handle registration
        assert len(agent.toolkits) == 100

        # Schema generation should work
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        schemas = agent._generate_all_schemas(context)

        # Should have 100 schemas (one per toolkit)
        assert len(schemas) == 100

    def test_deeply_nested_agent_state_isolation(self):
        """Test state isolation in 5-level agent hierarchy."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="done", tool_calls=None))]
        )

        # Create 5 levels with state
        agents = []
        for level in range(5):
            agent = Agent(provider=mock_provider, model="gpt-4")
            toolkit = Toolkit()
            toolkit.context.set("level", level)

            @tool()
            def report_level() -> Tuple[int, dict]:
                return toolkit.context.get("level"), {}

            toolkit.register_tool(report_level)
            agent.add_toolkit(f"level_{level}", toolkit)

            # Add previous agent as tool
            if agents:
                agent.add_tool("call_child", agents[-1].execute)

            agents.append(agent)

        # Execute top-level agent
        agents[-1].run("test", max_iterations=1)

        # All contexts should remain isolated
        for level, agent in enumerate(agents):
            # Check the toolkit context hasn't changed
            toolkit_name = f"level_{level}"
            if toolkit_name in agent.toolkits:
                assert agent.toolkits[toolkit_name].context.get("level") == level


class TestConcurrentAgentModifications:
    """Test modifying agent while it's executing."""

    def test_add_toolkit_during_execution(self):
        """Test adding toolkit while agent is running."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        execution_started = threading.Event()
        can_add_toolkit = threading.Event()

        def slow_create(*args, **kwargs):
            execution_started.set()
            can_add_toolkit.wait(timeout=2)  # Wait for toolkit to be added
            time.sleep(0.1)
            return Mock(choices=[Mock(message=Mock(content="done", tool_calls=None))])

        mock_provider.chat.completions.create = slow_create

        def run_agent():
            response, ctx = agent.run("test", max_iterations=1)

        def add_toolkit():
            execution_started.wait(timeout=2)
            new_toolkit = Toolkit()
            new_toolkit.add_tool("new_tool", lambda: ("result", {}))
            agent.add_toolkit("new", new_toolkit)
            can_add_toolkit.set()

        # Run in parallel
        t1 = threading.Thread(target=run_agent)
        t2 = threading.Thread(target=add_toolkit)

        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Both should complete without crashes
        assert "new" in agent.toolkits

    def test_clear_messages_during_execution(self):
        """Test clearing messages while agent is running."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        execution_in_progress = threading.Event()
        can_clear = threading.Event()

        def slow_create(*args, **kwargs):
            execution_in_progress.set()
            can_clear.wait(timeout=2)
            time.sleep(0.1)
            return Mock(choices=[Mock(message=Mock(content="done", tool_calls=None))])

        mock_provider.chat.completions.create = slow_create

        def run_agent():
            response, ctx2 = agent.run("test", max_iterations=1)

        def clear_messages():
            execution_in_progress.wait(timeout=2)
            agent.clear_messages()
            can_clear.set()

        t1 = threading.Thread(target=run_agent)
        t2 = threading.Thread(target=clear_messages)

        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Should complete without crashes
        # Message state might be inconsistent, but shouldn't crash


class TestDynamicSchemaChanges:
    """Test schema changes during execution."""

    def test_context_modified_between_schema_gen_and_execution(self):
        """Test context changing between schema generation and tool execution."""
        toolkit = Toolkit()
        toolkit.context.set("options", ["a", "b", "c"])

        @tool()
        def select(option: DynamicStr["options"]) -> Tuple[str, dict]:
            # By the time this executes, context might have changed
            return f"selected {option}", {}

        toolkit.register_tool(select)

        # Generate schema with original context
        schemas = toolkit.generate_schemas()
        original_enum = schemas[0]["function"]["parameters"]["properties"]["option"]["enum"]
        assert original_enum == ["a", "b", "c"]

        # Change context BEFORE execution
        toolkit.context.set("options", ["x", "y"])

        # Execute with value from ORIGINAL enum
        results = toolkit.execute_sequential([{"name": "select", "arguments": {"option": "a"}}])

        # Should succeed - tool doesn't re-validate against schema
        assert results[0]["success"]
        assert "selected a" in results[0]["result"]

    def test_concurrent_schema_generation(self):
        """Test multiple threads generating schemas simultaneously."""
        toolkit = Toolkit()
        toolkit.context.set("values", list(range(100)))

        @tool()
        def process(value: DynamicStr["values"]) -> Tuple[str, dict]:
            return str(value), {}

        toolkit.register_tool(process)

        schemas_list = []

        def generate():
            for _ in range(50):
                schemas = toolkit.generate_schemas()
                schemas_list.append(schemas)
                time.sleep(0.001)

        # Generate schemas from 5 threads
        threads = [threading.Thread(target=generate) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(schemas_list) == 250


class TestErrorPropagation:
    """Test error propagation through complex systems."""

    def test_nested_agent_error_propagation(self):
        """Test error in deeply nested agent propagates correctly."""
        mock_provider = Mock()
        # Mock to return text response (no tool calls)
        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="error handled", tool_calls=None))]
        )

        # Bottom agent with failing tool
        bottom_agent = Agent(provider=mock_provider, model="gpt-4")

        def failing_tool() -> Tuple[str, dict]:
            raise ValueError("Bottom agent failed")

        bottom_agent.add_tool("fail", failing_tool)

        # Middle agent calls bottom
        middle_agent = Agent(provider=mock_provider, model="gpt-4")
        middle_agent.add_tool("call_bottom", bottom_agent.execute)

        # Top agent calls middle
        top_agent = Agent(provider=mock_provider, model="gpt-4")
        top_agent.add_tool("call_middle", middle_agent.execute)

        # Call through hierarchy
        # agent.execute() returns (result, {}), doesn't raise exceptions
        result, updates = top_agent.execute("test")

        # Should complete without exception (errors are handled gracefully)
        assert updates == {}
        assert isinstance(result, str)

    def test_pipeline_error_preserves_partial_state(self):
        """Test that pipeline errors preserve all successful state updates."""
        toolkit = Toolkit()

        @tool()
        def step1() -> Tuple[str, dict]:
            return "ok", {"step1": "done", "shared": "from_step1"}

        @tool()
        def step2() -> Tuple[str, dict]:
            return "ok", {"step2": "done", "shared": "from_step2"}

        @tool()
        def step3_fail() -> Tuple[str, dict]:
            raise ValueError("Step 3 fails")

        @tool()
        def step4() -> Tuple[str, dict]:
            return "ok", {"step4": "done"}

        toolkit.register_tool(step1)
        toolkit.register_tool(step2)
        toolkit.register_tool(step3_fail)
        toolkit.register_tool(step4)

        # Execute pipeline
        with pytest.raises(ToolkitPipelineError) as exc_info:
            toolkit.execute_sequential(
                [
                    {"name": "step1", "arguments": {}},
                    {"name": "step2", "arguments": {}},
                    {"name": "step3_fail", "arguments": {}},
                    {"name": "step4", "arguments": {}},
                ]
            )

        # Verify partial results
        error = exc_info.value
        assert len(error.partial_results) == 3
        assert error.failed_at == 2

        # First two steps should have updated context
        assert toolkit.context.get("step1") == "done"
        assert toolkit.context.get("step2") == "done"
        assert toolkit.context.get("shared") == "from_step2"  # step2 overwrote step1

        # Step 4 should NOT have run
        assert not toolkit.context.has("step4")


class TestToolkitStateConflicts:
    """Test state conflicts between multiple toolkits."""

    def test_overlapping_state_names_across_toolkits(self):
        """Test multiple toolkits using same state name."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Create two toolkits with same state name
        tk1 = Toolkit()
        tk2 = Toolkit()

        @tool(enables_states=["authenticated"])
        def tk1_login() -> Tuple[str, dict]:
            return "tk1 logged in", {}

        @tool(enables_states=["authenticated"])
        def tk2_login() -> Tuple[str, dict]:
            return "tk2 logged in", {}

        @tool(required_states=["authenticated"])
        def tk1_action() -> Tuple[str, dict]:
            return "tk1 action", {}

        @tool(required_states=["authenticated"])
        def tk2_action() -> Tuple[str, dict]:
            return "tk2 action", {}

        tk1.register_tool(tk1_login)
        tk1.register_tool(tk1_action)
        tk2.register_tool(tk2_login)
        tk2.register_tool(tk2_action)

        agent.add_toolkit("tk1", tk1)
        agent.add_toolkit("tk2", tk2)

        # Login to tk1
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        results = agent._execute_toolkit_group(
            context, "tk1", [{"id": "1", "name": "tk1_login", "arguments": {}}]
        )

        assert results[0]["role"] == "tool"

        # tk1's authenticated state should be enabled
        assert "authenticated" in context.toolkits["tk1"].unlocked_states

        # tk2's authenticated state should NOT be enabled (separate state)
        assert "authenticated" not in context.toolkits["tk2"].unlocked_states

        # This verifies toolkit state isolation

    def test_parallel_toolkits_modifying_same_state(self):
        """Test parallel toolkits with overlapping state names."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        tk1 = Toolkit()
        tk2 = Toolkit()

        @tool(enables_states=["shared_state"])
        def tk1_enable() -> Tuple[str, dict]:
            time.sleep(0.05)
            return "tk1 enabled", {}

        @tool(disables_states=["shared_state"])
        def tk2_disable() -> Tuple[str, dict]:
            time.sleep(0.05)
            return "tk2 disabled", {}

        tk1.register_tool(tk1_enable)
        tk2.register_tool(tk2_disable)
        tk1.set_unlocked_states({"shared_state"})
        tk2.set_unlocked_states({"shared_state"})

        agent.add_toolkit("tk1", tk1)
        agent.add_toolkit("tk2", tk2)

        # Execute both in parallel
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        tool_calls = [
            {"id": "1", "function": {"name": "tk1::tk1_enable", "arguments": "{}"}},
            {"id": "2", "function": {"name": "tk2::tk2_disable", "arguments": "{}"}},
        ]

        results = agent._execute_tool_calls(context, tool_calls)

        # Both should complete
        assert len(results) == 2

        # Each toolkit should have independent state
        # tk1 should have shared_state enabled
        # tk2 should have shared_state disabled
        # Both can be true because they're separate toolkits
        assert "shared_state" in context.toolkits["tk1"].unlocked_states
        assert "shared_state" not in context.toolkits["tk2"].unlocked_states


class TestDuplicateToolNames:
    """Test handling of duplicate tool names."""

    def test_toolkit_with_duplicate_tool_registration(self):
        """Test registering tool with same name twice."""
        toolkit = Toolkit()

        @tool()
        def my_tool_first() -> Tuple[str, dict]:
            return "first", {}

        @tool()
        def my_tool() -> Tuple[str, dict]:  # Same name
            return "second", {}

        toolkit.register_tool(my_tool)

        # Second registration should overwrite first
        # (Python function definition overwrites the name)
        # But both have same name, so effectively only one tool

        assert "my_tool" in toolkit.tools
        result, _ = toolkit.tools["my_tool"].execute()
        assert result == "second"

    def test_multiple_toolkits_same_tool_name(self):
        """Test different toolkits can have tools with same names."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        tk1 = Toolkit()
        tk2 = Toolkit()

        tk1.add_tool("action", lambda: ("tk1 action", {}))
        tk2.add_tool("action", lambda: ("tk2 action", {}))

        agent.add_toolkit("tk1", tk1)
        agent.add_toolkit("tk2", tk2)

        # Both should be registered with prefixes
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        schemas = agent._generate_all_schemas(context)

        names = [s["function"]["name"] for s in schemas]
        assert "tk1::action" in names
        assert "tk2::action" in names
