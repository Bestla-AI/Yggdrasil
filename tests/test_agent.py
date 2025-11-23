"""Tests for Agent class."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from unittest.mock import Mock

import pytest

from bestla.yggdrasil import Agent, Toolkit, tool
from bestla.yggdrasil.agent import ExecutionContext


@pytest.fixture
def mock_provider():
    """Create a mock OpenAI provider."""
    return Mock()


class TestAgent:
    """Test Agent functionality."""

    def test_create_agent(self, mock_provider):
        """Test creating an agent."""
        agent = Agent(provider=mock_provider, model="gpt-4")
        assert agent.model == "gpt-4"
        assert agent.system_prompt is not None

    def test_add_toolkit(self, mock_provider):
        """Test adding toolkit with prefix."""
        agent = Agent(provider=mock_provider, model="gpt-4")
        toolkit = Toolkit()

        def test_tool() -> Tuple[str, dict]:
            return "result", {}

        toolkit.add_tool("test", test_tool)

        agent.add_toolkit("myprefix", toolkit)

        assert "myprefix" in agent.toolkits
        assert agent.toolkit_prefixes["myprefix::test"] == "myprefix"

    def test_add_independent_tool(self, mock_provider):
        """Test adding independent tool."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        def add(a: int, b: int) -> Tuple[int, dict]:
            return a + b, {}

        agent.add_tool("add", add)

        assert "add" in agent.independent_toolkit.tools
        assert agent.independent_toolkit.is_tool_available("add")

    def test_generate_all_schemas(self, mock_provider):
        """Test generating schemas from all toolkits."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Add toolkit
        toolkit = Toolkit()
        toolkit.add_tool("tool1", lambda: ("r", {}))
        agent.add_toolkit("prefix", toolkit)

        # Add independent tool
        agent.add_tool("add", lambda a, b: (a + b, {}))

        # Create execution context
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        schemas = agent._generate_all_schemas(context)

        # Should have both prefixed and independent tools
        names = [s["function"]["name"] for s in schemas]
        assert "prefix::tool1" in names
        assert "add" in names

    def test_execute_toolkit_group(self, mock_provider):
        """Test executing a toolkit group."""
        agent = Agent(provider=mock_provider, model="gpt-4")
        toolkit = Toolkit()

        @tool()
        def test_tool(x: int) -> Tuple[int, dict]:
            return x * 2, {}

        toolkit.register_tool(test_tool)
        agent.add_toolkit("test", toolkit)

        # Create execution context
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Execute group
        results = agent._execute_toolkit_group(
            context, "test", [{"id": "call_1", "name": "test_tool", "arguments": {"x": 5}}]
        )

        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call_1"
        assert results[0]["role"] == "tool"
        assert "10" in results[0]["content"]

    def test_execute_independent_tools(self, mock_provider):
        """Test executing independent tools."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        def add(a: int, b: int) -> Tuple[int, dict]:
            return a + b, {}

        def multiply(a: int, b: int) -> Tuple[int, dict]:
            return a * b, {}

        agent.add_tool("add", add)
        agent.add_tool("multiply", multiply)

        # Create execution context
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Execute both
        results = agent._execute_toolkit_group(
            context,
            "independent",
            [
                {"id": "call_1", "name": "add", "arguments": {"a": 5, "b": 3}},
                {"id": "call_2", "name": "multiply", "arguments": {"a": 4, "b": 2}},
            ],
        )

        assert len(results) == 2
        # Results can be in any order due to parallel execution
        contents = {r["tool_call_id"]: r["content"] for r in results}
        assert "8" in contents["call_1"]
        assert "8" in contents["call_2"]

    def test_agent_as_tool(self, mock_provider):
        """Test using agent as a tool (hierarchical composition)."""
        # Create sub-agent with mock provider
        sub_agent = Agent(provider=mock_provider, model="gpt-4")
        sub_agent.add_tool("add", lambda a, b: (a + b, {}))

        # Mock the run method to avoid OpenAI call
        sub_agent.run = Mock(return_value="Result: 5")

        # Sub-agent's execute should return (result, {})
        result, updates = sub_agent.execute("Calculate 2 + 3")
        assert updates == {}  # No context updates (isolation)
        assert result == "Result: 5"

    def test_clear_messages(self, mock_provider):
        """Test clearing conversation history."""
        agent = Agent(provider=mock_provider, model="gpt-4")
        agent.messages.append({"role": "user", "content": "test"})
        assert len(agent.messages) > 0

        agent.clear_messages()
        assert len(agent.messages) == 0

    def test_toolkit_isolation_in_execute(self, mock_provider):
        """Test that sub-agent toolkits are isolated via ExecutionContext."""
        toolkit = Toolkit()
        toolkit.context.set("key", "original")

        agent = Agent(provider=mock_provider, model="gpt-4")
        agent.add_toolkit("test", toolkit)

        # Mock the run method to modify context
        def mock_run(query):
            # Create a context (simulating what real run does)
            context = ExecutionContext(agent.toolkits, agent.independent_toolkit)
            # Modify the copied context
            context.toolkits["test"].context.set("key", "modified")
            return "result"

        agent.run = mock_run

        # Execute as tool
        result, _ = agent.execute("test query")

        # Original toolkit context should NOT be modified
        assert toolkit.context.get("key") == "original"

    def test_multiple_toolkits_parallel_execution(self, mock_provider):
        """Test that different toolkits execute in parallel."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Create two toolkits
        toolkit1 = Toolkit()
        toolkit2 = Toolkit()

        executed_order = []

        @tool()
        def tool1() -> Tuple[str, dict]:
            executed_order.append("tool1")
            return "result1", {}

        @tool()
        def tool2() -> Tuple[str, dict]:
            executed_order.append("tool2")
            return "result2", {}

        toolkit1.register_tool(tool1)
        toolkit2.register_tool(tool2)

        agent.add_toolkit("tk1", toolkit1)
        agent.add_toolkit("tk2", toolkit2)

        # Create execution context
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Execute both (different toolkits, should be parallel)
        tool_calls = [
            {"id": "1", "function": {"name": "tk1::tool1", "arguments": "{}"}},
            {"id": "2", "function": {"name": "tk2::tool2", "arguments": "{}"}},
        ]

        results = agent._execute_tool_calls(context, tool_calls)

        # Both should execute
        assert len(results) == 2

    def test_sequential_within_toolkit(self, mock_provider):
        """Test that tools in same toolkit execute sequentially."""
        agent = Agent(provider=mock_provider, model="gpt-4")
        toolkit = Toolkit()

        execution_log = []

        @tool(enables_states=["step2_enabled"])
        def step1() -> Tuple[str, dict]:
            execution_log.append("step1")
            return "step1", {"step1": True}

        @tool(required_context=["step1"], required_states=["step2_enabled"])
        def step2() -> Tuple[str, dict]:
            execution_log.append("step2")
            # This should only run after step1
            # Note: context will be the copied context from ExecutionContext
            return "step2", {}

        toolkit.register_tool(step1)
        toolkit.register_tool(step2)

        agent.add_toolkit("test", toolkit)

        # Create execution context
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Execute sequence
        tool_calls = [
            {"id": "1", "function": {"name": "test::step1", "arguments": "{}"}},
            {"id": "2", "function": {"name": "test::step2", "arguments": "{}"}},
        ]

        results = agent._execute_tool_calls(context, tool_calls)

        # Check execution order
        assert execution_log == ["step1", "step2"]
        assert len(results) == 2

    def test_unknown_toolkit_error(self, mock_provider):
        """Test error handling for unknown toolkit."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Create execution context
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        tool_calls = [{"id": "1", "name": "nonexistent_tool", "arguments": {}}]

        results = agent._execute_toolkit_group(context, "unknown_toolkit", tool_calls)

        assert len(results) == 1
        assert "Unknown toolkit" in results[0]["content"]

    def test_execution_context_deep_copy(self, mock_provider):
        """Test that ExecutionContext creates deep copies of toolkits."""
        agent = Agent(provider=mock_provider, model="gpt-4")
        toolkit = Toolkit()
        toolkit.context.set("key", "original")

        agent.add_toolkit("test", toolkit)

        # Create execution context
        context1 = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        context2 = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Modify context1's toolkit
        context1.toolkits["test"].context.set("key", "modified1")

        # Modify context2's toolkit
        context2.toolkits["test"].context.set("key", "modified2")

        # Original should be unchanged
        assert toolkit.context.get("key") == "original"

        # Each context should have its own state
        assert context1.toolkits["test"].context.get("key") == "modified1"
        assert context2.toolkits["test"].context.get("key") == "modified2"

    def test_simultaneous_run_calls_state_isolation(self, mock_provider):
        """Test that simultaneous run() calls maintain separate states."""
        agent = Agent(provider=mock_provider, model="gpt-4")
        toolkit = Toolkit()

        # Track which execution modified which state
        execution_states = {}
        state_lock = threading.Lock()

        @tool()
        def set_value(execution_id: str, value: str) -> Tuple[str, dict]:
            """Set a value in the toolkit context."""
            # Store in toolkit context
            return f"Set {execution_id}={value}", {"execution_id": execution_id, "value": value}

        toolkit.register_tool(set_value)
        agent.add_toolkit("test", toolkit)

        # Mock run to test state isolation

        def mock_run_with_state(query, execution_id):
            """Mock run that tracks state changes."""
            # Create execution context (simulating real run)
            context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

            # Simulate state modification in this execution
            context.toolkits["test"].context.set("execution_id", execution_id)

            # Simulate some delay to increase chance of race conditions
            time.sleep(0.01)

            # Store the state we saw
            with state_lock:
                execution_states[execution_id] = context.toolkits["test"].context.get(
                    "execution_id"
                )

            return f"Result for {execution_id}"

        # Run multiple simultaneous "executions"
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(mock_run_with_state, f"query_{i}", f"exec_{i}") for i in range(5)
            ]

            # Wait for all to complete
            for future in futures:
                future.result()

        # Verify each execution saw its own state
        for i in range(5):
            assert execution_states[f"exec_{i}"] == f"exec_{i}"

        # Original toolkit should be unchanged
        assert not toolkit.context.has("execution_id")

    def test_concurrent_execute_calls_isolation(self, mock_provider):
        """Test that concurrent execute() calls maintain separate states."""
        agent = Agent(provider=mock_provider, model="gpt-4")
        toolkit = Toolkit()
        toolkit.context.set("counter", 0)

        agent.add_toolkit("test", toolkit)

        # Track execution results
        results = {}
        results_lock = threading.Lock()

        # Mock run to simulate state-dependent operations
        def mock_run(query):
            # Create context (simulating real run behavior)
            context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

            # Get execution_id from query
            execution_id = query.split(":")[-1]

            # Read counter, increment, and store
            counter = context.toolkits["test"].context.get("counter")
            new_value = counter + int(execution_id)
            context.toolkits["test"].context.set("counter", new_value)

            # Add delay to increase race condition likelihood
            time.sleep(0.01)

            # Read the value we just set
            final_value = context.toolkits["test"].context.get("counter")

            with results_lock:
                results[execution_id] = final_value

            return f"Result: {final_value}"

        agent.run = mock_run

        # Execute concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(agent.execute, f"query:{i}")
                for i in range(1, 11)  # execution_ids 1-10
            ]

            # Wait for completion
            for future in futures:
                future.result()

        # Each execution should have seen its own incremented value
        # If state was shared, we'd see race conditions
        for i in range(1, 11):
            # Each execution started with counter=0, added i, so should see i
            assert results[str(i)] == i

        # Original toolkit counter should still be 0
        assert toolkit.context.get("counter") == 0

    def test_execution_context_independent_toolkit_isolation(self, mock_provider):
        """Test that independent toolkit is also deep copied."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Add tool to independent toolkit with state
        execution_log = []

        def stateful_tool(value: str) -> Tuple[str, dict]:
            execution_log.append(value)
            return f"Processed {value}", {}

        agent.add_tool("stateful", stateful_tool)

        # Create two contexts
        context1 = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        context2 = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Verify they are different instances
        assert context1.independent_toolkit is not context2.independent_toolkit
        assert context1.independent_toolkit is not agent.independent_toolkit
        assert context2.independent_toolkit is not agent.independent_toolkit

    def test_provider_property_setter(self, mock_provider):
        """Test provider property getter and setter."""
        agent = Agent(provider=mock_provider, model="gpt-4")
        assert agent.provider == mock_provider

        # Set new provider
        new_provider = Mock()
        agent.provider = new_provider
        assert agent.provider == new_provider

    def test_register_tool(self, mock_provider):
        """Test registering an existing Tool instance."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        @tool()
        def custom_tool(x: int) -> Tuple[int, dict]:
            return x * 2, {}

        # Register the tool
        agent.register_tool(custom_tool)

        # Verify it's in independent toolkit
        assert custom_tool.name in agent.independent_toolkit.tools
        assert agent.independent_toolkit.is_tool_available(custom_tool.name)

    def test_group_tool_calls_by_toolkit(self):
        """Test _group_tool_calls_by_toolkit function."""
        from bestla.yggdrasil.agent import _group_tool_calls_by_toolkit

        tool_calls = [
            {"id": "1", "function": {"name": "toolkit1::tool1", "arguments": '{"x": 1}'}},
            {"id": "2", "function": {"name": "toolkit1::tool2", "arguments": '{"y": 2}'}},
            {"id": "3", "function": {"name": "toolkit2::tool3", "arguments": '{"z": 3}'}},
            {"id": "4", "function": {"name": "independent_tool", "arguments": '{"a": 4}'}},
        ]

        groups = _group_tool_calls_by_toolkit(tool_calls)

        # Check grouping
        assert "toolkit1" in groups
        assert "toolkit2" in groups
        assert "independent" in groups

        # Check toolkit1 group
        assert len(groups["toolkit1"]) == 2
        assert groups["toolkit1"][0]["name"] == "tool1"
        assert groups["toolkit1"][1]["name"] == "tool2"

        # Check toolkit2 group
        assert len(groups["toolkit2"]) == 1
        assert groups["toolkit2"][0]["name"] == "tool3"

        # Check independent group
        assert len(groups["independent"]) == 1
        assert groups["independent"][0]["name"] == "independent_tool"

    def test_toolkit_pipeline_error_handling(self, mock_provider):
        """Test handling of ToolkitPipelineError with partial results."""

        agent = Agent(provider=mock_provider, model="gpt-4")
        toolkit = Toolkit()

        # Create tools where second one fails
        @tool(enables_states=["step2_enabled"])
        def step1() -> Tuple[str, dict]:
            return "success", {"step1": True}

        @tool(required_context=["step1"], required_states=["step2_enabled"])
        def step2_fail() -> Tuple[str, dict]:
            raise ValueError("Intentional failure")

        toolkit.register_tool(step1)
        toolkit.register_tool(step2_fail)
        agent.add_toolkit("test", toolkit)

        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Execute - should handle pipeline error
        results = agent._execute_toolkit_group(
            context,
            "test",
            [
                {"id": "call_1", "name": "step1", "arguments": {}},
                {"id": "call_2", "name": "step2_fail", "arguments": {}},
            ],
        )

        # Should have results for both calls
        assert len(results) == 2
        # First should succeed
        assert "success" in results[0]["content"]
        # Second should show error
        assert "Error" in results[1]["content"]

    def test_run_max_iterations(self, mock_provider):
        """Test that run() stops at max_iterations."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Add a simple tool
        agent.add_tool("test_tool", lambda: ("result", {}))

        # Create mock tool call that supports both attribute and dict access
        class MockToolCall:
            def __init__(self):
                self.id = "call_1"
                self.function = Mock()
                self.function.name = "test_tool"
                self.function.arguments = "{}"

            def __getitem__(self, key):
                if key == "id":
                    return self.id
                elif key == "function":
                    return {"name": self.function.name, "arguments": self.function.arguments}
                raise KeyError(key)

        # Mock provider to always return tool calls (infinite loop scenario)
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Thinking..."
        mock_response.choices[0].message.tool_calls = [MockToolCall()]

        mock_provider.chat.completions.create.return_value = mock_response

        # Run with low max_iterations
        result = agent.run("test query", max_iterations=3)

        # Should hit max iterations
        assert "Maximum iterations reached" in result
        # Should have made 3 LLM calls
        assert mock_provider.chat.completions.create.call_count == 3

    def test_run_no_tool_calls(self, mock_provider):
        """Test run() when LLM returns no tool calls (normal completion)."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Mock provider to return a simple response without tool calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Here is my answer"
        mock_response.choices[0].message.tool_calls = None

        mock_provider.chat.completions.create.return_value = mock_response

        result = agent.run("What is 2+2?")

        assert result == "Here is my answer"
        assert mock_provider.chat.completions.create.call_count == 1

    def test_run_system_message_insertion(self, mock_provider):
        """Test that run() properly inserts system message."""
        agent = Agent(provider=mock_provider, model="gpt-4", system_prompt="Custom system prompt")

        # Mock provider
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = None

        mock_provider.chat.completions.create.return_value = mock_response

        # Clear any existing messages
        agent.clear_messages()

        # Run
        agent.run("Test query")

        # Check messages
        assert len(agent.messages) >= 2
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] == "Custom system prompt"
        assert agent.messages[1]["role"] == "user"

    def test_run_system_message_not_duplicated(self, mock_provider):
        """Test that system message is not duplicated on subsequent runs."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Mock provider
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = None

        mock_provider.chat.completions.create.return_value = mock_response

        # First run
        agent.run("First query")

        # Second run
        agent.run("Second query")

        # Should only have one system message
        system_messages = [m for m in agent.messages if m.get("role") == "system"]
        assert len(system_messages) == 1

    def test_repr(self, mock_provider):
        """Test __repr__ method."""
        agent = Agent(provider=mock_provider, model="gpt-4o")

        # Add some toolkits and tools
        toolkit1 = Toolkit()
        toolkit2 = Toolkit()
        agent.add_toolkit("tk1", toolkit1)
        agent.add_toolkit("tk2", toolkit2)

        agent.add_tool("tool1", lambda: ("r", {}))
        agent.add_tool("tool2", lambda: ("r", {}))

        repr_str = repr(agent)

        assert "gpt-4o" in repr_str
        assert "toolkits=2" in repr_str
        assert "independent_tools=2" in repr_str

    def test_single_toolkit_group_execution(self, mock_provider):
        """Test execution optimization for single toolkit group."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        toolkit = Toolkit()

        @tool()
        def single_tool(x: int) -> Tuple[int, dict]:
            return x * 2, {}

        toolkit.register_tool(single_tool)
        agent.add_toolkit("test", toolkit)

        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Execute with only one toolkit group (should skip ThreadPoolExecutor)
        tool_calls = [
            {"id": "1", "function": {"name": "test::single_tool", "arguments": '{"x": 5}'}},
        ]

        results = agent._execute_tool_calls(context, tool_calls)

        assert len(results) == 1
        assert "10" in results[0]["content"]

    def test_tool_execution_error_handling(self, mock_provider):
        """Test error handling when tool execution fails."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        def failing_tool() -> Tuple[str, dict]:
            raise RuntimeError("Tool failed")

        agent.add_tool("failing_tool", failing_tool)

        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        results = agent._execute_toolkit_group(
            context, "independent", [{"id": "call_1", "name": "failing_tool", "arguments": {}}]
        )

        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert "Error" in results[0]["content"]
        assert "Tool failed" in results[0]["content"]

    def test_custom_system_prompt(self, mock_provider):
        """Test agent creation with custom system prompt."""
        custom_prompt = "You are a specialized assistant for testing."
        agent = Agent(provider=mock_provider, model="gpt-4", system_prompt=custom_prompt)

        assert agent.system_prompt == custom_prompt

    def test_default_system_prompt(self, mock_provider):
        """Test agent creation with default system prompt."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        assert agent.system_prompt == "You are a helpful assistant."

    def test_mixed_toolkit_and_independent_tools(self, mock_provider):
        """Test execution with both toolkit and independent tools in same call."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Add toolkit
        toolkit = Toolkit()

        @tool()
        def toolkit_tool(x: int) -> Tuple[int, dict]:
            return x + 1, {}

        toolkit.register_tool(toolkit_tool)
        agent.add_toolkit("tk", toolkit)

        # Add independent tool
        def independent_tool(y: int) -> Tuple[int, dict]:
            return y * 2, {}

        agent.add_tool("independent_tool", independent_tool)

        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Execute both types
        tool_calls = [
            {"id": "1", "function": {"name": "tk::toolkit_tool", "arguments": '{"x": 5}'}},
            {"id": "2", "function": {"name": "independent_tool", "arguments": '{"y": 3}'}},
        ]

        results = agent._execute_tool_calls(context, tool_calls)

        assert len(results) == 2
        # Find results by ID
        results_by_id = {r["tool_call_id"]: r for r in results}
        assert "6" in results_by_id["1"]["content"]
        assert "6" in results_by_id["2"]["content"]
