"""Tests for Agent class."""

import pytest
from typing import Tuple
from unittest.mock import Mock, MagicMock, patch
from bestla.yggdrasil import Agent, Toolkit, Tool, tool


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
        toolkit.set_available_tools({"test"})

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
        toolkit.set_available_tools({"tool1"})
        agent.add_toolkit("prefix", toolkit)

        # Add independent tool
        agent.add_tool("add", lambda a, b: (a + b, {}))

        schemas = agent._generate_all_schemas()

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
        toolkit.set_available_tools({"test_tool"})
        agent.add_toolkit("test", toolkit)

        # Execute group
        results = agent._execute_toolkit_group("test", [
            {"id": "call_1", "name": "test_tool", "arguments": {"x": 5}}
        ])

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

        # Execute both
        results = agent._execute_toolkit_group("independent", [
            {"id": "call_1", "name": "add", "arguments": {"a": 5, "b": 3}},
            {"id": "call_2", "name": "multiply", "arguments": {"a": 4, "b": 2}}
        ])

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
        """Test that sub-agent toolkits are isolated."""
        toolkit = Toolkit()
        toolkit.context.set("key", "original")

        agent = Agent(provider=mock_provider, model="gpt-4")
        agent.add_toolkit("test", toolkit)

        # Mock the run method to modify context
        original_run = agent.run

        def mock_run(query):
            agent.toolkits["test"].context.set("key", "modified")
            return "result"

        agent.run = mock_run

        # Execute as tool
        result, _ = agent.execute("test query")

        # Original toolkit context should be restored
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
        toolkit1.set_available_tools({"tool1"})
        toolkit2.set_available_tools({"tool2"})

        agent.add_toolkit("tk1", toolkit1)
        agent.add_toolkit("tk2", toolkit2)

        # Execute both (different toolkits, should be parallel)
        tool_calls = [
            {"id": "1", "function": {"name": "tk1::tool1", "arguments": "{}"}},
            {"id": "2", "function": {"name": "tk2::tool2", "arguments": "{}"}},
        ]

        results = agent._execute_tool_calls(tool_calls)

        # Both should execute
        assert len(results) == 2

    def test_sequential_within_toolkit(self, mock_provider):
        """Test that tools in same toolkit execute sequentially."""
        agent = Agent(provider=mock_provider, model="gpt-4")
        toolkit = Toolkit()

        execution_log = []

        @tool(provides=["step1"], unlocks=["step2"])
        def step1() -> Tuple[str, dict]:
            execution_log.append("step1")
            return "step1", {"step1": True}

        @tool(requires=["step1"])
        def step2() -> Tuple[str, dict]:
            execution_log.append("step2")
            # This should only run after step1
            assert toolkit.context.has("step1")
            return "step2", {}

        toolkit.register_tool(step1)
        toolkit.register_tool(step2)
        toolkit.set_available_tools({"step1"})

        agent.add_toolkit("test", toolkit)

        # Execute sequence
        tool_calls = [
            {"id": "1", "function": {"name": "test::step1", "arguments": "{}"}},
            {"id": "2", "function": {"name": "test::step2", "arguments": "{}"}},
        ]

        results = agent._execute_tool_calls(tool_calls)

        # Check execution order
        assert execution_log == ["step1", "step2"]
        assert len(results) == 2

    def test_unknown_toolkit_error(self, mock_provider):
        """Test error handling for unknown toolkit."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        tool_calls = [
            {"id": "1", "name": "nonexistent_tool", "arguments": {}}
        ]

        results = agent._execute_toolkit_group("unknown_toolkit", tool_calls)

        assert len(results) == 1
        assert "Unknown toolkit" in results[0]["content"]
