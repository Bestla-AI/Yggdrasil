"""Tests for parallel execution of independent tools."""

import time
from typing import Tuple
from unittest.mock import Mock

import pytest

from bestla.yggdrasil import Agent, Toolkit, tool
from bestla.yggdrasil.agent import ExecutionContext


@pytest.fixture
def mock_provider():
    """Create a mock OpenAI provider."""
    return Mock()


class TestParallelExecution:
    """Test parallel execution functionality."""

    def test_independent_tools_run_in_parallel(self, mock_provider):
        """Test that independent tools actually execute in parallel."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        execution_times = []

        def slow_tool_1() -> Tuple[str, dict]:
            """Tool that takes 100ms."""
            time.sleep(0.1)
            execution_times.append(("tool1", time.time()))
            return "result1", {}

        def slow_tool_2() -> Tuple[str, dict]:
            """Tool that takes 100ms."""
            time.sleep(0.1)
            execution_times.append(("tool2", time.time()))
            return "result2", {}

        def slow_tool_3() -> Tuple[str, dict]:
            """Tool that takes 100ms."""
            time.sleep(0.1)
            execution_times.append(("tool3", time.time()))
            return "result3", {}

        agent.add_tool("slow1", slow_tool_1)
        agent.add_tool("slow2", slow_tool_2)
        agent.add_tool("slow3", slow_tool_3)

        # Execute all three in parallel
        start_time = time.time()
        tool_calls = [
            {"id": "1", "function": {"name": "slow1", "arguments": "{}"}},
            {"id": "2", "function": {"name": "slow2", "arguments": "{}"}},
            {"id": "3", "function": {"name": "slow3", "arguments": "{}"}},
        ]

        # Create execution context
        context = ExecutionContext(
            toolkits=agent.toolkits,
            independent_toolkit=agent.independent_toolkit
        )
        results = agent._execute_tool_calls(context, tool_calls)
        end_time = time.time()

        # If parallel: ~0.1s, if sequential: ~0.3s
        total_time = end_time - start_time

        # Should complete in roughly 0.1s (parallel) not 0.3s (sequential)
        # Allow some overhead
        assert total_time < 0.2, f"Tools ran sequentially ({total_time:.2f}s), not in parallel"

        # All three should complete
        assert len(results) == 3
        assert all(r["role"] == "tool" for r in results)

    def test_toolkit_tools_run_sequentially(self):
        """Test that toolkit tools run sequentially (not parallel)."""
        toolkit = Toolkit()
        execution_order = []

        @tool()
        def step1() -> Tuple[str, dict]:
            execution_order.append("step1_start")
            time.sleep(0.05)
            execution_order.append("step1_end")
            return "step1", {}

        @tool()
        def step2() -> Tuple[str, dict]:
            execution_order.append("step2_start")
            time.sleep(0.05)
            execution_order.append("step2_end")
            return "step2", {}

        toolkit.register_tool(step1)
        toolkit.register_tool(step2)

        # Execute sequentially
        results = toolkit.execute_sequential([
            {"name": "step1", "arguments": {}},
            {"name": "step2", "arguments": {}}
        ])

        # Should execute in order: step1_start, step1_end, step2_start, step2_end
        assert execution_order == ["step1_start", "step1_end", "step2_start", "step2_end"]
        assert len(results) == 2

    def test_parallel_context_updates_merge(self, mock_provider):
        """Test that context updates from parallel tools merge correctly."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        def tool1() -> Tuple[str, dict]:
            return "result1", {"key1": "value1", "shared": "from_tool1"}

        def tool2() -> Tuple[str, dict]:
            return "result2", {"key2": "value2", "shared": "from_tool2"}

        def tool3() -> Tuple[str, dict]:
            return "result3", {"key3": "value3"}

        agent.add_tool("tool1", tool1)
        agent.add_tool("tool2", tool2)
        agent.add_tool("tool3", tool3)

        # Execute in parallel
        tool_calls = [
            {"id": "1", "function": {"name": "tool1", "arguments": "{}"}},
            {"id": "2", "function": {"name": "tool2", "arguments": "{}"}},
            {"id": "3", "function": {"name": "tool3", "arguments": "{}"}},
        ]

        # Create execution context
        context = ExecutionContext(
            toolkits=agent.toolkits,
            independent_toolkit=agent.independent_toolkit
        )
        results = agent._execute_tool_calls(context, tool_calls)

        # All should succeed
        assert len(results) == 3

        # Context should have all unique keys
        # Note: we check the execution context's independent_toolkit, not the agent's
        exec_context = context.independent_toolkit.context
        assert exec_context.has("key1")
        assert exec_context.has("key2")
        assert exec_context.has("key3")

        # Shared key should have one of the values (last-write-wins)
        assert exec_context.has("shared")
        assert exec_context.get("shared") in ["from_tool1", "from_tool2"]

    def test_parallel_tool_failure_doesnt_stop_others(self, mock_provider):
        """Test that failure of one parallel tool doesn't stop others."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        executed = []

        def success_tool1() -> Tuple[str, dict]:
            executed.append("tool1")
            return "success1", {}

        def failing_tool() -> Tuple[str, dict]:
            executed.append("failing")
            raise ValueError("Tool failed!")

        def success_tool2() -> Tuple[str, dict]:
            executed.append("tool2")
            return "success2", {}

        agent.add_tool("success1", success_tool1)
        agent.add_tool("failing", failing_tool)
        agent.add_tool("success2", success_tool2)

        # Execute in parallel
        tool_calls = [
            {"id": "1", "function": {"name": "success1", "arguments": "{}"}},
            {"id": "2", "function": {"name": "failing", "arguments": "{}"}},
            {"id": "3", "function": {"name": "success2", "arguments": "{}"}},
        ]

        # Create execution context
        context = ExecutionContext(
            toolkits=agent.toolkits,
            independent_toolkit=agent.independent_toolkit
        )
        results = agent._execute_tool_calls(context, tool_calls)

        # All three should have executed (failure doesn't stop others)
        assert set(executed) == {"tool1", "failing", "tool2"}

        # Should have 3 results: 2 successes, 1 failure
        assert len(results) == 3

        success_count = sum(1 for r in results if "Error" not in r["content"])
        failure_count = sum(1 for r in results if "Error" in r["content"])

        assert success_count == 2
        assert failure_count == 1

    def test_mixed_toolkit_and_independent_parallel(self, mock_provider):
        """Test that toolkit tools and independent tools can run in parallel."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Create a toolkit with sequential tools
        toolkit = Toolkit()

        toolkit_executed = []
        independent_executed = []

        @tool()
        def toolkit_tool1() -> Tuple[str, dict]:
            toolkit_executed.append("tk1_start")
            time.sleep(0.05)
            toolkit_executed.append("tk1_end")
            return "tk1", {}

        @tool()
        def toolkit_tool2() -> Tuple[str, dict]:
            toolkit_executed.append("tk2_start")
            time.sleep(0.05)
            toolkit_executed.append("tk2_end")
            return "tk2", {}

        toolkit.register_tool(toolkit_tool1)
        toolkit.register_tool(toolkit_tool2)

        agent.add_toolkit("mykit", toolkit)

        # Add independent tools
        def ind_tool() -> Tuple[str, dict]:
            independent_executed.append("ind")
            time.sleep(0.05)
            return "ind", {}

        agent.add_tool("ind", ind_tool)

        # Execute both toolkit and independent
        start_time = time.time()
        tool_calls = [
            {"id": "1", "function": {"name": "mykit::toolkit_tool1", "arguments": "{}"}},
            {"id": "2", "function": {"name": "mykit::toolkit_tool2", "arguments": "{}"}},
            {"id": "3", "function": {"name": "ind", "arguments": "{}"}},
        ]

        # Create execution context
        context = ExecutionContext(
            toolkits=agent.toolkits,
            independent_toolkit=agent.independent_toolkit
        )
        results = agent._execute_tool_calls(context, tool_calls)
        end_time = time.time()

        # Toolkit tools should be sequential: tk1_start, tk1_end, tk2_start, tk2_end
        assert toolkit_executed == ["tk1_start", "tk1_end", "tk2_start", "tk2_end"]

        # Independent tool should have executed
        assert independent_executed == ["ind"]

        # Total time should be ~0.1s (toolkit sequential but parallel with independent)
        # not 0.15s (all sequential)
        total_time = end_time - start_time
        assert total_time < 0.13, f"Not enough parallelism ({total_time:.2f}s)"

        assert len(results) == 3
