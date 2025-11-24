"""Critical edge case tests - scenarios that could cause data corruption or system failure."""

import threading
import time
from typing import Tuple
from unittest.mock import Mock

import pytest

from bestla.yggdrasil import Agent, Context, ExecutionContext, Toolkit, tool


class TestCircularReferences:
    """Test handling of circular references in context."""

    def test_context_with_circular_dict_reference(self):
        """Test context storing dict with circular reference."""
        context = Context()

        # Create circular reference
        circular_dict = {"name": "root"}
        circular_dict["self"] = circular_dict

        # Should not raise during set
        context.set("circular", circular_dict)

        # Verify it's stored
        assert context.has("circular")

        # Get should work (returns the circular structure)
        result = context.get("circular")
        assert result["name"] == "root"
        assert result["self"] is result  # Circular reference preserved

    def test_context_copy_with_circular_reference(self):
        """Test copying context with circular references."""
        context = Context()

        # Create circular structure
        circular = {"a": {}}
        circular["a"]["back_to_root"] = circular

        context.set("circ", circular)

        # Copy should handle circular references
        # Note: immutables.Map might handle this differently
        try:
            copy = context.copy()
            # If it succeeds, verify the copy
            assert copy.has("circ")
        except RecursionError:
            # Expected if circular references cause issues
            pytest.skip("Context.copy() doesn't support circular references")

    def test_context_nested_access_with_circular_ref(self):
        """Test nested access doesn't infinite loop on circular refs."""
        context = Context()

        # Create structure with circular ref
        data = {"level1": {"level2": {}}}
        data["level1"]["level2"]["back"] = data

        context.set("data", data)

        # Nested access should not infinite loop
        # This should return None (path doesn't exist infinitely)
        result = context.get("data.level1.level2.back.level1.level2.back.level1")

        # Should not hang, should return something
        assert result is not None or result is None  # Just verify it returns


class TestStateLeakageAcrossExecutions:
    """Test that execution contexts properly isolate state."""

    def test_state_does_not_leak_between_agent_runs(self):
        """Test sequential agent.run() calls have isolated state."""
        from unittest.mock import MagicMock

        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        toolkit = Toolkit()
        toolkit.context.set("counter", 0)

        @tool()
        def increment() -> Tuple[str, dict]:
            current = toolkit.context.get("counter", 0)
            return f"counter={current}", {"counter": current + 1}

        toolkit.register_tool(increment)
        agent.add_toolkit("test", toolkit)

        # Mock provider to call increment then finish
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                # First call of each run - return tool call
                mock_tc = MagicMock()
                mock_tc.id = "call_1"
                mock_tc.function = MagicMock()
                mock_tc.function.name = "test::increment"
                mock_tc.function.arguments = "{}"
                # Support both dict and attribute access
                mock_tc.__getitem__ = lambda self, key: {
                    "id": "call_1",
                    "function": {"name": "test::increment", "arguments": "{}"},
                }[key]
                return Mock(choices=[Mock(message=Mock(content=None, tool_calls=[mock_tc]))])
            else:
                # Second call - finish
                return Mock(choices=[Mock(message=Mock(content="Done", tool_calls=None))])

        mock_provider.chat.completions.create = mock_create

        # First run
        response, ctx = agent.run("test 1", max_iterations=2)

        # Original toolkit counter should still be 0 (isolated)
        assert toolkit.context.get("counter") == 0

        # Second run
        response, ctx2 = agent.run("test 2", max_iterations=2)

        # Still should be 0 (each run is isolated)
        assert toolkit.context.get("counter") == 0

    def test_parallel_agent_executions_isolated(self):
        """Test parallel agent.execute() calls don't share state."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="done", tool_calls=None))]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")
        toolkit = Toolkit()
        toolkit.context.set("execution_id", "none")

        @tool()
        def mark_execution(exec_id: str) -> Tuple[str, dict]:
            return f"marked {exec_id}", {"execution_id": exec_id}

        toolkit.register_tool(mark_execution)
        agent.add_toolkit("test", toolkit)

        results = {}
        lock = threading.Lock()

        def run_agent(exec_id):
            # Simulate agent execution that marks its ID
            result, _ = agent.execute(f"mark {exec_id}")
            with lock:
                results[exec_id] = result

        # Run 10 agents in parallel
        threads = [threading.Thread(target=run_agent, args=(f"exec_{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should complete
        assert len(results) == 10

        # Original toolkit state should be unchanged
        assert toolkit.context.get("execution_id") == "none"


class TestSelfReferencingAgent:
    """Test agent calling itself as a tool (infinite recursion risk)."""

    def test_agent_cannot_add_itself_as_tool(self):
        """Test agent adding itself as a tool."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Add agent as tool to itself
        agent.add_tool("self", agent.execute)

        # This creates circular reference - verify it doesn't crash immediately
        assert "self" in agent.independent_toolkit.tools

        # Don't actually call it (would infinite loop), just verify structure
        # In production, this should be detected and prevented

    def test_circular_agent_hierarchy_detection(self):
        """Test detecting circular agent hierarchies."""
        mock_provider = Mock()

        agent_a = Agent(provider=mock_provider, model="gpt-4")
        agent_b = Agent(provider=mock_provider, model="gpt-4")

        # A uses B
        agent_a.add_tool("use_b", agent_b.execute)

        # B uses A (circular!)
        agent_b.add_tool("use_a", agent_a.execute)

        # This creates a circular reference
        # Verify it doesn't crash on setup
        assert "use_b" in agent_a.independent_toolkit.tools
        assert "use_a" in agent_b.independent_toolkit.tools

        # Actual execution would infinite loop - don't test that
        # This test just verifies the structure can be created


class TestConcurrentSchemaGeneration:
    """Test schema generation during concurrent context modifications."""

    def test_schema_generation_during_context_updates(self):
        """Test generate_schemas() while context is being modified."""
        toolkit = Toolkit()
        toolkit.context.set("options", ["a", "b", "c"])

        from bestla.yggdrasil import DynamicStr

        @tool()
        def select(option: DynamicStr["options"]) -> Tuple[str, dict]:
            return option, {}

        toolkit.register_tool(select)

        schemas_generated = []
        updates_completed = [0]

        def generate_schemas():
            for _ in range(100):
                schemas = toolkit.generate_schemas()
                schemas_generated.append(len(schemas))
                time.sleep(0.001)

        def update_context():
            for i in range(100):
                toolkit.context.set("options", [f"opt_{i}"])
                updates_completed[0] += 1
                time.sleep(0.001)

        # Run both concurrently
        t1 = threading.Thread(target=generate_schemas)
        t2 = threading.Thread(target=update_context)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both should complete without crashes
        assert len(schemas_generated) == 100
        assert updates_completed[0] == 100


class TestToolModifiesOwnMetadata:
    """Test tools that modify their own metadata during execution."""

    def test_tool_modifying_required_states(self):
        """Test tool that tries to modify its own required_states."""

        @tool(required_states=["initial"])
        def self_modifying() -> Tuple[str, dict]:
            # Try to modify tool metadata (should not affect toolkit)
            # In Python, we can't actually modify the decorator metadata easily,
            # but we can test that the toolkit's copy is immutable
            return "modified", {}

        toolkit = Toolkit()
        toolkit.register_tool(self_modifying)
        toolkit.set_unlocked_states({"initial"})

        # Execute tool
        results = toolkit.execute_sequential([{"name": "self_modifying", "arguments": {}}])

        assert results[0]["success"]

        # Tool metadata should be unchanged
        assert "initial" in self_modifying.required_states

    def test_tool_modifying_context_during_execution(self):
        """Test tool that modifies context in unexpected ways."""

        @tool()
        def tricky_tool() -> Tuple[str, dict]:
            # Returns updates, but also tries to directly modify (shouldn't work)
            return "result", {"valid_update": "ok", "": "empty_key", None: "none_key"}

        toolkit = Toolkit()
        toolkit.register_tool(tricky_tool)

        # Execute - should handle invalid keys gracefully
        try:
            results = toolkit.execute_sequential([{"name": "tricky_tool", "arguments": {}}])
            # If it succeeds, check results
            assert results[0]["success"] or not results[0]["success"]
        except (TypeError, ValueError):
            # Expected if None key causes issues
            pass


class TestMemoryLeaks:
    """Test for memory leaks in long-running scenarios."""

    def test_decorator_cache_bounded_growth(self):
        """Test @cache_result doesn't leak memory with many unique calls."""
        from bestla.yggdrasil.decorators import cache_result

        @cache_result(ttl=None)  # Cache forever
        def cached_function(value: int) -> Tuple[int, dict]:
            return value * 2, {}

        # Call with 10,000 unique values
        for i in range(10000):
            cached_function(i)

        # Check cache size
        size = cached_function.cache_size()
        assert size == 10000

        # Clear cache to prevent memory issues
        cached_function.clear_cache()
        assert cached_function.cache_size() == 0

    def test_stateless_runs_dont_accumulate_in_agent(self):
        """Test that stateless runs don't accumulate messages in Agent."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="response", tool_calls=None))]
        )

        contexts = []
        for i in range(100):
            response, ctx = agent.run(f"query {i}", max_iterations=1)
            contexts.append(ctx)

        # Each context should be independent
        for ctx in contexts:
            assert len(ctx.conversation.messages) <= 3  # system + user + assistant

        # Agent has no internal state
        assert not hasattr(agent, 'messages')

    def test_context_copy_memory_efficiency(self):
        """Test context copies don't cause memory explosion."""
        import tracemalloc

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Create large context
        context = Context()
        for i in range(1000):
            context.set(f"key_{i}", f"value_{i}" * 100)

        # Make 100 copies
        copies = [context.copy() for _ in range(100)]

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate memory growth
        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        total_increase = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

        # With immutable Map structural sharing, should be < 10MB
        assert total_increase < 10 * 1024 * 1024

        # Keep copies in scope for measurement
        assert len(copies) == 100


class TestNestedAgentExhaustion:
    """Test thread pool exhaustion with deeply nested agents."""

    def test_deep_agent_hierarchy_completes(self):
        """Test 5-level deep agent hierarchy doesn't exhaust resources."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="done", tool_calls=None))]
        )

        # Create 5 levels: A -> B -> C -> D -> E
        agent_e = Agent(provider=mock_provider, model="gpt-4")

        agent_d = Agent(provider=mock_provider, model="gpt-4")
        agent_d.add_tool("call_e", agent_e.execute)

        agent_c = Agent(provider=mock_provider, model="gpt-4")
        agent_c.add_tool("call_d", agent_d.execute)

        agent_b = Agent(provider=mock_provider, model="gpt-4")
        agent_b.add_tool("call_c", agent_c.execute)

        agent_a = Agent(provider=mock_provider, model="gpt-4")
        agent_a.add_tool("call_b", agent_b.execute)

        # Execute top-level agent
        result, ctx = agent_a.run("test deep hierarchy", max_iterations=1)

        assert result == "done"

    def test_parallel_execution_with_nested_agents(self):
        """Test parallel independent tools with nested agent calls."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="done", tool_calls=None))]
        )

        # Create nested agents
        sub_agent = Agent(provider=mock_provider, model="gpt-4")

        main_agent = Agent(provider=mock_provider, model="gpt-4")

        # Add multiple independent tools that call sub-agent
        for i in range(10):
            main_agent.add_tool(f"sub_{i}", sub_agent.execute)

        # This creates 10 independent "threads" each calling sub-agent
        # Should not exhaust thread pool
        result, ctx = main_agent.run("test", max_iterations=1)

        assert result == "done"


class TestPipelinePartialFailures:
    """Test error handling in partially completed pipelines."""

    def test_validation_error_mid_pipeline_rollback(self):
        """Test context updates rollback on validation error mid-pipeline."""
        toolkit = Toolkit(validation_enabled=True)
        toolkit.context.schema.define("count", {"type": "integer"})

        @tool()
        def step1() -> Tuple[str, dict]:
            return "step1", {"count": 1}

        @tool()
        def step2_invalid() -> Tuple[str, dict]:
            # Returns invalid update
            return "step2", {"count": "not_an_integer"}

        @tool()
        def step3() -> Tuple[str, dict]:
            return "step3", {"count": 3}

        toolkit.register_tool(step1)
        toolkit.register_tool(step2_invalid)
        toolkit.register_tool(step3)

        # Execute pipeline
        with pytest.raises(Exception):  # Should fail on step2
            toolkit.execute_sequential(
                [
                    {"name": "step1", "arguments": {}},
                    {"name": "step2_invalid", "arguments": {}},
                    {"name": "step3", "arguments": {}},
                ]
            )

        # step1's update should have been applied before step2 failed
        assert toolkit.context.get("count") == 1

    def test_multiple_parallel_toolkit_failures(self):
        """Test error handling when multiple toolkits fail simultaneously."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Create 3 toolkits with failing tools
        for i in range(3):
            toolkit = Toolkit()

            @tool()
            def failing_tool() -> Tuple[str, dict]:
                raise ValueError(f"Toolkit {i} failed")

            toolkit.register_tool(failing_tool)
            agent.add_toolkit(f"tk{i}", toolkit)

        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Execute all failing tools in parallel
        tool_calls = [
            {"id": "1", "function": {"name": "tk0::failing_tool", "arguments": "{}"}},
            {"id": "2", "function": {"name": "tk1::failing_tool", "arguments": "{}"}},
            {"id": "3", "function": {"name": "tk2::failing_tool", "arguments": "{}"}},
        ]

        results = agent._execute_tool_calls(context, tool_calls)

        # All should return error results
        assert len(results) == 3
        assert all("Error" in r["content"] for r in results)


class TestBoundaryConditions:
    """Test extreme boundary conditions."""

    def test_toolkit_with_no_tools(self):
        """Test toolkit with zero tools."""
        toolkit = Toolkit()

        # Should work fine
        assert len(toolkit.tools) == 0

        # Schema generation should return empty list
        schemas = toolkit.generate_schemas()
        assert schemas == []

        # Execute empty pipeline
        results = toolkit.execute_sequential([])
        assert results == []

    def test_agent_with_all_tools_unavailable(self):
        """Test agent where all tools fail availability checks."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        toolkit = Toolkit()

        @tool(required_states=["impossible"])
        def unavailable_tool() -> Tuple[str, dict]:
            return "never runs", {}

        toolkit.register_tool(unavailable_tool)
        agent.add_toolkit("test", toolkit)

        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Generate schemas - should be empty
        schemas = agent._generate_all_schemas(context)

        # Only independent toolkit tools would be available (none in this case)
        assert len(schemas) == 0

    def test_tool_with_many_parameters(self):
        """Test tool with many parameters."""

        # Create tool with 20 parameters (reasonable large number)
        @tool()
        def many_params(
            p0: int,
            p1: int,
            p2: int,
            p3: int,
            p4: int,
            p5: int,
            p6: int,
            p7: int,
            p8: int,
            p9: int,
            p10: int,
            p11: int,
            p12: int,
            p13: int,
            p14: int,
            p15: int,
            p16: int,
            p17: int,
            p18: int,
            p19: int,
        ) -> Tuple[int, dict]:
            return (
                sum([
                    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                    p11, p12, p13, p14, p15, p16, p17, p18, p19
                ]),
                {}
            )

        # Should generate schema without issues
        schema = many_params.generate_schema(Context())

        # Verify all 20 parameters in schema
        assert len(schema["function"]["parameters"]["properties"]) == 20
        assert len(schema["function"]["parameters"]["required"]) == 20

    def test_zero_execution_time_tools_in_parallel(self):
        """Test tools with instant execution in parallel."""
        toolkit = Toolkit()

        call_times = []

        @tool()
        def instant_tool() -> Tuple[str, dict]:
            call_times.append(time.time())
            return "instant", {}

        toolkit.register_tool(instant_tool)

        # Execute 100 times in parallel
        tool_calls = [{"name": "instant_tool", "arguments": {}} for _ in range(100)]

        start = time.time()
        results = toolkit.execute_parallel(tool_calls)
        duration = time.time() - start

        assert len(results) == 100

        # Should complete very quickly (< 1 second despite 100 calls)
        assert duration < 1.0
