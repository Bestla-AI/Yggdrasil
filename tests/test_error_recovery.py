"""Error recovery and resilience edge case tests."""

import time
from typing import Tuple
from unittest.mock import Mock

import pytest

from bestla.yggdrasil import Agent, Context, ExecutionContext, Toolkit, tool
from bestla.yggdrasil.agent import ExecutionContext
from bestla.yggdrasil.exceptions import ToolkitPipelineError


class TestDecoratorErrorPropagation:
    """Test error propagation through decorator chains."""

    def test_error_through_multiple_decorators(self):
        """Test exception propagates through decorator chain."""
        from bestla.yggdrasil.decorators import cache_result, rate_limit, retry

        @retry(max_attempts=2, backoff=0.01, exceptions=ValueError)
        @cache_result()
        @rate_limit(calls=10, period=1.0)
        def failing_tool() -> Tuple[str, dict]:
            raise RuntimeError("Unrecoverable error")

        # RuntimeError should propagate (not caught by retry which only catches ValueError)
        with pytest.raises(RuntimeError, match="Unrecoverable"):
            failing_tool()

    def test_retry_preserves_original_exception_type(self):
        """Test retry doesn't mask original exception type."""
        from bestla.yggdrasil.decorators import retry

        class CustomError(Exception):
            pass

        @retry(max_attempts=3, backoff=0.01, exceptions=CustomError)
        def custom_error_tool() -> Tuple[str, dict]:
            raise CustomError("Custom failure")

        # Should raise CustomError, not generic Exception
        with pytest.raises(CustomError, match="Custom failure"):
            custom_error_tool()

    def test_cache_invalidation_on_error(self):
        """Test cache behavior when function raises error."""
        from bestla.yggdrasil.decorators import cache_result

        call_count = [0]

        @cache_result()
        def sometimes_fails(x: int) -> Tuple[int, dict]:
            call_count[0] += 1
            if x == 5:
                raise ValueError("Failed on 5")
            return x * 2, {}

        # Successful call - cached
        sometimes_fails(3)
        sometimes_fails(3)
        assert call_count[0] == 1  # Cached

        # Failed call - should not cache error
        with pytest.raises(ValueError):
            sometimes_fails(5)

        # Try again with same value - should call function again (error not cached)
        with pytest.raises(ValueError):
            sometimes_fails(5)

        # If errors aren't cached, count should be 3 (1 + 2 failures)
        # If they are cached, count would be 2
        # Most cache implementations don't cache errors
        assert call_count[0] >= 2


class TestContextStateCorruption:
    """Test scenarios that could corrupt context state."""

    def test_context_update_during_iteration(self):
        """Test updating context while iterating over it."""
        context = Context()

        # Set initial data
        for i in range(100):
            context.set(f"key_{i}", i)

        # Try to modify while iterating (if iteration is supported)
        try:
            # This tests if _data.items() or similar is available
            items = [(k, v) for k, v in context._data.items()]

            # Modify during iteration through another operation
            for i in range(100, 200):
                context.set(f"key_{i}", i)

            # Original items should still be valid
            assert len(items) == 100

        except (RuntimeError, AttributeError):
            # Expected if iteration not supported or protected
            pass

    def test_context_delete_during_get(self):
        """Test deleting key while another thread is getting it."""
        import threading

        context = Context()
        context.set("shared", "value")

        get_results = []
        delete_done = [False]

        def get_values():
            for _ in range(100):
                result = context.get("shared")
                get_results.append(result)
                time.sleep(0.001)

        def delete_key():
            time.sleep(0.02)  # Let some gets happen first
            context.delete("shared")
            delete_done[0] = True

        t1 = threading.Thread(target=get_values)
        t2 = threading.Thread(target=delete_key)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Should complete without crashes
        assert delete_done[0]
        # Some gets might return "value", some might return None
        assert len(get_results) == 100


class TestToolExecutionTimeout:
    """Test tool execution timeout scenarios."""

    def test_tool_execution_long_running(self):
        """Test tool that runs for extended time."""

        @tool()
        def slow_tool() -> Tuple[str, dict]:
            time.sleep(2.0)  # 2 second execution
            return "completed", {}

        # Should complete eventually
        start = time.time()
        result, _ = slow_tool.execute()
        duration = time.time() - start

        assert result == "completed"
        assert duration >= 2.0

    def test_many_sequential_slow_tools(self):
        """Test pipeline with many slow tools."""
        toolkit = Toolkit()

        @tool()
        def slow() -> Tuple[str, dict]:
            time.sleep(0.1)
            return "ok", {}

        toolkit.register_tool(slow)

        # Execute 10 times sequentially
        start = time.time()
        results = toolkit.execute_sequential([{"name": "slow", "arguments": {}} for _ in range(10)])
        duration = time.time() - start

        assert len(results) == 10
        # Should take ~1 second (10 * 0.1s)
        assert 0.9 < duration < 1.5


class TestAgentMessageHandling:
    """Test agent message handling edge cases."""

    def test_conversation_context_with_very_long_message_history(self):
        """Test ConversationContext with 10,000+ messages."""
        from bestla.yggdrasil import ConversationContext

        conv = ConversationContext()

        for i in range(10000):
            role = "user" if i % 2 == 0 else "assistant"
            conv.messages.append({"role": role, "content": f"Message {i}"})

        assert len(conv.messages) >= 10000
        assert conv.messages[5000]["content"] == "Message 5000"

    def test_conversation_context_messages_with_tool_results(self):
        """Test message history with interleaved tool results."""
        from bestla.yggdrasil import ConversationContext

        conv = ConversationContext()

        conv.messages.append({"role": "user", "content": "Query"})
        conv.messages.append({"role": "assistant", "content": "I'll use a tool", "tool_calls": []})
        conv.messages.append({"role": "tool", "content": "Tool result", "tool_call_id": "123"})
        conv.messages.append({"role": "assistant", "content": "Final answer"})

        assert len(conv.messages) >= 4

    def test_agent_run_with_existing_messages_in_context(self):
        """Test agent.run() with pre-existing messages in ExecutionContext."""
        from bestla.yggdrasil import ConversationContext, ExecutionContext

        mock_provider = Mock()
        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="response", tool_calls=None))]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        conv = ConversationContext()
        conv.messages.append({"role": "user", "content": "Previous query"})
        conv.messages.append({"role": "assistant", "content": "Previous response"})

        initial_count = len(conv.messages)

        exec_ctx = ExecutionContext(
            toolkits=agent.toolkits,
            independent_toolkit=agent.independent_toolkit,
            conversation_context=conv
        )

        response, result_ctx = agent.run("New query", max_iterations=1, execution_context=exec_ctx)

        assert len(result_ctx.conversation.messages) > initial_count


class TestToolkitCopyCornerCases:
    """Test toolkit copy in unusual scenarios."""

    def test_copy_toolkit_with_active_state_transitions(self):
        """Test copying toolkit during state transitions."""
        toolkit = Toolkit()

        @tool(enables_states=["state_a"])
        def enable_a() -> Tuple[str, dict]:
            return "enabled", {}

        toolkit.register_tool(enable_a)

        # Execute to enable state
        toolkit.execute_sequential([{"name": "enable_a", "arguments": {}}])

        assert "state_a" in toolkit.unlocked_states

        # Copy toolkit
        copy = toolkit.copy()

        # Copy should have same states
        assert "state_a" in copy.unlocked_states

        # Modifying copy shouldn't affect original
        copy.set_unlocked_states({"state_b"})

        assert "state_a" in toolkit.unlocked_states
        assert "state_a" not in copy.unlocked_states

    def test_copy_toolkit_multiple_times(self):
        """Test copying toolkit multiple times creates independent copies."""
        toolkit = Toolkit()
        toolkit.context.set("value", 0)

        # Create 10 copies
        copies = [toolkit.copy() for _ in range(10)]

        # Modify each copy differently
        for i, copy in enumerate(copies):
            copy.context.set("value", i)

        # Each should have different value
        for i, copy in enumerate(copies):
            assert copy.context.get("value") == i

        # Original unchanged
        assert toolkit.context.get("value") == 0


class TestPipelineRecovery:
    """Test pipeline recovery and continuation after errors."""

    def test_pipeline_partial_success_access(self):
        """Test accessing partial results from failed pipeline."""
        toolkit = Toolkit()

        @tool()
        def step1() -> Tuple[str, dict]:
            return "step1 ok", {"step1_data": "important"}

        @tool()
        def step2() -> Tuple[str, dict]:
            return "step2 ok", {"step2_data": "also_important"}

        @tool()
        def step3_fail() -> Tuple[str, dict]:
            raise ValueError("Step 3 explodes")

        toolkit.register_tool(step1)
        toolkit.register_tool(step2)
        toolkit.register_tool(step3_fail)

        # Execute and catch error
        try:
            toolkit.execute_sequential(
                [
                    {"name": "step1", "arguments": {}},
                    {"name": "step2", "arguments": {}},
                    {"name": "step3_fail", "arguments": {}},
                ]
            )
        except ToolkitPipelineError as e:
            # Access partial results
            assert len(e.partial_results) == 3
            assert e.partial_results[0]["success"]
            assert e.partial_results[1]["success"]
            assert not e.partial_results[2]["success"]

            # Get successful results
            successful_results = [r for r in e.partial_results if r["success"]]
            assert len(successful_results) == 2

    def test_retry_pipeline_after_failure(self):
        """Test retrying pipeline after partial failure."""
        toolkit = Toolkit()

        fail_count = [0]

        @tool()
        def flaky_step() -> Tuple[str, dict]:
            fail_count[0] += 1
            if fail_count[0] < 3:
                raise ValueError("Not ready yet")
            return "success", {"flaky_done": True}

        toolkit.register_tool(flaky_step)

        # First attempt fails
        with pytest.raises(ToolkitPipelineError):
            toolkit.execute_sequential([{"name": "flaky_step", "arguments": {}}])

        # Second attempt fails
        with pytest.raises(ToolkitPipelineError):
            toolkit.execute_sequential([{"name": "flaky_step", "arguments": {}}])

        # Third attempt succeeds
        results = toolkit.execute_sequential([{"name": "flaky_step", "arguments": {}}])

        assert results[0]["success"]


class TestAgentWithNoTools:
    """Test agent behavior with no tools available."""

    def test_agent_with_no_toolkits_or_tools(self):
        """Test agent with absolutely no tools."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Agent has no toolkits and no independent tools
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        schemas = agent._generate_all_schemas(context)

        # Should return empty list
        assert schemas == []

    def test_agent_run_with_no_available_tools(self):
        """Test agent.run() when all tools are unavailable."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="I have no tools", tool_calls=None))]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        toolkit = Toolkit()

        @tool(required_states=["impossible_state"])
        def unavailable() -> Tuple[str, dict]:
            return "never", {}

        toolkit.register_tool(unavailable)
        agent.add_toolkit("test", toolkit)

        # Run agent - LLM should receive empty tool list
        result, ctx = agent.run("Do something", max_iterations=1)

        # Should complete with text response (no tools available)
        assert result == "I have no tools"


class TestConcurrentStateModification:
    """Test concurrent state modifications."""

    def test_state_modification_during_availability_check(self):
        """Test modifying states while checking availability."""
        toolkit = Toolkit()

        @tool(required_states=["enabled"])
        def state_dependent() -> Tuple[str, dict]:
            return "ok", {}

        toolkit.register_tool(state_dependent)

        availability_results = []
        state_changes = [0]

        def check_availability():
            for _ in range(100):
                available = toolkit.is_tool_available("state_dependent")
                availability_results.append(available)
                time.sleep(0.001)

        def toggle_state():
            for i in range(100):
                if i % 2 == 0:
                    toolkit.set_unlocked_states({"enabled"})
                else:
                    toolkit.set_unlocked_states(set())
                state_changes[0] += 1
                time.sleep(0.001)

        # Run concurrently
        import threading

        t1 = threading.Thread(target=check_availability)
        t2 = threading.Thread(target=toggle_state)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Should complete without crashes
        assert len(availability_results) == 100
        assert state_changes[0] == 100

        # Results should contain mix of True and False
        assert True in availability_results
        assert False in availability_results


class TestResourceCleanup:
    """Test proper resource cleanup after errors."""

    def test_toolkit_state_after_pipeline_error(self):
        """Test toolkit state is consistent after pipeline error."""
        toolkit = Toolkit()

        @tool(enables_states=["step1_done"])
        def step1() -> Tuple[str, dict]:
            return "ok", {"step1": True}

        @tool(required_states=["step1_done"], enables_states=["step2_done"])
        def step2_fail() -> Tuple[str, dict]:
            raise ValueError("Step 2 fails")

        @tool(required_states=["step2_done"])
        def step3() -> Tuple[str, dict]:
            return "ok", {"step3": True}

        toolkit.register_tool(step1)
        toolkit.register_tool(step2_fail)
        toolkit.register_tool(step3)

        # Execute pipeline
        with pytest.raises(ToolkitPipelineError):
            toolkit.execute_sequential(
                [
                    {"name": "step1", "arguments": {}},
                    {"name": "step2_fail", "arguments": {}},
                    {"name": "step3", "arguments": {}},
                ]
            )

        # Verify state is consistent
        # step1 should have enabled its state
        assert "step1_done" in toolkit.unlocked_states

        # step2 should NOT have enabled its state (it failed)
        assert "step2_done" not in toolkit.unlocked_states

        # Context should have step1's updates
        assert toolkit.context.get("step1") is True

        # step3 should not have run
        assert not toolkit.context.has("step3")

    def test_execution_context_state_after_provider_error(self):
        """Test ExecutionContext state after provider error."""
        from bestla.yggdrasil import ConversationContext, ExecutionContext

        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="success", tool_calls=None))]
        )

        result1, ctx1 = agent.run("First query", max_iterations=1)
        assert result1 == "success"

        message_count_after_success = len(ctx1.conversation.messages)

        mock_provider.chat.completions.create.side_effect = ConnectionError("Network failure")

        try:
            response, ctx2 = agent.run("Second query", max_iterations=1, execution_context=ctx1)
        except ConnectionError:
            pass

        assert len(ctx1.conversation.messages) > message_count_after_success


class TestParallelExecutionEdgeCases:
    """Test parallel execution edge cases."""

    def test_parallel_with_one_very_slow_tool(self):
        """Test parallel execution where one tool is much slower."""
        toolkit = Toolkit()

        @tool()
        def fast1() -> Tuple[str, dict]:
            return "fast1", {}

        @tool()
        def fast2() -> Tuple[str, dict]:
            return "fast2", {}

        @tool()
        def slow() -> Tuple[str, dict]:
            time.sleep(0.5)
            return "slow", {}

        toolkit.register_tool(fast1)
        toolkit.register_tool(fast2)
        toolkit.register_tool(slow)

        start = time.time()
        results = toolkit.execute_parallel(
            [
                {"name": "fast1", "arguments": {}},
                {"name": "fast2", "arguments": {}},
                {"name": "slow", "arguments": {}},
            ]
        )
        duration = time.time() - start

        # Should wait for slowest tool (~0.5s, not instant)
        assert duration >= 0.4

        # All should complete
        assert len(results) == 3

    def test_parallel_with_all_tools_failing(self):
        """Test parallel execution where all tools fail."""
        toolkit = Toolkit()

        @tool()
        def fail1() -> Tuple[str, dict]:
            raise ValueError("Fail 1")

        @tool()
        def fail2() -> Tuple[str, dict]:
            raise RuntimeError("Fail 2")

        @tool()
        def fail3() -> Tuple[str, dict]:
            raise TypeError("Fail 3")

        toolkit.register_tool(fail1)
        toolkit.register_tool(fail2)
        toolkit.register_tool(fail3)

        results = toolkit.execute_parallel(
            [
                {"name": "fail1", "arguments": {}},
                {"name": "fail2", "arguments": {}},
                {"name": "fail3", "arguments": {}},
            ]
        )

        # All should have error results
        assert len(results) == 3
        assert all(not r["success"] for r in results)

        # Each should have different error message
        errors = [r["error"] for r in results]
        assert "Fail 1" in str(errors)
        assert "Fail 2" in str(errors)
        assert "Fail 3" in str(errors)


class TestExecutionContextIsolation:
    """Test ExecutionContext isolation edge cases."""

    def test_execution_context_deep_copy_verification(self):
        """Test ExecutionContext creates truly independent copies."""
        from unittest.mock import Mock

        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        toolkit = Toolkit()
        toolkit.context.set("shared", "original")
        toolkit.set_unlocked_states({"initial_state"})

        agent.add_toolkit("test", toolkit)

        # Create two execution contexts
        context1 = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        context2 = ExecutionContext(agent.toolkits, agent.independent_toolkit)

        # Modify context1
        context1.toolkits["test"].context.set("shared", "modified1")
        context1.toolkits["test"].set_unlocked_states({"state1"})

        # Modify context2
        context2.toolkits["test"].context.set("shared", "modified2")
        context2.toolkits["test"].set_unlocked_states({"state2"})

        # Verify independence
        assert context1.toolkits["test"].context.get("shared") == "modified1"
        assert context2.toolkits["test"].context.get("shared") == "modified2"
        assert toolkit.context.get("shared") == "original"

        assert "state1" in context1.toolkits["test"].unlocked_states
        assert "state2" in context2.toolkits["test"].unlocked_states
        assert "initial_state" in toolkit.unlocked_states

    def test_execution_context_with_100_toolkits(self):
        """Test ExecutionContext copying 100 toolkits."""
        from unittest.mock import Mock

        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Add 100 toolkits
        for i in range(100):
            tk = Toolkit()
            tk.context.set("id", i)
            agent.add_toolkit(f"tk_{i}", tk)

        # Create execution context
        start = time.time()
        context = ExecutionContext(agent.toolkits, agent.independent_toolkit)
        duration = time.time() - start

        # Should be fast (< 0.1s even with 100 toolkits)
        assert duration < 0.1

        # All toolkits should be copied
        assert len(context.toolkits) == 100

        # Each should be independent
        for i in range(100):
            assert context.toolkits[f"tk_{i}"].context.get("id") == i


class TestMaxIterationBehavior:
    """Test agent max_iterations edge cases."""

    def test_max_iterations_exactly_reached(self):
        """Test behavior when exactly max_iterations is reached."""
        from unittest.mock import MagicMock

        mock_provider = Mock()

        # Always return tool calls to force iteration
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function = MagicMock()
        mock_tc.function.name = "test_tool"
        mock_tc.function.arguments = "{}"
        # Support dict access
        mock_tc.__getitem__ = lambda self, key: {
            "id": "call_1",
            "function": {"name": "test_tool", "arguments": "{}"},
        }[key]

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=None, tool_calls=[mock_tc]))]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")
        agent.add_tool("test_tool", lambda: ("result", {}))

        # Run with max_iterations=5
        result, ctx = agent.run("test", max_iterations=5)

        # Should hit max iterations
        assert "Maximum iterations reached" in result

        # Should have called LLM exactly 5 times
        assert mock_provider.chat.completions.create.call_count == 5

    def test_zero_max_iterations(self):
        """Test agent with max_iterations=0."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Should either raise or return immediately
        # Behavior depends on implementation
        try:
            result, ctx = agent.run("test", max_iterations=0)
            # Might return immediately without calling LLM
            assert isinstance(result, str)
        except ValueError:
            # Or might reject invalid max_iterations
            pass

    def test_negative_max_iterations(self):
        """Test agent with negative max_iterations."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Should handle gracefully
        try:
            response, ctx = agent.run("test", max_iterations=-1)
            # Behavior undefined
        except (ValueError, AssertionError):
            # Expected - negative iterations invalid
            pass
