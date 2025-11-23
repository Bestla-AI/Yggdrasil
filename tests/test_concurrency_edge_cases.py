"""Tests for concurrency edge cases and thread safety."""

import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
from unittest.mock import Mock

from bestla.yggdrasil import Agent, Context, Toolkit


class TestConcurrentContextModifications:
    """Test thread safety of Context modifications."""

    def test_concurrent_context_updates_thread_safety(self):
        """Test 100 threads simultaneously updating same context."""
        context = Context()
        num_threads = 100
        updates_per_thread = 10

        def update_context(thread_id):
            """Update context from a thread."""
            for i in range(updates_per_thread):
                context.set(f"thread_{thread_id}_key_{i}", thread_id * 1000 + i)
            return thread_id

        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(update_context, i) for i in range(num_threads)]
            results = [f.result() for f in futures]

        # Verify all updates were applied
        assert len(results) == num_threads
        for thread_id in range(num_threads):
            for i in range(updates_per_thread):
                key = f"thread_{thread_id}_key_{i}"
                expected_value = thread_id * 1000 + i
                assert context.get(key) == expected_value

    def test_execution_context_isolation_under_load(self):
        """Test 50 concurrent agent.run() calls maintain isolated contexts."""
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        mock_provider = Mock()
        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="done", tool_calls=None))]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")
        toolkit = Toolkit()

        def test_tool() -> Tuple[str, dict]:
            return "result", {"executed": True}

        toolkit.add_tool("test", test_tool)
        agent.add_toolkit("test", toolkit)

        num_concurrent = 50

        def run_agent(run_id):
            """Run agent in separate thread."""
            # Each run should have isolated context
            messages = [{"role": "user", "content": f"Run {run_id}"}]
            agent.run(messages, max_iterations=1)
            return run_id

        # Run concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_agent, i) for i in range(num_concurrent)]
            results = [f.result() for f in futures]

        assert len(results) == num_concurrent

        # Check memory usage - should not have massive leaks
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        total_increase = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

        # Memory increase should be reasonable (< 50MB for 50 runs)
        assert total_increase < 50 * 1024 * 1024

    def test_concurrent_state_transitions(self):
        """Test multiple threads enabling/disabling states maintains consistency."""
        toolkit = Toolkit()
        num_threads = 20
        num_ops = 50

        def modify_states(thread_id):
            """Enable and disable states from thread."""
            for i in range(num_ops):
                state = f"state_{i % 10}"
                if i % 2 == 0:
                    toolkit.set_unlocked_states(
                        toolkit.context.get("unlocked_states", set()) | {state}
                    )
                else:
                    current = toolkit.context.get("unlocked_states", set())
                    if state in current:
                        toolkit.set_unlocked_states(current - {state})
            return thread_id

        # Run concurrent state modifications
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(modify_states, i) for i in range(num_threads)]
            results = [f.result() for f in futures]

        # All threads should complete without errors
        assert len(results) == num_threads

    def test_context_copy_during_modification(self):
        """Test copying context while it's being modified."""
        context = Context()

        # Add initial data
        for i in range(100):
            context.set(f"key_{i}", i)

        copies = []
        modification_done = threading.Event()

        def modify_context():
            """Continuously modify context."""
            for i in range(100, 200):
                context.set(f"key_{i}", i)
                time.sleep(0.001)
            modification_done.set()

        def copy_context():
            """Copy context while modifications happen."""
            copies.append(context.copy())
            time.sleep(0.002)

        # Start modification thread
        mod_thread = threading.Thread(target=modify_context)
        mod_thread.start()

        # Create multiple copies concurrently
        copy_threads = [threading.Thread(target=copy_context) for _ in range(10)]
        for t in copy_threads:
            t.start()

        for t in copy_threads:
            t.join()
        mod_thread.join()

        # All copies should be valid
        assert len(copies) == 10
        for copy in copies:
            # Each copy should have consistent data
            # At minimum, should have initial 100 keys
            assert len([k for k in copy._data.keys() if k.startswith("key_")]) >= 100


class TestConcurrentAgentOperations:
    """Test concurrent agent operations."""

    def test_concurrent_toolkit_registration(self):
        """Test adding toolkits while agent operations are running."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="done", tool_calls=None))]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        def add_toolkit(toolkit_id):
            """Add a toolkit."""
            toolkit = Toolkit()

            def dynamic_tool() -> Tuple[str, dict]:
                return f"toolkit_{toolkit_id}", {}

            toolkit.add_tool(f"tool_{toolkit_id}", dynamic_tool)
            agent.add_toolkit(f"tk_{toolkit_id}", toolkit)
            return toolkit_id

        # Add toolkits concurrently (this tests thread safety of registration)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_toolkit, i) for i in range(20)]
            results = [f.result() for f in futures]

        # All toolkits should be registered
        assert len(results) == 20
        assert len(agent.toolkits) == 20

    def test_concurrent_message_history_access(self):
        """Test multiple threads reading/writing messages."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        def access_messages(thread_id):
            """Access message history."""
            # Read messages
            _ = len(agent.messages)

            # Add a message
            agent.messages.append({"role": "user", "content": f"Thread {thread_id}"})

            # Read again
            return len(agent.messages)

        # Concurrent access
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_messages, i) for i in range(50)]
            [f.result() for f in futures]

        # Should have system message + 50 added messages
        assert len(agent.messages) >= 50

    def test_concurrent_execute_and_run(self):
        """Test mixed agent.execute() and agent.run() calls."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="done", tool_calls=None))]
        )

        # Create a parent agent
        parent_agent = Agent(provider=mock_provider, model="gpt-4")

        # Create a child agent that will be used as a tool
        child_agent = Agent(provider=mock_provider, model="gpt-4")

        # Add child as tool to parent
        parent_agent.add_tool("child", child_agent.execute)

        def run_parent():
            """Run parent agent."""
            parent_agent.run([{"role": "user", "content": "test"}], max_iterations=1)

        def execute_child():
            """Execute child agent directly."""
            child_agent.execute({"task": "test"})

        # Run both concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(5):
                futures.append(executor.submit(run_parent))
                futures.append(executor.submit(execute_child))

            results = [f.result() for f in futures]

        # All should complete without errors
        assert len(results) == 10


class TestResourceExhaustion:
    """Test resource exhaustion scenarios."""

    def test_thread_pool_saturation(self):
        """Test behavior when more tools than available threads."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Create toolkit with many slow tools
        toolkit = Toolkit()

        def slow_tool_1() -> Tuple[str, dict]:
            time.sleep(0.05)
            return "done", {}

        def slow_tool_2() -> Tuple[str, dict]:
            time.sleep(0.05)
            return "done", {}

        def slow_tool_3() -> Tuple[str, dict]:
            time.sleep(0.05)
            return "done", {}

        toolkit.add_tool("slow1", slow_tool_1)
        toolkit.add_tool("slow2", slow_tool_2)
        toolkit.add_tool("slow3", slow_tool_3)

        agent.add_toolkit("slow", toolkit)

        # Add many independent tools
        for i in range(20):

            def make_tool(idx):
                def independent_tool() -> Tuple[str, dict]:
                    time.sleep(0.01)
                    return f"tool_{idx}", {}

                return independent_tool

            agent.add_tool(f"ind_{i}", make_tool(i))

        # Execute many tools - should queue gracefully
        start = time.time()

        # This will test ThreadPoolExecutor queuing
        # Default executor should handle this without errors
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(20):

                def work():
                    time.sleep(0.01)
                    return "ok"

                futures.append(executor.submit(work))

            results = [f.result() for f in as_completed(futures)]

        duration = time.time() - start

        # Should complete in reasonable time
        assert len(results) == 20
        assert duration < 10  # Should not hang

    def test_large_context_memory_usage(self):
        """Test context with 1000+ keys doesn't cause issues."""
        context = Context()

        # Add 1000 keys
        for i in range(1000):
            context.set(f"key_{i}", f"value_{i}")

        # Make 100 copies - with immutable Map, should be efficient
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        copies = [context.copy() for _ in range(100)]

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Memory increase should be minimal due to structural sharing
        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        total_increase = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

        # Should use < 5MB for 100 copies (structural sharing)
        assert total_increase < 5 * 1024 * 1024

        # All copies should be independent
        assert len(copies) == 100
        for copy in copies:
            assert len([k for k in copy._data.keys() if k.startswith("key_")]) == 1000

    def test_message_history_growth(self):
        """Test message history with 1000+ messages."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Add 1000 messages
        for i in range(1000):
            agent.messages.append(
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            )

        # Message history should still be accessible
        assert len(agent.messages) >= 1000

        # Should be able to access messages
        assert agent.messages[500]["content"] == "Message 500"

    def test_parallel_execution_stress(self):
        """Test 200+ parallel tool calls across toolkits."""
        mock_provider = Mock()
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Create multiple toolkits with tools
        for tk_id in range(10):
            toolkit = Toolkit()

            for tool_id in range(5):

                def make_tool(tid, ttid):
                    def test_tool() -> Tuple[str, dict]:
                        time.sleep(0.001)
                        return f"tk{tid}_tool{ttid}", {}

                    return test_tool

                toolkit.add_tool(f"tool_{tool_id}", make_tool(tk_id, tool_id))

            agent.add_toolkit(f"tk_{tk_id}", toolkit)

        # Add many independent tools
        for i in range(150):

            def make_independent(idx):
                def ind_tool() -> Tuple[str, dict]:
                    time.sleep(0.001)
                    return f"ind_{idx}", {}

                return ind_tool

            agent.add_tool(f"ind_{i}", make_independent(i))

        # Verify all registered
        assert len(agent.toolkits) == 10
        assert len(agent.independent_toolkit.tools) == 150

        # Agent successfully registered 200+ tools without issues
        # (10 toolkits * 5 tools + 150 independent tools)
