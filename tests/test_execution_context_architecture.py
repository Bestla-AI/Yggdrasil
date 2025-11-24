"""Tests for ExecutionContext architecture with integrated ConversationContext.

These tests demonstrate the new architecture where ConversationContext is a
component within ExecutionContext, as conversation is tied to specific execution.
"""

import pytest
from unittest.mock import Mock
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
)

from bestla.yggdrasil import (
    Agent,
    ExecutionContext,
    ConversationContext,
    ContextManager,
    Toolkit,
)


@pytest.fixture
def mock_provider():
    """Create mock OpenAI provider."""
    provider = Mock()
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Response"
    response.choices[0].message.tool_calls = None
    provider.chat.completions.create.return_value = response
    return provider


class TestExecutionContextArchitecture:
    """Test ExecutionContext contains ConversationContext."""

    def test_execution_context_has_conversation(self):
        """Test that ExecutionContext has conversation attribute."""
        toolkit = Toolkit()
        exec_ctx = ExecutionContext(
            toolkits={},
            independent_toolkit=toolkit,
        )

        assert hasattr(exec_ctx, 'conversation')
        assert isinstance(exec_ctx.conversation, ConversationContext)

    def test_execution_context_with_custom_conversation(self):
        """Test creating ExecutionContext with custom ConversationContext."""
        # Create custom conversation context with messages and context manager
        cm = ContextManager(threshold=1000)
        conv = ConversationContext(
            messages=[
                ChatCompletionSystemMessageParam(role="system", content="Custom system")
            ],
            context_manager=cm,
        )

        toolkit = Toolkit()
        exec_ctx = ExecutionContext(
            toolkits={},
            independent_toolkit=toolkit,
            conversation_context=conv,
        )

        # Should use the provided conversation context
        assert exec_ctx.conversation is conv
        assert len(exec_ctx.conversation.messages) == 1
        assert exec_ctx.conversation.messages[0]["content"] == "Custom system"
        assert exec_ctx.conversation.context_manager is cm

    def test_execution_context_creates_default_conversation(self):
        """Test that ExecutionContext creates default conversation if none provided."""
        toolkit = Toolkit()
        exec_ctx = ExecutionContext(
            toolkits={},
            independent_toolkit=toolkit,
        )

        # Should have created a default conversation context
        assert exec_ctx.conversation is not None
        assert exec_ctx.conversation.messages == []
        assert exec_ctx.conversation.context_manager is None

    def test_conversation_isolated_per_execution_context(self):
        """Test that each ExecutionContext has its own conversation."""
        toolkit = Toolkit()

        # Create two execution contexts
        exec_ctx1 = ExecutionContext(toolkits={}, independent_toolkit=toolkit)
        exec_ctx2 = ExecutionContext(toolkits={}, independent_toolkit=toolkit)

        # Add message to first conversation
        exec_ctx1.conversation.messages.append(
            ChatCompletionUserMessageParam(role="user", content="Message 1")
        )

        # Add different message to second conversation
        exec_ctx2.conversation.messages.append(
            ChatCompletionUserMessageParam(role="user", content="Message 2")
        )

        # Should be independent
        assert len(exec_ctx1.conversation.messages) == 1
        assert len(exec_ctx2.conversation.messages) == 1
        assert exec_ctx1.conversation.messages[0]["content"] == "Message 1"
        assert exec_ctx2.conversation.messages[0]["content"] == "Message 2"

    def test_shared_conversation_across_execution_contexts(self):
        """Test sharing a ConversationContext across multiple ExecutionContexts."""
        # Create shared conversation
        shared_conv = ConversationContext()
        shared_conv.messages.append(
            ChatCompletionUserMessageParam(role="user", content="Shared message")
        )

        toolkit = Toolkit()

        # Create two execution contexts with same conversation
        exec_ctx1 = ExecutionContext(
            toolkits={},
            independent_toolkit=toolkit,
            conversation_context=shared_conv
        )
        exec_ctx2 = ExecutionContext(
            toolkits={},
            independent_toolkit=toolkit,
            conversation_context=shared_conv
        )

        # Both should reference same conversation
        assert exec_ctx1.conversation is exec_ctx2.conversation
        assert exec_ctx1.conversation is shared_conv

        # Modifications via one affect the other
        exec_ctx1.conversation.messages.append(
            ChatCompletionUserMessageParam(role="user", content="Via exec_ctx1")
        )

        assert len(exec_ctx2.conversation.messages) == 2
        assert exec_ctx2.conversation.messages[1]["content"] == "Via exec_ctx1"


class TestAgentWithExecutionContext:
    """Test Agent.run() with custom ExecutionContext."""

    def test_agent_run_with_custom_execution_context(self, mock_provider):
        """Test running agent with custom ExecutionContext."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Create custom conversation with pre-existing messages
        conv = ConversationContext()
        conv.messages.append(
            ChatCompletionSystemMessageParam(role="system", content="Custom system")
        )
        conv.messages.append(
            ChatCompletionUserMessageParam(role="user", content="Previous message")
        )

        # Create execution context with this conversation
        exec_ctx = ExecutionContext(
            toolkits=agent.toolkits,
            independent_toolkit=agent.independent_toolkit,
            conversation_context=conv,
        )

        # Run with custom execution context
        response, ctx = agent.run("New query", execution_context=exec_ctx)

        assert response == "Response"

        # Conversation should have accumulated messages
        assert len(conv.messages) >= 3  # system, previous, new query, new response

    def test_agent_run_default_creates_fresh_context(self, mock_provider):
        """Test that default run() creates fresh ExecutionContext (stateless)."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Run without execution_context parameter (should create fresh context)
        response, ctx = agent.run("Query")

        assert response == "Response"

        # Should have created ExecutionContext with fresh conversation
        assert ctx is not None
        assert ctx.conversation is not None
        assert len(ctx.conversation.messages) >= 2  # system + user + assistant

    def test_agent_stateless_execution_with_fresh_context(self, mock_provider):
        """Test stateless execution by providing fresh ExecutionContext each time."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # First run with fresh execution context
        exec_ctx1 = ExecutionContext(
            toolkits=agent.toolkits,
            independent_toolkit=agent.independent_toolkit,
        )
        response1, ctx1 = agent.run("Query 1", execution_context=exec_ctx1)

        # Second run with different fresh execution context
        exec_ctx2 = ExecutionContext(
            toolkits=agent.toolkits,
            independent_toolkit=agent.independent_toolkit,
        )
        response2, ctx2 = agent.run("Query 2", execution_context=exec_ctx2)

        assert response1 == "Response"
        assert response2 == "Response"

        # Each execution context should have its own conversation
        assert exec_ctx1.conversation is not exec_ctx2.conversation

        # Agent has no internal conversation state (stateless)
        assert not hasattr(agent, 'messages')
        assert not hasattr(agent, '_conversation_context')

    def test_agent_with_context_manager_in_execution_context(self, mock_provider):
        """Test ExecutionContext with ConversationContext that has ContextManager."""
        # Create conversation with context manager
        cm = ContextManager(threshold=50, strategy="tool_result_clearing")
        conv = ConversationContext(context_manager=cm)

        # Create agent
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Create execution context with managed conversation
        exec_ctx = ExecutionContext(
            toolkits=agent.toolkits,
            independent_toolkit=agent.independent_toolkit,
            conversation_context=conv,
        )

        # Run multiple times to potentially trigger compaction
        for i in range(5):
            response, ctx = agent.run(f"Query {i} " * 20, execution_context=exec_ctx)

        # Conversation should exist and potentially be compacted
        assert len(conv.messages) > 0

    def test_multiple_agents_same_execution_context(self, mock_provider):
        """Test multiple agents can share an ExecutionContext (for conversation continuity)."""
        # Create shared execution context
        shared_conv = ConversationContext()
        shared_conv.messages.append(
            ChatCompletionSystemMessageParam(role="system", content="Shared system")
        )

        agent1 = Agent(provider=mock_provider, model="gpt-4")
        agent2 = Agent(provider=mock_provider, model="gpt-4")

        exec_ctx = ExecutionContext(
            toolkits={},
            independent_toolkit=Toolkit(),
            conversation_context=shared_conv,
        )

        # Agent 1 runs
        response, ctx = agent1.run("Query from agent 1", execution_context=exec_ctx)

        # Agent 2 can see agent 1's conversation
        initial_length = len(shared_conv.messages)

        # Agent 2 runs with same context
        response, ctx2 = agent2.run("Query from agent 2", execution_context=exec_ctx)

        # Conversation accumulated
        assert len(shared_conv.messages) > initial_length


class TestExecutionContextIsolation:
    """Test that ExecutionContext properly isolates toolkit state."""

    def test_toolkits_copied_in_execution_context(self):
        """Test that toolkits are copied (isolated) in ExecutionContext."""
        toolkit = Toolkit()
        toolkit.add_tool("test_tool", lambda: ("result", {}))

        # Original toolkit
        original_toolkit = toolkit

        # Create execution context
        exec_ctx = ExecutionContext(
            toolkits={"test": toolkit},
            independent_toolkit=Toolkit(),
        )

        # Toolkits should be copied
        assert exec_ctx.toolkits["test"] is not original_toolkit

    def test_conversation_not_copied_in_execution_context(self):
        """Test that conversation is NOT copied (shared with caller)."""
        conv = ConversationContext()
        conv.messages.append(
            ChatCompletionUserMessageParam(role="user", content="Original")
        )

        # Create execution context with conversation
        exec_ctx = ExecutionContext(
            toolkits={},
            independent_toolkit=Toolkit(),
            conversation_context=conv,
        )

        # Conversation should be same reference (not copied)
        assert exec_ctx.conversation is conv

        # Modifications affect original
        exec_ctx.conversation.messages.append(
            ChatCompletionUserMessageParam(role="user", content="Added via exec_ctx")
        )

        assert len(conv.messages) == 2
        assert conv.messages[1]["content"] == "Added via exec_ctx"


class TestExecutionContextDocumentedUseCases:
    """Test documented use cases from ExecutionContext/ConversationContext."""

    def test_use_case_stateful_conversation(self, mock_provider):
        """Use case: Stateful conversation across multiple runs (explicit)."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        # Stateful by explicitly passing context
        response1, ctx1 = agent.run("First query")
        response2, ctx2 = agent.run("Second query", execution_context=ctx1)
        response3, ctx3 = agent.run("Third query", execution_context=ctx2)

        # Conversation accumulates (ctx3 has all messages)
        assert ctx3 is ctx2 is ctx1  # Same context throughout
        assert len(ctx3.conversation.messages) >= 6  # 3 queries + 3 responses (+ system)

    def test_use_case_stateless_execution(self, mock_provider):
        """Use case: Stateless execution with fresh context each time (default)."""
        agent = Agent(provider=mock_provider, model="gpt-4")

        results = []
        contexts = []
        for i in range(3):
            # Each run creates fresh context (default behavior)
            result, ctx = agent.run(f"Query {i}")
            results.append(result)
            contexts.append(ctx)

        # Each run is independent
        assert len(results) == 3

        # Each context is different
        assert contexts[0] is not contexts[1] is not contexts[2]

        # Each context only has its own query
        for i, ctx in enumerate(contexts):
            user_msgs = [m for m in ctx.conversation.messages if m.get("role") == "user"]
            assert f"Query {i}" in str(user_msgs[0].get("content"))

    def test_use_case_shared_conversation_between_agents(self, mock_provider):
        """Use case: Multiple agents share conversation context."""
        # Create shared conversation
        shared_conv = ConversationContext()

        agent1 = Agent(provider=mock_provider, model="gpt-4")
        agent2 = Agent(provider=mock_provider, model="gpt-4")

        # Create execution contexts with shared conversation
        exec_ctx1 = ExecutionContext(
            toolkits=agent1.toolkits,
            independent_toolkit=agent1.independent_toolkit,
            conversation_context=shared_conv,
        )

        exec_ctx2 = ExecutionContext(
            toolkits=agent2.toolkits,
            independent_toolkit=agent2.independent_toolkit,
            conversation_context=shared_conv,
        )

        # Agent 1 runs
        response, ctx = agent1.run("Agent 1 query", execution_context=exec_ctx1)

        # Agent 2 can see and continue Agent 1's conversation
        response, ctx2 = agent2.run("Agent 2 query", execution_context=exec_ctx2)

        # Shared conversation has both exchanges
        assert len(shared_conv.messages) >= 4  # 2 queries + 2 responses (+ system)
