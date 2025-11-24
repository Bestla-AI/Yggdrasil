"""Conversation context for agent runs.

Manages conversation state including messages and context compaction.
"""

from typing import TYPE_CHECKING, List

from openai.types.chat import (
    ChatCompletionMessageParam,
)

if TYPE_CHECKING:
    from bestla.yggdrasil.context_manager import ContextManager


class ConversationContext:
    """Manages conversation state for agent runs.

    Encapsulates conversation messages and optional context management strategy.
    Can be shared between agents or used for stateless execution.

    Attributes:
        messages: List of conversation messages
        context_manager: Optional context manager for automatic compaction

    Example:
        # Default usage with agent
        agent = Agent(provider=client, model="gpt-4")
        agent.run("Hello")  # Uses default conversation context

        # Shared context between agents
        ctx = ConversationContext()
        agent1 = Agent(provider=client, model="gpt-4", conversation_context=ctx)
        agent2 = Agent(provider=client, model="gpt-4", conversation_context=ctx)

        # Stateless execution
        agent = Agent(provider=client, model="gpt-4")
        ctx1 = ConversationContext()
        ctx2 = ConversationContext()
        agent.run("Query 1", conversation_context=ctx1)
        agent.run("Query 2", conversation_context=ctx2)
    """

    def __init__(
        self,
        messages: List[ChatCompletionMessageParam] | None = None,
        context_manager: "ContextManager | None" = None,
    ):
        """Initialize conversation context.

        Args:
            messages: Initial message list (defaults to empty list)
            context_manager: Optional context manager for automatic compaction

        Example:
            # Basic context
            ctx = ConversationContext()

            # With initial messages
            ctx = ConversationContext(messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content="You are helpful",
                )
            ])

            # With context management
            cm = ContextManager(threshold=100000)
            ctx = ConversationContext(context_manager=cm)
        """
        self._messages = messages or []
        self.context_manager = context_manager

    @property
    def messages(self) -> List[ChatCompletionMessageParam]:
        """Get conversation message history.

        Returns:
            List of chat completion messages
        """
        return self._messages

    @messages.setter
    def messages(self, value: List[ChatCompletionMessageParam]) -> None:
        """Set conversation message history.

        Args:
            value: New message list

        Raises:
            TypeError: If value is not a list
            ValueError: If value is None

        Example:
            ctx.messages = [
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Hello",
                ),
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content="Hi there!",
                )
            ]
        """
        if value is None:
            raise ValueError("messages cannot be None (use empty list instead)")
        if not isinstance(value, list):
            raise TypeError(
                f"messages must be a list, got {type(value).__name__}"
            )
        self._messages = value

    def clear_messages(self) -> None:
        """Clear all conversation messages.

        Example:
            ctx.clear_messages()
            assert len(ctx.messages) == 0
        """
        self._messages.clear()

    def should_compact(self) -> bool:
        """Check if context compaction should be triggered.

        Returns:
            True if context_manager exists and recommends compaction,
            False otherwise

        Example:
            cm = ContextManager(threshold=1000)
            ctx = ConversationContext(context_manager=cm)
            # Add many messages...
            if ctx.should_compact():
                ctx.compact()
        """
        if self.context_manager is None:
            return False
        return self.context_manager.should_compact(self._messages)

    def compact(self) -> None:
        """Apply context compaction if context_manager is configured.

        If no context_manager is set, this is a no-op.

        Example:
            cm = ContextManager(strategy="tool_result_clearing")
            ctx = ConversationContext(context_manager=cm)
            # Add messages...
            ctx.compact()  # Removes old tool results
        """
        if self.context_manager is None:
            return
        self._messages = self.context_manager.compact(self._messages)

    def __repr__(self) -> str:
        """Return string representation."""
        msg_count = len(self._messages)
        cm_status = "enabled" if self.context_manager else "disabled"
        return (
            f"ConversationContext(messages={msg_count}, "
            f"context_manager={cm_status})"
        )
