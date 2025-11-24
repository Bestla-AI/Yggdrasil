"""Tests for ContextManager class."""

from unittest.mock import Mock, patch

import pytest
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)


class TestContextManager:
    """Test ContextManager functionality."""

    def test_create_context_manager_defaults(self):
        """Test creating a ContextManager with default settings."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager()

        assert cm.strategy == "tool_result_clearing"
        assert cm.threshold is None  # Disabled by default
        assert cm.preserve_recent == 10

    def test_create_context_manager_custom(self):
        """Test creating a ContextManager with custom settings."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(
            strategy="summarization",
            threshold=100000,
            preserve_recent=20,
        )

        assert cm.strategy == "summarization"
        assert cm.threshold == 100000
        assert cm.preserve_recent == 20

    def test_should_compact_when_disabled(self):
        """Test should_compact returns False when threshold is None."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(threshold=None)
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="test" * 1000,
            )
        ]

        assert cm.should_compact(messages) is False

    def test_should_compact_under_threshold(self):
        """Test should_compact returns False when under threshold."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(threshold=10000)
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="short message",
            )
        ]

        # Mock estimate_tokens to return value under threshold
        with patch.object(cm, 'estimate_tokens', return_value=5000):
            assert cm.should_compact(messages) is False

    def test_should_compact_over_threshold(self):
        """Test should_compact returns True when over threshold."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(threshold=10000)
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="test" * 5000,
            )
        ]

        # Mock estimate_tokens to return value over threshold
        with patch.object(cm, 'estimate_tokens', return_value=15000):
            assert cm.should_compact(messages) is True

    def test_should_compact_at_threshold(self):
        """Test should_compact returns True when exactly at threshold."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(threshold=10000)
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="test",
            )
        ]

        # Mock estimate_tokens to return exact threshold
        with patch.object(cm, 'estimate_tokens', return_value=10000):
            assert cm.should_compact(messages) is True

    def test_estimate_tokens_empty_list(self):
        """Test token estimation with empty message list."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager()
        assert cm.estimate_tokens([]) == 0

    def test_estimate_tokens_single_message(self):
        """Test token estimation with single message."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager()
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Hello world",
            )
        ]

        tokens = cm.estimate_tokens(messages)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_multiple_messages(self):
        """Test token estimation with multiple messages."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager()
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="You are helpful",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Hello",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Hi there!",
            ),
        ]

        tokens = cm.estimate_tokens(messages)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_compact_tool_result_clearing_strategy(self):
        """Test compaction using tool_result_clearing strategy."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=2)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System prompt",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 1",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Response 1",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="old_1",
                content="Old tool result 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 2",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Response 2",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="old_2",
                content="Old tool result 2",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 3",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Response 3",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="recent_1",
                content="Recent tool result",
            ),
        ]

        compacted = cm.compact(messages)

        # System message should be preserved
        assert compacted[0]["role"] == "system"

        # Old tool messages should be removed
        tool_messages = [m for m in compacted if m.get("role") == "tool"]
        assert len(tool_messages) < len([m for m in messages if m.get("role") == "tool"])

        # Recent messages should be preserved
        assert len(compacted) < len(messages)

    def test_compact_preserves_system_message(self):
        """Test that compaction always preserves system message."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=1)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="Important system prompt",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 1",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Tool result",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 2",
            ),
        ]

        compacted = cm.compact(messages)

        assert compacted[0]["role"] == "system"
        assert compacted[0]["content"] == "Important system prompt"

    def test_compact_preserves_recent_messages(self):
        """Test that compaction preserves specified number of recent messages."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=3)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Old 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Old 2",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent 2",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent 3",
            ),
        ]

        compacted = cm.compact(messages)

        # Should have system + 3 recent messages
        assert len(compacted) >= 4
        assert compacted[-1]["content"] == "Recent 3"
        assert compacted[-2]["content"] == "Recent 2"
        assert compacted[-3]["content"] == "Recent 1"

    def test_compact_empty_messages(self):
        """Test compaction with empty message list."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing")

        compacted = cm.compact([])
        assert compacted == []

    def test_compact_only_system_message(self):
        """Test compaction with only system message."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing")

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        compacted = cm.compact(messages)
        assert len(compacted) == 1
        assert compacted[0]["role"] == "system"

    def test_compact_no_tool_messages(self):
        """Test compaction when there are no tool messages."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=2)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 1",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Response 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 2",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Response 2",
            ),
        ]

        compacted = cm.compact(messages)

        # Should preserve system + recent messages
        assert len(compacted) >= 3  # system + preserve_recent * 2 (user + assistant)

    def test_compact_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError at init."""
        from bestla.yggdrasil.context_manager import ContextManager

        # Invalid strategy now raises at __init__ time
        with pytest.raises(ValueError, match="Invalid compaction strategy"):
            ContextManager(strategy="invalid_strategy")

    def test_compact_preserves_message_order(self):
        """Test that compaction preserves chronological message order."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=5)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User 1",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Assistant 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User 2",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Assistant 2",
            ),
        ]

        compacted = cm.compact(messages)

        # Verify chronological order is maintained
        for i in range(1, len(compacted)):
            # Messages should be in the same order
            original_idx = messages.index(compacted[i])
            prev_original_idx = messages.index(compacted[i-1])
            assert original_idx > prev_original_idx

    def test_multiple_compaction_cycles(self):
        """Test that multiple compaction cycles work correctly."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=2)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 1",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Tool 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 2",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="2",
                content="Tool 2",
            ),
        ]

        # First compaction
        compacted1 = cm.compact(messages)

        # Add more messages
        compacted1.extend([
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 3",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="3",
                content="Tool 3",
            ),
        ])

        # Second compaction
        compacted2 = cm.compact(compacted1)

        # Should still have system message
        assert compacted2[0]["role"] == "system"
        # Should be smaller or equal to first compaction + new messages
        assert len(compacted2) <= len(compacted1)

    def test_estimate_tokens_with_tool_messages(self):
        """Test token estimation includes tool messages."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager()

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Tool result with some content",
            ),
        ]

        tokens = cm.estimate_tokens(messages)
        assert tokens > 0

    def test_compact_with_preserve_recent_larger_than_message_list(self):
        """Test compaction when preserve_recent is larger than message count."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=100)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Tool",
            ),
        ]

        compacted = cm.compact(messages)

        # All messages should be preserved if preserve_recent > total messages
        assert len(compacted) == len(messages)

    def test_context_manager_repr(self):
        """Test __repr__ method of ContextManager."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", threshold=50000, preserve_recent=15)

        repr_str = repr(cm)
        assert "ContextManager" in repr_str
        assert "tool_result_clearing" in repr_str
        assert "50000" in repr_str

    def test_compact_with_none_content(self):
        """Test compaction handles messages with None content gracefully."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=2)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=None,
            ),
        ]

        # Should not raise an error
        compacted = cm.compact(messages)
        assert compacted[0]["role"] == "system"

    def test_threshold_validation(self):
        """Test that threshold must be positive if set."""
        from bestla.yggdrasil.context_manager import ContextManager

        # Valid: None
        cm1 = ContextManager(threshold=None)
        assert cm1.threshold is None

        # Valid: positive number
        cm2 = ContextManager(threshold=10000)
        assert cm2.threshold == 10000

        # Invalid: negative number
        with pytest.raises(ValueError, match="threshold must be positive"):
            ContextManager(threshold=-1000)

        # Invalid: zero
        with pytest.raises(ValueError, match="threshold must be positive"):
            ContextManager(threshold=0)

    def test_preserve_recent_validation(self):
        """Test that preserve_recent must be positive."""
        from bestla.yggdrasil.context_manager import ContextManager

        # Valid: positive number
        cm1 = ContextManager(preserve_recent=5)
        assert cm1.preserve_recent == 5

        # Invalid: zero
        with pytest.raises(ValueError, match="preserve_recent must be at least 1"):
            ContextManager(preserve_recent=0)

        # Invalid: negative
        with pytest.raises(ValueError, match="preserve_recent must be at least 1"):
            ContextManager(preserve_recent=-5)

    def test_custom_token_counter(self):
        """Test using custom token counter."""
        from bestla.yggdrasil.context_manager import ContextManager

        # Create custom counter that always returns 1000
        def custom_counter(messages):
            return 1000

        cm = ContextManager(threshold=500, token_counter=custom_counter)

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="test",
            )
        ]

        # Should use custom counter
        assert cm.estimate_tokens(messages) == 1000

        # Should trigger compaction (1000 >= 500)
        assert cm.should_compact(messages) is True

    def test_default_token_counter(self):
        """Test that default token counter is used when none provided."""
        from bestla.yggdrasil.context_manager import ContextManager
        from bestla.yggdrasil.context_utils import estimate_tokens

        cm = ContextManager()

        # Should use default estimator
        assert cm.token_counter is estimate_tokens

    def test_token_counter_in_repr(self):
        """Test that token counter appears in repr."""
        from bestla.yggdrasil.context_manager import ContextManager

        def my_custom_counter(messages):
            return 100

        cm = ContextManager(token_counter=my_custom_counter)

        repr_str = repr(cm)
        assert "my_custom_counter" in repr_str

    def test_custom_counter_with_different_behavior(self):
        """Test custom counter with length-based counting."""
        from bestla.yggdrasil.context_manager import ContextManager

        # Counter that counts 1 token per message
        def simple_counter(messages):
            return len(messages)

        cm = ContextManager(threshold=5, token_counter=simple_counter)

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="msg1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="msg2",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="msg3",
            ),
        ]

        # 3 messages = 3 tokens
        assert cm.estimate_tokens(messages) == 3
        assert cm.should_compact(messages) is False

        # Add more messages
        messages.extend([
            ChatCompletionUserMessageParam(
                role="user",
                content="msg4",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="msg5",
            ),
        ])

        # 5 messages = 5 tokens, should trigger at threshold
        assert cm.estimate_tokens(messages) == 5
        assert cm.should_compact(messages) is True

    def test_token_counter_called_during_compaction_check(self):
        """Test that token counter is called when checking compaction."""
        from bestla.yggdrasil.context_manager import ContextManager

        call_count = {"count": 0}

        def counting_counter(messages):
            call_count["count"] += 1
            return 100

        cm = ContextManager(threshold=50, token_counter=counting_counter)

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="test",
            )
        ]

        # Should call counter when checking
        cm.should_compact(messages)
        assert call_count["count"] == 1

        # Should call again
        cm.estimate_tokens(messages)
        assert call_count["count"] == 2

    def test_set_summarization_config(self):
        """Test setting summarization configuration."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="summarization")

        # Initially provider and model should be None
        assert cm._provider is None
        assert cm._model is None

        # Set configuration
        mock_provider = Mock()
        cm.set_summarization_config(mock_provider, "gpt-4")

        # Verify configuration is set
        assert cm._provider is mock_provider
        assert cm._model == "gpt-4"

    def test_compact_summarization_without_config_raises_error(self):
        """Test that summarization strategy raises error when not configured."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="summarization")

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 2",
            ),
        ]

        # Should raise ValueError because provider/model not set
        with pytest.raises(ValueError, match="Summarization strategy requires provider and model"):
            cm.compact(messages)

    def test_compact_summarization_strategy_success(self):
        """Test successful compaction with summarization strategy."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(
            strategy="summarization",
            preserve_recent=2,
        )

        # Configure provider
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Summary of previous conversation about topics A and B."
        )
        mock_provider.chat.completions.create.return_value = mock_response

        cm.set_summarization_config(mock_provider, "gpt-4")

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Old message 1",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Old response 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Old message 2",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Old response 2",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent 1",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Recent response 1",
            ),
        ]

        compacted = cm.compact(messages)

        # Should have: system + first_user + summary + 2 recent messages
        assert compacted[0]["role"] == "system"
        assert compacted[0]["content"] == "System"

        # Second message should be first user message (task directive, preserved)
        assert compacted[1]["role"] == "user"
        assert compacted[1]["content"] == "Old message 1"

        # Third message should be the summary
        assert compacted[2]["role"] == "user"
        assert "[COMPACTED CONTEXT" in compacted[2]["content"]
        assert "Summary of previous conversation" in compacted[2]["content"]

        # Last 2 messages should be preserved
        assert compacted[-1]["content"] == "Recent response 1"
        assert compacted[-2]["content"] == "Recent 1"

        # Provider should have been called to generate summary
        assert mock_provider.chat.completions.create.called

    def test_compact_summarization_with_no_old_messages(self):
        """Test summarization when all messages are recent."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(
            strategy="summarization",
            preserve_recent=10,
        )

        mock_provider = Mock()
        cm.set_summarization_config(mock_provider, "gpt-4")

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 2",
            ),
        ]

        compacted = cm.compact(messages)

        # No old messages to summarize, should just preserve all
        assert len(compacted) == 3
        assert compacted[0]["role"] == "system"

        # Provider should not be called since nothing to summarize
        assert not mock_provider.chat.completions.create.called

    def test_compact_summarization_preserves_system_message(self):
        """Test that summarization always preserves system message."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(
            strategy="summarization",
            preserve_recent=1,
        )

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Summary"
        mock_provider.chat.completions.create.return_value = mock_response

        cm.set_summarization_config(mock_provider, "gpt-4")

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="Important system prompt",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Old 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Old 2",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent",
            ),
        ]

        compacted = cm.compact(messages)

        # System should be first
        assert compacted[0]["role"] == "system"
        assert compacted[0]["content"] == "Important system prompt"

    def test_compact_summarization_calls_utils_correctly(self):
        """Test that summarization strategy calls utility functions with correct params."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(
            strategy="summarization",
            preserve_recent=3,
        )

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test summary"
        mock_provider.chat.completions.create.return_value = mock_response

        cm.set_summarization_config(mock_provider, "gpt-4")

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Msg 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Msg 2",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Msg 3",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Msg 4",
            ),
        ]

        compacted = cm.compact(messages)

        # Verify the summarization happened
        call_args = mock_provider.chat.completions.create.call_args
        assert call_args is not None

        # Verify model was passed
        assert call_args.kwargs["model"] == "gpt-4"

        # Verify summary message is in result
        summary_messages = [
            m for m in compacted if "[COMPACTED CONTEXT" in str(m.get("content", ""))
        ]
        assert len(summary_messages) == 1


class TestContextManagerValidation:
    """Test ContextManager validation and error handling."""

    def test_strategy_validated_at_init(self):
        """Test that invalid strategy IS validated at initialization."""
        from bestla.yggdrasil.context_manager import ContextManager

        # Invalid strategy now raises error at __init__ time (updated behavior)
        with pytest.raises(ValueError, match="Invalid compaction strategy"):
            ContextManager(strategy="invalid_strategy")

    def test_valid_strategies_constant(self):
        """Test VALID_STRATEGIES constant is accessible."""
        from bestla.yggdrasil.context_manager import ContextManager

        # VALID_STRATEGIES should be defined
        assert hasattr(ContextManager, "VALID_STRATEGIES")
        assert "tool_result_clearing" in ContextManager.VALID_STRATEGIES
        assert "summarization" in ContextManager.VALID_STRATEGIES
        assert isinstance(ContextManager.VALID_STRATEGIES, list)

    def test_strategy_validation_with_valid_strategies(self):
        """Test creating ContextManager with each valid strategy."""
        from bestla.yggdrasil.context_manager import ContextManager

        for strategy in ContextManager.VALID_STRATEGIES:
            cm = ContextManager(strategy=strategy)
            assert cm.strategy == strategy

    def test_attribute_modification_after_init_threshold(self):
        """Test modifying threshold after initialization."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(threshold=10000)
        assert cm.threshold == 10000

        # Modify threshold
        cm.threshold = 20000
        assert cm.threshold == 20000

        # Setting to None should disable compaction
        cm.threshold = None
        messages = [ChatCompletionUserMessageParam(
                        role="user",
                        content="test" * 1000,
                    )]
        assert cm.should_compact(messages) is False

        # Setting to negative (no validation after init)
        cm.threshold = -100
        # This might cause unexpected behavior, documenting current state
        assert cm.threshold == -100

    def test_attribute_modification_after_init_preserve_recent(self):
        """Test modifying preserve_recent after initialization."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(preserve_recent=10)
        assert cm.preserve_recent == 10

        # Modify preserve_recent
        cm.preserve_recent = 5
        assert cm.preserve_recent == 5

        # Setting to 0 (no validation after init)
        cm.preserve_recent = 0
        assert cm.preserve_recent == 0

    def test_attribute_modification_after_init_strategy(self):
        """Test modifying strategy after initialization."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing")
        assert cm.strategy == "tool_result_clearing"

        # Modify strategy
        cm.strategy = "summarization"
        assert cm.strategy == "summarization"

        # Invalid strategy (will fail at compact time)
        cm.strategy = "nonexistent"
        assert cm.strategy == "nonexistent"

    def test_none_token_counter(self):
        """Test that None token_counter falls back to default."""
        from bestla.yggdrasil.context_manager import ContextManager
        from bestla.yggdrasil.context_utils import estimate_tokens

        cm = ContextManager(token_counter=None)

        # Should use default estimator
        assert cm.token_counter is estimate_tokens

    def test_threshold_none_explicitly_set(self):
        """Test that explicitly setting threshold to None disables compaction."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(threshold=None)

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="test" * 10000,
            )
        ]

        # Should never trigger compaction
        assert cm.should_compact(messages) is False

    def test_preserve_recent_boundary_values(self):
        """Test preserve_recent with boundary values."""
        from bestla.yggdrasil.context_manager import ContextManager

        # Minimum valid value
        cm1 = ContextManager(preserve_recent=1)
        assert cm1.preserve_recent == 1

        # Large value
        cm2 = ContextManager(preserve_recent=10000)
        assert cm2.preserve_recent == 10000

    def test_threshold_boundary_values(self):
        """Test threshold with boundary values."""
        from bestla.yggdrasil.context_manager import ContextManager

        # Minimum valid value
        cm1 = ContextManager(threshold=1)
        assert cm1.threshold == 1

        # Large value
        cm2 = ContextManager(threshold=1000000)
        assert cm2.threshold == 1000000

    def test_compact_with_uninitialized_summarization(self):
        """Test that summarization requires configuration."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="summarization")

        # Provider and model not set
        assert cm._provider is None
        assert cm._model is None

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 2",
            ),
        ]

        # Should raise error
        with pytest.raises(ValueError, match="Summarization strategy requires provider and model"):
            cm.compact(messages)

    def test_set_summarization_config_updates_attributes(self):
        """Test set_summarization_config properly updates internal state."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="summarization")

        mock_provider = Mock()
        cm.set_summarization_config(mock_provider, "gpt-4o")

        assert cm._provider is mock_provider
        assert cm._model == "gpt-4o"

        # Can update configuration
        mock_provider2 = Mock()
        cm.set_summarization_config(mock_provider2, "gpt-4-turbo")

        assert cm._provider is mock_provider2
        assert cm._model == "gpt-4-turbo"

    def test_custom_token_counter_with_invalid_return(self):
        """Test that custom token counter returning invalid values raises error."""
        from bestla.yggdrasil.context_manager import ContextManager

        # Counter that returns None
        def none_counter(messages):
            return None

        cm = ContextManager(threshold=100, token_counter=none_counter)

        messages = [ChatCompletionUserMessageParam(
                        role="user",
                        content="test",
                    )]

        # Now raises ValueError with validation (updated behavior)
        with pytest.raises(ValueError, match="Token counter must return int"):
            cm.should_compact(messages)

    def test_repr_with_disabled_threshold(self):
        """Test __repr__ shows 'disabled' for None threshold."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(threshold=None)

        repr_str = repr(cm)
        assert "disabled" in repr_str
        assert "ContextManager" in repr_str


class TestStrategyEffectiveness:
    """Test that compaction strategies actually reduce token counts."""

    def test_tool_result_clearing_reduces_tokens(self):
        """Test that tool_result_clearing strategy reduces token count."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=5)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User 1",
            ),
        ]

        # Add many tool messages
        for i in range(20):
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"tool_{i}",
                    content=f"This is a tool result with some content {i}"
                )
            )

        # Add recent messages
        messages.extend([
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent user",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Recent assistant",
            ),
        ])

        tokens_before = cm.estimate_tokens(messages)
        compacted = cm.compact(messages)
        tokens_after = cm.estimate_tokens(compacted)

        # Should reduce token count
        assert tokens_after < tokens_before
        # Should be significant reduction
        reduction_percentage = ((tokens_before - tokens_after) / tokens_before) * 100
        assert reduction_percentage > 10  # At least 10% reduction

    def test_summarization_reduces_tokens(self):
        """Test that summarization strategy reduces token count."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="summarization", preserve_recent=3)

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        # Short summary (much shorter than original messages)
        mock_response.choices[0].message.content = "Brief summary"
        mock_provider.chat.completions.create.return_value = mock_response

        cm.set_summarization_config(mock_provider, "gpt-4")

        # Create long messages
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        for i in range(15):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=(
                        f"This is a long user message with lots of content that "
                        f"should be summarized. Message number {i}. "
                    )
                    * 10,
                )
            )
            messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=(
                        f"This is a long assistant response with detailed "
                        f"information. Response number {i}. "
                    )
                    * 10,
                )
            )

        tokens_before = cm.estimate_tokens(messages)
        compacted = cm.compact(messages)
        tokens_after = cm.estimate_tokens(compacted)

        # Should reduce token count
        assert tokens_after < tokens_before

    def test_tool_clearing_vs_no_compaction(self):
        """Test that tool clearing is better than no compaction."""
        from bestla.yggdrasil.context_manager import ContextManager

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        # Add mix of messages with many tool results
        for i in range(30):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"User message {i}",
                )
            )
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"tool_{i}",
                    content=f"Tool result with data {i}" * 5
                )
            )

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=10)

        original_count = len(messages)
        tokens_original = cm.estimate_tokens(messages)

        compacted = cm.compact(messages)
        compacted_count = len(compacted)
        tokens_compacted = cm.estimate_tokens(compacted)

        # Should have fewer messages
        assert compacted_count < original_count

        # Should have fewer tokens
        assert tokens_compacted < tokens_original

        # Recent messages should still be there
        assert compacted_count >= 10  # At least preserve_recent messages

    def test_compaction_meaningful_not_trivial(self):
        """Test that compaction provides meaningful reduction, not trivial."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=5)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System prompt",
            ),
        ]

        # Create scenario with lots of tool results
        for i in range(50):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Query {i}",
                )
            )
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"call_{i}",
                    content=f"Detailed tool result with lots of information for query {i}. " * 20
                )
            )

        tokens_before = cm.estimate_tokens(messages)
        compacted = cm.compact(messages)
        tokens_after = cm.estimate_tokens(compacted)

        reduction = tokens_before - tokens_after

        # Should have meaningful reduction (not just 1-2%)
        reduction_percentage = (reduction / tokens_before) * 100
        assert reduction_percentage > 20  # At least 20% reduction

        # Should save significant tokens
        assert reduction > 100  # Save at least 100 tokens

    def test_strategy_comparison_token_counts(self):
        """Compare token counts between different strategies."""
        from bestla.yggdrasil.context_manager import ContextManager

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        for i in range(20):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"User message {i} with some content"
                )
            )
            messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=f"Assistant response {i} with detailed information"
                )
            )
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"tool_{i}",
                    content=f"Tool result {i}"
                )
            )

        # Test tool clearing strategy
        cm_tool_clear = ContextManager(strategy="tool_result_clearing", preserve_recent=5)
        tokens_original = cm_tool_clear.estimate_tokens(messages)
        compacted_tool_clear = cm_tool_clear.compact(messages)
        tokens_tool_clear = cm_tool_clear.estimate_tokens(compacted_tool_clear)

        # Test summarization strategy
        cm_summarize = ContextManager(strategy="summarization", preserve_recent=5)
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Summary of conversation"
        mock_provider.chat.completions.create.return_value = mock_response
        cm_summarize.set_summarization_config(mock_provider, "gpt-4")

        compacted_summarize = cm_summarize.compact(messages)
        tokens_summarize = cm_summarize.estimate_tokens(compacted_summarize)

        # Both should reduce tokens
        assert tokens_tool_clear < tokens_original
        assert tokens_summarize < tokens_original

        # Both strategies should provide meaningful reduction
        assert tokens_tool_clear < tokens_original * 0.9  # At least 10% reduction
        assert tokens_summarize < tokens_original * 0.9  # At least 10% reduction

    def test_repeated_compaction_stabilizes(self):
        """Test that repeated compaction eventually stabilizes."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=5)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        for i in range(30):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Message {i}",
                )
            )
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"tool_{i}",
                    content=f"Result {i}"
                )
            )

        # First compaction
        compacted1 = cm.compact(messages)
        tokens1 = cm.estimate_tokens(compacted1)

        # Second compaction on already compacted messages
        compacted2 = cm.compact(compacted1)
        tokens2 = cm.estimate_tokens(compacted2)

        # Third compaction
        compacted3 = cm.compact(compacted2)
        tokens3 = cm.estimate_tokens(compacted3)

        # Should stabilize (second and third should be similar or identical)
        assert abs(tokens2 - tokens3) < tokens1 * 0.1  # Less than 10% change

    def test_compaction_with_no_compactable_content(self):
        """Test compaction when there's nothing to compact."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=10)

        # Only user and assistant messages, no tool messages
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User 1",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Assistant 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User 2",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Assistant 2",
            ),
        ]

        tokens_before = cm.estimate_tokens(messages)
        compacted = cm.compact(messages)
        tokens_after = cm.estimate_tokens(compacted)

        # Should not change much (nothing to compact)
        assert len(compacted) == len(messages)
        assert tokens_after == tokens_before

    def test_token_count_accuracy_with_real_content(self):
        """Test that token estimation is reasonable with realistic content."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager()

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Hello, can you help me understand how context compaction works?"
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=(
                    "Of course! Context compaction is a technique to reduce token "
                    "usage by removing or summarizing old messages."
                ),
            ),
        ]

        tokens = cm.estimate_tokens(messages)

        # Should be reasonable estimate (roughly 4 chars per token)
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        estimated_tokens_from_chars = total_chars // 4

        # Should be within reasonable range
        assert tokens > estimated_tokens_from_chars * 0.5
        assert tokens < estimated_tokens_from_chars * 2.0


class TestMessagePreservationGuarantees:
    """Test that compaction preserves messages according to guarantees."""

    def test_system_message_always_at_index_zero(self):
        """Test that system message is always preserved at index 0."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=5)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="Important system prompt",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 1",
            ),
        ]

        for i in range(30):
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"tool_{i}",
                    content=f"Result {i}"
                )
            )

        messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent message",
            )
        )

        compacted = cm.compact(messages)

        # System message must be at index 0
        assert compacted[0]["role"] == "system"
        assert compacted[0]["content"] == "Important system prompt"

        # Run multiple compactions
        for _ in range(3):
            compacted = cm.compact(compacted)
            assert compacted[0]["role"] == "system"
            assert compacted[0]["content"] == "Important system prompt"

    def test_preserve_recent_exact_count(self):
        """Test that exactly preserve_recent messages are preserved."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=7)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        # Add many old messages
        for i in range(50):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Old {i}",
                )
            )

        # Add exactly 7 recent messages
        recent_contents = []
        for i in range(7):
            content = f"Recent {i}"
            recent_contents.append(content)
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=content,
                )
            )

        compacted = cm.compact(messages)

        # Should have system + at least preserve_recent messages
        # (may have more if old messages don't need compaction)
        assert len(compacted) >= 8  # system + 7 recent

        # Check that all recent messages are present
        compacted_contents = [m.get("content", "") for m in compacted]
        for recent_content in recent_contents:
            assert recent_content in compacted_contents

    def test_chronological_order_maintained(self):
        """Test that chronological order is maintained after compaction."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=5)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User 1",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Assistant 1",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Tool 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User 2",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Assistant 2",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="2",
                content="Tool 2",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User 3",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Assistant 3",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="3",
                content="Tool 3",
            ),
        ]

        compacted = cm.compact(messages)

        # Verify chronological order is preserved
        for i in range(1, len(compacted)):
            current_msg = compacted[i]
            prev_msg = compacted[i-1]

            # Find their original indices
            current_idx = messages.index(current_msg)
            prev_idx = messages.index(prev_msg)

            # Current should come after previous
            assert current_idx > prev_idx

    def test_no_message_duplication(self):
        """Test that compaction doesn't duplicate messages."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=5)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        for i in range(20):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Unique message {i}",
                )
            )
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"unique_tool_{i}",
                    content=f"Unique result {i}"
                )
            )

        compacted = cm.compact(messages)

        # Check for duplicates by comparing content
        contents = [str(m.get("content", "")) for m in compacted]

        # Should have no duplicates (except possibly empty strings)
        non_empty_contents = [c for c in contents if c]
        assert len(non_empty_contents) == len(set(non_empty_contents))

    def test_recent_messages_relative_order(self):
        """Test that recent messages maintain their relative order."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=5)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        # Add old messages
        for i in range(20):
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"old_{i}",
                    content=f"Old {i}",
                )
            )

        # Add recent messages with specific order
        recent_messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent A",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Recent B",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent C",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Recent D",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent E",
            ),
        ]
        messages.extend(recent_messages)

        compacted = cm.compact(messages)

        # Extract recent messages from compacted
        compacted_recent = []
        for msg in compacted:
            content = msg.get("content", "")
            if content.startswith("Recent"):
                compacted_recent.append(content)

        # Should be in order: A, B, C, D, E
        expected_order = ["Recent A", "Recent B", "Recent C", "Recent D", "Recent E"]
        assert compacted_recent == expected_order

    def test_system_message_preserved_with_summarization(self):
        """Test that summarization strategy also preserves system message."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="summarization", preserve_recent=3)

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Summary"
        mock_provider.chat.completions.create.return_value = mock_response
        cm.set_summarization_config(mock_provider, "gpt-4")

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="Critical system instructions",
            ),
        ]

        for i in range(15):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Message {i}",
                )
            )

        compacted = cm.compact(messages)

        # System message must be first
        assert compacted[0]["role"] == "system"
        assert compacted[0]["content"] == "Critical system instructions"

    def test_preserve_recent_with_mixed_message_types(self):
        """Test preserve_recent counts all message types correctly."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=6)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        # Add old messages
        for i in range(20):
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"old_{i}",
                    content=f"Old {i}",
                )
            )

        # Add exactly 6 recent messages of mixed types
        recent_messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="R1",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="R2",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="r3",
                content="R3",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="R4",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="R5",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="r6",
                content="R6",
            ),
        ]
        messages.extend(recent_messages)

        compacted = cm.compact(messages)

        # All 6 recent messages should be present
        recent_in_compacted = [
            m
            for m in compacted
            if m.get("content", "") in ["R1", "R2", "R3", "R4", "R5", "R6"]
        ]
        assert len(recent_in_compacted) == 6

    def test_empty_system_message_preserved(self):
        """Test that even empty system messages are preserved."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=2)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="",
            ),  # Empty content
        ]

        for i in range(10):
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"tool_{i}",
                    content=f"Result {i}",
                )
            )

        messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent",
            )
        )

        compacted = cm.compact(messages)

        # System message should still be first, even with empty content
        assert compacted[0]["role"] == "system"
        assert compacted[0]["content"] == ""


class TestCompactionLifecycle:
    """Test compaction lifecycle and repeated compaction scenarios."""

    def test_idempotent_compaction(self):
        """Test that compacting already-compacted messages is idempotent."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=5)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        for i in range(20):
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"tool_{i}",
                    content=f"Result {i}"
                )
            )

        # Recent messages
        for i in range(5):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Recent {i}",
                )
            )

        # First compaction
        compacted1 = cm.compact(messages)
        count1 = len(compacted1)

        # Second compaction (should be idempotent)
        compacted2 = cm.compact(compacted1)
        count2 = len(compacted2)

        # Third compaction
        compacted3 = cm.compact(compacted2)
        count3 = len(compacted3)

        # Should stabilize
        assert count2 == count3  # Idempotent after first compaction
        assert count2 <= count1  # Can't grow

    def test_compaction_threshold_boundary(self):
        """Test compaction behavior at threshold boundaries."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(
            strategy="tool_result_clearing",
            threshold=1000,
            preserve_recent=5
        )

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        # Add messages to just under threshold
        while cm.estimate_tokens(messages) < 950:
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Message " * 10,
                )
            )

        # Should not trigger compaction yet
        assert not cm.should_compact(messages)

        # Add one more message to exceed threshold
        messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content="Final message" * 20,
            )
        )

        # Should now trigger compaction
        assert cm.should_compact(messages)

    def test_alternating_compaction_and_growth(self):
        """Test alternating between compaction and message growth."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="tool_result_clearing", preserve_recent=5)

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        for cycle in range(5):
            # Add many messages
            for i in range(20):
                messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=f"cycle{cycle}_tool{i}",
                        content=f"Result {i}"
                    )
                )

            # Compact
            messages = cm.compact(messages)

            # Should not grow unbounded
            assert len(messages) < 100

    def test_compaction_after_threshold_changes(self):
        """Test compaction behavior after changing threshold."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(
            strategy="tool_result_clearing",
            threshold=100000,
            preserve_recent=5
        )

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        # Add messages with tool results to ensure compaction is meaningful
        for i in range(50):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Message with content " * 10
                )
            )
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"tool_{i}",
                    content="Tool result with data " * 10
                )
            )

        # Verify current token count
        current_tokens = cm.estimate_tokens(messages)

        # Should not trigger with high threshold
        assert not cm.should_compact(messages)

        # Lower threshold to below current token count
        cm.threshold = current_tokens - 1

        # Should now trigger
        assert cm.should_compact(messages)

        # Compact
        compacted = cm.compact(messages)

        # Should be smaller (tool messages removed)
        assert len(compacted) < len(messages)

    def test_summarization_lifecycle(self):
        """Test summarization strategy lifecycle."""
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(strategy="summarization", preserve_recent=3)

        mock_provider = Mock()

        # First summarization
        mock_response1 = Mock()
        mock_response1.choices = [Mock()]
        mock_response1.choices[0].message.content = "First summary"

        # Second summarization
        mock_response2 = Mock()
        mock_response2.choices = [Mock()]
        mock_response2.choices[0].message.content = "Second summary"

        mock_provider.chat.completions.create.side_effect = [mock_response1, mock_response2]
        cm.set_summarization_config(mock_provider, "gpt-4")

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        for i in range(15):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Message {i}",
                )
            )

        # First compaction with summarization
        compacted1 = cm.compact(messages)

        # Should have summary
        summaries1 = [m for m in compacted1 if "[COMPACTED CONTEXT" in str(m.get("content", ""))]
        assert len(summaries1) == 1

        # Add more messages
        for i in range(15, 25):
            compacted1.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Message {i}",
                )
            )

        # Second compaction
        compacted2 = cm.compact(compacted1)

        # Should have new summary (old summary gets summarized with other old messages)
        # Or keeps existing summary plus new one
        summaries2 = [m for m in compacted2 if "[COMPACTED CONTEXT" in str(m.get("content", ""))]
        assert len(summaries2) >= 1


@pytest.fixture
def mock_provider_for_exec_ctx():
    """Create a mock OpenAI provider for ExecutionContext tests."""
    provider = Mock()
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Response"
    response.choices[0].message.tool_calls = None
    provider.chat.completions.create.return_value = response
    return provider


class TestExecutionContextWithContextManager:
    """Test ExecutionContext with ContextManager for context compaction."""

    def test_execution_context_with_disabled_context_manager(self, mock_provider_for_exec_ctx):
        """Test ExecutionContext with ContextManager disabled (threshold=None)."""
        from bestla.yggdrasil import Agent, ConversationContext, ExecutionContext
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(threshold=None)
        conv = ConversationContext(context_manager=cm)

        for i in range(50):
            conv.messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Message {i}",
                )
            )

        initial_count = len(conv.messages)

        exec_ctx = ExecutionContext(
            toolkits={},
            independent_toolkit=Agent(
                provider=mock_provider_for_exec_ctx, model="gpt-4"
            ).independent_toolkit,
            conversation_context=conv,
        )

        agent = Agent(provider=mock_provider_for_exec_ctx, model="gpt-4")
        response, result_ctx = agent.run("Test query", execution_context=exec_ctx)

        assert len(result_ctx.conversation.messages) > initial_count

    def test_execution_context_with_compaction_enabled(self, mock_provider_for_exec_ctx):
        """Test ExecutionContext with ContextManager compaction enabled."""
        from bestla.yggdrasil import Agent, ConversationContext, ExecutionContext
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(
            strategy="tool_result_clearing",
            threshold=100,
            preserve_recent=5,
        )
        conv = ConversationContext(context_manager=cm)

        conv.messages.append(
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            )
        )
        for i in range(20):
            conv.messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Old message {i}",
                )
            )
            conv.messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"old_{i}",
                    content=f"Old tool result {i}",
                )
            )

        exec_ctx = ExecutionContext(
            toolkits={},
            independent_toolkit=Agent(
                provider=mock_provider_for_exec_ctx, model="gpt-4"
            ).independent_toolkit,
            conversation_context=conv,
        )

        agent = Agent(provider=mock_provider_for_exec_ctx, model="gpt-4")
        response, result_ctx = agent.run("Test query", execution_context=exec_ctx)

        assert result_ctx.conversation.messages[0]["role"] == "system"

        tool_count = len([m for m in result_ctx.conversation.messages if m.get("role") == "tool"])
        assert tool_count < 20

    def test_multiple_runs_with_same_execution_context(self, mock_provider_for_exec_ctx):
        """Test multiple runs sharing the same ExecutionContext (stateful)."""
        from bestla.yggdrasil import Agent, ConversationContext, ExecutionContext
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(
            strategy="tool_result_clearing",
            threshold=50,
            preserve_recent=3,
        )
        conv = ConversationContext(context_manager=cm)

        exec_ctx = ExecutionContext(
            toolkits={},
            independent_toolkit=Agent(
                provider=mock_provider_for_exec_ctx, model="gpt-4"
            ).independent_toolkit,
            conversation_context=conv,
        )

        agent = Agent(provider=mock_provider_for_exec_ctx, model="gpt-4")

        response1, ctx1 = agent.run("Query 1", execution_context=exec_ctx)
        response2, ctx2 = agent.run("Query 2", execution_context=ctx1)
        response3, ctx3 = agent.run("Query 3", execution_context=ctx2)

        assert ctx3 is ctx2 is ctx1 is exec_ctx

    def test_compaction_during_long_conversation(self, mock_provider_for_exec_ctx):
        """Test that compaction triggers during a long conversation."""
        from bestla.yggdrasil import Agent, ConversationContext, ExecutionContext
        from bestla.yggdrasil.context_manager import ContextManager

        cm = ContextManager(
            strategy="tool_result_clearing",
            threshold=50,
            preserve_recent=3,
        )
        conv = ConversationContext(context_manager=cm)

        exec_ctx = ExecutionContext(
            toolkits={},
            independent_toolkit=Agent(
                provider=mock_provider_for_exec_ctx, model="gpt-4"
            ).independent_toolkit,
            conversation_context=conv,
        )

        agent = Agent(provider=mock_provider_for_exec_ctx, model="gpt-4")

        for i in range(10):
            exec_ctx.conversation.messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Long message {i}" * 20,
                )
            )
            exec_ctx.conversation.messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=f"tool_{i}",
                    content=f"Tool result {i}" * 20
                )
            )

        response, result_ctx = agent.run("Final query", execution_context=exec_ctx)

        assert len(result_ctx.conversation.messages) < 30
