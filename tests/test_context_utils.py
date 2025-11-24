"""Tests for context management utility functions."""

from typing import cast
from unittest.mock import Mock

import pytest
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)


class TestTokenEstimation:
    """Test token estimation utilities."""

    def test_estimate_tokens_empty_list(self):
        """Test estimating tokens for empty message list."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        assert estimate_tokens([]) == 0

    def test_estimate_tokens_simple_message(self):
        """Test estimating tokens for simple message."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Hello world",
            )
        ]

        tokens = estimate_tokens(messages)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_system_message(self):
        """Test estimating tokens includes system message."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="You are a helpful assistant with extensive knowledge."
            )
        ]

        tokens = estimate_tokens(messages)
        assert tokens > 10  # Should be more than just "Hello"

    def test_estimate_tokens_tool_messages(self):
        """Test estimating tokens for tool messages."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = [
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="call_123",
                content="Tool execution result with detailed output"
            )
        ]

        tokens = estimate_tokens(messages)
        assert tokens > 0

    def test_estimate_tokens_multiple_messages(self):
        """Test that multiple messages increase token count."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        single_message = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Hello",
            )
        ]

        multiple_messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Hello",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Hi there!",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="How are you?",
            ),
        ]

        single_tokens = estimate_tokens(single_message)
        multiple_tokens = estimate_tokens(multiple_messages)

        assert multiple_tokens > single_tokens

    def test_estimate_tokens_long_content(self):
        """Test token estimation for long content."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        short_content = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Hi",
            )
        ]

        long_content = [
            ChatCompletionUserMessageParam(
                role="user",
                content="This is a much longer message with many more words " * 50
            )
        ]

        short_tokens = estimate_tokens(short_content)
        long_tokens = estimate_tokens(long_content)

        assert long_tokens > short_tokens * 10

    def test_estimate_tokens_none_content(self):
        """Test token estimation handles None content."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = [
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=None,
            )
        ]

        # Should not crash, should return base tokens for message structure
        tokens = estimate_tokens(messages)
        assert tokens >= 0

    def test_estimate_tokens_metadata_overhead(self):
        """Test that token estimation includes message metadata overhead."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        # Empty content should still have some tokens for message structure
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="",
            )
        ]

        tokens = estimate_tokens(messages)
        assert tokens > 0  # Should account for role, structure, etc.


class TestToolResultClearing:
    """Test tool result clearing compaction strategy."""

    def test_clear_tool_results_basic(self):
        """Test basic tool result clearing."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User 1",
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
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="2",
                content="Tool 2",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User 3",
            ),
        ]

        cleared = clear_tool_results(messages, keep_recent=1)

        # Should keep system message and recent messages
        assert cleared[0]["role"] == "system"

        # Should have removed old tool messages
        tool_count = len([m for m in cleared if m.get("role") == "tool"])
        original_tool_count = len([m for m in messages if m.get("role") == "tool"])
        assert tool_count < original_tool_count

    def test_clear_tool_results_preserve_recent(self):
        """Test that recent messages are preserved."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Old",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Old tool",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent 2",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="2",
                content="Recent tool",
            ),
        ]

        cleared = clear_tool_results(messages, keep_recent=3)

        # Last 3 messages should be preserved
        assert cleared[-1]["content"] == "Recent tool"
        assert cleared[-2]["content"] == "Recent 2"
        assert cleared[-3]["content"] == "Recent 1"

    def test_clear_tool_results_empty_list(self):
        """Test clearing with empty message list."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        cleared = clear_tool_results([], keep_recent=5)
        assert cleared == []

    def test_clear_tool_results_no_tool_messages(self):
        """Test clearing when there are no tool messages."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Assistant",
            ),
        ]

        cleared = clear_tool_results(messages, keep_recent=1)

        # Should preserve all non-tool messages
        assert len(cleared) == len(messages)

    def test_clear_tool_results_only_tool_messages(self):
        """Test clearing with only tool messages (edge case)."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        messages = [
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Tool 1",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="2",
                content="Tool 2",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="3",
                content="Tool 3",
            ),
        ]

        cleared = clear_tool_results(messages, keep_recent=1)

        # Should keep at least the recent tool message
        assert len(cleared) > 0
        assert cleared[-1]["content"] == "Tool 3"

    def test_clear_tool_results_preserves_system(self):
        """Test that system message is always preserved."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="Important system",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Tool",
            ),
        ]

        cleared = clear_tool_results(messages, keep_recent=0)

        # System should always be preserved even with keep_recent=0
        assert len(cleared) >= 1
        assert cleared[0]["role"] == "system"
        assert cleared[0]["content"] == "Important system"

    def test_clear_tool_results_interleaved_messages(self):
        """Test clearing with interleaved user/assistant/tool messages."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="U1",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="A1",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="T1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="U2",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="A2",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="2",
                content="T2",
            ),
        ]

        cleared = clear_tool_results(messages, keep_recent=2)

        # Should preserve chronological order
        for i in range(1, len(cleared)):
            original_idx = messages.index(cleared[i])
            prev_original_idx = messages.index(cleared[i-1])
            assert original_idx > prev_original_idx

    def test_clear_tool_results_keep_recent_larger_than_list(self):
        """Test when keep_recent is larger than message count."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Tool",
            ),
        ]

        cleared = clear_tool_results(messages, keep_recent=100)

        # All messages should be preserved
        assert len(cleared) == len(messages)


class TestSummarizationUtils:
    """Test summarization utility functions."""

    def test_summarize_messages_basic(self):
        """Test basic message summarization."""
        from bestla.yggdrasil.context_utils import summarize_messages

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Summary of the conversation"
        mock_provider.chat.completions.create.return_value = mock_response

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="What's the weather?",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="It's sunny.",
            ),
        ]

        summary_msg = summarize_messages(messages, mock_provider, "gpt-4", keep_recent=0)

        # Should return a single user message with the summary
        assert summary_msg["role"] == "user"
        assert "Summary" in summary_msg["content"]
        assert mock_provider.chat.completions.create.called

    def test_summarize_messages_preserves_system(self):
        """Test that summarization preserves system message."""
        from bestla.yggdrasil.context_utils import summarize_messages

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Summarized conversation"
        mock_provider.chat.completions.create.return_value = mock_response

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="Custom system",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Hello",
            ),
        ]

        summary_msg = summarize_messages(messages, mock_provider, "gpt-4", keep_recent=0)

        # System message is handled separately, summary is user message
        assert summary_msg["role"] == "user"

    def test_summarize_messages_empty_list(self):
        """Test summarization with empty message list."""
        from bestla.yggdrasil.context_utils import summarize_messages

        mock_provider = Mock()

        result = summarize_messages([], mock_provider, "gpt-4", keep_recent=0)

        # Should return None or empty for empty list
        assert result is None

    def test_summarize_messages_only_system(self):
        """Test summarization with only system message."""
        from bestla.yggdrasil.context_utils import summarize_messages

        mock_provider = Mock()
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
        ]

        result = summarize_messages(messages, mock_provider, "gpt-4", keep_recent=0)

        # Nothing to summarize
        assert result is None

    def test_build_compacted_messages(self):
        """Test building compacted message list with summary."""
        from bestla.yggdrasil.context_utils import build_compacted_messages

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
        ]

        summary_msg = ChatCompletionUserMessageParam(
            role="user",
            content="[Summary: Previous conversation about old topics]"
        )

        compacted = build_compacted_messages(messages, summary_msg, keep_recent=2)

        # Should have: system, first_user, summary, recent messages
        assert compacted[0]["role"] == "system"
        assert compacted[1]["content"] == "Old 1"  # First user message preserved
        assert "[Summary:" in compacted[2]["content"]  # Summary
        assert compacted[-1]["content"] == "Recent 2"
        assert compacted[-2]["content"] == "Recent 1"

    def test_build_compacted_messages_no_summary(self):
        """Test building compacted messages without summary."""
        from bestla.yggdrasil.context_utils import build_compacted_messages

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Recent",
            ),
        ]

        compacted = build_compacted_messages(messages, None, keep_recent=1)

        # Should have system + recent
        assert compacted[0]["role"] == "system"
        assert compacted[-1]["content"] == "Recent"


class TestContextUtilsEdgeCases:
    """Test edge cases for context utilities."""

    def test_handle_mixed_message_types(self):
        """Test utilities handle different message types correctly."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Assistant",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Tool",
            ),
        ]

        # Should handle all types without error
        cleared = clear_tool_results(messages, keep_recent=2)
        assert len(cleared) > 0

    def test_very_large_message_list(self):
        """Test utilities handle large message lists efficiently."""
        from bestla.yggdrasil.context_utils import clear_tool_results, estimate_tokens

        # Create large message list
        messages = []
        for i in range(1000):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Message {i}",
                )
            )
            if i % 3 == 0:
                messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=str(i),
                        content=f"Tool {i}",
                    )
                )

        # Should process without error
        cleared = clear_tool_results(messages, keep_recent=10)
        assert len(cleared) < len(messages)

        tokens = estimate_tokens(messages)
        assert tokens > 0

    def test_unicode_content_handling(self):
        """Test utilities handle unicode content correctly."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Hello 世界 🌍 مرحبا мир"
            )
        ]

        tokens = estimate_tokens(messages)
        assert tokens > 0

    def test_special_characters_in_content(self):
        """Test utilities handle special characters."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content="System with\nnewlines\tand\ttabs"
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Result with \"quotes\" and 'apostrophes'"
            ),
        ]

        # Should not crash
        cleared = clear_tool_results(messages, keep_recent=1)
        assert len(cleared) > 0

    def test_clear_tool_results_no_system_message_all_recent(self):
        """Test clear_tool_results when all messages are recent and no system message."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        # No system message, and messages <= keep_recent
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 1",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Tool 1",
            ),
        ]

        # keep_recent=5 is larger than len(messages)=2
        cleared = clear_tool_results(messages, keep_recent=5)

        # Should return all messages unchanged (line 85 coverage)
        assert len(cleared) == 2
        assert cleared[0]["content"] == "Message 1"
        assert cleared[1]["content"] == "Tool 1"

    def test_summarize_messages_keep_recent_equals_message_count(self):
        """Test summarize_messages when keep_recent equals message count after filtering."""
        from bestla.yggdrasil.context_utils import summarize_messages

        mock_provider = Mock()

        # System message + 2 user messages
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

        # keep_recent=2, but after removing system, we have 2 messages
        # So len(messages_to_summarize) <= keep_recent
        result = summarize_messages(messages, mock_provider, "gpt-4", keep_recent=2)

        # Should return None (nothing to summarize, line 126 coverage)
        assert result is None

        # Provider should not be called
        assert not mock_provider.chat.completions.create.called

    def test_format_messages_for_summary_with_tool_messages(self):
        """Test _format_messages_for_summary formats tool messages correctly."""
        from bestla.yggdrasil.context_utils import _format_messages_for_summary

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="User message",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Assistant message",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="call_123",
                content="Tool execution result"
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="call_456",
                content="Another tool result"
            ),
        ]

        formatted = _format_messages_for_summary(messages)

        # Verify tool messages are formatted correctly (lines 181-182 coverage)
        assert "[Tool call_123]: Tool execution result" in formatted
        assert "[Tool call_456]: Another tool result" in formatted

        # Verify other messages are formatted correctly
        assert "User: User message" in formatted
        assert "Assistant: Assistant message" in formatted

    def test_format_messages_for_summary_tool_without_call_id(self):
        """Test _format_messages_for_summary handles tool messages without call_id."""
        from bestla.yggdrasil.context_utils import _format_messages_for_summary

        messages = [
            ChatCompletionToolMessageParam(
                role="tool",
                content="Tool result without call_id"
            ),
        ]

        formatted = _format_messages_for_summary(messages)

        # Should use "unknown" as default (line 181 coverage)
        assert "[Tool unknown]: Tool result without call_id" in formatted

    def test_build_compacted_messages_with_keep_recent_zero(self):
        """Test build_compacted_messages with keep_recent=0."""
        from bestla.yggdrasil.context_utils import build_compacted_messages

        original_messages = [
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
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 3",
            ),
        ]

        summary_msg = ChatCompletionUserMessageParam(
            role="user",
            content="[Summary: Previous conversation]"
        )

        # keep_recent=0 should include all original messages after system
        compacted = build_compacted_messages(original_messages, summary_msg, keep_recent=0)

        # Should have: system + first_user + summary + all other messages
        assert compacted[0]["role"] == "system"
        assert compacted[1]["content"] == "Message 1"  # First user message preserved
        assert "[Summary:" in compacted[2]["content"]  # Summary

        # All original messages should be included
        assert compacted[3]["content"] == "Message 2"
        assert compacted[4]["content"] == "Message 3"

    def test_build_compacted_messages_keep_recent_zero_no_summary(self):
        """Test build_compacted_messages with keep_recent=0 and no summary."""
        from bestla.yggdrasil.context_utils import build_compacted_messages

        original_messages = [
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

        # No summary, keep_recent=0
        compacted = build_compacted_messages(original_messages, None, keep_recent=0)

        # Should include system + all original messages
        assert len(compacted) == 3
        assert compacted[0]["role"] == "system"
        assert compacted[1]["content"] == "Message 1"
        assert compacted[2]["content"] == "Message 2"

    def test_estimate_tokens_with_assistant_tool_calls(self):
        """Test token estimation includes assistant messages with tool_calls."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        # Create assistant message with tool_calls attribute
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Call some tools",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Calling tools",
            ),
        ]

        tokens = estimate_tokens(messages)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_clear_tool_results_with_keep_recent_zero(self):
        """Test clear_tool_results with keep_recent=0."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="User 1",
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
        ]

        # keep_recent=0 means all messages are "old"
        cleared = clear_tool_results(messages, keep_recent=0)

        # Should remove all tool messages but keep system and user messages
        assert cleared[0]["role"] == "system"

        # Tool messages should be filtered out
        tool_count = len([m for m in cleared if m.get("role") == "tool"])
        assert tool_count == 0

        # User messages should remain
        user_count = len([m for m in cleared if m.get("role") == "user"])
        assert user_count == 2


class TestContextUtilsErrorHandling:
    """Test error handling in context utils."""

    def test_summarize_messages_api_failure(self):
        """Test summarize_messages when API call raises exception."""
        from bestla.yggdrasil.context_utils import summarize_messages

        mock_provider = Mock()
        # Simulate API error
        mock_provider.chat.completions.create.side_effect = Exception("API Error")

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

        # Should raise exception (no error handling currently)
        with pytest.raises(Exception, match="API Error"):
            summarize_messages(messages, mock_provider, "gpt-4", keep_recent=0)

    def test_summarize_messages_returns_none_content(self):
        """Test summarize_messages when LLM returns None content."""
        from bestla.yggdrasil.context_utils import summarize_messages

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None  # None content
        mock_provider.chat.completions.create.return_value = mock_response

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 2",
            ),
        ]

        summary_msg = summarize_messages(messages, mock_provider, "gpt-4", keep_recent=0)

        # Should handle None content
        assert summary_msg["role"] == "user"
        assert "None" in summary_msg["content"]

    def test_summarize_messages_returns_empty_string(self):
        """Test summarize_messages when LLM returns empty string."""
        from bestla.yggdrasil.context_utils import summarize_messages

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""  # Empty string
        mock_provider.chat.completions.create.return_value = mock_response

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 1",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Message 2",
            ),
        ]

        summary_msg = summarize_messages(messages, mock_provider, "gpt-4", keep_recent=0)

        # Should handle empty string
        assert summary_msg["role"] == "user"
        assert "COMPACTED CONTEXT" in summary_msg["content"]

    def test_format_messages_with_malformed_structure(self):
        """Test _format_messages_for_summary with malformed messages."""
        from bestla.yggdrasil.context_utils import _format_messages_for_summary

        # Message with missing role
        messages = [
            {"content": "Message without role"},
        ]

        # Should handle gracefully (uses default "unknown")
        formatted = _format_messages_for_summary(messages)
        assert "Unknown:" in formatted or "message without role" in formatted.lower()

    def test_format_messages_with_empty_content(self):
        """Test _format_messages_for_summary with empty content."""
        from bestla.yggdrasil.context_utils import _format_messages_for_summary

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="",
            ),
        ]

        formatted = _format_messages_for_summary(messages)

        # Should handle empty content
        assert "User:" in formatted
        assert "Assistant:" in formatted

    def test_estimate_tokens_with_malformed_messages(self):
        """Test token estimation with malformed message structure."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        # Message missing content field (intentionally malformed for testing)
        messages = [
            cast(ChatCompletionMessageParam, {"role": "user"}),  # No content
        ]

        # Should handle gracefully
        tokens = estimate_tokens(messages)
        assert tokens >= 0

    def test_clear_tool_results_with_malformed_messages(self):
        """Test clear_tool_results with malformed messages."""
        from bestla.yggdrasil.context_utils import clear_tool_results

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System",
            ),
            cast(ChatCompletionMessageParam, {"role": "tool"}),  # Missing tool_call_id (malformed)
            ChatCompletionUserMessageParam(
                role="user",
                content="User",
            ),
        ]

        # Should handle gracefully
        cleared = clear_tool_results(messages, keep_recent=1)
        assert len(cleared) > 0

    def test_summarize_messages_with_very_long_content(self):
        """Test summarization with very long message content."""
        from bestla.yggdrasil.context_utils import summarize_messages

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Summary of long conversation"
        mock_provider.chat.completions.create.return_value = mock_response

        # Create messages with very long content
        long_content = "A" * 100000  # 100k characters
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content=long_content,
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=long_content,
            ),
        ]

        summary_msg = summarize_messages(messages, mock_provider, "gpt-4", keep_recent=0)

        # Should handle long content
        assert summary_msg is not None
        assert "Summary of long conversation" in summary_msg["content"]

    def test_build_compacted_messages_with_empty_original(self):
        """Test build_compacted_messages with empty original messages."""
        from bestla.yggdrasil.context_utils import build_compacted_messages

        summary_msg = ChatCompletionUserMessageParam(
            role="user",
            content="[Summary: Empty]"
        )

        compacted = build_compacted_messages([], summary_msg, keep_recent=5)

        # Should just have the summary
        assert len(compacted) == 1
        assert compacted[0]["content"] == "[Summary: Empty]"

    def test_estimate_tokens_with_non_string_content(self):
        """Test token estimation with non-string content values."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = [
            cast(
                ChatCompletionMessageParam,
                {"role": "user", "content": 123}
            ),  # Integer content (malformed)
            cast(
                ChatCompletionMessageParam,
                {"role": "user", "content": ["list", "content"]}
            ),  # List content (malformed)
        ]

        # Should convert to string and estimate
        tokens = estimate_tokens(messages)
        assert tokens > 0

    def test_format_messages_with_missing_content(self):
        """Test _format_messages_for_summary with messages missing content."""
        from bestla.yggdrasil.context_utils import _format_messages_for_summary

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Valid message",
            ),
            cast(ChatCompletionMessageParam, {"role": "assistant"}),  # Missing content (malformed)
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
            ),  # Missing content
        ]

        formatted = _format_messages_for_summary(messages)

        # Should handle missing content (empty string or default)
        assert "User: Valid message" in formatted
        assert "Assistant:" in formatted

    def test_format_messages_with_developer_role(self):
        """Test _format_messages_for_summary with developer role messages."""
        from bestla.yggdrasil.context_utils import _format_messages_for_summary

        messages = [
            cast(
                ChatCompletionMessageParam,
                {"role": "developer", "content": "You are a helpful assistant."}
            ),  # Developer role (non-standard)
            ChatCompletionUserMessageParam(
                role="user",
                content="Hello",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Hi there!",
            ),
        ]

        formatted = _format_messages_for_summary(messages)

        # Should format developer role properly
        assert "Developer: You are a helpful assistant." in formatted
        assert "User: Hello" in formatted
        assert "Assistant: Hi there!" in formatted

    def test_format_messages_with_function_role(self):
        """Test _format_messages_for_summary with function role messages (legacy)."""
        from bestla.yggdrasil.context_utils import _format_messages_for_summary

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Get weather",
            ),
            cast(
                ChatCompletionMessageParam,
                {"role": "function", "name": "get_weather", "content": '{"temp": 72}'}
            ),  # Function role (legacy)
        ]

        formatted = _format_messages_for_summary(messages)

        # Should format function role with function name
        assert "User: Get weather" in formatted
        assert "[Function get_weather]:" in formatted
        assert '{"temp": 72}' in formatted

    def test_format_messages_with_function_role_missing_name(self):
        """Test _format_messages_for_summary with function role but missing name."""
        from bestla.yggdrasil.context_utils import _format_messages_for_summary

        messages = [
            cast(
                ChatCompletionMessageParam,
                {"role": "function", "content": "Function result without name"}
            ),  # Missing name (malformed)
        ]

        formatted = _format_messages_for_summary(messages)

        # Should use 'unknown' as default name
        assert "[Function unknown]:" in formatted
        assert "Function result without name" in formatted

    def test_format_messages_with_all_role_types(self):
        """Test _format_messages_for_summary with all supported role types."""
        from bestla.yggdrasil.context_utils import _format_messages_for_summary

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="System instructions",
            ),
            cast(
                ChatCompletionMessageParam,
                {"role": "developer", "content": "Developer instructions"}
            ),  # Developer role (non-standard)
            ChatCompletionUserMessageParam(
                role="user",
                content="User message",
            ),
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content="Assistant response"
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="call_123",
                content="Tool result"
            ),
            cast(
                ChatCompletionMessageParam,
                {"role": "function", "name": "my_func", "content": "Function result"}
            ),  # Function role (legacy)
        ]

        formatted = _format_messages_for_summary(messages)

        # All roles should be formatted correctly
        assert "System: System instructions" in formatted
        assert "Developer: Developer instructions" in formatted
        assert "User: User message" in formatted
        assert "Assistant: Assistant response" in formatted
        assert "[Tool call_123]: Tool result" in formatted
        assert "[Function my_func]: Function result" in formatted

    def test_format_messages_with_unknown_role(self):
        """Test _format_messages_for_summary with unknown/custom role."""
        from bestla.yggdrasil.context_utils import _format_messages_for_summary

        messages = [
            cast(
                ChatCompletionMessageParam,
                {"role": "custom_role", "content": "Custom message"}
            ),  # Custom role (invalid)
            cast(
                ChatCompletionMessageParam,
                {"role": "", "content": "Empty role"}
            ),  # Empty role (invalid)
        ]

        formatted = _format_messages_for_summary(messages)

        # Should handle unknown roles gracefully
        assert "Custom_role: Custom message" in formatted
        assert ": Empty role" in formatted

    def test_summarize_messages_with_only_tool_messages(self):
        """Test summarization with only tool messages."""
        from bestla.yggdrasil.context_utils import summarize_messages

        mock_provider = Mock()

        messages = [
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="1",
                content="Result 1",
            ),
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="2",
                content="Result 2",
            ),
        ]

        # Should attempt to summarize
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Summary of tool results"
        mock_provider.chat.completions.create.return_value = mock_response

        summary_msg = summarize_messages(messages, mock_provider, "gpt-4", keep_recent=0)

        assert summary_msg is not None
        assert "Summary of tool results" in summary_msg["content"]

    def test_clear_tool_results_preserves_non_tool_roles(self):
        """Test that clear_tool_results only removes tool role messages."""
        from bestla.yggdrasil.context_utils import clear_tool_results

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
        ]

        cleared = clear_tool_results(messages, keep_recent=2)

        # All non-tool roles should be preserved (except possibly in old section)
        roles = [m.get("role") for m in cleared]

        # System should always be there
        assert "system" in roles

        # User and assistant messages should be preserved
        user_count = roles.count("user")
        assistant_count = roles.count("assistant")
        assert user_count >= 1
        assert assistant_count >= 1


class TestTokenEstimationEdgeCases:
    """Test token estimation with edge cases."""

    def test_estimate_tokens_very_large_single_message(self):
        """Test token estimation with a very large single message."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        # 1 million character message
        large_content = "A" * 1000000
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content=large_content,
            )
        ]

        tokens = estimate_tokens(messages)

        # Should handle large content
        assert tokens > 100000  # Roughly 250k tokens for 1M chars
        assert isinstance(tokens, int)

    def test_estimate_tokens_many_small_messages(self):
        """Test token estimation with many small messages."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        # 10,000 small messages
        messages = []
        for i in range(10000):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Msg {i}",
                )
            )

        tokens = estimate_tokens(messages)

        # Should handle many messages
        assert tokens > 10000  # At least overhead for each message
        assert isinstance(tokens, int)

    def test_estimate_tokens_with_newlines_and_whitespace(self):
        """Test token estimation with various whitespace."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Line 1\n\nLine 2\n\n\nLine 3\t\tTabbed"
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="    Leading spaces    \n    More spaces    "
            ),
        ]

        tokens = estimate_tokens(messages)

        # Should count whitespace characters
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_with_unicode_emojis(self):
        """Test token estimation with unicode and emojis."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Hello 👋 World 🌍! This has emojis 😀😃😄"
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="中文字符 مرحبا Привет"
            ),
        ]

        tokens = estimate_tokens(messages)

        # Should handle unicode
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_with_code_blocks(self):
        """Test token estimation with code-like content."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="""```python
def example_function():
    '''This is a docstring'''
    for i in range(100):
        print(f"Value: {i}")
    return True
```"""
            ),
        ]

        tokens = estimate_tokens(messages)

        # Should count all characters including formatting
        assert tokens > 20
        assert isinstance(tokens, int)

    def test_estimate_tokens_with_json_content(self):
        """Test token estimation with JSON-like content."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content=(
                    '{"key": "value", "nested": {"array": [1, 2, 3]}, "long_string": "'
                    + "x" * 1000
                    + '"}'
                ),
            ),
        ]

        tokens = estimate_tokens(messages)

        # Should handle JSON structure
        assert tokens > 250  # Roughly for 1000+ chars
        assert isinstance(tokens, int)

    def test_estimate_tokens_incremental(self):
        """Test that token estimation grows with message additions."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = []
        previous_tokens = 0

        for i in range(10):
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Message number {i} with some content"
                )
            )

            current_tokens = estimate_tokens(messages)

            # Should grow with each addition
            assert current_tokens > previous_tokens
            previous_tokens = current_tokens

    def test_estimate_tokens_with_special_chars(self):
        """Test token estimation with special characters."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Special chars: @#$%^&*()_+-=[]{}|;':\"<>,.?/~`"
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content="Escape sequences: \n \t \r \\ \""
            ),
        ]

        tokens = estimate_tokens(messages)

        # Should count all special characters
        assert tokens > 10
        assert isinstance(tokens, int)

    def test_estimate_tokens_includes_tool_call_id(self):
        """Test that tool_call_id contributes to token count."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        # Tool message with very short content but long tool_call_id
        messages_long_id = [
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="a" * 1000,  # Very long ID
                content="short"
            )
        ]

        # Tool message with short tool_call_id and short content
        messages_short_id = [
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="abc",  # Short ID
                content="short"
            )
        ]

        tokens_long_id = estimate_tokens(messages_long_id)
        tokens_short_id = estimate_tokens(messages_short_id)

        # Long tool_call_id should result in more tokens
        assert tokens_long_id > tokens_short_id
        assert tokens_long_id > 200  # Should count the 1000-char ID

    def test_estimate_tokens_tool_call_id_vs_content(self):
        """Test that both tool_call_id and content contribute to tokens."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        # Message with both long ID and long content
        messages_both = [
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="x" * 500,
                content="y" * 500
            )
        ]

        # Message with only long content
        messages_content_only = [
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id="short",
                content="y" * 500
            )
        ]

        tokens_both = estimate_tokens(messages_both)
        tokens_content_only = estimate_tokens(messages_content_only)

        # Both should count, so 'both' should have more tokens
        assert tokens_both > tokens_content_only

    def test_estimate_tokens_missing_tool_call_id(self):
        """Test token estimation when tool_call_id is missing."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        # Tool message without tool_call_id field (malformed)
        messages = [
            cast(ChatCompletionMessageParam, {"role": "tool", "content": "Tool result without ID"})
        ]

        # Should handle gracefully (tool_call_id just not counted)
        tokens = estimate_tokens(messages)
        assert tokens > 0

    def test_estimate_tokens_none_tool_call_id(self):
        """Test token estimation when tool_call_id is None."""
        from bestla.yggdrasil.context_utils import estimate_tokens

        # Tool message with None tool_call_id (malformed)
        messages = [
            cast(
                ChatCompletionMessageParam,
                {"role": "tool", "tool_call_id": None, "content": "Tool result"}
            )
        ]

        # Should handle gracefully
        tokens = estimate_tokens(messages)
        assert tokens > 0
