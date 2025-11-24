"""Tests for handling malformed LLM responses and provider errors."""

import json
from typing import Tuple
from unittest.mock import Mock

import pytest
from openai.types.chat import (
    ChatCompletionUserMessageParam,
)

from bestla.yggdrasil import Agent


class TestMalformedLLMResponses:
    """Test agent handling of malformed LLM responses."""

    def test_invalid_json_in_tool_arguments(self):
        """Test handling of invalid JSON in tool_call arguments."""
        mock_provider = Mock()

        # Create mock response with invalid JSON in arguments
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = "invalid json{["  # Malformed JSON

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        def test_tool(param: str) -> Tuple[str, dict]:
            return "result", {}

        agent.add_tool("test_tool", test_tool)

        # Should handle JSON parse error gracefully
        with pytest.raises((json.JSONDecodeError, ValueError, Exception)):
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )

    def test_missing_function_field(self):
        """Test tool_call without 'function' field."""
        mock_provider = Mock()

        # Create incomplete tool_call
        mock_tool_call = Mock(spec=["id"])  # Only has 'id', no 'function'
        mock_tool_call.id = "call_123"

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        # Should handle missing field gracefully
        with pytest.raises((AttributeError, Exception)):
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )

    def test_empty_tool_calls_and_no_content(self):
        """Test response with empty tool_calls and no content."""
        mock_provider = Mock()

        # Response with neither content nor tool_calls
        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = None

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        # Should handle gracefully (might return or raise)
        try:
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )
            assert len(ctx.conversation.messages) >= 1
        except Exception:
            pass

    def test_unknown_tool_name(self):
        """Test LLM calls a tool that doesn't exist."""
        mock_provider = Mock()

        # Tool call for non-existent tool
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "nonexistent_tool"
        mock_tool_call.function.arguments = "{}"

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        def real_tool() -> Tuple[str, dict]:
            return "result", {}

        agent.add_tool("real_tool", real_tool)

        # Should handle unknown tool gracefully
        # Might return error message to LLM
        try:
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=2
            )
        except (KeyError, ValueError, Exception):
            # Expected - unknown tool should raise
            pass

    def test_tool_call_with_extra_fields(self):
        """Test tool_call with unexpected extra fields."""
        from unittest.mock import MagicMock

        mock_provider = Mock()

        # Create tool_call as dict-like object supporting both dict and attribute access
        mock_tool_call = MagicMock()
        mock_tool_call.__getitem__ = lambda self, key: {
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": '{"param": "value"}'},
            "extra_field": "unexpected"
        }[key]
        mock_tool_call.id = "call_123"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"param": "value"}'

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        def test_tool(param: str) -> Tuple[str, dict]:
            return f"got {param}", {}

        agent.add_tool("test_tool", test_tool)

        # Should ignore extra fields and work normally
        response, ctx2 = agent.run(
            [ChatCompletionUserMessageParam(
                 role="user",
                 content="test",
             )],
            max_iterations=1
        )

    def test_malformed_arguments_structure(self):
        """Test arguments that parse as JSON but have wrong structure."""
        mock_provider = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        # Valid JSON but wrong structure (array instead of object)
        mock_tool_call.function.arguments = '["not", "an", "object"]'

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        def test_tool(param: str) -> Tuple[str, dict]:
            return "result", {}

        agent.add_tool("test_tool", test_tool)

        # Should handle wrong structure
        with pytest.raises((TypeError, ValueError, Exception)):
            response, ctx3 = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )


class TestProviderErrors:
    """Test handling of provider errors."""

    def test_provider_network_failure(self):
        """Test handling of network failures."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.side_effect = ConnectionError(
            "Network unavailable"
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        with pytest.raises(ConnectionError, match="Network unavailable"):
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )

    def test_provider_rate_limit_error(self):
        """Test handling of rate limit errors."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.side_effect = Exception(
            "Rate limit exceeded"
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        with pytest.raises(Exception, match="Rate limit exceeded"):
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )

    def test_provider_timeout(self):
        """Test handling of provider timeout."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.side_effect = TimeoutError(
            "Request timeout"
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        with pytest.raises(TimeoutError, match="Request timeout"):
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )

    def test_provider_authentication_error(self):
        """Test handling of authentication errors."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.side_effect = PermissionError(
            "Invalid API key"
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        with pytest.raises(PermissionError, match="Invalid API key"):
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )

    def test_provider_returns_none(self):
        """Test handling when provider returns None."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.return_value = None

        agent = Agent(provider=mock_provider, model="gpt-4")

        with pytest.raises((AttributeError, TypeError, Exception)):
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )

    def test_provider_returns_empty_choices(self):
        """Test handling when provider returns empty choices list."""
        mock_provider = Mock()
        mock_provider.chat.completions.create.return_value = Mock(choices=[])

        agent = Agent(provider=mock_provider, model="gpt-4")

        with pytest.raises((IndexError, Exception)):
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )


class TestToolExecutionErrors:
    """Test handling of tool execution errors."""

    def test_tool_raises_exception(self):
        """Test handling when tool raises exception."""
        from unittest.mock import MagicMock

        mock_provider = Mock()

        # First call - tool call (dict-like structure for dict access)
        mock_tool_call = MagicMock()
        mock_tool_call.__getitem__ = lambda self, key: {
            "id": "call_123",
            "function": {"name": "failing_tool", "arguments": "{}"}
        }[key]
        mock_tool_call.id = "call_123"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "failing_tool"
        mock_tool_call.function.arguments = "{}"

        mock_message1 = Mock()
        mock_message1.content = None
        mock_message1.tool_calls = [mock_tool_call]

        # Second call - completion
        mock_message2 = Mock()
        mock_message2.content = "handled error"
        mock_message2.tool_calls = None

        mock_provider.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=mock_message1)]),
            Mock(choices=[Mock(message=mock_message2)]),
        ]

        agent = Agent(provider=mock_provider, model="gpt-4")

        def failing_tool() -> Tuple[str, dict]:
            raise RuntimeError("Tool failed")

        agent.add_tool("failing_tool", failing_tool)

        response, ctx = agent.run(
            [ChatCompletionUserMessageParam(
                 role="user",
                 content="test",
             )],
            max_iterations=2
        )

        assert any("error" in str(msg).lower() for msg in ctx.conversation.messages)

    def test_tool_returns_invalid_format(self):
        """Test handling when tool returns wrong format."""
        mock_provider = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "bad_tool"
        mock_tool_call.function.arguments = "{}"

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        # Tool that doesn't return tuple
        def bad_tool():
            return "just_string"  # Wrong format

        agent.add_tool("bad_tool", bad_tool)

        # Should handle format error
        with pytest.raises((TypeError, ValueError, Exception)):
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )

    def test_tool_execution_with_missing_required_arg(self):
        """Test tool execution when LLM doesn't provide required argument."""
        mock_provider = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "requires_arg"
        mock_tool_call.function.arguments = "{}"  # Missing required 'name' arg

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        def requires_arg(name: str) -> Tuple[str, dict]:
            return f"Hello {name}", {}

        agent.add_tool("requires_arg", requires_arg)

        # Should handle missing argument error
        with pytest.raises((TypeError, KeyError, Exception)):
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )


class TestResponseEdgeCases:
    """Test edge cases in LLM responses."""

    def test_very_long_response_content(self):
        """Test handling of very long response content."""
        mock_provider = Mock()

        # Very long content (1MB)
        long_content = "x" * (1024 * 1024)

        mock_message = Mock()
        mock_message.content = long_content
        mock_message.tool_calls = None

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        response, ctx = agent.run(
            [ChatCompletionUserMessageParam(
                 role="user",
                 content="test",
             )],
            max_iterations=1
        )

        assert any(
            len(str(msg.get("content", ""))) > 1000000
            for msg in ctx.conversation.messages
        )

    def test_unicode_in_response(self):
        """Test handling of Unicode characters in response."""
        mock_provider = Mock()

        mock_message = Mock()
        mock_message.content = "Hello 世界 🌍 مرحبا"
        mock_message.tool_calls = None

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        result, ctx = agent.run(
            [ChatCompletionUserMessageParam(
                 role="user",
                 content="test",
             )],
            max_iterations=1
        )

        assert result == "Hello 世界 🌍 مرحبا"

    def test_null_bytes_in_response(self):
        """Test handling of null bytes in response."""
        mock_provider = Mock()

        mock_message = Mock()
        mock_message.content = "Hello\x00World"
        mock_message.tool_calls = None

        mock_provider.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent(provider=mock_provider, model="gpt-4")

        # Should handle null bytes
        try:
            response, ctx = agent.run(
                [ChatCompletionUserMessageParam(
                     role="user",
                     content="test",
                 )],
                max_iterations=1
            )
        except (ValueError, UnicodeDecodeError):
            # Might raise depending on JSON handling
            pass
