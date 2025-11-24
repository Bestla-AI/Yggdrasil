"""Utility functions for context management and compaction."""

import re
from typing import List
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam


# Token counter type for better documentation
TokenCounter = type[List[ChatCompletionMessageParam], int]


def estimate_tokens(messages: List[ChatCompletionMessageParam]) -> int:
    """Estimate token count for a list of messages.

    Uses a simple heuristic: ~4 characters per token, plus overhead for message structure.
    For production use, consider using tiktoken library for accurate counting.

    Args:
        messages: List of chat completion messages

    Returns:
        Estimated token count
    """
    if not messages:
        return 0

    total_chars = 0
    message_overhead = 4  # Tokens per message for role, structure, etc.

    for message in messages:
        total_chars += message_overhead * 4

        content = message.get("content")
        if content:
            total_chars += len(str(content))

        role = message.get("role", "")
        total_chars += len(role)

        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            total_chars += len(str(tool_call_id))

    estimated_tokens = total_chars // 4
    return estimated_tokens


def clear_tool_results(
    messages: List[ChatCompletionMessageParam],
    keep_recent: int = 10,
) -> List[ChatCompletionMessageParam]:
    """Clear old tool result messages while preserving recent context.

    This is the lightest-touch compaction strategy. It removes tool messages
    that are older than keep_recent messages from the end of the list.

    Args:
        messages: List of messages to compact
        keep_recent: Number of recent messages to preserve (from the end)

    Returns:
        Compacted message list with old tool results removed
    """
    if not messages:
        return []

    system_message = None
    if messages and messages[0].get("role") == "system":
        system_message = messages[0]
        messages = messages[1:]

    if len(messages) <= keep_recent:
        if system_message:
            return [system_message] + messages
        return messages

    old_messages = messages[:-keep_recent] if keep_recent > 0 else messages
    recent_messages = messages[-keep_recent:] if keep_recent > 0 else []

    filtered_old = [m for m in old_messages if m.get("role") != "tool"]

    result = []
    if system_message:
        result.append(system_message)
    result.extend(filtered_old)
    result.extend(recent_messages)

    return result


def summarize_messages(
    messages: List[ChatCompletionMessageParam],
    provider: OpenAI,
    model: str,
    keep_recent: int = 10,
) -> ChatCompletionUserMessageParam | None:
    """Summarize old messages using LLM.

    Args:
        messages: List of messages to summarize
        provider: OpenAI provider instance
        model: Model to use for summarization
        keep_recent: Number of recent messages to exclude from summary

    Returns:
        User message containing summary, or None if nothing to summarize
    """
    # Remove system message for summarization
    messages_to_summarize = [m for m in messages if m.get("role") != "system"]

    # Exclude recent messages
    if len(messages_to_summarize) <= keep_recent:
        return None

    old_messages = messages_to_summarize[:-keep_recent] if keep_recent > 0 else messages_to_summarize

    if not old_messages:
        return None

    conversation_text = _format_messages_for_summary(old_messages)

    summarization_prompt = f"""You are summarizing an older portion of a conversation that exceeded the context window.
This summary will REPLACE the original messages to save tokens while preserving critical information.

IMPORTANT: This is a COMPACTED CONTEXT SUMMARY. The model will see this summary instead of the full conversation history.

Your summary must include:
1. Key decisions and conclusions reached
2. Important facts, data, and context established
3. Any unresolved issues or pending questions
4. Critical implementation details or requirements
5. State of any ongoing work or tasks
6. If there are previous summaries in the conversation, integrate their information

CONVERSATION TO SUMMARIZE (this will be removed and replaced with your summary):
{conversation_text}

Provide a comprehensive summary in 2-4 well-structured paragraphs. Be thorough - missing information cannot be recovered."""

    response = provider.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": summarization_prompt}
        ],
    )

    summary = response.choices[0].message.content

    return ChatCompletionUserMessageParam(
        role="user",
        content=f"""[COMPACTED CONTEXT - Previous conversation has been summarized to save tokens]

{summary}

[END OF COMPACTED CONTEXT - Continue from here with recent messages below]"""
    )


def _format_messages_for_summary(messages: List[ChatCompletionMessageParam]) -> str:
    """Format messages into readable text for summarization.

    Handles all OpenAI message roles: system, user, assistant, tool, function, developer.

    Args:
        messages: Messages to format

    Returns:
        Formatted string representation

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ...     {"role": "tool", "tool_call_id": "call_123", "content": "Result"}
        ... ]
        >>> formatted = _format_messages_for_summary(messages)
        >>> assert "User: Hello" in formatted
        >>> assert "[Tool call_123]: Result" in formatted
    """
    formatted = []

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "tool":
            tool_id = msg.get("tool_call_id", "unknown")
            formatted.append(f"[Tool {tool_id}]: {content}")
        elif role == "function":
            func_name = msg.get("name", "unknown")
            formatted.append(f"[Function {func_name}]: {content}")
        else:
            formatted.append(f"{role.capitalize()}: {content}")

    return "\n\n".join(formatted)


def build_compacted_messages(
    original_messages: List[ChatCompletionMessageParam],
    summary_message: ChatCompletionUserMessageParam | None,
    keep_recent: int = 10,
) -> List[ChatCompletionMessageParam]:
    """Build compacted message list with summary and recent messages.

    Preserves:
    - System message (if present, always first)
    - Developer message (if present, right after system)
    - First user message (task directive)
    - Summary of old messages
    - Recent messages

    Args:
        original_messages: Original message list
        summary_message: Summary message to insert, or None
        keep_recent: Number of recent messages to preserve

    Returns:
        Compacted message list: [system, developer?, first_user?, summary, recent messages]

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Build a REST API"},  # Task directive
        ...     # ... many messages ...
        ... ]
        >>> compacted = build_compacted_messages(messages, summary, keep_recent=5)
        >>> assert compacted[0]["role"] == "system"
        >>> assert compacted[1]["content"] == "Build a REST API"  # Preserved!
        >>> assert "[Context Summary:" in compacted[2]["content"]
    """
    result = []
    preserved_messages = []

    if original_messages and original_messages[0].get("role") == "system":
        result.append(original_messages[0])
        original_messages = original_messages[1:]

    if original_messages and original_messages[0].get("role") == "developer":
        result.append(original_messages[0])
        original_messages = original_messages[1:]

    if original_messages and original_messages[0].get("role") == "user":
        result.append(original_messages[0])
        preserved_messages.append(original_messages[0])
        original_messages = original_messages[1:]

    if summary_message:
        result.append(summary_message)

    if keep_recent > 0 and original_messages:
        recent = original_messages[-keep_recent:]
        result.extend(recent)
    elif not keep_recent and original_messages:
        result.extend(original_messages)

    return result
