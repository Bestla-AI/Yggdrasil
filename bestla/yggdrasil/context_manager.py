"""Context manager for automated conversation context compaction."""

from typing import Callable, List

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from bestla.yggdrasil.context_utils import (
    build_compacted_messages,
    clear_tool_results,
    estimate_tokens,
    summarize_messages,
)


class ContextManager:
    """Manages conversation context and applies compaction strategies.

    The ContextManager monitors token usage and applies compaction strategies
    when the context window approaches its limit. It supports multiple
    compaction strategies and is designed to be used externally with Agent.

    Attributes:
        strategy: Compaction strategy to use ("tool_result_clearing" or "summarization")
        threshold: Token count at which to trigger compaction (None = disabled)
        preserve_recent: Number of recent messages to always preserve
        token_counter: Custom token counting function (defaults to built-in estimator)
    """

    VALID_STRATEGIES = ["tool_result_clearing", "summarization"]

    def __init__(
        self,
        strategy: str = "tool_result_clearing",
        threshold: int | None = None,
        preserve_recent: int = 10,
        token_counter: Callable[[List[ChatCompletionMessageParam]], int] | None = None,
    ):
        """Initialize context manager.

        Args:
            strategy: Compaction strategy ("tool_result_clearing" or "summarization")
            threshold: Token count to trigger compaction (None = disabled)
            preserve_recent: Number of recent messages to preserve
            token_counter: Custom function to count tokens. Defaults to built-in estimator.
                          Must accept List[ChatCompletionMessageParam] and return int.

        Raises:
            ValueError: If threshold is not positive or preserve_recent < 1

        Example:
            # Use default estimator
            cm = ContextManager(threshold=100000)

            # Use custom counter
            def my_counter(messages):
                # Your custom token counting logic
                total = 0
                for msg in messages:
                    content = msg.get("content", "")
                    total += len(str(content)) // 4  # Rough estimate
                return total

            cm = ContextManager(threshold=100000, token_counter=my_counter)
        """
        if threshold is not None and threshold <= 0:
            raise ValueError("threshold must be positive or None")

        if preserve_recent < 1:
            raise ValueError("preserve_recent must be at least 1")

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid compaction strategy: '{strategy}'. "
                f"Must be one of {self.VALID_STRATEGIES}"
            )

        self.strategy = strategy
        self.threshold = threshold
        self.preserve_recent = preserve_recent
        self.token_counter = token_counter or estimate_tokens

        # Provider and model for summarization strategy
        self._provider: OpenAI | None = None
        self._model: str | None = None

    def set_summarization_config(self, provider: OpenAI, model: str) -> None:
        """Configure provider and model for summarization strategy.

        Required if using "summarization" strategy.

        Args:
            provider: OpenAI provider instance
            model: Model name to use for summarization
        """
        self._provider = provider
        self._model = model

    def should_compact(self, messages: List[ChatCompletionMessageParam]) -> bool:
        """Check if compaction should be triggered.

        Args:
            messages: Current message list

        Returns:
            True if token count >= threshold, False otherwise
        """
        if self.threshold is None:
            return False

        current_tokens = self.estimate_tokens(messages)
        return current_tokens >= self.threshold

    def estimate_tokens(self, messages: List[ChatCompletionMessageParam]) -> int:
        """Estimate token count for messages using configured counter.

        Args:
            messages: Message list to estimate

        Returns:
            Estimated token count (guaranteed to be non-negative integer)

        Raises:
            ValueError: If token counter returns invalid value (non-int or negative)
            TypeError: If token counter is not callable

        Example:
            >>> cm = ContextManager()
            >>> messages = [
            ...     ChatCompletionUserMessageParam(
            ...         role="user",
            ...         content="Hello",
            ...     )
            ... ]
            >>> tokens = cm.estimate_tokens(messages)
            >>> assert isinstance(tokens, int) and tokens >= 0
        """
        result = self.token_counter(messages)

        # Validate result type
        if not isinstance(result, int):
            counter_name = getattr(self.token_counter, '__name__', 'custom')
            raise ValueError(
                f"Token counter must return int, got {type(result).__name__}. "
                f"Counter: {counter_name}, returned: {result!r}"
            )

        # Validate result value
        if result < 0:
            counter_name = getattr(self.token_counter, '__name__', 'custom')
            raise ValueError(
                f"Token counter returned negative value: {result}. "
                f"Counter: {counter_name}"
            )

        return result

    def compact(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        """Apply compaction strategy to messages.

        Args:
            messages: Message list to compact

        Returns:
            Compacted message list

        Raises:
            ValueError: If strategy is unknown or required config is missing
        """
        if not messages:
            return []

        if self.strategy == "tool_result_clearing":
            return self._compact_tool_result_clearing(messages)
        elif self.strategy == "summarization":
            return self._compact_summarization(messages)
        else:
            raise ValueError(f"Unknown compaction strategy: {self.strategy}")

    def _compact_tool_result_clearing(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        """Apply tool result clearing strategy.

        Removes old tool result messages while preserving recent context.

        Args:
            messages: Messages to compact

        Returns:
            Compacted messages
        """
        return clear_tool_results(messages, keep_recent=self.preserve_recent)

    def _compact_summarization(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        """Apply summarization strategy.

        Summarizes old messages using LLM and replaces them with summary.

        Args:
            messages: Messages to compact

        Returns:
            Compacted messages with summary

        Raises:
            ValueError: If provider/model not configured
        """
        if not self._provider or not self._model:
            raise ValueError(
                "Summarization strategy requires provider and model. "
                "Call set_summarization_config() first."
            )

        summary = summarize_messages(
            messages,
            self._provider,
            self._model,
            keep_recent=self.preserve_recent,
        )

        return build_compacted_messages(
            messages,
            summary,
            keep_recent=self.preserve_recent,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        threshold_str = f"{self.threshold}" if self.threshold else "disabled"
        counter_name = getattr(self.token_counter, "__name__", "custom")
        return (
            f"ContextManager(strategy='{self.strategy}', "
            f"threshold={threshold_str}, "
            f"preserve_recent={self.preserve_recent}, "
            f"token_counter={counter_name})"
        )
