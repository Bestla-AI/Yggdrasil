"""Tests for decorator edge cases, including untested decorators."""

import asyncio
import platform
import time
from typing import Tuple

import pytest

from bestla.yggdrasil import tool
from bestla.yggdrasil.decorators import cache_result, rate_limit, retry, retry_async, timeout
from bestla.yggdrasil.tool import Tool


class TestRetryAsyncDecorator:
    """Test the @retry_async decorator (currently untested)."""

    @pytest.mark.asyncio
    async def test_retry_async_basic_success(self):
        """Test async function succeeds on first attempt."""
        call_count = 0

        @retry_async(max_attempts=3, backoff=0.01)
        async def successful_async() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            return "success", {}

        result, updates = await successful_async()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_async_fails_then_succeeds(self):
        """Test async function fails twice, succeeds on third attempt."""
        call_count = 0

        @retry_async(max_attempts=3, backoff=0.01)
        async def unstable_async() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success", {}

        result, updates = await unstable_async()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_exhaustion(self):
        """Test async function always fails - raises after max_attempts."""
        call_count = 0

        @retry_async(max_attempts=3, backoff=0.01)
        async def always_fails() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError, match="Always fails"):
            await always_fails()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_exponential_backoff(self):
        """Test exponential backoff timing."""
        call_times = []

        @retry_async(max_attempts=4, backoff=0.1)
        async def timed_failures() -> Tuple[str, dict]:
            call_times.append(time.time())
            if len(call_times) < 4:
                raise ValueError("Not yet")
            return "done", {}

        await timed_failures()

        # Check backoff intervals: 0.1, 0.2, 0.4 seconds
        assert len(call_times) == 4

        # First retry should wait ~0.1s
        assert 0.08 < (call_times[1] - call_times[0]) < 0.15

        # Second retry should wait ~0.2s
        assert 0.18 < (call_times[2] - call_times[1]) < 0.25

        # Third retry should wait ~0.4s
        assert 0.38 < (call_times[3] - call_times[2]) < 0.50

    @pytest.mark.asyncio
    async def test_retry_async_specific_exceptions(self):
        """Test retry only on specific exception types."""
        call_count = 0

        @retry_async(max_attempts=3, backoff=0.01, exceptions=(ValueError, TypeError))
        async def selective_retry() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retryable")
            elif call_count == 2:
                raise RuntimeError("Not retryable")
            return "done", {}

        # RuntimeError should not be retried
        with pytest.raises(RuntimeError, match="Not retryable"):
            await selective_retry()

        # Should have called twice (initial + 1 retry on ValueError)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_async_with_tool_decorator(self):
        """Test @retry_async combined with @tool decorator."""

        call_count = 0

        # Apply decorators: retry_async wraps function, then tool wraps result
        # Tool.execute() is not async, so we can't await it
        # Instead, test that decorators can be combined without error
        @tool()
        @retry_async(max_attempts=3, backoff=0.01)
        async def retryable_tool() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network issue")
            return "recovered", {"attempts": call_count}

        # Tool object created successfully
        assert isinstance(retryable_tool, Tool)

        # Execute via tool's function attribute directly for async execution
        result, updates = await retryable_tool.function()

        assert result == "recovered"
        assert updates["attempts"] == 2


class TestTimeoutDecorator:
    """Test the @timeout decorator (currently untested)."""

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="signal.SIGALRM not available on Windows"
    )
    def test_timeout_basic_success(self):
        """Test function completes within timeout."""

        @timeout(1.0)
        def fast_function() -> Tuple[str, dict]:
            time.sleep(0.1)
            return "completed", {}

        result, updates = fast_function()
        assert result == "completed"

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="signal.SIGALRM not available on Windows"
    )
    def test_timeout_exceeded(self):
        """Test function exceeds timeout."""

        @timeout(0.1)
        def slow_function() -> Tuple[str, dict]:
            time.sleep(1.0)
            return "should not reach", {}

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            slow_function()

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="signal.SIGALRM not available on Windows"
    )
    def test_timeout_exact_boundary(self):
        """Test function at exact timeout boundary."""

        @timeout(0.5)
        def boundary_function() -> Tuple[str, dict]:
            time.sleep(0.48)  # Just under timeout
            return "made it", {}

        result, updates = boundary_function()
        assert result == "made it"

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="signal.SIGALRM not available on Windows"
    )
    def test_timeout_with_tool_decorator(self):
        """Test @timeout combined with @tool decorator."""

        @tool()
        @timeout(0.5)
        def timed_tool() -> Tuple[str, dict]:
            time.sleep(0.1)
            return "done", {}

        result, updates = timed_tool.execute({})
        assert result == "done"

    @pytest.mark.skipif(platform.system() != "Windows", reason="Test Windows-specific behavior")
    def test_timeout_on_windows_unavailable(self):
        """Test that timeout decorator fails gracefully on Windows."""
        # On Windows, signal.SIGALRM is not available
        # The decorator might raise an AttributeError or similar

        try:

            @timeout(1.0)
            def windows_function() -> Tuple[str, dict]:
                return "result", {}

            # If it doesn't raise during decoration, try calling it
            windows_function()

        except (AttributeError, NotImplementedError, OSError):
            # Expected on Windows
            pass


class TestDecoratorCombinations:
    """Test combining multiple decorators."""

    def test_all_decorators_combined(self):
        """Test stacking all decorators together."""
        call_count = 0

        @retry(max_attempts=3, backoff=0.01)
        @cache_result(ttl=1.0)
        @rate_limit(calls=10, period=1.0)
        def combined_tool() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            return f"call_{call_count}", {}

        # First call
        result1, _ = combined_tool()

        # Second call should return cached result
        result2, _ = combined_tool()

        # Should be same due to cache
        assert result1 == result2
        assert call_count == 1  # Only called once due to cache

    def test_decorator_execution_order(self):
        """Test decorator execution order matters."""
        execution_log = []

        # Custom decorator to log execution
        def logger(name):
            def decorator(func):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    execution_log.append(f"enter_{name}")
                    result = func(*args, **kwargs)
                    execution_log.append(f"exit_{name}")
                    return result

                return wrapper

            return decorator

        import functools

        @logger("outer")
        @logger("middle")
        @logger("inner")
        def logged_function() -> Tuple[str, dict]:
            execution_log.append("function")
            return "done", {}

        logged_function()

        # Verify execution order (decorators execute bottom-up)
        assert execution_log == [
            "enter_outer",
            "enter_middle",
            "enter_inner",
            "function",
            "exit_inner",
            "exit_middle",
            "exit_outer",
        ]

    def test_retry_with_rate_limit(self):
        """Test retry decorator with rate limiting."""
        call_count = 0

        @retry(max_attempts=3, backoff=0.01)
        @rate_limit(calls=5, period=1.0)
        def limited_retry() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry me")
            return "success", {}

        result, _ = limited_retry()
        assert result == "success"
        assert call_count == 2

    def test_cache_with_rate_limit(self):
        """Test cache bypasses rate limit."""
        call_count = 0

        @cache_result(ttl=10.0)
        @rate_limit(calls=2, period=1.0)
        def cached_limited() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}", {}

        # First call - executes function
        result1, _ = cached_limited()

        # These calls should return cached result, bypassing rate limit
        for i in range(10):
            result_i, _ = cached_limited()
            assert result_i == result1

        # Function only called once due to cache
        assert call_count == 1


class TestDecoratorEdgeCases:
    """Test edge cases for all decorators."""

    def test_retry_with_zero_backoff(self):
        """Test retry with zero backoff time."""
        call_count = 0

        @retry(max_attempts=3, backoff=0.0)
        def instant_retry() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Try again")
            return "done", {}

        result, _ = instant_retry()
        assert call_count == 3

    def test_retry_max_attempts_one(self):
        """Test retry with max_attempts=1 (no retries)."""
        call_count = 0

        @retry(max_attempts=1, backoff=0.01)
        def no_retry() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            raise ValueError("Fail")

        with pytest.raises(ValueError):
            no_retry()

        assert call_count == 1

    def test_cache_with_no_ttl(self):
        """Test cache with TTL=None (cache forever)."""
        call_count = 0

        @cache_result(ttl=None)
        def forever_cached() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}", {}

        result1, _ = forever_cached()
        time.sleep(0.1)
        result2, _ = forever_cached()

        # Should be same even after time passes
        assert result1 == result2
        assert call_count == 1

    def test_cache_with_different_arguments(self):
        """Test cache differentiates between different arguments."""
        call_count = 0

        @cache_result(ttl=10.0)
        def cached_with_args(x: int, y: int) -> Tuple[int, dict]:
            nonlocal call_count
            call_count += 1
            return x + y, {}

        result1, _ = cached_with_args(1, 2)
        result2, _ = cached_with_args(1, 2)  # Cached
        result3, _ = cached_with_args(2, 3)  # Different args, not cached

        assert result1 == result2 == 3
        assert result3 == 5
        assert call_count == 2  # Called twice: once for (1,2), once for (2,3)

    def test_cache_clear(self):
        """Test cache clearing functionality."""
        call_count = 0

        @cache_result(ttl=10.0)
        def clearable_cache() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}", {}

        result1, _ = clearable_cache()

        # Clear cache
        clearable_cache.clear_cache()

        # Next call should execute function again
        result2, _ = clearable_cache()

        assert result1 != result2
        assert call_count == 2

    def test_cache_size(self):
        """Test cache size tracking."""

        @cache_result(ttl=10.0)
        def sized_cache(x: int) -> Tuple[int, dict]:
            return x * 2, {}

        # Cache different values
        for i in range(5):
            sized_cache(i)

        assert sized_cache.cache_size() == 5

    def test_rate_limit_exact_boundary(self):
        """Test rate limit at exact boundary."""

        @rate_limit(calls=3, period=1.0)
        def limited() -> Tuple[str, dict]:
            return "ok", {}

        # Should allow exactly 3 calls
        limited()
        limited()
        limited()

        # 4th call should fail
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            limited()

    def test_rate_limit_resets_after_period(self):
        """Test rate limit resets after period expires."""

        @rate_limit(calls=2, period=0.2)
        def time_limited() -> Tuple[str, dict]:
            return "ok", {}

        # Use 2 calls
        time_limited()
        time_limited()

        # 3rd call should fail
        with pytest.raises(RuntimeError):
            time_limited()

        # Wait for period to expire
        time.sleep(0.25)

        # Should work now
        result, _ = time_limited()
        assert result == "ok"

    def test_rate_limit_independent_per_function(self):
        """Test rate limits are independent per function."""

        @rate_limit(calls=2, period=1.0)
        def func1() -> Tuple[str, dict]:
            return "func1", {}

        @rate_limit(calls=2, period=1.0)
        def func2() -> Tuple[str, dict]:
            return "func2", {}

        # Each function should have independent limits
        func1()
        func1()

        func2()
        func2()

        # Both should be at limit
        with pytest.raises(RuntimeError):
            func1()

        with pytest.raises(RuntimeError):
            func2()

    def test_decorator_preserves_function_name(self):
        """Test decorators preserve original function name."""

        @retry(max_attempts=2)
        @cache_result(ttl=1.0)
        @rate_limit(calls=5, period=1.0)
        def named_function() -> Tuple[str, dict]:
            return "result", {}

        assert named_function.__name__ == "named_function"

    def test_decorator_with_exceptions_during_decoration(self):
        """Test decorator behavior with invalid parameters."""

        # Decorators don't validate parameters at decoration time
        # Invalid parameters would cause issues at execution time, not decoration time

        # Invalid max_attempts (decorator doesn't validate at decoration time)
        @retry(max_attempts=-1)
        def invalid_retry():
            return "result", {}

        # Decorator applied successfully (no validation at decoration time)
        assert callable(invalid_retry)

        # Invalid backoff (decorator doesn't validate at decoration time)
        @retry(max_attempts=3, backoff=-1.0)
        def negative_backoff():
            return "result", {}

        # Decorator applied successfully
        assert callable(negative_backoff)

    @pytest.mark.asyncio
    async def test_retry_async_preserves_async_nature(self):
        """Test @retry_async preserves async function nature."""

        @retry_async(max_attempts=2, backoff=0.01)
        async def async_func() -> Tuple[str, dict]:
            await asyncio.sleep(0.01)
            return "async_result", {}

        # Should be awaitable
        result, _ = await async_func()
        assert result == "async_result"

        # Should return coroutine
        coro = async_func()
        assert asyncio.iscoroutine(coro)
        await coro  # Clean up


class TestDecoratorWithToolExecution:
    """Test decorators specifically with @tool decorator."""

    def test_tool_with_retry(self):
        """Test @tool + @retry combination."""

        call_count = 0

        @tool()
        @retry(max_attempts=3, backoff=0.01)
        def retryable_tool() -> Tuple[str, dict]:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry")
            return "success", {"attempts": call_count}

        result, updates = retryable_tool.execute()

        assert result == "success"
        assert updates["attempts"] == 2

    def test_tool_with_cache(self):
        """Test @tool + @cache_result combination."""

        call_count = 0

        @tool()
        @cache_result(ttl=1.0)
        def cached_tool(value: int) -> Tuple[int, dict]:
            nonlocal call_count
            call_count += 1
            return value * 2, {}

        result1, _ = cached_tool.execute(value=5)
        result2, _ = cached_tool.execute(value=5)

        assert result1 == result2 == 10
        assert call_count == 1  # Only called once due to cache

    def test_tool_with_rate_limit(self):
        """Test @tool + @rate_limit combination."""

        @tool()
        @rate_limit(calls=2, period=1.0)
        def limited_tool() -> Tuple[str, dict]:
            return "ok", {}

        # Should allow 2 calls
        limited_tool.execute()
        limited_tool.execute()

        # 3rd should fail
        with pytest.raises(RuntimeError):
            limited_tool.execute()
