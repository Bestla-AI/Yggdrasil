"""Tests for decorators."""

import time
from typing import Tuple

import pytest

from bestla.yggdrasil.decorators import cache_result, rate_limit, retry


class TestRetryDecorator:
    """Test @retry decorator."""

    def test_retry_succeeds_on_first_try(self):
        """Test function that succeeds immediately."""
        call_count = [0]

        @retry(max_attempts=3, backoff=0.01)
        def successful_func() -> Tuple[str, dict]:
            call_count[0] += 1
            return "success", {}

        result, updates = successful_func()

        assert result == "success"
        assert updates == {}
        assert call_count[0] == 1  # Only called once

    def test_retry_succeeds_on_second_try(self):
        """Test function that fails once then succeeds."""
        call_count = [0]

        @retry(max_attempts=3, backoff=0.01)
        def flaky_func() -> Tuple[str, dict]:
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First attempt fails")
            return "success", {}

        result, updates = flaky_func()

        assert result == "success"
        assert call_count[0] == 2  # Called twice

    def test_retry_exhausts_attempts(self):
        """Test function that always fails."""
        call_count = [0]

        @retry(max_attempts=3, backoff=0.01)
        def always_fails() -> Tuple[str, dict]:
            call_count[0] += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        assert call_count[0] == 3  # All attempts exhausted

    def test_retry_with_specific_exceptions(self):
        """Test retry only catches specific exceptions."""
        call_count = [0]

        @retry(max_attempts=3, backoff=0.01, exceptions=(ConnectionError, TimeoutError))
        def specific_exception_func() -> Tuple[str, dict]:
            call_count[0] += 1
            raise ValueError("Not a connection error")

        # Should not retry ValueError (not in exceptions tuple)
        with pytest.raises(ValueError, match="Not a connection error"):
            specific_exception_func()

        assert call_count[0] == 1  # Only called once, no retries

    def test_retry_connection_errors(self):
        """Test retry with connection errors."""
        call_count = [0]

        @retry(max_attempts=3, backoff=0.01, exceptions=ConnectionError)
        def connection_func() -> Tuple[str, dict]:
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Connection failed")
            return "connected", {}

        result, _ = connection_func()

        assert result == "connected"
        assert call_count[0] == 3  # Retried twice, succeeded on third

    def test_retry_exponential_backoff(self):
        """Test exponential backoff timing."""
        call_times = []

        @retry(max_attempts=3, backoff=0.1)
        def timed_func() -> Tuple[str, dict]:
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Not yet")
            return "done", {}

        timed_func()

        # Check backoff times (0.1s, then 0.2s)
        if len(call_times) >= 2:
            first_gap = call_times[1] - call_times[0]
            assert 0.08 < first_gap < 0.15  # ~0.1s

        if len(call_times) >= 3:
            second_gap = call_times[2] - call_times[1]
            assert 0.18 < second_gap < 0.25  # ~0.2s (doubled)

    def test_retry_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @retry(max_attempts=3)
        def documented_func() -> Tuple[str, dict]:
            """This is a docstring."""
            return "result", {}

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."


class TestCacheResultDecorator:
    """Test @cache_result decorator."""

    def test_cache_caches_result(self):
        """Test that results are cached."""
        call_count = [0]

        @cache_result()
        def expensive_func(x: int) -> Tuple[int, dict]:
            call_count[0] += 1
            return x * 2, {}

        # First call
        result1, _ = expensive_func(5)
        assert result1 == 10
        assert call_count[0] == 1

        # Second call with same args - should use cache
        result2, _ = expensive_func(5)
        assert result2 == 10
        assert call_count[0] == 1  # Not called again

        # Different args - should call function
        result3, _ = expensive_func(10)
        assert result3 == 20
        assert call_count[0] == 2

    def test_cache_with_ttl(self):
        """Test cache with time-to-live."""
        call_count = [0]

        @cache_result(ttl=0.1)  # 100ms TTL
        def ttl_func(x: int) -> Tuple[int, dict]:
            call_count[0] += 1
            return x * 2, {}

        # First call
        result1, _ = ttl_func(5)
        assert result1 == 10
        assert call_count[0] == 1

        # Immediate second call - cached
        result2, _ = ttl_func(5)
        assert call_count[0] == 1

        # Wait for cache to expire
        time.sleep(0.15)

        # Should call function again
        result3, _ = ttl_func(5)
        assert call_count[0] == 2

    def test_cache_with_kwargs(self):
        """Test cache with keyword arguments."""
        call_count = [0]

        @cache_result()
        def kwarg_func(x: int, y: int = 10) -> Tuple[int, dict]:
            call_count[0] += 1
            return x + y, {}

        # Different calls
        result1, _ = kwarg_func(5)
        result2, _ = kwarg_func(5, y=10)  # Different cache key (explicit kwarg)
        result3, _ = kwarg_func(5, y=20)  # Different

        assert result1 == 15
        assert result2 == 15
        assert result3 == 25
        # Note: (5) and (5, y=10) have different cache keys
        assert call_count[0] == 3  # All three are cached separately

        # But calling the exact same way uses cache
        result4, _ = kwarg_func(5)
        assert call_count[0] == 3  # Used cache from first call

    def test_cache_clear(self):
        """Test clearing cache."""
        call_count = [0]

        @cache_result()
        def cached_func(x: int) -> Tuple[int, dict]:
            call_count[0] += 1
            return x * 2, {}

        # First call
        cached_func(5)
        assert call_count[0] == 1

        # Cached call
        cached_func(5)
        assert call_count[0] == 1

        # Clear cache
        cached_func.clear_cache()

        # Should call function again
        cached_func(5)
        assert call_count[0] == 2

    def test_cache_size(self):
        """Test getting cache size."""
        @cache_result()
        def sized_func(x: int) -> Tuple[int, dict]:
            return x * 2, {}

        assert sized_func.cache_size() == 0

        sized_func(1)
        assert sized_func.cache_size() == 1

        sized_func(2)
        assert sized_func.cache_size() == 2

        sized_func(1)  # Cached, doesn't increase size
        assert sized_func.cache_size() == 2


class TestRateLimitDecorator:
    """Test @rate_limit decorator."""

    def test_rate_limit_allows_calls_within_limit(self):
        """Test that calls within limit are allowed."""
        @rate_limit(calls=3, period=1.0)
        def limited_func() -> Tuple[str, dict]:
            return "success", {}

        # Should allow 3 calls
        limited_func()
        limited_func()
        limited_func()

        # Fourth call should fail
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            limited_func()

    def test_rate_limit_window_sliding(self):
        """Test that rate limit window slides."""
        @rate_limit(calls=2, period=0.2)
        def windowed_func() -> Tuple[str, dict]:
            return "success", {}

        # First two calls succeed
        windowed_func()
        windowed_func()

        # Third call fails
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            windowed_func()

        # Wait for window to slide
        time.sleep(0.25)

        # Should work again
        result, _ = windowed_func()
        assert result == "success"

    def test_rate_limit_independent_per_function(self):
        """Test that rate limits are independent per function."""
        @rate_limit(calls=2, period=1.0)
        def func1() -> Tuple[str, dict]:
            return "func1", {}

        @rate_limit(calls=2, period=1.0)
        def func2() -> Tuple[str, dict]:
            return "func2", {}

        # Each function has its own limit
        func1()
        func1()

        func2()
        func2()

        # Both should fail on third call
        with pytest.raises(RuntimeError):
            func1()

        with pytest.raises(RuntimeError):
            func2()

    def test_rate_limit_fast_calls(self):
        """Test rate limit with very fast calls."""
        call_count = [0]

        @rate_limit(calls=5, period=0.1)
        def fast_func() -> Tuple[int, dict]:
            call_count[0] += 1
            return call_count[0], {}

        # Make 5 fast calls
        for i in range(5):
            result, _ = fast_func()
            assert result == i + 1

        # Sixth call should fail
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            fast_func()

        assert call_count[0] == 5


class TestDecoratorsCombined:
    """Test combining multiple decorators."""

    def test_retry_and_cache(self):
        """Test combining @retry and @cache_result."""
        call_count = [0]

        @retry(max_attempts=3, backoff=0.01)
        @cache_result()
        def combined_func(x: int) -> Tuple[int, dict]:
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First fails")
            return x * 2, {}

        # First call with retry
        result1, _ = combined_func(5)
        assert result1 == 10
        assert call_count[0] == 2  # Failed once, succeeded on retry

        # Second call - should use cache, no additional calls
        result2, _ = combined_func(5)
        assert result2 == 10
        assert call_count[0] == 2  # Still 2, used cache

    def test_cache_and_rate_limit(self):
        """Test combining @cache_result and @rate_limit."""
        call_count = [0]

        # Note: decorator order matters!
        # cache_result on top means cache is checked before rate limit
        @cache_result()
        @rate_limit(calls=2, period=1.0)
        def combined_func(x: int) -> Tuple[int, dict]:
            call_count[0] += 1
            return x * 2, {}

        # Two different calls - uses rate limit
        combined_func(1)
        combined_func(2)

        # Third new call fails rate limit
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            combined_func(3)

        # But cached call works (cache is checked first, before rate limit)
        result, _ = combined_func(1)
        assert result == 2  # From cache
        assert call_count[0] == 2  # Didn't call function again
