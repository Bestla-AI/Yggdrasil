"""Decorators for tool functions."""

import functools
import time
from typing import Callable, Tuple, Type, Union


def retry(
    max_attempts: int = 3,
    backoff: float = 1.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
):
    """Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        backoff: Initial backoff time in seconds (default: 1.0)
        exceptions: Exception type(s) to catch and retry (default: Exception)

    Returns:
        Decorated function

    Example:
        @retry(max_attempts=3, backoff=1.0, exceptions=(ConnectionError, TimeoutError))
        def fetch_data() -> Tuple[str, dict]:
            # Try to fetch data, will retry up to 3 times on connection errors
            return data, {}

        @retry(max_attempts=5, backoff=0.5)
        def unstable_operation() -> Tuple[str, dict]:
            # Retries on any exception, with 0.5s initial backoff
            return result, {}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_backoff = backoff

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # If this was the last attempt, re-raise
                    if attempt == max_attempts:
                        raise

                    # Wait before retrying (exponential backoff)
                    time.sleep(current_backoff)
                    current_backoff *= 2  # Double the backoff each time

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def retry_async(
    max_attempts: int = 3,
    backoff: float = 1.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
):
    """Async retry decorator with exponential backoff.

    Same as @retry but for async functions.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        backoff: Initial backoff time in seconds (default: 1.0)
        exceptions: Exception type(s) to catch and retry (default: Exception)

    Returns:
        Decorated async function

    Example:
        @retry_async(max_attempts=3, backoff=1.0)
        async def fetch_data_async() -> Tuple[str, dict]:
            return data, {}
    """
    import asyncio

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_backoff = backoff

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # If this was the last attempt, re-raise
                    if attempt == max_attempts:
                        raise

                    # Wait before retrying (exponential backoff)
                    await asyncio.sleep(current_backoff)
                    current_backoff *= 2

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def timeout(seconds: float):
    """Timeout decorator for tool functions.

    Args:
        seconds: Maximum execution time in seconds

    Returns:
        Decorated function

    Raises:
        TimeoutError: If function execution exceeds timeout

    Example:
        @timeout(5.0)
        def slow_operation() -> Tuple[str, dict]:
            # Must complete within 5 seconds
            return result, {}
    """
    import signal

    def decorator(func: Callable) -> Callable:
        def handler(signum, frame):
            raise TimeoutError(f"Function {func.__name__} exceeded timeout of {seconds}s")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set the signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the alarm and restore old handler
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper

    return decorator


def cache_result(ttl: float | None = None):
    """Cache decorator for tool results.

    Caches the result of a tool function. Useful for expensive operations
    that don't change frequently.

    Args:
        ttl: Time-to-live in seconds (None = cache forever)

    Returns:
        Decorated function

    Example:
        @cache_result(ttl=60.0)  # Cache for 60 seconds
        def expensive_lookup(key: str) -> Tuple[str, dict]:
            # Result will be cached for 60 seconds
            return data, {}

        @cache_result()  # Cache forever
        def static_data() -> Tuple[str, dict]:
            return data, {}
    """

    def decorator(func: Callable) -> Callable:
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            cache_key = (args, tuple(sorted(kwargs.items())))

            # Check if we have a cached result
            if cache_key in cache:
                cached_result, cached_time = cache[cache_key]

                # Check if cache is still valid
                if ttl is None or (time.time() - cached_time) < ttl:
                    return cached_result

            # Call function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())

            return result

        # Add cache management methods
        wrapper.clear_cache = lambda: cache.clear()
        wrapper.cache_size = lambda: len(cache)

        return wrapper

    return decorator


def rate_limit(calls: int, period: float):
    """Rate limit decorator for tool functions.

    Limits the number of times a tool can be called within a time period.

    Args:
        calls: Maximum number of calls allowed
        period: Time period in seconds

    Returns:
        Decorated function

    Raises:
        RuntimeError: If rate limit is exceeded

    Example:
        @rate_limit(calls=10, period=60.0)  # Max 10 calls per minute
        def api_call() -> Tuple[str, dict]:
            return response, {}
    """
    import collections

    def decorator(func: Callable) -> Callable:
        call_times = collections.deque()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()

            # Remove old calls outside the time window
            while call_times and call_times[0] < now - period:
                call_times.popleft()

            # Check if we're at the rate limit
            if len(call_times) >= calls:
                raise RuntimeError(
                    f"Rate limit exceeded: {calls} calls per {period}s for {func.__name__}"
                )

            # Record this call
            call_times.append(now)

            return func(*args, **kwargs)

        return wrapper

    return decorator
