# Decorators

Yggdrasil provides production-grade decorators for common operational patterns: retry logic, timeout enforcement, result caching, and rate limiting.

## Table of Contents
- [retry](#retry)
- [retry_async](#retry_async)
- [timeout](#timeout)
- [cache_result](#cache_result)
- [rate_limit](#rate_limit)
- [Combining Decorators](#combining-decorators)
- [Best Practices](#best-practices)

---

## retry

Automatically retry a function on failure with exponential backoff.

### Signature

```python
from bestla.yggdrasil import retry

@retry(
    max_attempts: int = 3,
    backoff: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
)
```

### Parameters

- `max_attempts`: Maximum number of attempts (including initial call)
- `backoff`: Initial backoff duration in seconds (doubles each retry)
- `exceptions`: Tuple of exception types to catch and retry

### Usage

```python
from bestla.yggdrasil import retry, tool
import requests

@retry(max_attempts=3, backoff=1.0, exceptions=(requests.RequestException,))
@tool()
def fetch_data(url: str):
    """Fetch data with automatic retry on network errors"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json(), {}
```

### Behavior

**Retry Schedule** (with `backoff=1.0`):
1. Attempt 1: Immediate
2. Attempt 2: After 1.0s
3. Attempt 3: After 2.0s (1.0 * 2)
4. Attempt 4: After 4.0s (2.0 * 2)

**Example Execution:**

```python
@retry(max_attempts=3, backoff=1.0)
@tool()
def flaky_operation():
    if random.random() < 0.7:  # 70% failure rate
        raise ConnectionError("Network error")
    return "Success", {}

# Execution timeline:
# T=0.0s: Attempt 1 → ConnectionError
# T=1.0s: Attempt 2 → ConnectionError
# T=3.0s: Attempt 3 → Success
# Returns: ("Success", {})
```

### Exception Filtering

Only retry specific exceptions:

```python
@retry(
    max_attempts=5,
    backoff=2.0,
    exceptions=(requests.Timeout, requests.ConnectionError)
)
@tool()
def api_call():
    # Retries on Timeout or ConnectionError
    # Does NOT retry on other exceptions (e.g., HTTPError 404)
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()  # Raises HTTPError on 4xx/5xx
    return response.json(), {}
```

### All Attempts Failed

If all attempts fail, the last exception is raised:

```python
@retry(max_attempts=3)
@tool()
def always_fails():
    raise ValueError("Always fails")

# After 3 attempts, raises: ValueError("Always fails")
```

---

## retry_async

Async version of `@retry` for async functions.

### Signature

```python
from bestla.yggdrasil import retry_async

@retry_async(
    max_attempts: int = 3,
    backoff: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
)
```

### Usage

```python
import aiohttp
from bestla.yggdrasil import retry_async, tool

@retry_async(max_attempts=3, backoff=1.0, exceptions=(aiohttp.ClientError,))
@tool()
async def fetch_data_async(url: str):
    """Async fetch with retry"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.json()
            return data, {}
```

**Note**: Use `retry_async` for `async def` functions, `retry` for sync functions.

---

## timeout

Enforce maximum execution time for a function.

### Signature

```python
from bestla.yggdrasil import timeout

@timeout(seconds: float)
```

### Parameters

- `seconds`: Maximum execution time in seconds

### Usage

```python
from bestla.yggdrasil import timeout, tool
import time

@timeout(5.0)
@tool()
def long_running_task():
    """Must complete within 5 seconds"""
    time.sleep(3)  # OK
    return "Completed", {}

@timeout(2.0)
@tool()
def too_slow():
    """Will timeout"""
    time.sleep(5)  # Exceeds timeout
    return "Never reached", {}
```

### Behavior

If the function exceeds the timeout, a `TimeoutError` is raised:

```python
@timeout(1.0)
@tool()
def slow_operation():
    time.sleep(3)
    return "Done", {}

# Raises: TimeoutError after 1 second
```

### Use Cases

1. **Prevent Hanging**: Ensure tools don't block indefinitely
2. **SLA Enforcement**: Guarantee response times
3. **Resource Protection**: Limit long-running operations

### Implementation Note

Uses threading (`threading.Timer`) to interrupt execution. Works for most blocking operations but may not interrupt certain syscalls.

---

## cache_result

Cache function results with time-to-live (TTL).

### Signature

```python
from bestla.yggdrasil import cache_result

@cache_result(ttl: float)
```

### Parameters

- `ttl`: Time-to-live in seconds (how long to cache results)

### Usage

```python
from bestla.yggdrasil import cache_result, tool
import requests

@cache_result(ttl=300.0)  # Cache for 5 minutes
@tool()
def get_exchange_rate(currency: str):
    """Fetch exchange rate (cached)"""
    response = requests.get(f"https://api.example.com/rate/{currency}")
    rate = response.json()["rate"]
    return rate, {}
```

### Behavior

**Cache Key**: Function name + arguments (as tuple)

```python
@cache_result(ttl=60.0)
@tool()
def expensive_operation(x: int, y: str):
    time.sleep(2)  # Simulate expensive work
    return f"Result: {x}, {y}", {}

# First call: executes function (2s delay)
result1, _ = expensive_operation(1, "a")  # Cache miss

# Second call (within 60s): returns cached result (instant)
result2, _ = expensive_operation(1, "a")  # Cache hit

# Different args: executes function (2s delay)
result3, _ = expensive_operation(2, "a")  # Cache miss (different key)

# After 60s: cache expired, executes function
time.sleep(61)
result4, _ = expensive_operation(1, "a")  # Cache miss (expired)
```

### Cache Expiration

Cache entries expire after TTL:

```python
@cache_result(ttl=10.0)  # 10 second TTL
@tool()
def get_data():
    return fetch_fresh_data(), {}

# T=0s: First call → Cache miss, executes function
get_data()

# T=5s: Second call → Cache hit (within TTL)
get_data()

# T=11s: Third call → Cache miss (TTL expired)
get_data()
```

### Use Cases

1. **API Rate Limiting**: Reduce repeated API calls
2. **Performance**: Cache expensive computations
3. **External Service**: Reduce load on third-party services

### Caveats

- **Arguments must be hashable**: Dicts/lists won't work as arguments
- **Memory**: Cache grows with unique argument combinations
- **No cache invalidation**: Can't manually clear cache (TTL only)

---

## rate_limit

Limit function call rate (calls per time period).

### Signature

```python
from bestla.yggdrasil import rate_limit

@rate_limit(calls: int, period: float)
```

### Parameters

- `calls`: Maximum number of calls allowed
- `period`: Time period in seconds

### Usage

```python
from bestla.yggdrasil import rate_limit, tool

@rate_limit(calls=10, period=60.0)  # Max 10 calls per minute
@tool()
def api_call():
    """Rate-limited API call"""
    response = requests.get("https://api.example.com/data")
    return response.json(), {}
```

### Behavior

Tracks call timestamps in a sliding window:

```python
@rate_limit(calls=3, period=10.0)  # Max 3 calls per 10 seconds
@tool()
def limited_function():
    return "Called", {}

# T=0s: Call 1 → OK
limited_function()

# T=1s: Call 2 → OK
limited_function()

# T=2s: Call 3 → OK
limited_function()

# T=3s: Call 4 → Blocks until T=10s (waits 7 seconds)
limited_function()

# T=11s: Call 5 → OK (Call 1 aged out of 10s window)
limited_function()
```

### Sliding Window

Uses a sliding window algorithm:

```
Time: 0s  1s  2s  3s  4s  5s  6s  7s  8s  9s  10s 11s
Call: X   X   X   [Rate limit hit]              X

Window at T=3s:  [0s, 1s, 2s] = 3 calls → FULL
Window at T=11s: [1s, 2s, 11s] = 3 calls → FULL (0s aged out)
```

### Use Cases

1. **API Quotas**: Respect external API rate limits
2. **Resource Protection**: Prevent resource exhaustion
3. **Fair Usage**: Enforce usage policies

### Blocking Behavior

When rate limit is exceeded, **the function blocks** until a slot becomes available:

```python
@rate_limit(calls=2, period=5.0)
@tool()
def limited():
    print(f"Called at {time.time()}")
    return "Done", {}

# T=0.0s: Call 1 → Executes immediately
limited()

# T=0.5s: Call 2 → Executes immediately
limited()

# T=1.0s: Call 3 → BLOCKS for 4 seconds (until T=5.0s)
limited()  # Prints "Called at 5.0"
```

**Note**: Blocking can cause the LLM conversation loop to pause. Use timeouts to prevent indefinite blocking.

---

## Combining Decorators

Decorators can be stacked to combine behaviors.

### Order Matters

Decorators are applied **bottom-up**:

```python
@decorator_a
@decorator_b
@tool()
def my_function():
    pass

# Execution order:
# 1. tool() wraps function
# 2. decorator_b wraps result from (1)
# 3. decorator_a wraps result from (2)
#
# Call flow:
# decorator_a → decorator_b → tool → function
```

### Common Combinations

#### Retry + Timeout

```python
@retry(max_attempts=3, backoff=1.0)
@timeout(5.0)
@tool()
def resilient_api_call():
    """Retry up to 3 times, each attempt times out after 5s"""
    response = requests.get("https://api.example.com/data", timeout=10)
    return response.json(), {}

# Flow:
# 1. retry starts attempt 1
# 2. timeout enforces 5s limit on attempt 1
# 3. If attempt 1 fails, retry waits and starts attempt 2
# 4. timeout enforces 5s limit on attempt 2
# ... and so on
```

#### Cache + Rate Limit

```python
@cache_result(ttl=60.0)
@rate_limit(calls=10, period=60.0)
@tool()
def cached_limited_api():
    """Cache results for 1 min, limit to 10 calls/min"""
    return requests.get("https://api.example.com/data").json(), {}

# Flow:
# 1. cache_result checks cache
# 2. If cache miss, rate_limit enforces quota
# 3. Function executes
# 4. Result cached
```

#### Retry + Rate Limit + Timeout

```python
@retry(max_attempts=3, backoff=2.0)
@rate_limit(calls=5, period=60.0)
@timeout(10.0)
@tool()
def robust_api_call():
    """
    - Max 5 calls per minute (rate limit)
    - Each attempt times out after 10s
    - Retry up to 3 times with exponential backoff
    """
    return requests.get("https://api.example.com/data").json(), {}
```

### Decorator Ordering Guidelines

**Recommended order** (outermost to innermost):

1. **@retry** - Outermost (retries entire wrapped function)
2. **@cache_result** - Second (caches after rate limiting)
3. **@rate_limit** - Third (limits calls)
4. **@timeout** - Fourth (times each call)
5. **@tool** - Innermost (always last)

```python
@retry(max_attempts=3, backoff=1.0)
@cache_result(ttl=300.0)
@rate_limit(calls=10, period=60.0)
@timeout(5.0)
@tool()
def fully_decorated():
    pass
```

---

## Best Practices

### 1. Set Reasonable Retry Limits

```python
# ❌ Bad: Too many retries
@retry(max_attempts=100, backoff=0.1)
@tool()
def flaky():
    pass

# ✅ Good: Reasonable limit
@retry(max_attempts=3, backoff=1.0)
@tool()
def flaky():
    pass
```

### 2. Specify Exception Types

```python
# ❌ Bad: Retries ALL exceptions (including bugs)
@retry(max_attempts=3, exceptions=(Exception,))
@tool()
def api_call():
    pass

# ✅ Good: Retry only transient errors
@retry(
    max_attempts=3,
    exceptions=(requests.Timeout, requests.ConnectionError)
)
@tool()
def api_call():
    pass
```

### 3. Combine Timeout with Retry

```python
# ✅ Prevent infinite blocking
@retry(max_attempts=3)
@timeout(10.0)
@tool()
def network_call():
    # Each retry attempt limited to 10s
    pass
```

### 4. Cache Expensive, Stable Data

```python
# ✅ Good: Cache stable reference data
@cache_result(ttl=3600.0)  # 1 hour
@tool()
def get_country_list():
    return fetch_countries(), {}

# ❌ Bad: Cache rapidly changing data
@cache_result(ttl=3600.0)
@tool()
def get_stock_price():
    # Stale data for up to 1 hour!
    pass
```

### 5. Align Rate Limits with External APIs

```python
# If API allows 100 requests/minute
@rate_limit(calls=90, period=60.0)  # Leave 10% buffer
@tool()
def api_call():
    pass
```

### 6. Log Decorator Events

```python
import logging

@retry(max_attempts=3)
@tool()
def logged_retry():
    try:
        result = risky_operation()
        return result, {}
    except Exception as e:
        logging.warning(f"Attempt failed: {e}")
        raise  # Re-raise for retry mechanism
```

### 7. Test Decorator Behavior

```python
# Test retry logic
def test_retry():
    call_count = 0

    @retry(max_attempts=3, backoff=0.1)
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Fail")
        return "Success", {}

    result, _ = flaky()
    assert result == "Success"
    assert call_count == 3  # Called 3 times

# Test cache
def test_cache():
    call_count = 0

    @cache_result(ttl=10.0)
    def cached():
        nonlocal call_count
        call_count += 1
        return f"Call {call_count}", {}

    result1, _ = cached()
    result2, _ = cached()

    assert result1 == result2  # Same result
    assert call_count == 1  # Only called once
```

### 8. Document Decorator Configuration

```python
@retry(
    max_attempts=5,
    backoff=2.0,
    exceptions=(requests.Timeout,)
)
@cache_result(ttl=600.0)
@rate_limit(calls=20, period=60.0)
@tool(description="Fetch user data with resilience")
def get_user_data(user_id: str):
    """
    Fetch user data from external API.

    Resilience:
        - Retries: Up to 5 attempts with 2s exponential backoff on timeout
        - Cache: Results cached for 10 minutes
        - Rate Limit: Max 20 calls per minute

    Args:
        user_id: User identifier

    Returns:
        User data dict
    """
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json(), {}
```

---

## Performance Considerations

### Memory Usage

- **@cache_result**: Cache grows with unique argument combinations
- **@rate_limit**: Maintains call timestamp list (grows with call rate)

### Latency

- **@retry**: Adds latency on failures (backoff time)
- **@rate_limit**: Blocks when limit exceeded
- **@timeout**: Minimal overhead (threading setup)
- **@cache_result**: Fast (dict lookup)

### Thread Safety

All decorators are **thread-safe** and can be used in concurrent environments.

---

## Common Patterns

### API Client Tool

```python
@retry(max_attempts=3, backoff=1.0, exceptions=(requests.RequestException,))
@cache_result(ttl=300.0)
@rate_limit(calls=60, period=60.0)
@timeout(10.0)
@tool()
def fetch_api_data(endpoint: str):
    """Production-ready API client"""
    response = requests.get(f"https://api.example.com/{endpoint}")
    response.raise_for_status()
    return response.json(), {}
```

### Database Query Tool

```python
@retry(max_attempts=3, backoff=0.5, exceptions=(DatabaseError,))
@cache_result(ttl=60.0)
@timeout(5.0)
@tool()
def query_database(query: str):
    """Cached database query with retry"""
    conn = get_db_connection()
    result = conn.execute(query).fetchall()
    return result, {}
```

### Background Job Trigger

```python
@rate_limit(calls=10, period=60.0)
@timeout(30.0)
@tool()
def trigger_job(job_type: str):
    """Rate-limited job trigger"""
    job_id = enqueue_job(job_type)
    return f"Job {job_id} enqueued", {}
```

---

**Next**: [Context Management](context-management.md) - Conversation compaction strategies
