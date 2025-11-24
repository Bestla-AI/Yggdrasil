# Best Practices

This guide provides design patterns, anti-patterns, and recommendations for building robust Yggdrasil applications.

## Table of Contents
- [Architecture Patterns](#architecture-patterns)
- [State Management](#state-management)
- [Performance](#performance)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Security](#security)
- [Common Anti-Patterns](#common-anti-patterns)

---

## Architecture Patterns

### Pattern: Stateless Agent, Stateful Execution

**✅ Good:**
```python
# Agent is stateless and reusable
agent = Agent(provider=client, model="gpt-4")

# State lives in ExecutionContext
ctx1 = None
ctx2 = None

# User 1
r1, ctx1 = agent.run("User 1 query", execution_context=ctx1)

# User 2 (concurrent with user 1)
r2, ctx2 = agent.run("User 2 query", execution_context=ctx2)
```

**❌ Bad:**
```python
# Don't store state in agent
agent.user_sessions = {}  # Global state - race conditions!
```

### Pattern: Toolkit per Domain

**✅ Good:**
```python
# Each toolkit represents a domain
auth_toolkit = Toolkit()  # Authentication domain
project_toolkit = Toolkit()  # Project management domain
github_toolkit = Toolkit()  # GitHub integration domain

agent.add_toolkit("auth", auth_toolkit)
agent.add_toolkit("project", project_toolkit)
agent.add_toolkit("github", github_toolkit)
```

**Benefits:**
- Clear separation of concerns
- Parallel execution across domains
- Independent state management

### Pattern: Progressive Enhancement

Build workflows that progressively enhance context:

```python
@tool(enables_states=["authenticated"])
def login(username: str):
    return "Logged in", {"user": username, "permissions": ["read", "write"]}

@tool(
    required_states=["authenticated"],
    enables_states=["project_selected"]
)
def select_project(name: DynamicStr["projects"]):
    issues = fetch_issues(name)
    return f"Selected {name}", {
        "current_project": name,
        "issues": [i.name for i in issues]
    }

@tool(
    required_states=["authenticated", "project_selected"],
    enables_states=["issue_selected"]
)
def select_issue(name: DynamicStr["issues"]):
    return f"Selected {name}", {"current_issue": name}
```

**Flow**: login → context populated → select_project → context enhanced → select_issue

### Pattern: Hierarchical Agents

**✅ Good:**
```python
# Specialized sub-agents
research_agent = Agent(provider=client, model="gpt-4")
research_agent.add_toolkit("web", web_search_toolkit)

analysis_agent = Agent(provider=client, model="gpt-4")
analysis_agent.add_toolkit("data", data_analysis_toolkit)

# Coordinator agent
coordinator = Agent(provider=client, model="gpt-4")

@tool()
def research(topic: str):
    response, _ = research_agent.run(f"Research {topic}")
    return response, {}

@tool()
def analyze(data: str):
    response, _ = analysis_agent.run(f"Analyze {data}")
    return response, {}

coordinator.add_tool("research", research)
coordinator.add_tool("analyze", analyze)
```

**Benefits:**
- Modular design
- Specialized agents
- Parallel execution

---

## State Management

### Use Abstract State Names

**✅ Good:**
```python
enables_states=["authenticated", "resource_locked", "processing"]
```

**❌ Bad:**
```python
enables_states=["user_alice_logged_in_at_2pm"]  # Too specific
```

### Keep States Orthogonal

**✅ Good:**
```python
# Independent states
unlocked_states = {
    "authenticated",
    "premium_user",
    "admin_mode"
}
```

**❌ Bad:**
```python
# Redundant states
unlocked_states = {
    "authenticated",
    "not_authenticated",  # Redundant (inverse)
    "logged_in"  # Duplicate of authenticated
}
```

### Combine FSM with Context

**✅ Good:**
```python
# State controls availability, context provides data
@tool(
    required_states=["project_selected"],
    required_context=["issues"]
)
def select_issue(name: DynamicStr["issues"]):
    return name, {}
```

### Document State Transitions

```python
@tool(
    description="Login to system",
    enables_states=["authenticated"],
    disables_states=["guest_mode"]
)
def login(username: str, password: str):
    """
    Authenticate user and enable authenticated state.

    State Transitions:
        - Enables: authenticated
        - Disables: guest_mode

    Context Updates:
        - user: username
        - session_id: generated session ID
    """
    session_id = create_session(username, password)
    return f"Logged in as {username}", {
        "user": username,
        "session_id": session_id
    }
```

---

## Performance

### Minimize Toolkit Size

**✅ Good:**
```python
# Small toolkits (3-7 tools each)
auth_toolkit = Toolkit()  # 4 tools
project_toolkit = Toolkit()  # 5 tools
issue_toolkit = Toolkit()  # 6 tools

# Toolkits execute in parallel
```

**❌ Bad:**
```python
# Huge toolkit (50 tools)
mega_toolkit = Toolkit()
for i in range(50):
    mega_toolkit.add_tool(f"tool_{i}", tool)
# All 50 tools execute sequentially!
```

### Use Independent Tools for Parallelism

**✅ Good:**
```python
# Stateless operations as independent tools
agent.add_tool("add", add)
agent.add_tool("multiply", multiply)
agent.add_tool("format_date", format_date)
# All execute in parallel
```

### Cache Expensive Operations

**✅ Good:**
```python
@cache_result(ttl=300.0)
@tool()
def get_exchange_rates():
    # Expensive API call cached for 5 minutes
    return fetch_rates(), {}
```

### Rate Limit External APIs

**✅ Good:**
```python
@rate_limit(calls=100, period=60.0)
@tool()
def api_call():
    # Respects API quota
    return call_external_api(), {}
```

### Profile Critical Paths

```python
import time

@tool()
def critical_tool():
    start = time.time()
    result = expensive_operation()
    duration = time.time() - start

    if duration > 1.0:
        logging.warning(f"Slow tool: {duration:.2f}s")

    return result, {}
```

---

## Error Handling

### Use Retry for Transient Errors

**✅ Good:**
```python
@retry(
    max_attempts=3,
    backoff=1.0,
    exceptions=(requests.Timeout, requests.ConnectionError)
)
@tool()
def resilient_api_call():
    return requests.get("https://api.example.com/data").json(), {}
```

### Return Meaningful Errors

**✅ Good:**
```python
@tool()
def process_data(data: dict):
    try:
        result = process(data)
        return result, {}
    except ValueError as e:
        # Return error message for LLM to understand
        return f"Error: Invalid data format - {e}", {}
```

**❌ Bad:**
```python
@tool()
def process_data(data: dict):
    result = process(data)  # Uncaught exception crashes tool
    return result, {}
```

### Validate Inputs

**✅ Good:**
```python
@tool()
def create_user(age: int):
    if age < 0 or age > 150:
        return "Error: Invalid age (must be 0-150)", {}

    user = create_user_record(age)
    return user, {}
```

### Handle Pipeline Failures

```python
@tool()
def step1():
    try:
        result = risky_operation()
        return result, {"step1_complete": True}
    except Exception as e:
        # Log error but don't crash
        logging.error(f"Step 1 failed: {e}")
        return f"Step 1 failed: {e}", {"step1_failed": True}
```

---

## Testing

### Test State Transitions

```python
def test_authentication_flow():
    toolkit = Toolkit()
    toolkit.add_tool("login", login)
    toolkit.add_tool("logout", logout)

    # Initial state
    assert toolkit.unlocked_states == set()

    # Login
    toolkit.execute_tool("login", {"username": "alice", "password": "secret"})
    assert "authenticated" in toolkit.unlocked_states
    assert toolkit.context.get("user") == "alice"

    # Logout
    toolkit.execute_tool("logout", {})
    assert "authenticated" not in toolkit.unlocked_states
```

### Test Dynamic Schemas

```python
def test_dynamic_schema():
    toolkit = Toolkit()
    toolkit.context.set("projects", ["alpha", "beta"])

    @tool()
    def select_project(name: DynamicStr["projects"]):
        return name, {}

    toolkit.add_tool("select_project", select_project)

    # Generate schema
    schemas = toolkit.generate_schemas()
    schema = schemas[0]

    # Verify enum
    assert schema["function"]["parameters"]["properties"]["name"]["enum"] == ["alpha", "beta"]
```

### Test Concurrent Execution

```python
def test_concurrent_runs():
    agent = Agent(provider=client, model="gpt-4")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(agent.run, f"Query {i}") for i in range(10)]
        results = [f.result() for f in futures]

    assert len(results) == 10
    # No race conditions or crashes
```

### Mock External Dependencies

```python
from unittest.mock import patch

def test_api_tool():
    @tool()
    def fetch_data():
        return requests.get("https://api.example.com/data").json(), {}

    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"data": "mocked"}

        result, _ = fetch_data()
        assert result == {"data": "mocked"}
```

### Test Error Conditions

```python
def test_tool_error_handling():
    @tool()
    def may_fail(should_fail: bool):
        if should_fail:
            raise ValueError("Intentional failure")
        return "Success", {}

    # Test success
    result, _ = may_fail(False)
    assert result == "Success"

    # Test failure
    with pytest.raises(ValueError):
        may_fail(True)
```

---

## Security

### Validate User Inputs

**✅ Good:**
```python
@tool()
def execute_query(query: str):
    # Validate and sanitize
    if not is_safe_query(query):
        return "Error: Invalid query", {}

    # Use parameterized queries
    result = db.execute(query, sanitized=True)
    return result, {}
```

**❌ Bad:**
```python
@tool()
def execute_query(query: str):
    # SQL injection risk!
    result = db.execute(f"SELECT * FROM users WHERE name = '{query}'")
    return result, {}
```

### Use Required States for Authorization

**✅ Good:**
```python
@tool(required_states=["authenticated", "admin"])
def delete_user(user_id: str):
    # Only available when both authenticated AND admin
    delete_user_record(user_id)
    return "User deleted", {}
```

### Sanitize Tool Outputs

**✅ Good:**
```python
@tool()
def get_user_data(user_id: str):
    user = fetch_user(user_id)

    # Don't expose sensitive fields
    safe_user = {
        "id": user.id,
        "name": user.name,
        # DON'T include: password_hash, ssn, etc.
    }

    return safe_user, {}
```

### Rate Limit Sensitive Operations

**✅ Good:**
```python
@rate_limit(calls=5, period=60.0)
@tool()
def send_email(to: str, subject: str):
    # Prevent email spam
    send_email_message(to, subject)
    return "Email sent", {}
```

### Validate Context Schema

**✅ Good:**
```python
context = Context(validation_enabled=True)

# Define schema for sensitive data
context.schema.define("user_id", {
    "type": "string",
    "pattern": "^[a-zA-Z0-9]{8,32}$"  # Only allow alphanumeric
})

# Invalid values rejected
context.set("user_id", "valid123")  # OK
context.set("user_id", "../../../etc/passwd")  # ContextValidationError
```

---

## Common Anti-Patterns

### ❌ Storing State in Global Variables

```python
# ❌ Bad
global_user_state = {}

@tool()
def login(username: str):
    global_user_state["user"] = username  # Race conditions!
    return "Logged in", {}

# ✅ Good
@tool()
def login(username: str):
    return "Logged in", {"user": username}  # Use context
```

### ❌ Forgetting Return Tuple

```python
# ❌ Bad
@tool()
def bad_tool():
    return "Result"  # Missing context_updates!

# ✅ Good
@tool()
def good_tool():
    return "Result", {}
```

### ❌ Mutating Context Directly

```python
# ❌ Bad
@tool()
def bad_mutation():
    toolkit.context._data["key"] = "value"  # Direct mutation!
    return "Done", {}

# ✅ Good
@tool()
def good_mutation():
    return "Done", {"key": "value"}  # Return updates
```

### ❌ Blocking Operations Without Timeout

```python
# ❌ Bad
@tool()
def may_hang():
    response = requests.get("https://slow-api.com", timeout=None)
    return response.json(), {}

# ✅ Good
@timeout(10.0)
@tool()
def with_timeout():
    response = requests.get("https://slow-api.com", timeout=5.0)
    return response.json(), {}
```

### ❌ Too Many States

```python
# ❌ Bad
enables_states=[
    "step1_complete", "step2_complete", ..., "step50_complete"
]

# ✅ Good: Use context for progression
@tool()
def complete_step(step_num: int):
    return f"Step {step_num} done", {"completed_steps": step_num}
```

### ❌ Catching All Exceptions

```python
# ❌ Bad
@retry(exceptions=(Exception,))  # Retries everything, even bugs!
@tool()
def bad_retry():
    pass

# ✅ Good
@retry(exceptions=(requests.Timeout, requests.ConnectionError))
@tool()
def good_retry():
    pass
```

### ❌ Stale Context Data

```python
# ❌ Bad
context.set("users", fetch_users())  # Set once, never updated

# ... 100 tool calls later ...

@tool()
def select_user(name: DynamicStr["users"]):
    # Stale data - users may have changed!
    pass

# ✅ Good: Refresh context periodically
@tool(enables_states=["users_loaded"])
def load_users():
    users = fetch_users()
    return users, {"users": [u.name for u in users]}
```

### ❌ Over-Constraining Tools

```python
# ❌ Bad
@tool(
    required_states=["authenticated", "premium", "verified", "active"],
    forbidden_states=["suspended", "rate_limited", "maintenance", "readonly"],
    required_context=["user", "session", "project", "permissions"]
)
def overly_constrained():
    pass

# ✅ Good
@tool(required_states=["authenticated"])
def reasonably_constrained():
    pass
```

---

## Design Checklist

### Before Building

- [ ] Identify stateful vs. stateless tools
- [ ] Group related tools into toolkits
- [ ] Define FSM states and transitions
- [ ] Plan context schema
- [ ] Identify parallel vs. sequential operations

### During Development

- [ ] Return `(result, context_updates)` tuple from all tools
- [ ] Use abstract state names
- [ ] Document state requirements
- [ ] Add error handling
- [ ] Use decorators for resilience (@retry, @timeout)
- [ ] Validate inputs
- [ ] Test state transitions

### Before Deployment

- [ ] Add rate limiting for external APIs
- [ ] Enable context validation for sensitive data
- [ ] Add logging and monitoring
- [ ] Test concurrent execution
- [ ] Profile performance
- [ ] Review security (input validation, authorization)
- [ ] Set up ContextManager for long conversations

---

## Recommended Toolkit Structure

```python
# Domain toolkit (3-7 tools)
domain_toolkit = Toolkit(context=Context(validation_enabled=True))

# Define context schema
domain_toolkit.context.schema.define("domain_key", {...})

# Tool 1: Initialize (enable base state)
@tool(enables_states=["initialized"])
def initialize():
    return "Initialized", {...}

# Tool 2-5: Operations (require base state)
@tool(required_states=["initialized"])
def operation():
    return "Done", {}

# Tool 6: Cleanup (disable states)
@tool(
    required_states=["initialized"],
    disables_states=["initialized"]
)
def cleanup():
    return "Cleaned up", {}

# Add to toolkit
domain_toolkit.add_tool("initialize", initialize)
domain_toolkit.add_tool("operation", operation)
domain_toolkit.add_tool("cleanup", cleanup)

# Add to agent
agent.add_toolkit("domain", domain_toolkit)
```

---

**Next**: [Examples](examples.md) - Complete workflow examples
