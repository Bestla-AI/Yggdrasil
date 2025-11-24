# Concurrency

Yggdrasil uses a sophisticated three-tier concurrency model to maximize parallelism while preventing race conditions.

## Table of Contents
- [Three-Tier Model](#three-tier-model)
- [Thread Safety](#thread-safety)
- [Execution Isolation](#execution-isolation)
- [Concurrent Patterns](#concurrent-patterns)
- [Best Practices](#best-practices)

---

## Three-Tier Model

Yggdrasil executes tools at three concurrency levels:

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Independent Tools (Parallel)                        │
│  - Tools not in any toolkit                                 │
│  - Execute in parallel via ThreadPoolExecutor               │
│  - No ordering guarantees                                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Tier 2: Toolkit Groups (Parallel to Each Other)             │
│  - Different toolkits execute in parallel                   │
│  - plane:: and github:: run simultaneously                  │
│  - Each toolkit has isolated context (deep copy)            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Within-Toolkit Tools (Sequential)                   │
│  - Tools within same toolkit execute sequentially           │
│  - Ensures FSM state consistency                            │
│  - Context updates propagate immediately                    │
└─────────────────────────────────────────────────────────────┘
```

### Why Three Tiers?

**Design Goals:**
1. **Maximize Parallelism**: Execute independent operations concurrently
2. **Prevent Race Conditions**: Avoid concurrent writes to shared state
3. **Guarantee Ordering**: Preserve sequential semantics where needed

**Trade-offs:**
- ✅ Safe: No race conditions on toolkit state
- ✅ Fast: Parallel execution across toolkits
- ❌ Limited: Can't parallelize within a toolkit

---

## Tier 1: Independent Tools (Parallel)

Tools added via `agent.add_tool()` execute in parallel.

### Example

```python
from bestla.yggdrasil import Agent, tool

@tool()
def add(a: int, b: int):
    time.sleep(1)  # Simulate work
    return a + b, {}

@tool()
def multiply(a: int, b: int):
    time.sleep(1)  # Simulate work
    return a * b, {}

agent = Agent(provider=client, model="gpt-4")
agent.add_tool("add", add)
agent.add_tool("multiply", multiply)

# LLM calls both tools
# Execution: add() and multiply() run in parallel
# Total time: ~1 second (not 2)
```

### Characteristics

- **Execution**: `ThreadPoolExecutor` with futures
- **State**: No shared state (must be stateless)
- **Ordering**: No guarantees (may complete in any order)
- **Isolation**: Each tool runs independently

### Use Cases

- Math operations
- Stateless utilities
- Independent API calls

---

## Tier 2: Toolkit Groups (Parallel to Each Other)

Different toolkits execute in parallel, but each toolkit's context is isolated.

### Example

```python
from bestla.yggdrasil import Agent, Toolkit, tool

# Plane toolkit
plane_toolkit = Toolkit()

@tool()
def plane_select_project(id: str):
    time.sleep(1)
    return f"Project {id}", {"project": id}

plane_toolkit.add_tool("select_project", plane_select_project)

# GitHub toolkit
github_toolkit = Toolkit()

@tool()
def github_list_prs():
    time.sleep(1)
    return ["PR-1", "PR-2"], {}

github_toolkit.add_tool("list_prs", github_list_prs)

# Agent
agent = Agent(provider=client, model="gpt-4")
agent.add_toolkit("plane", plane_toolkit)
agent.add_toolkit("github", github_toolkit)

# LLM calls:
# - plane::select_project("alpha")
# - github::list_prs()
#
# Execution: Both toolkits run in parallel
# Total time: ~1 second (not 2)
```

### Isolation Mechanism

Each toolkit is **deep copied** when ExecutionContext is created:

```python
# Inside agent.run()
execution_context.toolkits = {
    "plane": plane_toolkit.copy(),    # Deep copy
    "github": github_toolkit.copy()   # Deep copy
}
```

**Benefits:**
- Parallel runs don't interfere
- Each execution has independent state
- Thread-safe without locking

### Characteristics

- **Execution**: `ThreadPoolExecutor` across toolkit groups
- **State**: Isolated per toolkit (deep copy)
- **Ordering**: Toolkit groups execute in parallel
- **Context**: Each toolkit has independent context

### Use Cases

- Multi-service workflows (Plane + GitHub + Slack)
- Domain-separated operations
- Independent resource access

---

## Tier 3: Within-Toolkit Tools (Sequential)

Tools within the same toolkit execute sequentially to ensure state consistency.

### Example

```python
toolkit = Toolkit()

@tool(enables_states=["authenticated"])
def login(username: str):
    time.sleep(1)
    return "Logged in", {"user": username}

@tool(required_states=["authenticated"])
def get_profile():
    time.sleep(1)
    user = toolkit.context.get("user")
    return f"Profile for {user}", {}

toolkit.add_tool("login", login)
toolkit.add_tool("get_profile", get_profile)

agent = Agent(provider=client, model="gpt-4")
agent.add_toolkit("auth", toolkit)

# LLM calls:
# - auth::login("alice")
# - auth::get_profile()
#
# Execution: Sequential within toolkit
# 1. login() executes → updates context, enables "authenticated"
# 2. get_profile() executes → sees updated context
# Total time: ~2 seconds
```

### Sequential Pipeline

```python
# Execution flow
for tool_call in toolkit_tool_calls:
    # 1. Execute tool
    result, updates = tool.execute(args)

    # 2. Update context (immediately visible to next tool)
    toolkit.context.update(updates)

    # 3. Update FSM states (immediately visible to next tool)
    toolkit.unlocked_states.update(tool.enables_states)
    toolkit.unlocked_states.difference_update(tool.disables_states)

    # 4. Next tool sees updated state
```

### Pipeline Failure

If a tool fails, the pipeline stops:

```python
@tool()
def step1():
    return "OK", {"step": 1}

@tool()
def step2():
    raise ValueError("Failed")

@tool()
def step3():
    return "Never reached", {"step": 3}

# Execution:
# - step1() → Success, context updated
# - step2() → Failure, pipeline stops
# - step3() → NOT EXECUTED
```

### Characteristics

- **Execution**: Sequential loop
- **State**: Shared context within toolkit
- **Ordering**: Strict sequential order (tool N before N+1)
- **Updates**: Immediate propagation

### Use Cases

- Stateful workflows (login → fetch → update)
- FSM-driven processes
- Multi-step transactions

---

## Thread Safety

### Immutable Context

Context uses `immutables.Map` for thread-safe updates:

```python
# Each update creates new Map (structural sharing)
context._data = self._data.set(key, value)  # O(1) update, new Map instance
```

**Benefits:**
- No locks needed
- Safe to read from multiple threads
- Fast shallow copies

### ExecutionContext Isolation

Each `agent.run()` creates or reuses ExecutionContext with deep-copied toolkits:

```python
# Parallel runs are safe
from concurrent.futures import ThreadPoolExecutor

agent = Agent(provider=client, model="gpt-4")

with ThreadPoolExecutor() as executor:
    future1 = executor.submit(agent.run, "Query 1")
    future2 = executor.submit(agent.run, "Query 2")
    # Safe: Separate ExecutionContexts
```

### Concurrent Agent Instances

Multiple agents can run concurrently:

```python
agent1 = Agent(provider=client, model="gpt-4")
agent2 = Agent(provider=client, model="gpt-4")

with ThreadPoolExecutor() as executor:
    future1 = executor.submit(agent1.run, "Query")
    future2 = executor.submit(agent2.run, "Query")
    # Safe: Separate agent instances
```

---

## Execution Isolation

### Pattern 1: Stateless (Full Isolation)

```python
agent = Agent(provider=client, model="gpt-4")

# Each run is completely isolated
r1, ctx1 = agent.run("Query 1")
r2, ctx2 = agent.run("Query 2")

# ctx1 and ctx2 are independent
```

### Pattern 2: Stateful (Shared Context)

```python
agent = Agent(provider=client, model="gpt-4")

# First run creates context
r1, ctx1 = agent.run("Login")

# Second run reuses context
r2, ctx2 = agent.run("Get profile", execution_context=ctx1)

# ctx2 is updated ctx1 (same conversation history)
```

### Pattern 3: Concurrent with Isolation

```python
agent = Agent(provider=client, model="gpt-4")

def process_user(user_id):
    # Each user has isolated context
    response, ctx = agent.run(f"Process user {user_id}")
    return response

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_user, i) for i in range(10)]
    results = [f.result() for f in futures]
    # Safe: Each call creates separate ExecutionContext
```

---

## Concurrent Patterns

### Multi-User Service

```python
from threading import Lock

class UserService:
    def __init__(self, agent):
        self.agent = agent
        self.sessions = {}  # user_id -> ExecutionContext
        self.lock = Lock()  # Protect sessions dict

    def handle_request(self, user_id, prompt):
        with self.lock:
            ctx = self.sessions.get(user_id)

        # Run agent (no lock needed, ExecutionContext is isolated)
        response, ctx = self.agent.run(prompt, execution_context=ctx)

        with self.lock:
            self.sessions[user_id] = ctx

        return response

# Usage
service = UserService(agent)

with ThreadPoolExecutor() as executor:
    # Concurrent requests from different users
    futures = [
        executor.submit(service.handle_request, "user1", "Hello"),
        executor.submit(service.handle_request, "user2", "Hi"),
        executor.submit(service.handle_request, "user1", "Follow-up")
    ]
```

### Parallel Task Processing

```python
def process_task(agent, task):
    """Process single task (isolated execution)"""
    response, ctx = agent.run(f"Process {task}")
    return response

agent = Agent(provider=client, model="gpt-4")

tasks = ["task1", "task2", "task3", "task4"]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(lambda t: process_task(agent, t), tasks))

# All tasks processed in parallel with isolated contexts
```

### Hierarchical Agents (Concurrent Sub-Agents)

```python
# Research sub-agent
research_agent = Agent(provider=client, model="gpt-4")
# ... add research tools ...

# Analysis sub-agent
analysis_agent = Agent(provider=client, model="gpt-4")
# ... add analysis tools ...

# Coordinator
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

# When coordinator calls both tools, they execute in parallel (Tier 1)
response, ctx = coordinator.run("Research topic X and analyze data Y")
```

---

## Best Practices

### 1. Use Appropriate Tier

```python
# ✅ Good: Independent operations as independent tools
agent.add_tool("add", add_function)
agent.add_tool("multiply", multiply_function)
# Executes in parallel

# ❌ Bad: Putting independent tools in toolkit
toolkit.add_tool("add", add_function)
toolkit.add_tool("multiply", multiply_function)
# Executes sequentially (unnecessary)
```

### 2. Group Related Tools in Toolkits

```python
# ✅ Good: Stateful operations in toolkit
auth_toolkit = Toolkit()
auth_toolkit.add_tool("login", login)
auth_toolkit.add_tool("get_profile", get_profile)
# Sequential execution ensures state consistency

# ❌ Bad: Stateful tools as independent
agent.add_tool("login", login)
agent.add_tool("get_profile", get_profile)
# May execute in parallel → race condition on shared state
```

### 3. Minimize Toolkit Size

```python
# ❌ Bad: Too many tools in one toolkit
big_toolkit = Toolkit()
for i in range(100):
    big_toolkit.add_tool(f"tool_{i}", tool_function)
# All 100 tools execute sequentially!

# ✅ Good: Split into logical toolkits
auth_toolkit = Toolkit()  # 3-5 tools
data_toolkit = Toolkit()  # 3-5 tools
# Toolkits execute in parallel
```

### 4. Don't Share Mutable State Across Toolkits

```python
# ❌ Bad: Shared mutable global
global_cache = {}

@tool()
def toolkit1_tool():
    global_cache["key"] = "value"  # RACE CONDITION
    return "OK", {}

@tool()
def toolkit2_tool():
    value = global_cache.get("key")  # May be None or "value"
    return value, {}

# ✅ Good: Use toolkit context
@tool()
def toolkit1_tool():
    return "OK", {"key": "value"}  # Isolated context
```

### 5. Handle Parallel Execution Failures

```python
# When independent tools run in parallel, some may fail

@tool()
def may_fail():
    if random.random() < 0.5:
        raise ValueError("Failed")
    return "Success", {}

# Agent handles partial failures gracefully
# Failed tools return error messages to LLM
```

### 6. Test Concurrent Execution

```python
def test_concurrent_runs():
    agent = Agent(provider=client, model="gpt-4")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(agent.run, f"Query {i}") for i in range(10)]
        results = [f.result() for f in futures]

    assert len(results) == 10  # All completed
    # No race conditions or errors
```

### 7. Monitor Parallel Performance

```python
import time

@tool()
def slow_tool_a():
    time.sleep(2)
    return "A", {}

@tool()
def slow_tool_b():
    time.sleep(2)
    return "B", {}

agent.add_tool("tool_a", slow_tool_a)
agent.add_tool("tool_b", slow_tool_b)

# If LLM calls both:
# Sequential: 4 seconds
# Parallel (Yggdrasil): 2 seconds
```

---

## Performance Implications

### Parallelism Benefits

**Independent Tools:**
- 10 tools, 1s each, parallel → ~1s total
- Same sequentially → 10s total

**Toolkit Groups:**
- 3 toolkits, 2s each, parallel → ~2s total
- Same sequentially → 6s total

### Sequential Overhead

**Within-Toolkit:**
- 5 tools, 1s each, sequential → 5s total
- No parallelism possible

**Recommendation**: Keep toolkits small (3-7 tools) to minimize sequential overhead.

### Optimal Structure

```python
# ✅ Good: Balanced parallelism
agent.add_tool("independent1", tool1)  # Tier 1: Parallel
agent.add_tool("independent2", tool2)  # Tier 1: Parallel

agent.add_toolkit("auth", auth_toolkit)    # Tier 2: Parallel with data_toolkit
agent.add_toolkit("data", data_toolkit)    # Tier 2: Parallel with auth_toolkit
# Within each toolkit: Tier 3 sequential (3-5 tools each)

# Total parallelism: 2 independent + 2 toolkits = ~4x speedup
```

---

## Concurrency Guarantees

### What Yggdrasil Guarantees

1. **No Race Conditions**: Toolkit state is never accessed concurrently
2. **Sequential Consistency**: Tools within toolkit execute in order
3. **Isolation**: ExecutionContexts are independent
4. **Thread Safety**: Immutable context, deep copies

### What Yggdrasil Does NOT Guarantee

1. **Ordering Across Toolkits**: Toolkit A may finish before or after Toolkit B
2. **Independent Tool Order**: `tool1` and `tool2` may complete in any order
3. **Fair Scheduling**: No guarantees on thread pool scheduling

---

## Debugging Concurrency

### Enable Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Yggdrasil logs execution flow:
# DEBUG: Executing toolkit group: plane
# DEBUG: Executing toolkit group: github (parallel)
# DEBUG: Toolkit plane: tool select_project started
# DEBUG: Toolkit github: tool list_prs started
```

### Add Timing

```python
import time

@tool()
def timed_tool():
    start = time.time()
    result = do_work()
    print(f"Tool took {time.time() - start:.2f}s")
    return result, {}
```

### Trace Execution

```python
@tool()
def traced_tool():
    print(f"[{threading.current_thread().name}] Tool started")
    result = do_work()
    print(f"[{threading.current_thread().name}] Tool finished")
    return result, {}
```

---

**Next**: [Best Practices](best-practices.md) - Patterns and anti-patterns
