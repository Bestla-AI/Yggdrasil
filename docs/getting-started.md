# Getting Started

This guide will get you up and running with Yggdrasil in minutes.

## Installation

```bash
pip install bestla-yggdrasil
```

**Requirements:**
- Python 3.10+
- OpenAI API key

**Dependencies:**
- `openai>=1.0.0`
- `jsonschema>=4.0.0`
- `immutables>=0.21`

## Your First Agent

### 1. Simple Stateless Tools

```python
from openai import OpenAI
from bestla.yggdrasil import Agent, tool

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key")

# Define tools
@tool(description="Add two numbers")
def add(a: int, b: int):
    """All tools must return (result, context_updates)"""
    return a + b, {}

@tool(description="Multiply two numbers")
def multiply(a: int, b: int):
    return a * b, {}

# Create agent
agent = Agent(
    provider=client,
    model="gpt-4",
    system_prompt="You are a helpful math assistant."
)

# Add independent tools (executed in parallel)
agent.add_tool("add", add)
agent.add_tool("multiply", multiply)

# Run query
response, context = agent.run("What is (5 + 3) * 2?")
print(response)
```

**Key Points:**
- All tools return `(result, context_updates)` tuple
- Independent tools run in parallel
- Agent is stateless; each `run()` is independent

### 2. Stateful Workflow with FSM

```python
from bestla.yggdrasil import Agent, Toolkit, Context, tool

# Create toolkit with context
auth_toolkit = Toolkit(context=Context())

# Tool 1: Login (enables "authenticated" state)
@tool(
    description="Login to the system",
    enables_states=["authenticated"]
)
def login(username: str, password: str):
    # Simulate authentication
    if password == "secret":
        return f"Logged in as {username}", {"user": username}
    return "Login failed", {}

# Tool 2: Get profile (requires "authenticated" state)
@tool(
    description="Get user profile",
    required_states=["authenticated"],
    required_context=["user"]  # Requires "user" in context
)
def get_profile():
    # Access context via toolkit
    username = auth_toolkit.context.get("user")
    return f"Profile for {username}", {}

# Tool 3: Logout (requires and disables "authenticated")
@tool(
    description="Logout from the system",
    required_states=["authenticated"],
    disables_states=["authenticated"]
)
def logout():
    return "Logged out", {"user": None}

# Add tools to toolkit
auth_toolkit.add_tool("login", login)
auth_toolkit.add_tool("get_profile", get_profile)
auth_toolkit.add_tool("logout", logout)

# Create agent with named toolkit
agent = Agent(provider=client, model="gpt-4")
agent.add_toolkit("auth", auth_toolkit)

# Run workflow
response, ctx = agent.run("Login as alice with password secret, then get my profile")
print(response)
```

**Key Points:**
- `enables_states`: Tool enables states after execution
- `required_states`: Tool only available when states are enabled
- `required_context`: Tool requires context keys to exist
- Tools within a toolkit execute sequentially
- Context updates propagate immediately to next tool

### 3. Dynamic Schemas

```python
from bestla.yggdrasil import Agent, Toolkit, Context, tool, DynamicStr

# Create toolkit with initial context
project_toolkit = Toolkit(context=Context())
project_toolkit.context.set("projects", ["alpha", "beta", "gamma"])

# Tool with dynamic parameter
@tool(description="Select a project")
def select_project(name: DynamicStr["projects"]):
    """
    name parameter schema is generated from context["projects"]
    Schema: {"type": "string", "enum": ["alpha", "beta", "gamma"]}
    """
    return f"Selected {name}", {"selected_project": name}

project_toolkit.add_tool("select_project", select_project)

agent = Agent(provider=client, model="gpt-4")
agent.add_toolkit("project", project_toolkit)

response, ctx = agent.run("Select project alpha")
print(response)
```

**Key Points:**
- `DynamicStr["projects"]` generates enum from `context["projects"]`
- Schema adapts as context changes
- LLM sees updated constraints in real-time

## Execution Patterns

### Pattern 1: Stateless (Independent Runs)

```python
agent = Agent(provider=client, model="gpt-4")
response1, ctx1 = agent.run("Query 1")
response2, ctx2 = agent.run("Query 2")  # Independent, no shared state
```

### Pattern 2: Stateful (Continued Conversation)

```python
agent = Agent(provider=client, model="gpt-4")
response1, ctx1 = agent.run("Login as alice")
response2, ctx2 = agent.run("Get my profile", execution_context=ctx1)  # Continues ctx1
```

### Pattern 3: Concurrent (Multiple Agents)

```python
from concurrent.futures import ThreadPoolExecutor

agent = Agent(provider=client, model="gpt-4")

with ThreadPoolExecutor() as executor:
    future1 = executor.submit(agent.run, "Query 1")
    future2 = executor.submit(agent.run, "Query 2")
    # Safe: separate ExecutionContexts
```

## Tool Return Format

**All tools must return a tuple: `(result, context_updates)`**

```python
@tool()
def my_tool(x: int):
    result = x * 2
    context_updates = {"last_result": result}
    return result, context_updates
```

**Context updates:**
- Dictionary of key-value pairs
- Merged into toolkit context after execution
- Available to subsequent tools in the pipeline

## Toolkit Naming

Toolkits are registered with prefixes:

```python
agent.add_toolkit("plane", plane_toolkit)
agent.add_toolkit("github", github_toolkit)

# Tools are namespaced:
# - plane::select_project
# - plane::list_issues
# - github::create_pr
```

**Benefits:**
- Avoid name collisions
- Clear tool organization
- LLM sees semantic grouping

## Next Steps

- [**Core Concepts**](core-concepts.md) - Deep dive into architecture
- [**Dynamic Types**](dynamic-types.md) - All 12+ dynamic type constructors
- [**State Management**](state-management.md) - FSM patterns and workflows
- [**Examples**](examples.md) - Complete real-world examples

## Common Gotchas

1. **Forgetting context_updates return value**
   ```python
   # ❌ Wrong
   @tool()
   def bad_tool(x: int):
       return x * 2

   # ✅ Correct
   @tool()
   def good_tool(x: int):
       return x * 2, {}
   ```

2. **Using toolkit context outside tools**
   ```python
   # ❌ Wrong
   @tool()
   def bad_tool():
       user = some_global_context.get("user")  # Don't do this
       return user, {}

   # ✅ Correct
   @tool(required_context=["user"])
   def good_tool():
       user = my_toolkit.context.get("user")  # Access via toolkit
       return user, {}
   ```

3. **Mixing parallel and sequential expectations**
   ```python
   # Independent tools run in parallel (no ordering guarantee)
   agent.add_tool("tool1", tool1)
   agent.add_tool("tool2", tool2)

   # Toolkit tools run sequentially (ordered execution)
   toolkit.add_tool("tool1", tool1)
   toolkit.add_tool("tool2", tool2)
   ```
