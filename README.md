# Bestla Yggdrasil

A stateful, hierarchical multi-agent tool framework for building sophisticated AI applications.

## Overview

Yggdrasil enables AI agents to use tools that:
- **Maintain state** across multiple interactions
- **Coordinate through finite state machines** (unlock/lock mechanisms)
- **Execute safely** with a three-tier concurrency model
- **Compose hierarchically** (agents can use other agents as tools)
- **Generate dynamic schemas** based on runtime context

### Core Innovation

Traditional AI tool frameworks treat tools as stateless functions. Yggdrasil introduces **stateful toolkits** where tool availability and parameters dynamically change based on execution history and context, enabling complex workflows like authenticated sessions, multi-step transactions, and project-scoped operations.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Example: Simple Tools

```python
from bestla.yggdrasil import Agent

# Create an agent
agent = Agent(model="gpt-4")

# Add independent tools (stateless)
def add(a: int, b: int):
    return a + b, {}

def multiply(a: int, b: int):
    return a * b, {}

agent.add_tool("add", add)
agent.add_tool("multiply", multiply)

# Run the agent
response = agent.run("What is 5 + 3, then multiply the result by 2?")
print(response)  # "The answer is 16"
```

### Stateful Toolkit Example

```python
from bestla.yggdrasil import Agent, Toolkit, tool, DynamicStr
from typing import Tuple

class ProjectToolkit(Toolkit):
    def __init__(self):
        super().__init__(validation_enabled=True)

        # Define context schema
        self.context.schema.define("selected_project", {"type": "string"})
        self.context.schema.define("issue_names", {
            "type": "array",
            "items": {"type": "string"}
        })

        # Add tools
        self.register_tool(select_project_tool)
        self.register_tool(list_issues_tool)
        self.register_tool(get_issue_tool)

        # Set initial availability
        self.set_available_tools({"select_project"})

# Define tools with metadata
@tool(
    unlocks=["list_issues"],
    provides=["selected_project"]
)
def select_project(project_id: str) -> Tuple[str, dict]:
    """Select a project to work with."""
    return f"Selected project {project_id}", {"selected_project": project_id}

@tool(
    requires=["selected_project"],
    provides=["issue_names"],
    unlocks=["get_issue"]
)
def list_issues() -> Tuple[str, dict]:
    """List all issues in the selected project."""
    # In real app, fetch from API
    issues = ["BUG-1", "BUG-2", "FEAT-3"]
    return f"Found {len(issues)} issues", {"issue_names": issues}

@tool(
    requires=["selected_project", "issue_names"]
)
def get_issue(name: DynamicStr["issue_names"]) -> Tuple[str, dict]:
    """Get details of a specific issue."""
    # name parameter will have an enum generated from context["issue_names"]
    return f"Details for {name}: ...", {}

# Use the toolkit
agent = Agent()
agent.add_toolkit("project", ProjectToolkit())

response = agent.run("Show me the issues in project-123 and get details for BUG-1")
```

## Key Concepts

### 1. Context

Toolkits own context (state) that persists across tool executions:

```python
from bestla.yggdrasil import Context

context = Context(validation_enabled=True)
context.schema.define("user_id", {"type": "string"})

# Set values
context.set("user_id", "user-123")

# Get values (supports nested access)
user_id = context.get("user_id")

# Nested access
context.set("project", {"id": "p1", "name": "My Project"})
project_id = context.get("project.id")  # "p1"
```

### 2. Dynamic Types

Parameters can have schemas that change based on context:

```python
from bestla.yggdrasil import DynamicStr, DynamicInt, DynamicArray

def select_issue(
    name: DynamicStr["issue_names"],  # Enum from context
    priority: DynamicInt["priority_range"]  # Min/max from context
):
    ...

# Context provides values:
# {"issue_names": ["BUG-1", "FEAT-2"], "priority_range": {"minimum": 1, "maximum": 5}}
```

Available dynamic types:
- `DynamicStr["key"]` - String enum from context
- `DynamicInt["key"]` - Integer with min/max constraints
- `DynamicFloat["key"]` - Float with constraints
- `DynamicArray["key"]` - Array with item enum
- `DynamicFormat["format"]` - String with format (e.g., "date-time")
- `DynamicPattern["key"]` - String with regex pattern
- `DynamicConst["key"]` - Constant value from context
- `DynamicFiltered[("key", "filter")]` - Filtered values
- `DynamicNested["key.subkey"]` - Nested context access
- `DynamicConstraints["key"]` - Generic schema from context
- `DynamicUnion[("key1", "key2")]` - Combine multiple context keys
- `DynamicConditional[("cond", "true_key", "false_key")]` - Conditional type selection

### 3. Finite State Machine (FSM)

Tools can unlock/lock other tools to enforce workflows:

```python
@tool(unlocks=["authenticated_action"])
def login(username: str, password: str):
    # After login, authenticated actions become available
    return "Logged in", {"auth_token": "..."}

@tool(locks=["authenticated_action"], unlocks=["login"])
def logout():
    # After logout, authenticated actions are locked
    return "Logged out", {}

@tool(requires=["auth_token"])
def authenticated_action():
    # Only available when logged in
    return "Action performed", {}
```

### 4. Hierarchical Agents

Agents can use other agents as tools:

```python
# Create specialized agents
research_agent = Agent()
research_agent.add_toolkit("web", WebSearchToolkit())

planning_agent = Agent()
planning_agent.add_toolkit("project", ProjectToolkit())

# Main agent uses sub-agents
main_agent = Agent()
main_agent.add_tool("research", research_agent.execute)
main_agent.add_tool("plan", planning_agent.execute)

# Sub-agents have isolated context
response = main_agent.run("Research AI trends and create a project plan")
```

### 5. Three-Tier Concurrency

Yggdrasil executes tools safely:

1. **Independent tools** (no toolkit) run in parallel
2. **Tools within same toolkit** run sequentially (FSM dependencies)
3. **Different toolkits** run in parallel to each other

```python
# AI calls:
# - plane::select_project
# - plane::list_issues
# - github::list_prs
# - add(2, 3)

# Execution:
# - plane::* runs sequentially (select, then list)
# - github::* runs in parallel with plane::*
# - add() runs in parallel with everything
```

## Advanced Features

### Advanced Dynamic Types

#### DynamicUnion - Combine Multiple Context Keys

```python
from bestla.yggdrasil import DynamicUnion

# Context
context.update({
    "users": ["alice", "bob"],
    "admins": ["charlie", "diana"]
})

@tool()
def select_person(name: DynamicUnion[("users", "admins")]):
    # name will have enum: ["alice", "bob", "charlie", "diana"]
    return f"Selected {name}", {}
```

#### DynamicConditional - Conditional Type Selection

```python
from bestla.yggdrasil import DynamicConditional

# Context
context.update({
    "advanced_mode": True,
    "simple_options": ["opt1", "opt2"],
    "advanced_options": ["opt1", "opt2", "opt3", "opt4"]
})

@tool()
def configure(value: DynamicConditional[("advanced_mode", "advanced_options", "simple_options")]):
    # If advanced_mode is True: enum is ["opt1", "opt2", "opt3", "opt4"]
    # If advanced_mode is False: enum is ["opt1", "opt2"]
    return f"Configured with {value}", {}
```

### Decorators

Yggdrasil provides powerful decorators for common patterns:

#### @retry - Automatic Retry with Exponential Backoff

```python
from bestla.yggdrasil import retry, tool

@tool()
@retry(max_attempts=3, backoff=1.0, exceptions=(ConnectionError, TimeoutError))
def fetch_from_api(endpoint: str) -> Tuple[str, dict]:
    # Will retry up to 3 times on connection errors
    # Backoff: 1s, 2s, 4s (exponential)
    response = api.get(endpoint)
    return response, {}
```

#### @cache_result - Result Caching

```python
from bestla.yggdrasil import cache_result

@tool()
@cache_result(ttl=300.0)  # Cache for 5 minutes
def expensive_operation(query: str) -> Tuple[str, dict]:
    # Result cached for 5 minutes
    result = slow_computation(query)
    return result, {}

# Cache management
expensive_operation.clear_cache()
size = expensive_operation.cache_size()
```

#### @rate_limit - Rate Limiting

```python
from bestla.yggdrasil import rate_limit

@tool()
@rate_limit(calls=10, period=60.0)  # Max 10 calls per minute
def api_call(endpoint: str) -> Tuple[str, dict]:
    # Rate limited to prevent API abuse
    return api.call(endpoint), {}
```

#### @timeout - Execution Timeout

```python
from bestla.yggdrasil import timeout

@tool()
@timeout(5.0)  # Must complete within 5 seconds
def time_sensitive_operation() -> Tuple[str, dict]:
    return result, {}
```

#### Combining Decorators

```python
@tool(requires=["auth_token"])
@retry(max_attempts=3, backoff=1.0)
@cache_result(ttl=60.0)
@rate_limit(calls=5, period=60.0)
def robust_api_call(endpoint: str) -> Tuple[str, dict]:
    # This tool:
    # - Requires authentication
    # - Retries on failure (up to 3 times)
    # - Caches results (1 minute)
    # - Rate limited (5 calls/minute)
    return api.call(endpoint), {}
```

### Custom Filters

```python
toolkit = Toolkit()

# Register custom filter
toolkit.register_filter("active_only",
    lambda users: [u for u in users if u.get("active")]
)

# Use in tool
@tool()
def assign_user(user: DynamicFiltered[("users", "active_only")]):
    # Gets filtered list of active users only
    ...
```

### Context Validation

```python
toolkit = Toolkit(validation_enabled=True)

# Define schema
toolkit.context.schema.define("priority", {
    "type": "integer",
    "minimum": 1,
    "maximum": 5
})

# Invalid updates raise ContextValidationError
toolkit.context.set("priority", 10)  # ❌ Raises error
toolkit.context.set("priority", 3)   # ✅ Valid
```

### Tool Return Format

All tools MUST return `(result, context_updates)` tuple:

```python
def my_tool(x: int) -> Tuple[Any, dict]:
    result = x * 2
    context_updates = {"last_result": result}
    return result, context_updates

# Stateless tools return empty dict
def stateless_tool(x: int) -> Tuple[int, dict]:
    return x + 1, {}
```

## Architecture

```
Agent
├── Multiple Toolkits (with prefixes: plane::, github::)
│   ├── Context (domain state)
│   ├── Tools (with dependencies)
│   └── FSM Logic (unlocks/locks)
└── Independent Toolkit (stateless tools, always parallel)
```

## Design Philosophy

- **Organizational Model**: Agents delegate to sub-agents like managers delegate to workers
- **Context Ownership**: Each toolkit owns its domain state (isolation)
- **Explicit State Changes**: Tools explicitly declare what context they modify
- **Safety First**: Sequential execution within toolkits prevents race conditions
- **Performance Where Safe**: Independent operations run in parallel

## Examples

### Authentication Workflow

```python
class AuthToolkit(Toolkit):
    def __init__(self):
        super().__init__()

        @tool(unlocks=["get_profile", "update_settings"])
        def login(username: str, password: str):
            # Authenticate and unlock user actions
            return "Logged in", {"auth_token": "...", "user_id": username}

        @tool(requires=["auth_token"])
        def get_profile():
            return "Profile data", {}

        @tool(locks=["get_profile", "update_settings"], unlocks=["login"])
        def logout():
            return "Logged out", {}

        self.register_tool(login)
        self.register_tool(get_profile)
        self.register_tool(logout)
        self.set_available_tools({"login"})
```

### Project Management

See the stateful toolkit example above for a complete project management workflow.

### Multi-Agent Research

```python
# Specialist agents
web_agent = Agent()
web_agent.add_toolkit("search", WebSearchToolkit())

analysis_agent = Agent()
analysis_agent.add_toolkit("data", DataAnalysisToolkit())

# Orchestrator
coordinator = Agent()
coordinator.add_tool("research", web_agent.execute)
coordinator.add_tool("analyze", analysis_agent.execute)

result = coordinator.run(
    "Research quantum computing papers from 2024 and analyze the trends"
)
```

## Contributing

Contributions welcome! Please check the issues page.

## License

MIT

## Links

- **Homepage**: https://github.com/BestlaAI/Yggdrasil
- **Issues**: https://github.com/BestlaAI/Yggdrasil/issues
