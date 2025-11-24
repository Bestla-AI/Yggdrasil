# Core Concepts

This document provides a deep dive into Yggdrasil's architecture and core components.

## Table of Contents
- [Agent](#agent)
- [ExecutionContext](#executioncontext)
- [Toolkit](#toolkit)
- [Tool](#tool)
- [Context](#context)
- [ConversationContext](#conversationcontext)

---

## Agent

The **Agent** is a stateless orchestrator that manages toolkits and drives the LLM conversation loop.

### Design Philosophy

**Key Decision**: Agent is stateless; all state lives in ExecutionContext.

**Benefits:**
- **Reusability**: Single agent instance can handle multiple concurrent requests
- **Thread Safety**: No shared mutable state between runs
- **Flexibility**: Choose between stateless, single-context, or multi-context patterns
- **Clean Lifecycle**: State can be discarded after run completion

### Constructor

```python
Agent(
    provider: OpenAI,           # OpenAI client instance
    model: str,                 # Model name (e.g., "gpt-4")
    system_prompt: str = "",    # Optional system message
    max_iterations: int = 10    # Max tool-calling loops
)
```

### Key Methods

#### `add_toolkit(prefix: str, toolkit: Toolkit)`

Register a named toolkit with a namespace prefix.

```python
agent.add_toolkit("plane", plane_toolkit)
agent.add_toolkit("github", github_toolkit)

# Tools become:
# - plane::select_project
# - plane::list_issues
# - github::create_pr
```

**Namespace Benefits:**
- Prevents tool name collisions
- Semantic grouping for LLM
- Clear ownership in multi-toolkit scenarios

#### `add_tool(name: str, function: Callable, description: str = "")`

Add an independent tool (not part of any toolkit).

```python
agent.add_tool("add", add_function, "Add two numbers")
```

**Independent Tool Characteristics:**
- Execute in parallel with other independent tools
- No access to toolkit context
- Must be stateless or manage own state

#### `run(prompt: str, execution_context: ExecutionContext = None)`

Execute the agent with a user prompt.

```python
response, context = agent.run("Your query here")
```

**Parameters:**
- `prompt`: User's natural language query
- `execution_context`: Optional context to continue from previous run

**Returns:**
- `response`: LLM's final text response
- `context`: ExecutionContext containing conversation history, toolkit state, FSM states

**Execution Flow:**
1. Create or reuse ExecutionContext
2. Deep-copy all toolkits for isolation
3. Generate tool schemas from available tools (based on FSM states + context)
4. Send prompt + schemas to LLM
5. Loop:
   - If LLM returns text: return response
   - If LLM returns tool calls: group by toolkit and execute
   - Add tool results to conversation
   - Continue loop (up to max_iterations)

### Execution Context Patterns

#### Pattern 1: Stateless (Default)

```python
agent = Agent(provider=client, model="gpt-4")
r1, ctx1 = agent.run("Query 1")  # Fresh context
r2, ctx2 = agent.run("Query 2")  # Fresh context
```

**Use Case**: Independent queries, no continuity needed

#### Pattern 2: Single-Context Continuation

```python
agent = Agent(provider=client, model="gpt-4")
r1, ctx1 = agent.run("Login as alice")
r2, ctx2 = agent.run("Get profile", execution_context=ctx1)  # Continue
r3, ctx3 = agent.run("Logout", execution_context=ctx2)       # Continue
```

**Use Case**: Multi-turn conversations, workflow continuation

#### Pattern 3: Multi-Context Management

```python
agent = Agent(provider=client, model="gpt-4")

# User sessions
sessions = {}

def handle_request(user_id, prompt):
    ctx = sessions.get(user_id)
    response, ctx = agent.run(prompt, execution_context=ctx)
    sessions[user_id] = ctx
    return response
```

**Use Case**: Multi-user applications, session management

---

## ExecutionContext

The **ExecutionContext** is a per-run state container that isolates state for concurrent executions.

### Structure

```python
class ExecutionContext:
    toolkits: Dict[str, Toolkit]              # Named toolkits (deep copied)
    independent_toolkit: Toolkit              # Independent tools
    conversation: ConversationContext         # Message history
```

### Isolation Mechanism

When an agent runs, it creates deep copies of all toolkits:

```python
# Inside agent.run()
execution_context.toolkits = {
    prefix: toolkit.copy()  # Deep copy!
    for prefix, toolkit in self.toolkits.items()
}
```

**Why Deep Copy?**
- Parallel runs don't interfere
- Each run has independent FSM states
- Context updates are isolated
- Thread-safe without locking

### Lifecycle

```python
# Created automatically if not provided
response, ctx = agent.run("Query")

# Reused if provided
response, ctx = agent.run("Query", execution_context=ctx)

# Can be inspected after run
print(ctx.conversation.messages)  # All messages
print(ctx.toolkits["auth"].context.data)  # Toolkit state
print(ctx.toolkits["auth"].unlocked_states)  # FSM states
```

### Accessing State

```python
# After execution
response, ctx = agent.run("Login and get profile")

# Access toolkit context
user = ctx.toolkits["auth"].context.get("user")

# Check FSM states
is_authenticated = "authenticated" in ctx.toolkits["auth"].unlocked_states

# View conversation
for msg in ctx.conversation.messages:
    print(msg["role"], msg["content"])
```

---

## Toolkit

The **Toolkit** manages a collection of related tools with shared context and FSM logic.

### Design Philosophy

**Key Decisions:**
- Tools in a toolkit share a `Context` instance
- Tools execute **sequentially** within a toolkit
- FSM states control tool availability
- Context updates propagate immediately to next tool

### Constructor

```python
Toolkit(
    context: Context = None,           # Shared domain state
    validation_enabled: bool = False   # Enable context schema validation
)
```

### Key Methods

#### `add_tool(name: str, function: Callable)`

Add a tool to the toolkit.

```python
toolkit.add_tool("login", login_function)
```

#### `generate_schemas()`

Generate JSON schemas for all available tools based on current context and FSM states.

```python
schemas = toolkit.generate_schemas()
# Returns only tools whose FSM requirements are met
```

#### `is_tool_available(tool_name: str) -> bool`

Check if a tool is currently available.

```python
if toolkit.is_tool_available("get_profile"):
    # Can be called
```

**Availability Criteria:**
1. All `required_states` are in `unlocked_states`
2. No `forbidden_states` are in `unlocked_states`
3. All `required_context` keys exist in context

### Sequential Execution Pipeline

Tools within a toolkit execute sequentially:

```python
# LLM returns:
# - plane::select_project(id="alpha")
# - plane::list_issues()

# Execution:
result1, updates1 = select_project(id="alpha")
toolkit.context.update(updates1)  # Apply immediately
toolkit.update_states(...)        # Process state transitions

result2, updates2 = list_issues()
toolkit.context.update(updates2)
toolkit.update_states(...)
```

**Pipeline Guarantees:**
- Tool N+1 sees updates from Tool N
- FSM state changes from Tool N affect Tool N+1 availability
- If Tool N fails, pipeline stops (Tool N+1 doesn't execute)

### FSM State Management

```python
# Initial state
toolkit.unlocked_states = set()  # Empty

# Tool executes with enables_states=["authenticated"]
toolkit.unlocked_states.add("authenticated")

# Tool executes with disables_states=["authenticated"]
toolkit.unlocked_states.remove("authenticated")
```

### Context Filters

Filters transform context values for dynamic schemas:

```python
# Register filter
toolkit.register_filter("active_only", lambda items: [i for i in items if i.active])

# Use in dynamic type
@tool()
def select_user(name: DynamicFiltered[("users", "active_only")]):
    # Schema uses filtered list
    pass
```

---

## Tool

The **Tool** class wraps Python functions with metadata for agent use.

### Decorator

```python
@tool(
    description: str = "",              # Tool description for LLM
    required_context: List[str] = [],   # Context keys that must exist
    required_states: List[str] = [],    # FSM states that must be enabled
    forbidden_states: List[str] = [],   # FSM states that must NOT be enabled
    enables_states: List[str] = [],     # States to enable after execution
    disables_states: List[str] = []     # States to disable after execution
)
def my_tool(param: type) -> Tuple[Any, dict]:
    return result, context_updates
```

### Return Format

**All tools must return `(result, context_updates)` tuple:**

```python
@tool()
def my_tool(x: int):
    result = x * 2
    updates = {"last_result": result}
    return result, updates
```

**Context Updates:**
- Dictionary of key-value pairs
- Merged into toolkit context
- Available to subsequent tools

### Metadata Example

```python
@tool(
    description="Get user profile",
    required_states=["authenticated"],      # Must be logged in
    required_context=["user_id"],           # Must have user_id in context
    forbidden_states=["rate_limited"],      # Can't be rate limited
    enables_states=["profile_loaded"],      # Enable this state after success
    disables_states=["needs_refresh"]       # Disable this state
)
def get_profile():
    user_id = my_toolkit.context.get("user_id")
    profile = fetch_profile(user_id)
    return profile, {"profile": profile}
```

### Schema Generation

Tool generates JSON schema from type hints:

```python
@tool()
def my_tool(
    name: str,
    age: int,
    tags: List[str],
    metadata: Optional[dict] = None
):
    return "result", {}

# Generated schema:
{
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"}
            },
            "required": ["name", "age", "tags"]
        }
    }
}
```

### Dynamic Type Integration

```python
from bestla.yggdrasil import DynamicStr

@tool()
def select_item(name: DynamicStr["available_items"]):
    # Schema generated from context["available_items"]
    return name, {}
```

See [Dynamic Types](dynamic-types.md) for details.

---

## Context

The **Context** is a dictionary-like state container with optional validation.

### Design

**Key Technology**: Uses `immutables.Map` for thread-safe updates with structural sharing.

**Benefits:**
- Fast shallow copies (O(1) due to immutability)
- No locks needed for concurrent access
- Efficient memory usage (shared structure)

### Constructor

```python
Context(
    validation_enabled: bool = False,   # Enable schema validation
    initial_data: dict = None           # Initial state
)
```

### Basic Usage

```python
context = Context()

# Set values
context.set("user", "alice")
context.set("score", 100)

# Get values
user = context.get("user")  # "alice"
missing = context.get("missing", default="N/A")  # "N/A"

# Check existence
if context.has("user"):
    print("User exists")

# Update multiple
context.update({"user": "bob", "score": 200})

# Access all data
data = context.data  # Immutable map
```

### Nested Access

```python
context.set("project", {"id": "alpha", "name": "Project Alpha"})

# Nested get
project_id = context.get("project.id")  # "alpha"
```

### Schema Validation

```python
# Enable validation
context = Context(validation_enabled=True)

# Define schema
context.schema.define("priority", {
    "type": "integer",
    "minimum": 1,
    "maximum": 5
})

# Valid set
context.set("priority", 3)  # OK

# Invalid set
context.set("priority", 10)  # Raises ContextValidationError
```

**Schema Definition:**

```python
context.schema.define("user", {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0}
    },
    "required": ["name"]
})

context.set("user", {"name": "Alice", "age": 30})  # OK
context.set("user", {"age": 30})  # Error: missing "name"
```

### Copy

```python
# Shallow copy (fast due to immutability)
context2 = context.copy()

# Deep copy (for complete isolation)
context3 = context.deep_copy()
```

---

## ConversationContext

The **ConversationContext** manages message history and optional automatic compaction.

### Constructor

```python
ConversationContext(
    messages: List[dict] = None,        # Initial messages
    context_manager: ContextManager = None  # Optional compaction
)
```

### Basic Usage

```python
from bestla.yggdrasil import ConversationContext

conversation = ConversationContext()

# Add messages
conversation.add_message({
    "role": "user",
    "content": "Hello"
})

conversation.add_message({
    "role": "assistant",
    "content": "Hi there!"
})

# Access messages
for msg in conversation.messages:
    print(msg["role"], msg["content"])

# Clear history
conversation.clear()
```

### With ContextManager

```python
from bestla.yggdrasil import ConversationContext, ContextManager

# Create manager with token limit
manager = ContextManager(
    max_tokens=4000,
    strategies=["tool_result_clearing", "summarization"]
)

conversation = ConversationContext(context_manager=manager)

# Manager automatically compacts when token limit is exceeded
# See context-management.md for details
```

### Message Format

Messages follow OpenAI format:

```python
{
    "role": "user" | "assistant" | "tool" | "system",
    "content": "...",
    "tool_calls": [...],  # For assistant messages
    "tool_call_id": "...",  # For tool messages
}
```

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│ Agent (Stateless Orchestrator)                              │
│  - Manages toolkits                                         │
│  - Drives LLM conversation loop                             │
│  - Groups and executes tool calls                           │
└─────────────────────────┬───────────────────────────────────┘
                          │ creates/reuses
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ExecutionContext (Per-Run State Container)                  │
│  - Isolates state between concurrent runs                   │
│  - Deep copies toolkits for thread safety                   │
├─────────────────────────────────────────────────────────────┤
│  Named Toolkits:                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Toolkit (Sequential Execution)                      │   │
│  │  - Shared Context (domain state)                    │   │
│  │  - FSM States (tool availability)                   │   │
│  │  - Tools (with metadata)                            │   │
│  │  - Filters (for dynamic types)                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Independent Toolkit (Parallel Execution):                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Stateless Tools                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Conversation:                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ConversationContext (Message History)               │   │
│  │  - Optional ContextManager (Compaction)             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Query** → Agent
2. **Agent** generates schemas from available tools (FSM + context filtered)
3. **LLM** receives schemas and returns tool calls
4. **Agent** groups tool calls by toolkit
5. **Toolkits execute sequentially** (within each toolkit)
6. **Context updates** propagate immediately
7. **FSM states** update immediately
8. **Tool results** added to conversation
9. **Loop** continues until LLM returns text or max iterations reached

---

**Next**: [Dynamic Types](dynamic-types.md) - Learn about runtime schema generation
