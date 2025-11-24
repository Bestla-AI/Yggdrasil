# Context Management

Long-running conversations can exceed LLM token limits. Yggdrasil provides automatic conversation compaction through the `ContextManager` to keep conversations within bounds.

## Table of Contents
- [Problem](#problem)
- [ContextManager](#contextmanager)
- [Compaction Strategies](#compaction-strategies)
- [Usage](#usage)
- [Best Practices](#best-practices)

---

## Problem

### Token Limit Exhaustion

LLMs have finite context windows:

| Model | Context Window |
|-------|----------------|
| GPT-4 | 8K - 128K tokens |
| GPT-3.5 | 4K - 16K tokens |

Long conversations accumulate messages:

```
User: Query 1
Assistant: Response 1 (+ tool calls & results)
User: Query 2
Assistant: Response 2 (+ tool calls & results)
...
User: Query 50
Assistant: Response 50 (+ tool calls & results)
```

**Problem**: After many turns, conversation exceeds token limit.

### Without Compaction

```python
agent = Agent(provider=client, model="gpt-4")
ctx = None

for i in range(100):
    response, ctx = agent.run(f"Query {i}", execution_context=ctx)

# Eventually: openai.error.InvalidRequestError: maximum context length exceeded
```

### With Compaction

```python
from bestla.yggdrasil import ContextManager, ConversationContext

manager = ContextManager(
    max_tokens=4000,
    strategies=["tool_result_clearing"]
)

conversation = ConversationContext(context_manager=manager)
ctx = ExecutionContext(conversation=conversation)

for i in range(100):
    response, ctx = agent.run(f"Query {i}", execution_context=ctx)
    # Manager automatically compacts when approaching limit
```

---

## ContextManager

The `ContextManager` monitors token usage and applies compaction strategies when thresholds are exceeded.

### Constructor

```python
from bestla.yggdrasil import ContextManager

ContextManager(
    max_tokens: int,                      # Maximum token budget
    strategies: List[str] = [],           # Compaction strategies to apply
    warning_threshold: float = 0.8,       # Warn at 80% of max_tokens
    compaction_threshold: float = 0.9     # Compact at 90% of max_tokens
)
```

### Parameters

- **max_tokens**: Hard token limit for conversation
- **strategies**: List of compaction strategies (applied in order)
  - `"tool_result_clearing"`: Remove old tool result messages
  - `"summarization"`: Summarize old messages to reduce tokens
- **warning_threshold**: Fraction of max_tokens at which to log warning (default 0.8)
- **compaction_threshold**: Fraction of max_tokens at which to compact (default 0.9)

### Token Monitoring

ContextManager estimates token usage:

```python
manager = ContextManager(max_tokens=4000)

# After each message addition
current_tokens = manager.estimate_tokens(messages)

if current_tokens > 0.9 * max_tokens:  # 90% threshold
    manager.compact(messages)
```

**Token Estimation**: Uses a simple heuristic (characters / 4). For precise counting, consider integrating `tiktoken`.

---

## Compaction Strategies

### tool_result_clearing

Removes old tool result messages while preserving conversation flow.

**What it removes:**
- Tool call messages (role: "tool")
- Assistant messages containing tool_calls (but keeps text content)

**What it keeps:**
- System messages
- Recent user/assistant text messages
- Latest N tool results (configurable)

**Example:**

```
Before compaction (10 messages):
1. system: "You are a helpful assistant"
2. user: "Login and get profile"
3. assistant: [tool_calls: login()]
4. tool: login result
5. assistant: [tool_calls: get_profile()]
6. tool: get_profile result
7. assistant: "Here's your profile"
8. user: "Update my name"
9. assistant: [tool_calls: update_profile()]
10. tool: update_profile result

After tool_result_clearing (keeps latest 2 tools):
1. system: "You are a helpful assistant"
2. user: "Login and get profile"
3. assistant: "Here's your profile"  # Tool calls removed
4. user: "Update my name"
5. assistant: [tool_calls: update_profile()]
6. tool: update_profile result
```

**Benefit**: Preserves conversation coherence while reducing tokens.

### summarization

Summarizes old messages to reduce token count.

**How it works:**
1. Identify old message range (older than N recent messages)
2. Send old messages to LLM with summarization prompt
3. Replace old messages with summary

**Example:**

```
Before summarization (15 messages):
1. user: "What's the weather?"
2. assistant: "It's sunny, 72°F"
3. user: "Should I bring an umbrella?"
4. assistant: "No, no rain expected"
5. user: "What about tomorrow?"
6. assistant: "Tomorrow will be cloudy"
...
15. user: "Current query"

After summarization (keep last 5, summarize rest):
1. system: "Summary: User asked about weather. Assistant provided
            forecast for today (sunny, 72°F) and tomorrow (cloudy).
            User asked about umbrella, advised not needed."
2. [messages 11-15]  # Last 5 messages preserved
```

**Configuration:**

```python
manager = ContextManager(
    max_tokens=4000,
    strategies=["summarization"],
    summarization_config={
        "keep_recent": 10,  # Keep last 10 messages
        "model": "gpt-3.5-turbo"  # Model for summarization
    }
)
```

**Benefit**: Retains semantic information while drastically reducing tokens.

### Strategy Combination

Apply multiple strategies in sequence:

```python
manager = ContextManager(
    max_tokens=4000,
    strategies=[
        "tool_result_clearing",  # First, remove old tool results
        "summarization"          # Then, summarize remaining old messages
    ]
)
```

**Execution Order:**
1. tool_result_clearing reduces conversation by removing tool messages
2. If still over threshold, summarization condenses old messages

---

## Usage

### Basic Setup

```python
from bestla.yggdrasil import (
    Agent, ExecutionContext, ConversationContext, ContextManager
)

# Create manager
manager = ContextManager(
    max_tokens=4000,
    strategies=["tool_result_clearing"]
)

# Create conversation with manager
conversation = ConversationContext(context_manager=manager)

# Create execution context with managed conversation
execution_context = ExecutionContext(conversation=conversation)

# Create agent
agent = Agent(provider=client, model="gpt-4")

# Use execution context for all runs
response, ctx = agent.run("First query", execution_context=execution_context)
response, ctx = agent.run("Second query", execution_context=ctx)
```

### Automatic Compaction

Compaction happens automatically when threshold is exceeded:

```python
manager = ContextManager(
    max_tokens=1000,
    compaction_threshold=0.9,  # Compact at 900 tokens
    strategies=["tool_result_clearing"]
)

conversation = ConversationContext(context_manager=manager)

# Add messages
for i in range(50):
    conversation.add_message({"role": "user", "content": f"Message {i}"})
    # When tokens exceed 900, manager automatically compacts
```

### Manual Compaction

Force compaction at any time:

```python
manager = ContextManager(max_tokens=4000, strategies=["summarization"])
conversation = ConversationContext(context_manager=manager)

# Add many messages
for i in range(100):
    conversation.add_message({"role": "user", "content": f"Message {i}"})

# Manually compact
manager.compact(conversation.messages)
```

### Monitoring Token Usage

```python
manager = ContextManager(max_tokens=4000)
conversation = ConversationContext(context_manager=manager)

# Check current usage
current = manager.estimate_tokens(conversation.messages)
print(f"Using {current}/{manager.max_tokens} tokens ({current/manager.max_tokens*100:.1f}%)")

# Check if approaching limit
if current > manager.warning_threshold * manager.max_tokens:
    print("Warning: Approaching token limit")
```

---

## Best Practices

### 1. Set Appropriate Token Limits

```python
# ❌ Bad: Limit higher than model capacity
manager = ContextManager(max_tokens=200000)  # GPT-4 is 128K max!

# ✅ Good: Leave headroom for responses
manager = ContextManager(max_tokens=6000)  # For 8K model, leaves 2K for response
```

### 2. Choose Strategies Based on Use Case

```python
# For tool-heavy workflows
manager = ContextManager(
    max_tokens=4000,
    strategies=["tool_result_clearing"]  # Remove old tool results
)

# For conversation-heavy workflows
manager = ContextManager(
    max_tokens=4000,
    strategies=["summarization"]  # Preserve conversation semantics
)

# For both
manager = ContextManager(
    max_tokens=4000,
    strategies=["tool_result_clearing", "summarization"]
)
```

### 3. Preserve System Messages

System messages are never removed by compaction strategies:

```python
conversation.add_message({
    "role": "system",
    "content": "You are a helpful assistant. [Important instructions]"
})
# System message always preserved
```

### 4. Monitor Compaction Events

```python
import logging

logging.basicConfig(level=logging.INFO)

manager = ContextManager(
    max_tokens=4000,
    strategies=["tool_result_clearing"]
)

# Manager logs compaction events:
# INFO: Token usage at 91% (3640/4000), applying compaction
# INFO: tool_result_clearing: Removed 15 tool messages
# INFO: Token usage after compaction: 68% (2720/4000)
```

### 5. Test Compaction Logic

```python
def test_compaction():
    manager = ContextManager(
        max_tokens=100,  # Small limit for testing
        compaction_threshold=0.9,
        strategies=["tool_result_clearing"]
    )

    conversation = ConversationContext(context_manager=manager)

    # Add messages until compaction triggers
    for i in range(50):
        conversation.add_message({
            "role": "user",
            "content": "x" * 10  # ~10 tokens each
        })

    # Verify compaction occurred
    tokens = manager.estimate_tokens(conversation.messages)
    assert tokens < 100  # Under limit
```

### 6. Balance History vs. Tokens

```python
# ❌ Bad: Too aggressive compaction (loses context)
manager = ContextManager(
    max_tokens=4000,
    compaction_threshold=0.5,  # Compacts at 50%!
    strategies=["summarization"]
)

# ✅ Good: Reasonable threshold
manager = ContextManager(
    max_tokens=4000,
    compaction_threshold=0.9,  # Compacts at 90%
    strategies=["tool_result_clearing"]
)
```

### 7. Custom Compaction Strategy

Implement custom strategy for specific needs:

```python
class CustomContextManager(ContextManager):
    def compact(self, messages):
        """Custom compaction logic"""
        # Remove all user messages older than 10 messages
        recent = messages[-10:]
        system_msgs = [m for m in messages if m["role"] == "system"]
        return system_msgs + recent
```

### 8. Handle Compaction Failures Gracefully

```python
try:
    response, ctx = agent.run("Query", execution_context=ctx)
except Exception as e:
    if "maximum context length" in str(e):
        # Compaction didn't help enough
        # Option 1: Start fresh conversation
        ctx = None
        response, ctx = agent.run("Query")

        # Option 2: More aggressive compaction
        manager.max_tokens = manager.max_tokens // 2
        manager.compact(ctx.conversation.messages)
```

---

## Advanced Configuration

### Custom Token Estimation

Override default token estimation:

```python
import tiktoken

class TiktokenContextManager(ContextManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def estimate_tokens(self, messages):
        """Precise token counting with tiktoken"""
        total = 0
        for message in messages:
            total += len(self.encoder.encode(str(message)))
        return total
```

### Selective Message Preservation

Keep specific messages even during compaction:

```python
conversation.add_message({
    "role": "user",
    "content": "Important context",
    "metadata": {"preserve": True}  # Custom metadata
})

# Custom compaction that respects metadata
class SelectiveContextManager(ContextManager):
    def compact(self, messages):
        preserved = [m for m in messages if m.get("metadata", {}).get("preserve")]
        recent = messages[-10:]
        return preserved + recent
```

---

## Performance Considerations

### Compaction Overhead

- **tool_result_clearing**: O(n) message filtering (fast)
- **summarization**: LLM API call (slow, incurs cost)

**Recommendation**: Use `tool_result_clearing` first, `summarization` as last resort.

### Frequency

Compaction triggers when threshold exceeded. Tune threshold to balance:
- **Low threshold (0.5)**: Frequent compaction, minimal token usage
- **High threshold (0.95)**: Rare compaction, risk of exceeding limit

### Cost

Summarization strategy makes LLM API calls:
- Cost per compaction: 1 API call with old messages as input
- Use cheaper model for summarization: `gpt-3.5-turbo`

---

## Complete Example

```python
from bestla.yggdrasil import (
    Agent, ExecutionContext, ConversationContext, ContextManager,
    Toolkit, tool
)

# Create toolkit
toolkit = Toolkit()

@tool()
def search(query: str):
    return f"Results for {query}", {}

toolkit.add_tool("search", search)

# Create manager with both strategies
manager = ContextManager(
    max_tokens=2000,
    compaction_threshold=0.9,
    strategies=["tool_result_clearing", "summarization"]
)

# Create conversation with manager
conversation = ConversationContext(context_manager=manager)

# Create execution context
ctx = ExecutionContext(conversation=conversation)

# Create agent
agent = Agent(provider=client, model="gpt-4")
agent.add_toolkit("search", toolkit)

# Long conversation
for i in range(100):
    response, ctx = agent.run(f"Search for topic {i}", execution_context=ctx)
    print(f"Turn {i}: {len(ctx.conversation.messages)} messages")
    # Compaction happens automatically when needed
```

---

**Next**: [Concurrency](concurrency.md) - Three-tier execution model
