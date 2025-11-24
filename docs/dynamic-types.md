# Dynamic Types

Dynamic types enable runtime schema generation based on context state. This allows tool parameters to adapt their constraints as the workflow progresses.

## Table of Contents
- [Overview](#overview)
- [Type Reference](#type-reference)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)

---

## Overview

### The Problem

Traditional tools have static schemas:

```python
@tool()
def select_issue(name: str):
    # LLM can pass ANY string
    # No validation against available issues
    return name, {}
```

**Issues:**
- LLM might hallucinate issue names
- No validation against actual data
- Error-prone user experience

### The Solution

Dynamic types generate schemas from context:

```python
@tool()
def select_issue(name: DynamicStr["issue_names"]):
    # Schema: {"type": "string", "enum": ["BUG-1", "FEAT-2", ...]}
    # Generated from context["issue_names"]
    return name, {}
```

**Benefits:**
- LLM sees actual valid options
- Type-safe parameter passing
- Schemas adapt as context changes

### How It Works

1. Toolkit context contains data: `context["issue_names"] = ["BUG-1", "FEAT-2"]`
2. Tool declares parameter: `name: DynamicStr["issue_names"]`
3. Schema generator reads context and produces: `{"type": "string", "enum": ["BUG-1", "FEAT-2"]}`
4. LLM receives constrained schema
5. As context changes, schema regenerates automatically

---

## Type Reference

### DynamicStr

Generate string enum from context list.

```python
from bestla.yggdrasil import DynamicStr

@tool()
def select_project(name: DynamicStr["projects"]):
    return name, {}
```

**Context:**
```python
context.set("projects", ["alpha", "beta", "gamma"])
```

**Generated Schema:**
```json
{
    "type": "string",
    "enum": ["alpha", "beta", "gamma"]
}
```

**Use Cases:**
- Selection from dynamic lists
- Enum-like parameters that change at runtime
- User/project/item selection

---

### DynamicInt

Generate integer with min/max constraints from context.

```python
from bestla.yggdrasil import DynamicInt

@tool()
def set_priority(level: DynamicInt["priority_range"]):
    return f"Priority set to {level}", {}
```

**Context:**
```python
context.set("priority_range", {"minimum": 1, "maximum": 5})
```

**Generated Schema:**
```json
{
    "type": "integer",
    "minimum": 1,
    "maximum": 5
}
```

**Use Cases:**
- Range-based inputs (priority, rating, quantity)
- Bounded numeric parameters
- Dynamic limits based on permissions

---

### DynamicFloat

Generate float with min/max constraints from context.

```python
from bestla.yggdrasil import DynamicFloat

@tool()
def set_rating(score: DynamicFloat["rating_range"]):
    return f"Rating: {score}", {}
```

**Context:**
```python
context.set("rating_range", {
    "minimum": 0.0,
    "maximum": 5.0,
    "exclusiveMinimum": False,
    "exclusiveMaximum": False
})
```

**Generated Schema:**
```json
{
    "type": "number",
    "minimum": 0.0,
    "maximum": 5.0
}
```

**Use Cases:**
- Decimal ratings
- Percentage inputs
- Continuous ranges

---

### DynamicArray

Generate array with enum items from context.

```python
from bestla.yggdrasil import DynamicArray

@tool()
def assign_tags(tags: DynamicArray["available_tags"]):
    return f"Tagged with {tags}", {}
```

**Context:**
```python
context.set("available_tags", ["bug", "feature", "urgent", "docs"])
```

**Generated Schema:**
```json
{
    "type": "array",
    "items": {
        "type": "string",
        "enum": ["bug", "feature", "urgent", "docs"]
    }
}
```

**Use Cases:**
- Multi-select from predefined options
- Tag assignment
- Bulk operations on enumerated items

---

### DynamicFormat

Generate string with format constraint.

```python
from bestla.yggdrasil import DynamicFormat

@tool()
def schedule_meeting(date: DynamicFormat["date"]):
    return f"Scheduled for {date}", {}
```

**Generated Schema:**
```json
{
    "type": "string",
    "format": "date"
}
```

**Supported Formats:**
- `date` (e.g., "2025-01-15")
- `date-time` (e.g., "2025-01-15T14:30:00Z")
- `email`
- `uri`
- `uuid`
- `ipv4`, `ipv6`

**Use Cases:**
- Date/time inputs
- Email validation
- URL parameters

---

### DynamicPattern

Generate string with regex pattern from context.

```python
from bestla.yggdrasil import DynamicPattern

@tool()
def create_ticket(ticket_id: DynamicPattern["ticket_pattern"]):
    return f"Created {ticket_id}", {}
```

**Context:**
```python
context.set("ticket_pattern", r"^[A-Z]+-\d+$")
```

**Generated Schema:**
```json
{
    "type": "string",
    "pattern": "^[A-Z]+-\\d+$"
}
```

**Use Cases:**
- Structured identifiers (JIRA-123, PR-456)
- Format validation
- Custom string constraints

---

### DynamicConst

Generate constant value from context.

```python
from bestla.yggdrasil import DynamicConst

@tool()
def process_project(project_id: DynamicConst["current_project"]):
    # LLM must use the current project ID
    return f"Processing {project_id}", {}
```

**Context:**
```python
context.set("current_project", "alpha")
```

**Generated Schema:**
```json
{
    "const": "alpha"
}
```

**Use Cases:**
- Implicit parameters (current user, active project)
- Single-value constraints
- Context-bound operations

---

### DynamicFiltered

Generate enum from filtered context list using custom filter.

```python
from bestla.yggdrasil import DynamicFiltered

@tool()
def select_user(name: DynamicFiltered[("users", "active_only")]):
    return f"Selected {name}", {}
```

**Context:**
```python
context.set("users", [
    {"name": "alice", "active": True},
    {"name": "bob", "active": False},
    {"name": "charlie", "active": True}
])
```

**Filter Registration:**
```python
toolkit.register_filter(
    "active_only",
    lambda users: [u["name"] for u in users if u["active"]]
)
```

**Generated Schema:**
```json
{
    "type": "string",
    "enum": ["alice", "charlie"]
}
```

**Use Cases:**
- Filtered lists (active users, available resources)
- Conditional options
- Permission-based filtering

---

### DynamicNested

Access nested context values using dot notation.

```python
from bestla.yggdrasil import DynamicNested

@tool()
def update_status(status: DynamicNested["project.allowed_statuses"]):
    return f"Status: {status}", {}
```

**Context:**
```python
context.set("project", {
    "id": "alpha",
    "allowed_statuses": ["open", "in_progress", "closed"]
})
```

**Generated Schema:**
```json
{
    "type": "string",
    "enum": ["open", "in_progress", "closed"]
}
```

**Use Cases:**
- Hierarchical data
- Nested configuration
- Complex state structures

---

### DynamicConstraints

Apply arbitrary JSON schema from context.

```python
from bestla.yggdrasil import DynamicConstraints

@tool()
def configure(settings: DynamicConstraints["settings_schema"]):
    return f"Configured with {settings}", {}
```

**Context:**
```python
context.set("settings_schema", {
    "type": "object",
    "properties": {
        "theme": {"type": "string", "enum": ["light", "dark"]},
        "notifications": {"type": "boolean"}
    },
    "required": ["theme"]
})
```

**Generated Schema:**
```json
{
    "type": "object",
    "properties": {
        "theme": {"type": "string", "enum": ["light", "dark"]},
        "notifications": {"type": "boolean"}
    },
    "required": ["theme"]
}
```

**Use Cases:**
- Complex object schemas
- Dynamic validation rules
- Flexible configuration

---

### DynamicUnion

Combine multiple context keys into a single enum.

```python
from bestla.yggdrasil import DynamicUnion

@tool()
def select_item(name: DynamicUnion[("projects", "teams")]):
    return f"Selected {name}", {}
```

**Context:**
```python
context.set("projects", ["alpha", "beta"])
context.set("teams", ["engineering", "design"])
```

**Generated Schema:**
```json
{
    "type": "string",
    "enum": ["alpha", "beta", "engineering", "design"]
}
```

**Use Cases:**
- Combined selection pools
- Multi-source options
- Unified namespaces

---

### DynamicConditional

Conditionally select schema based on context value.

```python
from bestla.yggdrasil import DynamicConditional

@tool()
def create_item(
    item_type: DynamicConditional[("mode", "admin_types", "user_types")]
):
    return f"Created {item_type}", {}
```

**Context:**
```python
# When mode is "admin"
context.set("mode", "admin")
context.set("admin_types", ["project", "team", "user"])
context.set("user_types", ["issue", "comment"])

# Schema uses admin_types
```

**Generated Schema (when mode == "admin"):**
```json
{
    "type": "string",
    "enum": ["project", "team", "user"]
}
```

**Generated Schema (when mode != "admin"):**
```json
{
    "type": "string",
    "enum": ["issue", "comment"]
}
```

**Use Cases:**
- Permission-based schemas
- Mode-dependent options
- Role-based parameters

---

## Advanced Usage

### Combining Dynamic Types with Static

```python
@tool()
def create_issue(
    title: str,                          # Static
    project: DynamicStr["projects"],     # Dynamic enum
    priority: DynamicInt["priority_range"],  # Dynamic range
    tags: DynamicArray["available_tags"] # Dynamic array
):
    return f"Created issue in {project}", {}
```

### Optional Dynamic Parameters

```python
from typing import Optional

@tool()
def update_issue(
    name: DynamicStr["issues"],
    assignee: Optional[DynamicStr["users"]] = None
):
    # assignee is optional but constrained when provided
    return f"Updated {name}", {}
```

### Multiple Filters

```python
# Register multiple filters
toolkit.register_filter("active", lambda items: [i for i in items if i.active])
toolkit.register_filter("admin", lambda items: [i for i in items if i.is_admin])

@tool()
def select_active_user(name: DynamicFiltered[("users", "active")]):
    pass

@tool()
def select_admin(name: DynamicFiltered[("users", "admin")]):
    pass
```

### Dynamic Schemas with Validation

```python
# Enable context validation
toolkit = Toolkit(context=Context(validation_enabled=True))

# Define schema for context
toolkit.context.schema.define("projects", {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1
})

# Now DynamicStr["projects"] will always have valid data
@tool()
def select_project(name: DynamicStr["projects"]):
    pass
```

---

## Best Practices

### 1. Keep Context Fresh

Dynamic types are only as good as the context data:

```python
# ❌ Bad: Stale data
context.set("issues", ["ISS-1"])  # Set once, never updated

# ✅ Good: Update context as data changes
@tool(enables_states=["project_selected"])
def select_project(id: str):
    issues = fetch_issues(id)  # Get fresh data
    return id, {"issues": [i.name for i in issues]}
```

### 2. Use Filters for Complex Transformations

```python
# ❌ Bad: Store transformed data in context
transformed = [u["name"] for u in users if u.active]
context.set("active_user_names", transformed)

# ✅ Good: Use filters
context.set("users", users)  # Store raw data
toolkit.register_filter("active_names",
    lambda users: [u["name"] for u in users if u.active])

@tool()
def select_user(name: DynamicFiltered[("users", "active_names")]):
    pass
```

### 3. Validate Context Before Schema Generation

```python
@tool()
def list_issues():
    issues = fetch_issues()

    if not issues:
        # Don't set empty list (would generate empty enum)
        return "No issues found", {}

    return issues, {"issue_names": [i.name for i in issues]}
```

### 4. Document Dynamic Type Dependencies

```python
@tool(required_context=["projects"])  # Explicit requirement
def select_project(name: DynamicStr["projects"]):
    """
    Select a project from available projects.

    Requires:
        - context["projects"]: List of project names
    """
    return name, {}
```

### 5. Use DynamicConst for Implicit Context

```python
# ❌ Bad: Force LLM to provide known value
@tool(required_context=["current_user"])
def get_my_profile(user_id: str):
    # LLM has to know current_user value
    pass

# ✅ Good: Use DynamicConst
@tool()
def get_my_profile(user_id: DynamicConst["current_user"]):
    # LLM sees: {"const": "alice"}
    # Automatically bound to current user
    pass
```

### 6. Avoid Over-Constraining

```python
# ❌ Too specific: LLM has no flexibility
@tool()
def create_issue(
    title: DynamicConst["default_title"],  # Title is always the same?
    project: DynamicStr["projects"],
    priority: DynamicInt["priority_range"]
):
    pass

# ✅ Balance: Constrain what matters
@tool()
def create_issue(
    title: str,  # Free text
    project: DynamicStr["projects"],  # Constrained to valid projects
    priority: DynamicInt["priority_range"]  # Constrained to valid range
):
    pass
```

### 7. Handle Missing Context Gracefully

Dynamic types require context keys to exist. Design workflows to populate context before tools need it:

```python
# Workflow design
@tool(enables_states=["authenticated"])
def login(username: str):
    return "Logged in", {"user": username, "permissions": ["read", "write"]}

@tool(
    required_states=["authenticated"],
    required_context=["permissions"]  # Ensured by login
)
def select_action(action: DynamicStr["permissions"]):
    return action, {}
```

---

## Complete Example

```python
from bestla.yggdrasil import (
    Agent, Toolkit, Context, tool,
    DynamicStr, DynamicInt, DynamicArray
)

# Create toolkit
project_toolkit = Toolkit(context=Context())

# Tool 1: Select project (populates issues)
@tool(
    description="Select a project",
    enables_states=["project_selected"]
)
def select_project(name: DynamicStr["projects"]):
    # Simulate fetching issues
    issues = {
        "alpha": ["ALPHA-1", "ALPHA-2"],
        "beta": ["BETA-1"]
    }
    return f"Selected {name}", {
        "current_project": name,
        "issues": issues.get(name, [])
    }

# Tool 2: Select issue (requires project selection)
@tool(
    description="Select an issue",
    required_states=["project_selected"],
    required_context=["issues"],
    enables_states=["issue_selected"]
)
def select_issue(name: DynamicStr["issues"]):
    return f"Selected {name}", {"current_issue": name}

# Tool 3: Set priority (requires issue selection)
@tool(
    description="Set issue priority",
    required_states=["issue_selected"],
    required_context=["current_issue"]
)
def set_priority(level: DynamicInt["priority_range"]):
    issue = project_toolkit.context.get("current_issue")
    return f"Set {issue} priority to {level}", {}

# Add tools to toolkit
project_toolkit.add_tool("select_project", select_project)
project_toolkit.add_tool("select_issue", select_issue)
project_toolkit.add_tool("set_priority", set_priority)

# Initialize context
project_toolkit.context.set("projects", ["alpha", "beta"])
project_toolkit.context.set("priority_range", {"minimum": 1, "maximum": 5})

# Create agent
agent = Agent(provider=client, model="gpt-4")
agent.add_toolkit("project", project_toolkit)

# Run workflow
response, ctx = agent.run(
    "Select project alpha, then select issue ALPHA-1, then set priority to 3"
)
```

**Execution Flow:**
1. Initial schemas:
   - `select_project`: `name: {"enum": ["alpha", "beta"]}`
   - `select_issue`: NOT AVAILABLE (requires "project_selected" state)
   - `set_priority`: NOT AVAILABLE (requires "issue_selected" state)

2. After `select_project("alpha")`:
   - Context: `{"current_project": "alpha", "issues": ["ALPHA-1", "ALPHA-2"]}`
   - `select_issue` becomes available: `name: {"enum": ["ALPHA-1", "ALPHA-2"]}`

3. After `select_issue("ALPHA-1")`:
   - Context: `{"current_issue": "ALPHA-1", ...}`
   - `set_priority` becomes available: `level: {"minimum": 1, "maximum": 5}`

4. After `set_priority(3)`:
   - Workflow complete

---

**Next**: [State Management](state-management.md) - FSM patterns and workflows
