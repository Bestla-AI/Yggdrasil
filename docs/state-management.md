# State Management

Yggdrasil uses Finite State Machines (FSM) to control tool availability based on workflow state. This ensures tools are only called in valid sequences and prevents invalid operations.

## Table of Contents
- [FSM Fundamentals](#fsm-fundamentals)
- [State Metadata](#state-metadata)
- [Common Patterns](#common-patterns)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)

---

## FSM Fundamentals

### The Problem

Without state management, tools can be called in any order:

```python
# ❌ Nothing prevents this invalid sequence:
get_profile()  # ERROR: Not logged in
login()
logout()      # Ok
get_profile()  # ERROR: Already logged out
```

### The Solution

FSM controls tool availability based on abstract states:

```python
@tool(enables_states=["authenticated"])
def login():
    return "Logged in", {}

@tool(required_states=["authenticated"])
def get_profile():
    return "Profile data", {}

@tool(required_states=["authenticated"], disables_states=["authenticated"])
def logout():
    return "Logged out", {}
```

**How It Works:**

1. **Initial State**: `unlocked_states = set()` (empty)
2. **login()** executes → adds "authenticated" to `unlocked_states`
3. **get_profile()** becomes available (requires "authenticated")
4. **logout()** executes → removes "authenticated" from `unlocked_states`
5. **get_profile()** becomes unavailable again

### State Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│ Toolkit State Machine                                       │
│                                                              │
│  unlocked_states: Set[str] = set()                          │
│                                                              │
│  When tool executes:                                        │
│    1. Check: required_states ⊆ unlocked_states              │
│    2. Check: forbidden_states ∩ unlocked_states = ∅         │
│    3. Execute tool                                          │
│    4. unlocked_states |= enables_states                     │
│    5. unlocked_states -= disables_states                    │
└─────────────────────────────────────────────────────────────┘
```

---

## State Metadata

### Tool Decorator Parameters

```python
@tool(
    required_states: List[str] = [],    # States that MUST be enabled
    forbidden_states: List[str] = [],   # States that MUST NOT be enabled
    enables_states: List[str] = [],     # States to enable after execution
    disables_states: List[str] = []     # States to disable after execution
)
```

### required_states

Tool is only available when ALL required states are enabled.

```python
@tool(required_states=["authenticated", "admin"])
def delete_user():
    # Only available when BOTH states are enabled
    return "User deleted", {}
```

**Availability Logic:**
```python
is_available = all(
    state in toolkit.unlocked_states
    for state in tool.required_states
)
```

### forbidden_states

Tool is unavailable if ANY forbidden state is enabled.

```python
@tool(forbidden_states=["rate_limited", "maintenance_mode"])
def make_api_call():
    # Unavailable if rate limited OR in maintenance
    return "API call made", {}
```

**Availability Logic:**
```python
is_available = not any(
    state in toolkit.unlocked_states
    for state in tool.forbidden_states
)
```

### enables_states

States to enable after successful execution.

```python
@tool(enables_states=["authenticated", "session_active"])
def login():
    # Enables BOTH states after execution
    return "Logged in", {}
```

**Update Logic:**
```python
toolkit.unlocked_states.update(tool.enables_states)
```

### disables_states

States to disable after successful execution.

```python
@tool(disables_states=["authenticated", "session_active"])
def logout():
    # Disables BOTH states after execution
    return "Logged out", {}
```

**Update Logic:**
```python
toolkit.unlocked_states.difference_update(tool.disables_states)
```

### Combining Metadata

```python
@tool(
    required_states=["authenticated"],      # Must be logged in
    forbidden_states=["rate_limited"],      # Must not be rate limited
    enables_states=["profile_loaded"],      # Enable after success
    disables_states=["needs_refresh"]       # Disable after success
)
def get_profile():
    return "Profile", {}
```

---

## Common Patterns

### Authentication Workflow

```python
from bestla.yggdrasil import Toolkit, tool

auth_toolkit = Toolkit()

@tool(
    description="Login to the system",
    enables_states=["authenticated"]
)
def login(username: str, password: str):
    # Authenticate user
    if authenticate(username, password):
        return f"Logged in as {username}", {"user": username}
    return "Login failed", {}

@tool(
    description="Get current user profile",
    required_states=["authenticated"],
    required_context=["user"]
)
def get_profile():
    user = auth_toolkit.context.get("user")
    profile = fetch_profile(user)
    return profile, {}

@tool(
    description="Update user profile",
    required_states=["authenticated"],
    required_context=["user"]
)
def update_profile(name: str, email: str):
    user = auth_toolkit.context.get("user")
    update_user(user, name, email)
    return "Profile updated", {}

@tool(
    description="Logout from the system",
    required_states=["authenticated"],
    disables_states=["authenticated"]
)
def logout():
    return "Logged out", {"user": None}

auth_toolkit.add_tool("login", login)
auth_toolkit.add_tool("get_profile", get_profile)
auth_toolkit.add_tool("update_profile", update_profile)
auth_toolkit.add_tool("logout", logout)
```

**State Flow:**
```
Initial: unlocked_states = {}
↓
login() → unlocked_states = {"authenticated"}
↓
get_profile() ✅ (authenticated required)
update_profile() ✅ (authenticated required)
↓
logout() → unlocked_states = {}
↓
get_profile() ❌ (authenticated required but not present)
```

### Multi-Step Wizard

```python
wizard_toolkit = Toolkit()

@tool(
    description="Start wizard",
    enables_states=["wizard_started"],
    disables_states=["wizard_completed"]
)
def start_wizard():
    return "Wizard started", {"current_step": 1}

@tool(
    description="Complete step 1",
    required_states=["wizard_started"],
    enables_states=["step1_completed"]
)
def complete_step1(data: dict):
    return "Step 1 complete", {"step1_data": data, "current_step": 2}

@tool(
    description="Complete step 2",
    required_states=["wizard_started", "step1_completed"],
    enables_states=["step2_completed"]
)
def complete_step2(data: dict):
    return "Step 2 complete", {"step2_data": data, "current_step": 3}

@tool(
    description="Finalize wizard",
    required_states=["wizard_started", "step1_completed", "step2_completed"],
    enables_states=["wizard_completed"],
    disables_states=["wizard_started", "step1_completed", "step2_completed"]
)
def finalize_wizard():
    step1 = wizard_toolkit.context.get("step1_data")
    step2 = wizard_toolkit.context.get("step2_data")
    process_wizard(step1, step2)
    return "Wizard complete", {}
```

**State Flow:**
```
Initial: {}
↓
start_wizard() → {"wizard_started"}
↓
complete_step1() → {"wizard_started", "step1_completed"}
↓
complete_step2() → {"wizard_started", "step1_completed", "step2_completed"}
↓
finalize_wizard() → {"wizard_completed"}
```

### Resource Lifecycle

```python
resource_toolkit = Toolkit()

@tool(
    description="Acquire resource",
    forbidden_states=["resource_locked"],
    enables_states=["resource_locked"]
)
def acquire_resource(resource_id: str):
    lock_resource(resource_id)
    return f"Acquired {resource_id}", {"locked_resource": resource_id}

@tool(
    description="Use resource",
    required_states=["resource_locked"],
    required_context=["locked_resource"]
)
def use_resource(action: str):
    resource_id = resource_toolkit.context.get("locked_resource")
    result = perform_action(resource_id, action)
    return result, {}

@tool(
    description="Release resource",
    required_states=["resource_locked"],
    disables_states=["resource_locked"]
)
def release_resource():
    resource_id = resource_toolkit.context.get("locked_resource")
    unlock_resource(resource_id)
    return f"Released {resource_id}", {"locked_resource": None}
```

**Key Feature**: `forbidden_states=["resource_locked"]` prevents acquiring while already locked.

### Transaction Pattern

```python
transaction_toolkit = Toolkit()

@tool(
    description="Begin transaction",
    forbidden_states=["in_transaction"],
    enables_states=["in_transaction"]
)
def begin_transaction():
    tx_id = start_transaction()
    return f"Transaction {tx_id} started", {"transaction_id": tx_id}

@tool(
    description="Add operation to transaction",
    required_states=["in_transaction"]
)
def add_operation(operation: str, data: dict):
    tx_id = transaction_toolkit.context.get("transaction_id")
    add_to_transaction(tx_id, operation, data)
    return f"Added {operation}", {}

@tool(
    description="Commit transaction",
    required_states=["in_transaction"],
    disables_states=["in_transaction"]
)
def commit_transaction():
    tx_id = transaction_toolkit.context.get("transaction_id")
    commit(tx_id)
    return "Transaction committed", {"transaction_id": None}

@tool(
    description="Rollback transaction",
    required_states=["in_transaction"],
    disables_states=["in_transaction"]
)
def rollback_transaction():
    tx_id = transaction_toolkit.context.get("transaction_id")
    rollback(tx_id)
    return "Transaction rolled back", {"transaction_id": None}
```

### Permission Elevation

```python
permission_toolkit = Toolkit()

@tool(
    description="Elevate to admin privileges",
    required_states=["authenticated"],
    forbidden_states=["admin_mode"],
    enables_states=["admin_mode"]
)
def elevate_privileges(admin_password: str):
    if verify_admin_password(admin_password):
        return "Admin mode enabled", {}
    return "Invalid password", {}

@tool(
    description="Perform admin action",
    required_states=["authenticated", "admin_mode"]
)
def admin_action(action: str):
    perform_admin_action(action)
    return f"Executed {action}", {}

@tool(
    description="Drop admin privileges",
    required_states=["admin_mode"],
    disables_states=["admin_mode"]
)
def drop_privileges():
    return "Admin mode disabled", {}
```

---

## Advanced Techniques

### Conditional State Transitions

```python
@tool(enables_states=["processed"])
def process_data(data: dict):
    result = process(data)

    # Conditionally enable states based on result
    updates = {}
    states_to_enable = ["processed"]

    if result.has_errors:
        states_to_enable.append("has_errors")
        updates["errors"] = result.errors
    else:
        states_to_enable.append("validated")

    # Manually update states (beyond decorator)
    # Note: Decorator states are applied first
    toolkit.unlocked_states.update(states_to_enable)

    return result, updates
```

**Note**: Decorator `enables_states` and `disables_states` are applied automatically. Manual state updates should be rare and documented.

### State Guards

```python
@tool(required_states=["authenticated"])
def protected_action():
    # State requirement acts as a guard
    # Tool is only available when authenticated
    return "Action performed", {}
```

**Alternative Pattern** (manual check):

```python
@tool()
def manual_guard_action():
    # ❌ Don't do this - use required_states instead
    if "authenticated" not in toolkit.unlocked_states:
        return "Not authenticated", {}

    return "Action performed", {}
```

### State Branches

Different tools enable different state paths:

```python
@tool(enables_states=["path_a"])
def choose_path_a():
    return "Path A selected", {"path": "a"}

@tool(enables_states=["path_b"])
def choose_path_b():
    return "Path B selected", {"path": "b"}

@tool(required_states=["path_a"])
def path_a_action():
    return "Path A action", {}

@tool(required_states=["path_b"])
def path_b_action():
    return "Path B action", {}
```

**Flow:**
```
choose_path_a() → unlocked_states = {"path_a"}
  → path_a_action() available
  → path_b_action() unavailable

choose_path_b() → unlocked_states = {"path_b"}
  → path_b_action() available
  → path_a_action() unavailable
```

### State Reset

```python
@tool(
    disables_states=["*"],  # Note: Not actually supported, shown for concept
    enables_states=["initial"]
)
def reset_workflow():
    # To reset all states, manually clear:
    toolkit.unlocked_states.clear()
    toolkit.unlocked_states.add("initial")
    return "Workflow reset", {}
```

**Best Practice**: Explicitly list states to disable:

```python
@tool(
    disables_states=["authenticated", "admin_mode", "resource_locked"],
    enables_states=["initial"]
)
def reset_workflow():
    return "Workflow reset", {}
```

### State Queries

Check current state in tool logic:

```python
@tool()
def get_available_actions():
    actions = []

    if "authenticated" in toolkit.unlocked_states:
        actions.append("logout")

    if "admin_mode" in toolkit.unlocked_states:
        actions.append("admin_action")

    return f"Available: {actions}", {"available_actions": actions}
```

---

## Best Practices

### 1. Use Abstract State Names

```python
# ❌ Bad: Concrete names tied to implementation
@tool(enables_states=["user_alice_logged_in"])
def login(username: str):
    pass

# ✅ Good: Abstract states
@tool(enables_states=["authenticated"])
def login(username: str):
    pass
```

### 2. Keep States Orthogonal

States should be independent concepts:

```python
# ✅ Good: Orthogonal states
unlocked_states = {
    "authenticated",      # Auth state
    "premium_user",       # Subscription state
    "admin_mode",         # Permission state
    "resource_locked"     # Resource state
}

# ❌ Bad: Redundant states
unlocked_states = {
    "authenticated",
    "not_authenticated",  # Redundant (inverse of authenticated)
}
```

### 3. Document State Dependencies

```python
@tool(
    description="Get user profile",
    required_states=["authenticated"],  # Clear requirement
    required_context=["user_id"]        # Clear context dependency
)
def get_profile():
    """
    Fetch the current user's profile.

    State Requirements:
        - authenticated: User must be logged in

    Context Requirements:
        - user_id: Current user identifier (set by login)
    """
    pass
```

### 4. Avoid State Explosion

```python
# ❌ Bad: Too many granular states
enables_states=[
    "step1_complete",
    "step2_complete",
    "step3_complete",
    "step4_complete",
    # ... 20 more steps
]

# ✅ Good: Use context for progression tracking
@tool()
def complete_step(step_number: int):
    current_step = toolkit.context.get("completed_steps", 0)
    new_step = current_step + 1
    return f"Step {step_number} complete", {"completed_steps": new_step}
```

### 5. Handle Failure States

```python
@tool(enables_states=["authenticated"])
def login(username: str, password: str):
    if authenticate(username, password):
        return "Logged in", {"user": username}
    else:
        # Don't enable "authenticated" state on failure
        # Decorator states are only applied on success
        toolkit.unlocked_states.add("login_failed")
        return "Login failed", {}
```

### 6. Use forbidden_states for Mutual Exclusion

```python
# ✅ Good: Prevent simultaneous states
@tool(
    forbidden_states=["editing_mode"],
    enables_states=["viewing_mode"]
)
def view_document():
    pass

@tool(
    forbidden_states=["viewing_mode"],
    enables_states=["editing_mode"]
)
def edit_document():
    pass
```

### 7. Combine FSM with Dynamic Types

```python
@tool(
    enables_states=["project_selected"],
    required_context=["projects"]
)
def select_project(name: DynamicStr["projects"]):
    # Fetch issues for selected project
    issues = fetch_issues(name)
    return f"Selected {name}", {
        "current_project": name,
        "issues": [i.name for i in issues]
    }

@tool(
    required_states=["project_selected"],
    required_context=["issues"]
)
def select_issue(name: DynamicStr["issues"]):
    # Issues enum is populated by select_project
    return f"Selected {name}", {"current_issue": name}
```

**Pattern**: State enables tool, context enables dynamic schema.

### 8. Test State Transitions

```python
# Unit test state transitions
def test_authentication_flow():
    toolkit = Toolkit()
    # ... add tools ...

    assert toolkit.unlocked_states == set()

    toolkit.execute_tool("login", {"username": "alice", "password": "secret"})
    assert "authenticated" in toolkit.unlocked_states

    toolkit.execute_tool("logout", {})
    assert "authenticated" not in toolkit.unlocked_states
```

---

## State Diagram Example

Complete authentication + resource workflow:

```
                    ┌─────────────┐
                    │   Initial   │
                    │   (empty)   │
                    └──────┬──────┘
                           │
                      login()
                           │
                           ▼
                    ┌─────────────┐
                    │authenticated│
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │                         │
       acquire_resource()          logout()
              │                         │
              ▼                         ▼
    ┌──────────────────┐         ┌──────────┐
    │  authenticated   │         │ Initial  │
    │ resource_locked  │         │ (empty)  │
    └────────┬─────────┘         └──────────┘
             │
        use_resource()
             │
             ▼
    ┌──────────────────┐
    │  authenticated   │
    │ resource_locked  │
    └────────┬─────────┘
             │
    release_resource()
             │
             ▼
       ┌─────────────┐
       │authenticated│
       └─────────────┘
```

---

## Common Pitfalls

### 1. Forgetting to Disable States

```python
# ❌ Bad: State never disabled
@tool(enables_states=["processing"])
def start_processing():
    return "Started", {}

@tool(enables_states=["processing_complete"])  # Doesn't disable "processing"
def finish_processing():
    return "Finished", {}

# Result: unlocked_states = {"processing", "processing_complete"}

# ✅ Good: Explicitly disable
@tool(
    enables_states=["processing_complete"],
    disables_states=["processing"]
)
def finish_processing():
    return "Finished", {}
```

### 2. Circular Dependencies

```python
# ❌ Bad: Deadlock
@tool(required_states=["state_b"], enables_states=["state_a"])
def tool_a():
    pass

@tool(required_states=["state_a"], enables_states=["state_b"])
def tool_b():
    pass

# Neither tool can execute initially!
```

### 3. Over-Constraining

```python
# ❌ Bad: Too restrictive
@tool(
    required_states=["authenticated", "premium", "verified", "active"],
    forbidden_states=["suspended", "rate_limited", "maintenance"]
)
def simple_action():
    pass

# ✅ Good: Minimal necessary constraints
@tool(required_states=["authenticated"])
def simple_action():
    pass
```

---

**Next**: [Decorators](decorators.md) - Production-grade tool decorators
