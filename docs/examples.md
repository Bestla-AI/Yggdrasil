# Examples

Complete, production-ready examples demonstrating Yggdrasil patterns and workflows.

## Table of Contents
- [Authentication Workflow](#authentication-workflow)
- [Project Management Workflow](#project-management-workflow)
- [E-commerce Transaction](#e-commerce-transaction)
- [Multi-Agent Research System](#multi-agent-research-system)
- [API Integration with Resilience](#api-integration-with-resilience)
- [Multi-User Service](#multi-user-service)

---

## Authentication Workflow

Complete authentication system with login, profile access, and logout.

```python
from openai import OpenAI
from bestla.yggdrasil import Agent, Toolkit, Context, tool

# Initialize OpenAI
client = OpenAI(api_key="your-api-key")

# Create authentication toolkit
auth_toolkit = Toolkit(context=Context())

# User database (simulate)
USERS = {
    "alice": {"password": "secret123", "email": "alice@example.com", "role": "admin"},
    "bob": {"password": "pass456", "email": "bob@example.com", "role": "user"}
}

@tool(
    description="Login to the system",
    enables_states=["authenticated"]
)
def login(username: str, password: str):
    """Authenticate user and create session"""
    if username in USERS and USERS[username]["password"] == password:
        user_data = USERS[username]
        return f"Logged in as {username}", {
            "user": username,
            "email": user_data["email"],
            "role": user_data["role"]
        }
    return "Login failed: Invalid credentials", {}

@tool(
    description="Get current user profile",
    required_states=["authenticated"],
    required_context=["user", "email"]
)
def get_profile():
    """Fetch profile for authenticated user"""
    user = auth_toolkit.context.get("user")
    email = auth_toolkit.context.get("email")
    role = auth_toolkit.context.get("role")

    profile = {
        "username": user,
        "email": email,
        "role": role
    }

    return f"Profile: {profile}", {}

@tool(
    description="Update user email",
    required_states=["authenticated"],
    required_context=["user"]
)
def update_email(new_email: str):
    """Update email for authenticated user"""
    user = auth_toolkit.context.get("user")

    # Simulate update
    USERS[user]["email"] = new_email

    return f"Email updated to {new_email}", {"email": new_email}

@tool(
    description="Logout from the system",
    required_states=["authenticated"],
    disables_states=["authenticated"]
)
def logout():
    """End user session"""
    user = auth_toolkit.context.get("user")
    return f"Logged out {user}", {
        "user": None,
        "email": None,
        "role": None
    }

# Add tools to toolkit
auth_toolkit.add_tool("login", login)
auth_toolkit.add_tool("get_profile", get_profile)
auth_toolkit.add_tool("update_email", update_email)
auth_toolkit.add_tool("logout", logout)

# Create agent
agent = Agent(
    provider=client,
    model="gpt-4",
    system_prompt="You are a helpful authentication assistant."
)
agent.add_toolkit("auth", auth_toolkit)

# Usage
if __name__ == "__main__":
    # Workflow 1: Login and view profile
    response, ctx = agent.run(
        "Login as alice with password secret123, then show my profile"
    )
    print(response)

    # Workflow 2: Update email (continuing session)
    response, ctx = agent.run(
        "Update my email to alice.new@example.com",
        execution_context=ctx
    )
    print(response)

    # Workflow 3: Logout
    response, ctx = agent.run("Logout", execution_context=ctx)
    print(response)

    # Workflow 4: Try to access profile after logout (should fail)
    response, ctx = agent.run("Show my profile", execution_context=ctx)
    print(response)  # Should indicate not authenticated
```

---

## Project Management Workflow

Multi-step workflow with dynamic schemas and progressive context enhancement.

```python
from bestla.yggdrasil import (
    Agent, Toolkit, Context, tool,
    DynamicStr, DynamicInt, DynamicArray
)

# Create project management toolkit
project_toolkit = Toolkit(context=Context())

# Simulate project database
PROJECTS = {
    "alpha": {
        "name": "Project Alpha",
        "issues": [
            {"id": "ALPHA-1", "title": "Setup database", "priority": 1},
            {"id": "ALPHA-2", "title": "Create API", "priority": 2}
        ]
    },
    "beta": {
        "name": "Project Beta",
        "issues": [
            {"id": "BETA-1", "title": "Design UI", "priority": 3}
        ]
    }
}

@tool(
    description="List all available projects",
    enables_states=["projects_loaded"]
)
def list_projects():
    """Fetch all projects"""
    project_names = list(PROJECTS.keys())
    return f"Projects: {project_names}", {
        "projects": project_names
    }

@tool(
    description="Select a project to work with",
    required_states=["projects_loaded"],
    required_context=["projects"],
    enables_states=["project_selected"]
)
def select_project(name: DynamicStr["projects"]):
    """Select a project and load its issues"""
    project = PROJECTS[name]
    issues = project["issues"]

    return f"Selected {project['name']}", {
        "current_project": name,
        "issues": [i["id"] for i in issues],
        "issue_details": {i["id"]: i for i in issues}
    }

@tool(
    description="List issues in selected project",
    required_states=["project_selected"],
    required_context=["issues"]
)
def list_issues():
    """List all issues in current project"""
    issues = project_toolkit.context.get("issues")
    return f"Issues: {issues}", {}

@tool(
    description="Get details of a specific issue",
    required_states=["project_selected"],
    required_context=["issues", "issue_details"],
    enables_states=["issue_selected"]
)
def get_issue(issue_id: DynamicStr["issues"]):
    """Get detailed information about an issue"""
    issue_details = project_toolkit.context.get("issue_details")
    issue = issue_details[issue_id]

    return f"Issue {issue_id}: {issue}", {
        "current_issue": issue_id
    }

@tool(
    description="Update issue priority",
    required_states=["project_selected", "issue_selected"],
    required_context=["current_issue", "issue_details"]
)
def set_priority(priority: DynamicInt["priority_range"]):
    """Set priority for current issue"""
    issue_id = project_toolkit.context.get("current_issue")
    issue_details = project_toolkit.context.get("issue_details")

    # Update priority
    issue_details[issue_id]["priority"] = priority

    return f"Set {issue_id} priority to {priority}", {
        "issue_details": issue_details
    }

@tool(
    description="Assign tags to current issue",
    required_states=["issue_selected"],
    required_context=["current_issue", "available_tags"]
)
def assign_tags(tags: DynamicArray["available_tags"]):
    """Assign tags to the current issue"""
    issue_id = project_toolkit.context.get("current_issue")
    return f"Assigned tags {tags} to {issue_id}", {}

# Add tools
project_toolkit.add_tool("list_projects", list_projects)
project_toolkit.add_tool("select_project", select_project)
project_toolkit.add_tool("list_issues", list_issues)
project_toolkit.add_tool("get_issue", get_issue)
project_toolkit.add_tool("set_priority", set_priority)
project_toolkit.add_tool("assign_tags", assign_tags)

# Initialize context with static data
project_toolkit.context.set("priority_range", {"minimum": 1, "maximum": 5})
project_toolkit.context.set("available_tags", ["bug", "feature", "urgent", "docs"])

# Create agent
agent = Agent(provider=client, model="gpt-4")
agent.add_toolkit("project", project_toolkit)

# Usage
if __name__ == "__main__":
    response, ctx = agent.run(
        "List projects, select alpha, get issue ALPHA-1, and set its priority to 5"
    )
    print(response)
```

---

## E-commerce Transaction

Transaction pattern with rollback capability.

```python
from bestla.yggdrasil import Agent, Toolkit, Context, tool

# Create transaction toolkit
transaction_toolkit = Toolkit(context=Context())

# Simulate database
INVENTORY = {"item1": 10, "item2": 5}
ORDERS = []

@tool(
    description="Begin a new transaction",
    forbidden_states=["in_transaction"],
    enables_states=["in_transaction"]
)
def begin_transaction():
    """Start a new transaction"""
    import uuid
    tx_id = str(uuid.uuid4())

    return f"Transaction {tx_id} started", {
        "transaction_id": tx_id,
        "operations": []
    }

@tool(
    description="Add item to cart",
    required_states=["in_transaction"],
    required_context=["operations"]
)
def add_to_cart(item_id: str, quantity: int):
    """Add item to transaction cart"""
    if item_id not in INVENTORY:
        return f"Error: Item {item_id} not found", {}

    if INVENTORY[item_id] < quantity:
        return f"Error: Insufficient inventory for {item_id}", {}

    operations = transaction_toolkit.context.get("operations")
    operations.append({"type": "add_to_cart", "item": item_id, "quantity": quantity})

    return f"Added {quantity}x {item_id} to cart", {
        "operations": operations
    }

@tool(
    description="Apply discount code",
    required_states=["in_transaction"]
)
def apply_discount(code: str):
    """Apply discount to transaction"""
    operations = transaction_toolkit.context.get("operations", [])
    operations.append({"type": "discount", "code": code})

    return f"Applied discount {code}", {
        "operations": operations,
        "discount_code": code
    }

@tool(
    description="Commit transaction and process order",
    required_states=["in_transaction"],
    required_context=["transaction_id", "operations"],
    disables_states=["in_transaction"]
)
def commit_transaction():
    """Finalize transaction"""
    tx_id = transaction_toolkit.context.get("transaction_id")
    operations = transaction_toolkit.context.get("operations")

    # Process operations
    for op in operations:
        if op["type"] == "add_to_cart":
            INVENTORY[op["item"]] -= op["quantity"]

    # Create order
    order = {"tx_id": tx_id, "operations": operations}
    ORDERS.append(order)

    return f"Transaction {tx_id} committed. Order placed.", {
        "transaction_id": None,
        "operations": None
    }

@tool(
    description="Rollback transaction and cancel",
    required_states=["in_transaction"],
    disables_states=["in_transaction"]
)
def rollback_transaction():
    """Cancel transaction without changes"""
    tx_id = transaction_toolkit.context.get("transaction_id")

    return f"Transaction {tx_id} rolled back", {
        "transaction_id": None,
        "operations": None
    }

# Add tools
transaction_toolkit.add_tool("begin_transaction", begin_transaction)
transaction_toolkit.add_tool("add_to_cart", add_to_cart)
transaction_toolkit.add_tool("apply_discount", apply_discount)
transaction_toolkit.add_tool("commit_transaction", commit_transaction)
transaction_toolkit.add_tool("rollback_transaction", rollback_transaction)

# Create agent
agent = Agent(provider=client, model="gpt-4")
agent.add_toolkit("transaction", transaction_toolkit)

# Usage
if __name__ == "__main__":
    # Successful transaction
    response, ctx = agent.run(
        "Begin transaction, add 2x item1 and 1x item2 to cart, "
        "apply discount SAVE10, then commit"
    )
    print(response)

    # Transaction with rollback
    response, ctx = agent.run(
        "Begin transaction, add 100x item1, then rollback"
    )
    print(response)
```

---

## Multi-Agent Research System

Hierarchical agents for research and analysis.

```python
from bestla.yggdrasil import Agent, Toolkit, tool, retry, cache_result
import requests

# Web Search Toolkit
web_toolkit = Toolkit()

@retry(max_attempts=3, exceptions=(requests.RequestException,))
@cache_result(ttl=3600.0)
@tool(description="Search the web")
def web_search(query: str):
    """Simulate web search"""
    # In production: integrate real search API
    results = [
        f"Result 1 for '{query}'",
        f"Result 2 for '{query}'",
        f"Result 3 for '{query}'"
    ]
    return results, {}

@tool(description="Fetch webpage content")
def fetch_webpage(url: str):
    """Fetch and parse webpage"""
    # In production: use requests + BeautifulSoup
    content = f"Content from {url}"
    return content, {}

web_toolkit.add_tool("search", web_search)
web_toolkit.add_tool("fetch", fetch_webpage)

# Data Analysis Toolkit
analysis_toolkit = Toolkit()

@tool(description="Analyze text data")
def analyze_text(text: str):
    """Perform text analysis"""
    # In production: NLP analysis
    analysis = {
        "word_count": len(text.split()),
        "sentiment": "positive"
    }
    return analysis, {}

@tool(description="Summarize content")
def summarize(text: str):
    """Generate summary"""
    # In production: use summarization model
    summary = f"Summary: {text[:100]}..."
    return summary, {}

analysis_toolkit.add_tool("analyze", analyze_text)
analysis_toolkit.add_tool("summarize", summarize)

# Create specialized agents
research_agent = Agent(
    provider=client,
    model="gpt-4",
    system_prompt="You are a research specialist. Find and gather information."
)
research_agent.add_toolkit("web", web_toolkit)

analysis_agent = Agent(
    provider=client,
    model="gpt-4",
    system_prompt="You are a data analyst. Analyze and summarize information."
)
analysis_agent.add_toolkit("analysis", analysis_toolkit)

# Create coordinator agent
coordinator = Agent(
    provider=client,
    model="gpt-4",
    system_prompt="You are a research coordinator. Delegate tasks to specialists."
)

@tool(description="Research a topic using web search")
def research(topic: str):
    """Delegate to research agent"""
    response, _ = research_agent.run(f"Research {topic} and gather key information")
    return response, {}

@tool(description="Analyze and summarize data")
def analyze(data: str):
    """Delegate to analysis agent"""
    response, _ = analysis_agent.run(f"Analyze and summarize: {data}")
    return response, {}

coordinator.add_tool("research", research)
coordinator.add_tool("analyze", analyze)

# Usage
if __name__ == "__main__":
    response, ctx = coordinator.run(
        "Research artificial intelligence trends in 2025, "
        "then analyze and summarize the findings"
    )
    print(response)
```

---

## API Integration with Resilience

Production-ready API client with retry, timeout, cache, and rate limiting.

```python
from bestla.yggdrasil import (
    Agent, Toolkit, tool,
    retry, timeout, cache_result, rate_limit
)
import requests

# API Toolkit
api_toolkit = Toolkit()

@retry(max_attempts=3, backoff=1.0, exceptions=(requests.Timeout, requests.ConnectionError))
@timeout(10.0)
@rate_limit(calls=100, period=60.0)
@tool(description="Fetch user data from API")
def get_user(user_id: str):
    """Fetch user with resilience"""
    response = requests.get(
        f"https://api.example.com/users/{user_id}",
        timeout=5.0
    )
    response.raise_for_status()
    return response.json(), {}

@retry(max_attempts=3, backoff=1.0, exceptions=(requests.RequestException,))
@cache_result(ttl=300.0)
@timeout(10.0)
@rate_limit(calls=60, period=60.0)
@tool(description="Search users by name")
def search_users(name: str):
    """Search users with caching"""
    response = requests.get(
        f"https://api.example.com/users/search",
        params={"name": name},
        timeout=5.0
    )
    response.raise_for_status()
    return response.json(), {}

@retry(max_attempts=5, backoff=2.0, exceptions=(requests.RequestException,))
@timeout(15.0)
@rate_limit(calls=50, period=60.0)
@tool(description="Create new user")
def create_user(name: str, email: str):
    """Create user with retry"""
    response = requests.post(
        "https://api.example.com/users",
        json={"name": name, "email": email},
        timeout=10.0
    )
    response.raise_for_status()
    return response.json(), {}

api_toolkit.add_tool("get_user", get_user)
api_toolkit.add_tool("search_users", search_users)
api_toolkit.add_tool("create_user", create_user)

# Create agent
agent = Agent(provider=client, model="gpt-4")
agent.add_toolkit("api", api_toolkit)

# Usage
if __name__ == "__main__":
    response, ctx = agent.run("Search for users named Alice, then get details for the first result")
    print(response)
```

---

## Multi-User Service

Concurrent multi-user service with session management.

```python
from bestla.yggdrasil import Agent, Toolkit, Context, ExecutionContext, ConversationContext, tool
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

# Create shared agent
auth_toolkit = Toolkit()

@tool(enables_states=["authenticated"])
def login(username: str):
    return f"Logged in as {username}", {"user": username}

@tool(required_states=["authenticated"], required_context=["user"])
def get_profile():
    user = auth_toolkit.context.get("user")
    return f"Profile for {user}", {}

auth_toolkit.add_tool("login", login)
auth_toolkit.add_tool("get_profile", get_profile)

agent = Agent(provider=client, model="gpt-4")
agent.add_toolkit("auth", auth_toolkit)

# User session manager
class SessionManager:
    def __init__(self, agent):
        self.agent = agent
        self.sessions = {}  # user_id -> ExecutionContext
        self.lock = Lock()

    def handle_request(self, user_id: str, prompt: str):
        """Handle user request with session isolation"""
        # Get user's execution context (thread-safe)
        with self.lock:
            ctx = self.sessions.get(user_id)

        # Process request (no lock needed - ExecutionContext is isolated)
        response, ctx = self.agent.run(prompt, execution_context=ctx)

        # Update session (thread-safe)
        with self.lock:
            self.sessions[user_id] = ctx

        return response

    def clear_session(self, user_id: str):
        """Clear user session"""
        with self.lock:
            self.sessions.pop(user_id, None)

# Usage
if __name__ == "__main__":
    manager = SessionManager(agent)

    # Simulate concurrent users
    def user_interaction(user_id, queries):
        results = []
        for query in queries:
            response = manager.handle_request(user_id, query)
            results.append(response)
        return results

    # Concurrent execution
    with ThreadPoolExecutor(max_workers=3) as executor:
        # User 1
        future1 = executor.submit(
            user_interaction,
            "user1",
            ["Login as Alice", "Get my profile"]
        )

        # User 2 (concurrent with user 1)
        future2 = executor.submit(
            user_interaction,
            "user2",
            ["Login as Bob", "Get my profile"]
        )

        # User 1 again (concurrent with user 2)
        future3 = executor.submit(
            user_interaction,
            "user1",
            ["Get my profile again"]  # Continues previous session
        )

        results1 = future1.result()
        results2 = future2.result()
        results3 = future3.result()

        print("User 1:", results1)
        print("User 2:", results2)
        print("User 1 (continued):", results3)
```

---

## Complete Production Example

Full-featured application with all Yggdrasil features.

```python
from bestla.yggdrasil import (
    Agent, Toolkit, Context, ExecutionContext, ConversationContext,
    ContextManager, tool,
    DynamicStr, DynamicInt,
    retry, timeout, cache_result, rate_limit
)
from openai import OpenAI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI
client = OpenAI(api_key="your-api-key")

# Create toolkit with validation
toolkit = Toolkit(context=Context(validation_enabled=True))

# Define context schema
toolkit.context.schema.define("user", {"type": "string"})
toolkit.context.schema.define("priority", {
    "type": "integer",
    "minimum": 1,
    "maximum": 5
})

# Initialize context
toolkit.context.set("projects", ["alpha", "beta", "gamma"])
toolkit.context.set("priority_range", {"minimum": 1, "maximum": 5})

# Tools with full decorators
@retry(max_attempts=3, backoff=1.0)
@timeout(5.0)
@tool(
    description="Login to system",
    enables_states=["authenticated"]
)
def login(username: str):
    """Authenticate user"""
    # Simulate authentication
    return f"Logged in as {username}", {"user": username}

@cache_result(ttl=60.0)
@tool(
    description="List available projects",
    required_states=["authenticated"],
    enables_states=["projects_loaded"]
)
def list_projects():
    """Fetch projects"""
    projects = toolkit.context.get("projects")
    return projects, {}

@tool(
    description="Select a project",
    required_states=["authenticated", "projects_loaded"],
    required_context=["projects"],
    enables_states=["project_selected"]
)
def select_project(name: DynamicStr["projects"]):
    """Select project by name"""
    return f"Selected {name}", {"current_project": name}

@rate_limit(calls=10, period=60.0)
@tool(
    description="Set project priority",
    required_states=["project_selected"],
    required_context=["current_project"]
)
def set_priority(level: DynamicInt["priority_range"]):
    """Set priority level"""
    project = toolkit.context.get("current_project")
    return f"Set {project} priority to {level}", {}

# Add tools
toolkit.add_tool("login", login)
toolkit.add_tool("list_projects", list_projects)
toolkit.add_tool("select_project", select_project)
toolkit.add_tool("set_priority", set_priority)

# Create context manager
manager = ContextManager(
    max_tokens=4000,
    strategies=["tool_result_clearing", "summarization"]
)

# Create conversation with manager
conversation = ConversationContext(context_manager=manager)

# Create execution context
execution_context = ExecutionContext(conversation=conversation)

# Create agent
agent = Agent(
    provider=client,
    model="gpt-4",
    system_prompt="You are a helpful project management assistant."
)
agent.add_toolkit("project", toolkit)

# Usage
if __name__ == "__main__":
    # Complete workflow
    response, ctx = agent.run(
        "Login as alice, list projects, select alpha, and set priority to 5",
        execution_context=execution_context
    )
    print(response)

    # Continue conversation
    response, ctx = agent.run(
        "What's the current project?",
        execution_context=ctx
    )
    print(response)
```

---

## Key Takeaways

1. **State Management**: Use FSM states to control tool availability
2. **Context Enhancement**: Progressively build context through workflows
3. **Dynamic Schemas**: Constrain parameters based on runtime context
4. **Resilience**: Add retry, timeout, cache, and rate limiting
5. **Concurrency**: Leverage three-tier model for parallelism
6. **Sessions**: Use ExecutionContext for multi-user isolation
7. **Production**: Combine all features for robust applications

---

**See Also:**
- [Core Concepts](core-concepts.md) - Architecture details
- [State Management](state-management.md) - FSM patterns
- [Dynamic Types](dynamic-types.md) - Schema generation
- [Best Practices](best-practices.md) - Design guidelines
