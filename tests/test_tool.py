"""Tests for Tool class."""

from typing import Tuple

from bestla.yggdrasil import Context, DynamicStr, Tool, tool


class TestTool:
    """Test Tool functionality."""

    def test_create_tool(self):
        """Test creating a basic tool."""
        def add(a: int, b: int) -> Tuple[int, dict]:
            return a + b, {}

        t = Tool(function=add)
        assert t.name == "add"
        assert t.function == add

    def test_tool_with_metadata(self):
        """Test tool with metadata."""
        def select_project(project_id: str) -> Tuple[str, dict]:
            """Select a project."""
            return f"Selected {project_id}", {"selected_project": project_id}

        t = Tool(
            function=select_project,
            required_context=["auth_token"],
            enables_states=["project_selected"],
            disables_states=["project_browsing"]
        )

        assert t.required_context == ["auth_token"]
        assert t.enables_states == ["project_selected"]
        assert t.disables_states == ["project_browsing"]

    def test_execute_tool(self):
        """Test executing a tool."""
        def add(a: int, b: int) -> Tuple[int, dict]:
            return a + b, {}

        t = Tool(function=add)
        result, updates = t.execute(a=5, b=3)

        assert result == 8
        assert updates == {}

    def test_execute_with_context_updates(self):
        """Test tool that returns context updates."""
        def create_issue(name: str) -> Tuple[str, dict]:
            return f"Created {name}", {"last_issue": name}

        t = Tool(function=create_issue)
        result, updates = t.execute(name="BUG-1")

        assert result == "Created BUG-1"
        assert updates == {"last_issue": "BUG-1"}

    def test_check_context_requirements(self):
        """Test checking context requirements."""
        def protected_action() -> Tuple[str, dict]:
            return "Done", {}

        t = Tool(
            function=protected_action,
            required_context=["auth_token", "user_id"]
        )

        context = Context()
        all_present, missing = t.check_context_requirements(context)
        assert not all_present
        assert set(missing) == {"auth_token", "user_id"}

        context.set("auth_token", "token123")
        all_present, missing = t.check_context_requirements(context)
        assert not all_present
        assert missing == ["user_id"]

        context.set("user_id", "user1")
        all_present, missing = t.check_context_requirements(context)
        assert all_present
        assert missing == []

    def test_generate_schema_basic(self):
        """Test generating basic JSON schema."""
        def add(a: int, b: int) -> Tuple[int, dict]:
            """Add two numbers."""
            return a + b, {}

        t = Tool(function=add)
        context = Context()
        schema = t.generate_schema(context)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"
        assert schema["function"]["description"] == "Add two numbers."
        assert "a" in schema["function"]["parameters"]["properties"]
        assert "b" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["properties"]["a"] == {"type": "integer"}

    def test_generate_schema_with_dynamic_type(self):
        """Test generating schema with dynamic types."""
        def select_issue(name: DynamicStr["issue_names"]) -> Tuple[str, dict]:
            return f"Selected {name}", {}

        t = Tool(function=select_issue)
        context = Context()
        context.set("issue_names", ["BUG-1", "FEAT-2"])

        schema = t.generate_schema(context)
        name_schema = schema["function"]["parameters"]["properties"]["name"]

        assert name_schema == {"type": "string", "enum": ["BUG-1", "FEAT-2"]}

    def test_tool_decorator(self):
        """Test @tool decorator."""
        @tool(
            required_context=["selected_project"],
            enables_states=["issues_loaded"]
        )
        def list_issues() -> Tuple[str, dict]:
            """List issues in project."""
            return "Found 5 issues", {"issue_names": ["BUG-1", "BUG-2"]}

        assert isinstance(list_issues, Tool)
        assert list_issues.name == "list_issues"
        assert list_issues.required_context == ["selected_project"]
        assert list_issues.enables_states == ["issues_loaded"]

    def test_tool_with_optional_params(self):
        """Test tool with optional parameters."""
        def greet(name: str, greeting: str = "Hello") -> Tuple[str, dict]:
            return f"{greeting}, {name}!", {}

        t = Tool(function=greet)
        context = Context()
        schema = t.generate_schema(context)

        # name should be required, greeting optional
        required = schema["function"]["parameters"]["required"]
        assert "name" in required
        assert "greeting" not in required

    def test_tool_without_type_hints(self):
        """Test tool without type hints defaults to string."""
        def untyped_tool(param):
            return "result", {}

        t = Tool(function=untyped_tool)
        context = Context()
        schema = t.generate_schema(context)

        # Should default to string type
        param_schema = schema["function"]["parameters"]["properties"]["param"]
        assert param_schema == {"type": "string"}
