"""Tool class for wrapping functions with metadata."""

import inspect
from typing import Any, Callable, Dict, List, Tuple, get_type_hints

from bestla.yggdrasil.dynamic_types import generate_param_schema


class Tool:
    """Wraps a Python function with metadata for agent use.

    Tools always return (result, context_updates) tuples.
    Tools can declare context dependencies, state requirements,
    and use dynamic types for context-driven schemas.
    """

    def __init__(
        self,
        function: Callable,
        name: str | None = None,
        description: str | None = None,
        required_context: List[str] | None = None,
        required_states: List[str] | None = None,
        forbidden_states: List[str] | None = None,
        enables_states: List[str] | None = None,
        disables_states: List[str] | None = None,
    ):
        """Initialize a tool.

        Args:
            function: Python function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
            required_context: List of context keys that must exist
            required_states: List of states that must be enabled
            forbidden_states: List of states that must NOT be enabled
            enables_states: List of states to enable after execution
            disables_states: List of states to disable after execution
        """
        self.function = function
        self.name = name or function.__name__
        self.description = description or (inspect.getdoc(function) or "")

        # Metadata
        self.required_context = required_context or []
        self.required_states = required_states or []
        self.forbidden_states = forbidden_states or []
        self.enables_states = enables_states or []
        self.disables_states = disables_states or []

        # Extract function signature
        self.signature = inspect.signature(function)
        try:
            self.type_hints = get_type_hints(function)
        except Exception:
            # get_type_hints can fail for forward references
            self.type_hints = {}

    def generate_schema(self, context: Any, filters: dict | None = None) -> dict:
        """Generate JSON schema for this tool based on current context.

        Args:
            context: Context instance to read dynamic type values from
            filters: Optional dictionary of filter functions for DynamicFiltered

        Returns:
            OpenAI-compatible tool schema dictionary
        """
        # Build parameters schema
        parameters = {"type": "object", "properties": {}, "required": []}

        for param_name, param in self.signature.parameters.items():
            # Skip self/cls
            if param_name in ("self", "cls"):
                continue

            # Get type annotation
            annotation = self.type_hints.get(param_name, param.annotation)
            if annotation == inspect.Parameter.empty:
                # No type hint, default to string
                param_schema = {"type": "string"}
            else:
                # Generate schema from annotation (handles dynamic types)
                param_schema = generate_param_schema(annotation, context, filters)

            parameters["properties"][param_name] = param_schema

            # Required if no default value
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        # Return OpenAI tool schema format
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def execute(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Execute the wrapped function.

        Args:
            **kwargs: Function arguments

        Returns:
            Tuple of (result, context_updates)
        """
        result = self.function(**kwargs)

        # Tools ALWAYS return (result, context_updates) tuple
        if isinstance(result, tuple) and len(result) == 2:
            return result
        else:
            # If function didn't return tuple, assume no context updates
            return result, {}

    def check_context_requirements(self, context: Any) -> Tuple[bool, List[str]]:
        """Check if required context keys are present.

        Args:
            context: Context instance to check

        Returns:
            Tuple of (all_present, missing_keys)
        """
        missing = [key for key in self.required_context if not context.has(key)]
        return len(missing) == 0, missing

    def __repr__(self) -> str:
        return (
            f"Tool(name='{self.name}', "
            f"required_context={self.required_context}, "
            f"required_states={self.required_states})"
        )


def tool(
    required_context: List[str] | None = None,
    required_states: List[str] | None = None,
    forbidden_states: List[str] | None = None,
    enables_states: List[str] | None = None,
    disables_states: List[str] | None = None,
    name: str | None = None,
    description: str | None = None,
):
    """Decorator to create a Tool from a function.

    Args:
        required_context: List of context keys that must exist
        required_states: List of states that must be enabled
        forbidden_states: List of states that must NOT be enabled
        enables_states: List of states to enable after execution
        disables_states: List of states to disable after execution
        name: Optional tool name (defaults to function name)
        description: Optional tool description (defaults to docstring)

    Returns:
        Decorator function

    Example:
        @tool(required_states=["authenticated"], enables_states=["project_selected"])
        def select_project(project_id: str) -> Tuple[str, dict]:
            return "Selected", {"project_id": project_id}
    """

    def decorator(func: Callable) -> Tool:
        return Tool(
            function=func,
            name=name,
            description=description,
            required_context=required_context,
            required_states=required_states,
            forbidden_states=forbidden_states,
            enables_states=enables_states,
            disables_states=disables_states,
        )

    return decorator
