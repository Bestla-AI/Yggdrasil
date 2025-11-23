"""Toolkit class for managing related tools with shared context."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Set

from bestla.yggdrasil.context import Context
from bestla.yggdrasil.exceptions import (
    ToolkitPipelineError,
)
from bestla.yggdrasil.tool import Tool


class Toolkit:
    """Manages collection of related tools with shared context and state-based FSM logic.

    Toolkits:
    - Own a Context instance for domain state
    - Manage tool availability via state-based FSM (enabled/disabled states)
    - Execute tools sequentially with immediate context updates
    - Generate dynamic schemas based on current context
    - Support custom filters for DynamicFiltered types
    """

    def __init__(self, validation_enabled: bool = False):
        """Initialize toolkit.

        Args:
            validation_enabled: Whether to enable context validation
        """
        self.context = Context(validation_enabled=validation_enabled)
        self.tools: Dict[str, Tool] = {}
        self.unlocked_states: Set[str] = set()
        self.filters: Dict[str, Callable] = {}

        # Register builtin filters
        self._register_builtin_filters()

    def _register_builtin_filters(self) -> None:
        """Register commonly used filters."""
        # Active items filter (assumes items have 'active' field)
        self.filters["active_only"] = lambda items: [
            i for i in items if isinstance(i, dict) and i.get("active", False)
        ]

        # Sort ascending (assumes items are comparable or dicts with 'name')
        self.filters["sort_asc"] = lambda items: sorted(
            items, key=lambda x: x if not isinstance(x, dict) else x.get("name", "")
        )

        # Sort descending
        self.filters["sort_desc"] = lambda items: sorted(
            items,
            key=lambda x: x if not isinstance(x, dict) else x.get("name", ""),
            reverse=True,
        )

    def add_tool(
        self,
        name: str,
        function: Callable,
        required_context: List[str] | None = None,
        required_states: List[str] | None = None,
        forbidden_states: List[str] | None = None,
        enables_states: List[str] | None = None,
        disables_states: List[str] | None = None,
        description: str | None = None,
    ) -> Tool:
        """Add a tool to this toolkit.

        Args:
            name: Tool name
            function: Python function to wrap
            required_context: Context keys required
            required_states: States that must be enabled
            forbidden_states: States that must NOT be enabled
            enables_states: States to enable after execution
            disables_states: States to disable after execution
            description: Tool description

        Returns:
            Created Tool instance
        """
        tool = Tool(
            function=function,
            name=name,
            description=description,
            required_context=required_context,
            required_states=required_states,
            forbidden_states=forbidden_states,
            enables_states=enables_states,
            disables_states=disables_states,
        )
        self.tools[name] = tool
        return tool

    def register_tool(self, tool: Tool) -> None:
        """Register an existing Tool instance.

        Args:
            tool: Tool to register
        """
        self.tools[tool.name] = tool

    def register_filter(self, name: str, filter_func: Callable) -> None:
        """Register a custom filter for DynamicFiltered types.

        Args:
            name: Filter name
            filter_func: Filter function that takes a list and returns filtered list
        """
        self.filters[name] = filter_func

    def set_unlocked_states(self, state_names: Set[str]) -> None:
        """Set which states are initially unlocked.

        Args:
            state_names: Set of state names to unlock
        """
        self.unlocked_states = state_names.copy()

    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is currently available.

        A tool is available if:
        1. All required_states are unlocked
        2. None of forbidden_states are unlocked
        3. All context requirements are met

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is available
        """
        if tool_name not in self.tools:
            return False

        tool = self.tools[tool_name]

        # Check required states
        if not all(state in self.unlocked_states for state in tool.required_states):
            return False

        # Check forbidden states
        if any(state in self.unlocked_states for state in tool.forbidden_states):
            return False

        # Check context requirements
        all_present, _ = tool.check_context_requirements(self.context)
        return all_present

    def get_availability_reason(self, tool_name: str) -> str:
        """Get human-readable reason why a tool is not available.

        Args:
            tool_name: Tool name

        Returns:
            Reason string
        """
        if tool_name not in self.tools:
            return "Tool does not exist"

        tool = self.tools[tool_name]

        # Check required states
        missing_states = [s for s in tool.required_states if s not in self.unlocked_states]
        if missing_states:
            return f"Missing required states: {', '.join(missing_states)}"

        # Check forbidden states
        present_forbidden = [s for s in tool.forbidden_states if s in self.unlocked_states]
        if present_forbidden:
            return f"Forbidden states are enabled: {', '.join(present_forbidden)}"

        # Check context requirements
        all_present, missing = tool.check_context_requirements(self.context)
        if not all_present:
            return f"Missing required context: {', '.join(missing)}"

        return "Available"

    def generate_schemas(self) -> List[dict]:
        """Generate JSON schemas for all currently available tools.

        Returns:
            List of OpenAI-compatible tool schemas
        """
        schemas = []
        for tool_name, tool in self.tools.items():
            if self.is_tool_available(tool_name):
                schema = tool.generate_schema(self.context, self.filters)
                schemas.append(schema)
        return schemas

    def execute_sequential(self, tool_calls: List[dict]) -> List[dict]:
        """Execute tool calls sequentially with immediate context updates.

        This implements the toolkit pipeline:
        1. Check availability with current context
        2. Execute tool
        3. Immediately apply context updates
        4. Update FSM (process unlocks/locks)
        5. Continue to next tool

        If any tool fails, stop pipeline and return partial results.

        Args:
            tool_calls: List of tool call dicts with 'name' and 'arguments'

        Returns:
            List of result dicts

        Raises:
            ToolkitPipelineError: If a tool fails during execution
        """
        results = []

        for idx, call in enumerate(tool_calls):
            tool_name = call["name"]
            arguments = call.get("arguments", {})

            # Check if tool exists
            if tool_name not in self.tools:
                error_result = {
                    "tool_name": tool_name,
                    "success": False,
                    "error": f"Tool '{tool_name}' not found in toolkit",
                }
                results.append(error_result)
                raise ToolkitPipelineError(
                    f"Tool '{tool_name}' not found", results, idx
                )

            tool = self.tools[tool_name]

            # Check availability (regenerate schemas implicitly via checks)
            if not self.is_tool_available(tool_name):
                reason = self.get_availability_reason(tool_name)
                error_result = {
                    "tool_name": tool_name,
                    "success": False,
                    "error": f"Tool not available: {reason}",
                }
                results.append(error_result)
                raise ToolkitPipelineError(
                    f"Tool '{tool_name}' not available: {reason}", results, idx
                )

            # Execute tool
            try:
                result, context_updates = tool.execute(**arguments)

                # Immediately apply context updates
                if context_updates:
                    self.context.update(context_updates)

                # Update FSM: process enables_states
                if tool.enables_states:
                    for state_name in tool.enables_states:
                        self.unlocked_states.add(state_name)

                # Update FSM: process disables_states
                if tool.disables_states:
                    for state_name in tool.disables_states:
                        self.unlocked_states.discard(state_name)

                # Store success result
                success_result = {
                    "tool_name": tool_name,
                    "success": True,
                    "result": result,
                    "context_updates": context_updates,
                }
                results.append(success_result)

            except Exception as e:
                # Tool execution failed
                error_result = {
                    "tool_name": tool_name,
                    "success": False,
                    "error": str(e),
                }
                results.append(error_result)
                raise ToolkitPipelineError(
                    f"Tool '{tool_name}' failed: {str(e)}", results, idx
                )

        return results

    def execute_parallel(self, tool_calls: List[dict]) -> List[dict]:
        """Execute tool calls in parallel (for independent tools).

        This implements parallel execution:
        1. Create context snapshot
        2. Execute all tools in parallel
        3. Collect results as they complete
        4. Merge context updates (last-write-wins)
        5. Return all results

        Individual tool failures don't stop other tools.

        Args:
            tool_calls: List of tool call dicts with 'name' and 'arguments'

        Returns:
            List of result dicts (may include both successes and failures)
        """

        def execute_single_tool(call: dict) -> dict:
            """Execute a single tool with context snapshot."""
            tool_name = call["name"]
            arguments = call.get("arguments", {})

            try:
                # Check if tool exists
                if tool_name not in self.tools:
                    return {
                        "tool_name": tool_name,
                        "success": False,
                        "error": f"Tool '{tool_name}' not found in toolkit",
                        "context_updates": {}
                    }

                tool = self.tools[tool_name]

                # Execute tool (tools see snapshot, not live context)
                result, context_updates = tool.execute(**arguments)

                return {
                    "tool_name": tool_name,
                    "success": True,
                    "result": result,
                    "context_updates": context_updates or {}
                }

            except Exception as e:
                return {
                    "tool_name": tool_name,
                    "success": False,
                    "error": str(e),
                    "context_updates": {}
                }

        # Execute all tools in parallel
        results = []
        all_context_updates = {}

        with ThreadPoolExecutor() as executor:
            # Submit all tool calls
            future_to_call = {
                executor.submit(execute_single_tool, call): call
                for call in tool_calls
            }

            # Collect results as they complete
            for future in as_completed(future_to_call):
                result = future.result()
                results.append(result)

                # Collect context updates (last-write-wins for conflicts)
                if result["success"] and result["context_updates"]:
                    all_context_updates.update(result["context_updates"])

        # Apply all context updates after all tools complete
        if all_context_updates:
            self.context.update(all_context_updates)

        return results

    def copy(self) -> "Toolkit":
        """Create an isolated copy of this toolkit for sub-agent execution.

        With immutable context data structures, this is now highly efficient:
        - Context copy is O(1) (reference to immutable Map)
        - Tool/filter dicts are shallow copied (tools are immutable)
        - Unlocked states set is shallow copied

        Returns:
            New Toolkit instance with isolated context and copied state
        """
        # Create new toolkit
        new_toolkit = Toolkit(validation_enabled=self.context.validation_enabled)

        # Copy context (O(1) with immutable Map!)
        new_toolkit.context = self.context.copy()

        # Copy tools (tools are immutable, so shallow copy is fine)
        new_toolkit.tools = self.tools.copy()

        # Copy state (shallow copy of set)
        new_toolkit.unlocked_states = self.unlocked_states.copy()

        # Copy filters
        new_toolkit.filters = self.filters.copy()

        return new_toolkit

    def __repr__(self) -> str:
        return (
            f"Toolkit(tools={len(self.tools)}, "
            f"unlocked_states={len(self.unlocked_states)}, "
            f"context_keys={len(self.context.to_dict())})"
        )
