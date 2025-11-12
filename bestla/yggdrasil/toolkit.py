"""Toolkit class for managing related tools with shared context."""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from bestla.yggdrasil.context import Context
from bestla.yggdrasil.tool import Tool
from bestla.yggdrasil.exceptions import (
    ToolNotAvailableError,
    ContextRequirementError,
    ToolkitPipelineError,
)


class Toolkit:
    """Manages collection of related tools with shared context and FSM logic.

    Toolkits:
    - Own a Context instance for domain state
    - Manage tool availability via FSM (unlocks/locks)
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
        self.available_tools: Set[str] = set()
        self.locked_tools: Set[str] = set()
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
        requires_context: Optional[List[str]] = None,
        provides_context: Optional[List[str]] = None,
        unlocks: Optional[List[str]] = None,
        locks: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> Tool:
        """Add a tool to this toolkit.

        Args:
            name: Tool name
            function: Python function to wrap
            requires_context: Context keys required
            provides_context: Context keys provided
            unlocks: Tools to unlock after execution
            locks: Tools to lock after execution
            description: Tool description

        Returns:
            Created Tool instance
        """
        tool = Tool(
            function=function,
            name=name,
            description=description,
            requires_context=requires_context,
            provides_context=provides_context,
            unlocks=unlocks,
            locks=locks,
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

    def set_available_tools(self, tool_names: Set[str]) -> None:
        """Set which tools are initially available.

        Args:
            tool_names: Set of tool names to make available
        """
        self.available_tools = tool_names.copy()

    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is currently available.

        A tool is available if:
        1. It's in the available_tools set
        2. It's not in the locked_tools set
        3. Its context requirements are met

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is available
        """
        if tool_name not in self.tools:
            return False

        # Check if locked
        if tool_name in self.locked_tools:
            return False

        # Check if in available set
        if tool_name not in self.available_tools:
            return False

        # Check context requirements
        tool = self.tools[tool_name]
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

        if tool_name in self.locked_tools:
            return "Tool is locked"

        if tool_name not in self.available_tools:
            return "Tool not in available set"

        tool = self.tools[tool_name]
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

                # Update FSM: process unlocks
                if tool.unlocks:
                    for unlock_name in tool.unlocks:
                        self.available_tools.add(unlock_name)
                        # Remove from locked set if present
                        self.locked_tools.discard(unlock_name)

                # Update FSM: process locks
                if tool.locks:
                    for lock_name in tool.locks:
                        self.locked_tools.add(lock_name)
                        # Optionally remove from available set
                        # (locked tools are checked separately in is_tool_available)

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
        # Create context snapshot for all tools
        context_snapshot = self.context.to_dict()

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
        """Create a deep copy of this toolkit for sub-agent isolation.

        Returns:
            New Toolkit instance with copied context and tools
        """
        # Create new toolkit
        new_toolkit = Toolkit(validation_enabled=self.context.validation_enabled)

        # Deep copy context
        new_toolkit.context = self.context.copy()

        # Copy tools (tools are immutable, so shallow copy is fine)
        new_toolkit.tools = self.tools.copy()

        # Copy availability state
        new_toolkit.available_tools = self.available_tools.copy()
        new_toolkit.locked_tools = self.locked_tools.copy()

        # Copy filters
        new_toolkit.filters = self.filters.copy()

        return new_toolkit

    def __repr__(self) -> str:
        return (
            f"Toolkit(tools={len(self.tools)}, "
            f"available={len(self.available_tools)}, "
            f"context_keys={len(self.context.to_dict())})"
        )
