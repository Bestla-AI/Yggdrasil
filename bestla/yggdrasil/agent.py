"""Agent class for orchestrating toolkits and LLM interactions."""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from bestla.yggdrasil.exceptions import ToolkitPipelineError
from bestla.yggdrasil.tool import Tool
from bestla.yggdrasil.toolkit import Toolkit


class ExecutionContext:
    """Holds execution-specific state for thread-safe agent execution.

    This context is created per execution to ensure isolation when agents
    are used as tools and run in parallel. All toolkits are copied using
    their custom copy() method to prevent state conflicts during simultaneous
    executions.

    Attributes:
        toolkits: Dict mapping prefix to copied Toolkit instances
        independent_toolkit: Copied Toolkit for independent tools (no prefix)
    """

    def __init__(
        self,
        toolkits: Dict[str, Toolkit],
        independent_toolkit: Toolkit,
    ):
        """Initialize execution context with copies of toolkits.

        Args:
            toolkits: Dict mapping prefix to Toolkit instance
            independent_toolkit: Toolkit for independent tools
        """
        # Copy all toolkits using their custom copy() method
        # This ensures proper handling of tools, filters, and context
        self.toolkits = {
            prefix: toolkit.copy()
            for prefix, toolkit in toolkits.items()
        }
        self.independent_toolkit = independent_toolkit.copy()


def _group_tool_calls_by_toolkit(
        tool_calls: List[dict]
) -> Dict[str, List[dict]]:
    """Group tool calls by their toolkit.

    Args:
        tool_calls: List of tool call dicts from LLM

    Returns:
        Dict mapping toolkit identifier to list of calls
        Special key "independent" for independent tools
    """
    groups = defaultdict(list)

    for call in tool_calls:
        tool_name = call["function"]["name"]

        # Check if it's a prefixed tool
        if "::" in tool_name:
            prefix, base_name = tool_name.split("::", 1)
            # Remove prefix for toolkit execution
            groups[prefix].append(
                {
                    "id": call["id"],
                    "name": base_name,
                    "arguments": json.loads(call["function"]["arguments"]),
                }
            )
        else:
            # Independent tool
            groups["independent"].append(
                {
                    "id": call["id"],
                    "name": tool_name,
                    "arguments": json.loads(call["function"]["arguments"]),
                }
            )

    return dict(groups)


class Agent:
    """Orchestrates multiple toolkits and manages LLM conversation loop.

    Agents:
    - Register toolkits with namespace prefixes (plane::, github::, etc.)
    - Group tool calls by toolkit
    - Execute toolkit groups in parallel
    - Execute tools within toolkit sequentially
    - Can be used as a Tool by other agents (hierarchical composition)
    """

    def __init__(
            self,
            provider: OpenAI,
            model: str,
            system_prompt: str | None = None,
    ):
        """Initialize agent.

        Args:
            provider: OpenAI client instance
            model: model name
            system_prompt: System prompt for the agent
        """
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self._provider = provider

        # Toolkit management
        self.toolkits: Dict[str, Toolkit] = {}
        self.toolkit_prefixes: Dict[str, str] = {}  # tool_name -> prefix

        # Independent tools (no prefix, always parallel)
        self.independent_toolkit = Toolkit()
        # Independent tools have no state requirements, so they're always available

        # Conversation history
        self.messages: List[ChatCompletionMessageParam] = []

    @property
    def provider(self) -> OpenAI:
        """Get OpenAI provider."""
        return self._provider

    @provider.setter
    def provider(self, value: OpenAI) -> None:
        """Set OpenAI provider."""
        self._provider = value

    def add_toolkit(self, prefix: str, toolkit: Toolkit) -> None:
        """Register a toolkit with a namespace prefix.

        Args:
            prefix: Namespace prefix (e.g., "plane", "github")
            toolkit: Toolkit instance

        Example:
            agent.add_toolkit("plane", PlaneToolkit())
            # Tools become: plane::list_issues, plane::get_issue, etc.
        """
        self.toolkits[prefix] = toolkit

        # Track which tools belong to this toolkit
        for tool_name in toolkit.tools.keys():
            prefixed_name = f"{prefix}::{tool_name}"
            self.toolkit_prefixes[prefixed_name] = prefix

    def add_tool(
            self,
            name: str,
            function: Callable,
            description: str | None = None,
    ) -> Tool:
        """Add an independent tool (executed in parallel).

        Args:
            name: Tool name
            function: Python function
            description: Tool description

        Returns:
            Created Tool instance
        """
        tool = self.independent_toolkit.add_tool(
            name=name,
            function=function,
            description=description,
        )
        # Independent tools have no state requirements, so they're automatically available
        return tool

    def register_tool(self, tool: Tool) -> None:
        """Register an existing Tool as independent tool.

        Args:
            tool: Tool instance
        """
        self.independent_toolkit.register_tool(tool)
        # Independent tools are automatically available based on their state requirements

    def _generate_all_schemas(self, context: ExecutionContext) -> List[dict]:
        """Generate schemas for all available tools from all toolkits.

        Args:
            context: Execution context containing toolkits

        Returns:
            List of OpenAI tool schemas with prefixed names
        """
        schemas = []

        # Get schemas from all toolkits (with prefixes)
        for prefix, toolkit in context.toolkits.items():
            toolkit_schemas = toolkit.generate_schemas()
            # Add prefix to tool names
            for schema in toolkit_schemas:
                schema["function"]["name"] = f"{prefix}::{schema['function']['name']}"
                schemas.append(schema)

        # Get schemas from independent tools (no prefix)
        independent_schemas = context.independent_toolkit.generate_schemas()
        schemas.extend(independent_schemas)

        return schemas

    def _execute_toolkit_group(
            self, context: ExecutionContext, prefix: str, tool_calls: List[dict]
    ) -> List[ChatCompletionToolMessageParam]:
        """Execute a group of tool calls for a specific toolkit.

        Args:
            context: Execution context containing isolated toolkit copies
            prefix: Toolkit prefix (or "independent")
            tool_calls: List of tool calls for this toolkit

        Returns:
            List of results with tool_call_id, content, and error info
        """
        if prefix == "independent":
            toolkit = context.independent_toolkit
        elif prefix in context.toolkits:
            toolkit = context.toolkits[prefix]
        else:
            # Unknown toolkit
            return [
                ChatCompletionToolMessageParam(
                    tool_call_id=call["id"],
                    role="tool",
                    content=f"Error: Unknown toolkit '{prefix}'",
                )
                for call in tool_calls
            ]

        try:
            # Execute tools: parallel for independent, sequential for toolkits
            if prefix == "independent":
                results = toolkit.execute_parallel(tool_calls)
            else:
                results = toolkit.execute_sequential(tool_calls)

            # Format results for LLM
            formatted_results = []
            for call, result in zip(tool_calls, results):
                if result["success"]:
                    formatted_results.append(
                        ChatCompletionToolMessageParam(
                            tool_call_id=call["id"],
                            role="tool",
                            content=str(result["result"]),
                        )
                    )
                else:
                    formatted_results.append(
                        ChatCompletionToolMessageParam(
                            tool_call_id=call["id"],
                            role="tool",
                            content=f"Error: {result['error']}",
                        )
                    )

            return formatted_results

        except ToolkitPipelineError as e:
            # Pipeline failed, return partial results
            formatted_results = []
            for idx, call in enumerate(tool_calls):
                if idx < len(e.partial_results):
                    result = e.partial_results[idx]
                    if result["success"]:
                        formatted_results.append(
                            ChatCompletionToolMessageParam(
                                tool_call_id=call["id"],
                                role="tool",
                                content=str(result["result"]),
                            )
                        )
                    else:
                        formatted_results.append(
                            ChatCompletionToolMessageParam(
                                tool_call_id=call["id"],
                                role="tool",
                                content=f"Error: {result['error']}",
                            )
                        )
                else:
                    formatted_results.append(
                        ChatCompletionToolMessageParam(
                            tool_call_id=call["id"],
                            role="tool",
                            content="Error: Pipeline failed before this tool executed",
                        )
                    )

            return formatted_results

    def _execute_tool_calls(
            self, context: ExecutionContext, tool_calls: List[Any]
    ) -> List[ChatCompletionToolMessageParam]:
        """Execute tool calls with three-tier concurrency model.

        1. Independent tools run in parallel
        2. Tools within same toolkit run sequentially
        3. Different toolkits run in parallel to each other

        Args:
            context: Execution context containing isolated toolkit copies
            tool_calls: List of tool call objects from LLM response

        Returns:
            List of tool result messages
        """
        # Group by toolkit
        groups = _group_tool_calls_by_toolkit(tool_calls)

        # Execute toolkit groups in parallel
        all_results = []

        if len(groups) == 1:
            # Only one group, execute directly (no parallelism needed)
            prefix, calls = next(iter(groups.items()))
            all_results = self._execute_toolkit_group(context, prefix, calls)
        else:
            # Multiple groups, execute in parallel
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self._execute_toolkit_group, context, prefix, calls): prefix
                    for prefix, calls in groups.items()
                }

                for future in as_completed(futures):
                    results = future.result()
                    all_results.extend(results)

        return all_results

    def run(
            self,
            user_message: str,
            max_iterations: int = 100,
            stream: bool = False,
    ) -> str:
        """Run the agent conversation loop.

        Args:
            user_message: User input
            max_iterations: Maximum number of LLM calls (prevents infinite loops)
            stream: Whether to stream responses (not implemented yet)

        Returns:
            Final agent response
        """
        # Create execution context with deep copies for state isolation
        context = ExecutionContext(
            toolkits=self.toolkits,
            independent_toolkit=self.independent_toolkit,
        )

        # Add system message if not already present
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(
                0, ChatCompletionSystemMessageParam(
                    role="system",
                    content=self.system_prompt
                )
            )

        # Add user message
        self.messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content=user_message,
            )
        )

        for iteration in range(max_iterations):
            # Generate tool schemas based on current context
            tools = self._generate_all_schemas(context)

            # Call LLM
            response = self.provider.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=tools if tools else [],
            )

            message = response.choices[0].message

            # Add assistant message to history
            self.messages.append(ChatCompletionAssistantMessageParam(
                role="assistant",
                content=message.content,
            ))

            # Check if we're done
            if not message.tool_calls:
                # No tool calls, return response
                return message.content or ""

            # Execute tool calls
            tool_results = self._execute_tool_calls(context, message.tool_calls)

            # Add tool results to messages
            self.messages.extend(tool_results)

            # Continue loop (AI will see tool results and respond)

        # Max iterations reached
        return "Maximum iterations reached. Please try rephrasing your request."

    def execute(self, query: str) -> Tuple[str, dict]:
        """Execute agent as a tool (for hierarchical composition).

        This allows agents to be called by other agents.
        State isolation is handled by ExecutionContext (deep copies).

        Args:
            query: Input query

        Returns:
            Tuple of (response_string, empty_dict)
            Context updates are always empty (organizational model)
        """
        # Run agent loop - ExecutionContext handles state isolation
        response = self.run(query)

        # Return response with no context updates (isolation)
        return response, {}

    def clear_messages(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    def __repr__(self) -> str:
        return (
            f"Agent(model='{self.model}', "
            f"toolkits={len(self.toolkits)}, "
            f"independent_tools={len(self.independent_toolkit.tools)})"
        )
