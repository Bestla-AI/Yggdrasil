"""Agent class for orchestrating toolkits and LLM interactions."""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from bestla.yggdrasil.conversation_context import ConversationContext
from bestla.yggdrasil.exceptions import ToolkitPipelineError
from bestla.yggdrasil.tool import Tool
from bestla.yggdrasil.toolkit import Toolkit

if TYPE_CHECKING:
    pass


class ExecutionContext:
    """Holds execution-specific state for thread-safe agent execution.

    This context is created per execution to ensure isolation when agents
    are used as tools and run in parallel. All toolkits are copied using
    their custom copy() method to prevent state conflicts during simultaneous
    executions.

    Attributes:
        toolkits: Dict mapping prefix to copied Toolkit instances
        independent_toolkit: Copied Toolkit for independent tools (no prefix)
        conversation: ConversationContext managing messages and compaction
    """

    def __init__(
        self,
        toolkits: Dict[str, Toolkit],
        independent_toolkit: Toolkit,
        conversation_context: ConversationContext | None = None,
    ):
        """Initialize execution context with copies of toolkits.

        Args:
            toolkits: Dict mapping prefix to Toolkit instance
            independent_toolkit: Toolkit for independent tools
            conversation_context: Optional ConversationContext to use.
                                If not provided, creates a new one.
        """
        # Deep copy toolkits for state isolation
        self.toolkits = {
            prefix: toolkit.copy()
            for prefix, toolkit in toolkits.items()
        }
        self.independent_toolkit = independent_toolkit.copy()

        # Conversation shared (not copied) to allow inspection after run
        self.conversation = conversation_context or ConversationContext()


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

        if "::" in tool_name:
            prefix, base_name = tool_name.split("::", 1)
            groups[prefix].append(
                {
                    "id": call["id"],
                    "name": base_name,
                    "arguments": json.loads(call["function"]["arguments"]),
                }
            )
        else:
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

    Warning:
        Agent instances are not thread-safe. Do not modify conversation state
        (messages, context_manager, etc.) from multiple threads concurrently.
        For concurrent execution, create separate Agent instances or use
        separate ExecutionContext instances.

    Example:
        # SAFE: Separate agents for concurrent execution
        agent1 = Agent(provider=client, model="gpt-4")
        agent2 = Agent(provider=client, model="gpt-4")
        with ThreadPoolExecutor() as executor:
            future1 = executor.submit(agent1.run, "Query 1")
            future2 = executor.submit(agent2.run, "Query 2")

        # UNSAFE: Same agent from multiple threads
        agent = Agent(provider=client, model="gpt-4")
        with ThreadPoolExecutor() as executor:
            future1 = executor.submit(agent.run, "Query 1")  # Race condition!
            future2 = executor.submit(agent.run, "Query 2")  # Race condition!
    """

    def __init__(
            self,
            provider: OpenAI,
            model: str,
            system_prompt: str | None = None,
    ):
        """Initialize agent (stateless - no conversation state).

        The Agent is a stateless, reusable tool. Each run() creates a fresh
        ExecutionContext by default, making runs independent. Conversation state
        is managed through ExecutionContext, not the Agent.

        Args:
            provider: OpenAI client instance
            model: Model name to use
            system_prompt: System prompt for the agent

        Example:

            agent = Agent(provider=client, model="gpt-4")


            response1, ctx1 = agent.run("Research quantum computing")
            response2, ctx2 = agent.run("Research neural networks")


            print(ctx1.conversation.messages)  # Quantum conversation
            print(ctx2.conversation.messages)  # Neural conversation


            response3, ctx3 = agent.run("What about applications?", execution_context=ctx1)


            cm = ContextManager(threshold=100000)
            conv = ConversationContext(context_manager=cm)
            ctx_with_cm = ExecutionContext(
                toolkits=agent.toolkits,
                independent_toolkit=agent.independent_toolkit,
                conversation_context=conv
            )
            response4, ctx4 = agent.run("Long task", execution_context=ctx_with_cm)
        """
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self._provider = provider

        self.toolkits: Dict[str, Toolkit] = {}
        self.toolkit_prefixes: Dict[str, str] = {}
        self.independent_toolkit = Toolkit()

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
        return tool

    def register_tool(self, tool: Tool) -> None:
        """Register an existing Tool as independent tool.

        Args:
            tool: Tool instance
        """
        self.independent_toolkit.register_tool(tool)

    def _generate_all_schemas(self, context: ExecutionContext) -> List[dict]:
        """Generate schemas for all available tools from all toolkits.

        Args:
            context: Execution context containing toolkits

        Returns:
            List of OpenAI tool schemas with prefixed names
        """
        schemas = []

        for prefix, toolkit in context.toolkits.items():
            toolkit_schemas = toolkit.generate_schemas()
            for schema in toolkit_schemas:
                schema["function"]["name"] = f"{prefix}::{schema['function']['name']}"
                schemas.append(schema)
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
            return [
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=call["id"],
                    content=f"Error: Unknown toolkit '{prefix}'",
                )
                for call in tool_calls
            ]

        try:
            if prefix == "independent":
                results = toolkit.execute_parallel(tool_calls)
            else:
                results = toolkit.execute_sequential(tool_calls)

            formatted_results = []
            for call, result in zip(tool_calls, results):
                if result["success"]:
                    formatted_results.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=call["id"],
                            content=str(result["result"]),
                        )
                    )
                else:
                    formatted_results.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=call["id"],
                            content=f"Error: {result['error']}",
                        )
                    )

            return formatted_results

        except ToolkitPipelineError as e:
            formatted_results = []
            for idx, call in enumerate(tool_calls):
                if idx < len(e.partial_results):
                    result = e.partial_results[idx]
                    if result["success"]:
                        formatted_results.append(
                            ChatCompletionToolMessageParam(
                                role="tool",
                                tool_call_id=call["id"],
                                content=str(result["result"]),
                            )
                        )
                    else:
                        formatted_results.append(
                            ChatCompletionToolMessageParam(
                                role="tool",
                                tool_call_id=call["id"],
                                content=f"Error: {result['error']}",
                            )
                        )
                else:
                    formatted_results.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=call["id"],
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
        groups = _group_tool_calls_by_toolkit(tool_calls)
        all_results = []

        if len(groups) == 1:
            prefix, calls = next(iter(groups.items()))
            all_results = self._execute_toolkit_group(context, prefix, calls)
        else:
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
            execution_context: ExecutionContext | None = None,
    ) -> Tuple[str, ExecutionContext]:
        """Run the agent conversation loop (stateless execution).

        The agent is stateless - each run creates a fresh ExecutionContext by default,
        making runs independent. Pass an ExecutionContext to continue a conversation.

        Args:
            user_message: User input
            max_iterations: Maximum number of LLM calls (prevents infinite loops)
            stream: Whether to stream responses (not implemented yet)
            execution_context: Optional ExecutionContext to continue from.
                             If None, creates fresh context (stateless execution).

        Returns:
            Tuple of (final_response, execution_context)
            - final_response: Agent's final text response
            - execution_context: ExecutionContext with full conversation history

        Example:
            agent = Agent(provider=client, model="gpt-4")

            # Stateless (default) - Each run is independent
            response1, ctx1 = agent.run("Research quantum computing")
            response2, ctx2 = agent.run("Research neural networks")


            print(ctx1.conversation.messages)  # Quantum conversation
            print(ctx2.conversation.messages)  # Neural conversation

            # Stateful (explicit continuity)
            response3, ctx3 = agent.run("What about applications?", execution_context=ctx1)

            # With context management
            cm = ContextManager(threshold=100000)
            conv = ConversationContext(context_manager=cm)
            ctx_managed = ExecutionContext(
                toolkits=agent.toolkits,
                independent_toolkit=agent.independent_toolkit,
                conversation_context=conv
            )
            response4, ctx4 = agent.run("Long task", execution_context=ctx_managed)
        """
        # Create FRESH ExecutionContext by default (stateless)
        if execution_context is None:
            execution_context = ExecutionContext(
                toolkits=self.toolkits,
                independent_toolkit=self.independent_toolkit,
                conversation_context=ConversationContext(),  # Fresh conversation!
            )

        conv = execution_context.conversation

        if not conv.messages or conv.messages[0].get("role") != "system":
            conv.messages.insert(
                0,
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=self.system_prompt,
                ),
            )

        conv.messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content=user_message,
            )
        )

        for iteration in range(max_iterations):
            if conv.should_compact():
                conv.compact()

            tools = self._generate_all_schemas(execution_context)

            response = self.provider.chat.completions.create(
                model=self.model,
                messages=conv.messages,
                tools=tools if tools else [],
            )

            message = response.choices[0].message

            conv.messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=message.content,
                )
            )

            if not message.tool_calls:
                return message.content or "", execution_context

            tool_results = self._execute_tool_calls(execution_context, message.tool_calls)
            conv.messages.extend(tool_results)

        return "Maximum iterations reached. Please try rephrasing your request.", execution_context

    def execute(self, query: str) -> Tuple[str, dict]:
        """Execute agent as a tool (for hierarchical composition).

        This allows agents to be called by other agents.
        Each execution is stateless - creates fresh ExecutionContext.

        Args:
            query: Input query

        Returns:
            Tuple of (response_string, empty_dict)
            Context updates are always empty (sub-agents are isolated)

        Example:
            # Agent as tool
            research_agent = Agent(provider=client, model="gpt-4")
            main_agent = Agent(provider=client, model="gpt-4")
            main_agent.add_tool("research", research_agent.execute)

            # When main agent calls research tool:
            # 1. research_agent.execute("query") is called
            # 2. Creates fresh ExecutionContext (isolated)
            # 3. Returns response, discards execution context
            # 4. Main agent sees only the response
        """
        response, _execution_context = self.run(query)
        return response, {}  # Discard execution context for sub-agent isolation

    def clear_messages(self) -> None:
        """DEPRECATED: Agent is now stateless.

        This method is a no-op for backward compatibility.
        To clear conversation state, create a fresh ExecutionContext instead.

        Deprecated:
            Use fresh ExecutionContext for each run() instead:

            # Old (deprecated):
            agent.clear_messages()
            agent.run("New query")

            # New (recommended):
            response, ctx = agent.run("New query")  # Fresh context automatically!
        """
        pass

    def __repr__(self) -> str:
        return (
            f"Agent(model='{self.model}', "
            f"toolkits={len(self.toolkits)}, "
            f"independent_tools={len(self.independent_toolkit.tools)})"
        )
