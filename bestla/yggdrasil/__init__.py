"""Yggdrasil - A stateful, hierarchical multi-agent tool framework.

Yggdrasil provides a framework for building AI agents that use tools with:
- Stateful toolkits with context management
- Dynamic schema generation based on runtime context
- Finite state machine (FSM) logic for tool availability
- Hierarchical agent composition (agents as tools)
- Three-tier concurrency model for safe parallel execution
"""

from bestla.yggdrasil.agent import Agent, ExecutionContext
from bestla.yggdrasil.context import Context, ContextSchema
from bestla.yggdrasil.context_manager import ContextManager
from bestla.yggdrasil.conversation_context import ConversationContext
from bestla.yggdrasil.decorators import (
    cache_result,
    rate_limit,
    retry,
    retry_async,
    timeout,
)
from bestla.yggdrasil.dynamic_types import (
    DynamicArray,
    DynamicConditional,
    DynamicConst,
    DynamicConstraints,
    DynamicFiltered,
    DynamicFloat,
    DynamicFormat,
    DynamicInt,
    DynamicNested,
    DynamicPattern,
    DynamicStr,
    DynamicUnion,
)
from bestla.yggdrasil.exceptions import (
    ContextRequirementError,
    ContextValidationError,
    ToolkitPipelineError,
    ToolNotAvailableError,
    YggdrasilError,
)
from bestla.yggdrasil.tool import Tool, tool
from bestla.yggdrasil.toolkit import Toolkit

__version__ = "0.2.0"

__all__ = [
    # Core classes
    "Agent",
    "ExecutionContext",
    "Toolkit",
    "Tool",
    "Context",
    "ContextSchema",
    "ContextManager",
    "ConversationContext",
    # Decorators
    "tool",
    # Dynamic types
    "DynamicStr",
    "DynamicInt",
    "DynamicFloat",
    "DynamicArray",
    "DynamicFormat",
    "DynamicPattern",
    "DynamicConst",
    "DynamicFiltered",
    "DynamicNested",
    "DynamicConstraints",
    "DynamicUnion",
    "DynamicConditional",
    # Exceptions
    "YggdrasilError",
    "ContextValidationError",
    "ToolNotAvailableError",
    "ToolkitPipelineError",
    "ContextRequirementError",
    # Decorators
    "retry",
    "retry_async",
    "timeout",
    "cache_result",
    "rate_limit",
]
