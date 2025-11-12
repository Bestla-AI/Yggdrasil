"""Custom exceptions for Yggdrasil framework."""


class YggdrasilError(Exception):
    """Base exception for all Yggdrasil errors."""
    pass


class ContextValidationError(YggdrasilError):
    """Raised when context validation fails."""
    pass


class ToolNotAvailableError(YggdrasilError):
    """Raised when a tool is called but not currently available."""

    def __init__(self, tool_name: str, reason: str):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' not available: {reason}")


class ToolkitPipelineError(YggdrasilError):
    """Raised when a toolkit pipeline fails partway through execution."""

    def __init__(self, message: str, partial_results: list, failed_at: int):
        self.partial_results = partial_results
        self.failed_at = failed_at
        super().__init__(message)


class ContextRequirementError(YggdrasilError):
    """Raised when required context keys are missing."""

    def __init__(self, tool_name: str, missing_keys: list):
        self.tool_name = tool_name
        self.missing_keys = missing_keys
        super().__init__(
            f"Tool '{tool_name}' requires context keys: {', '.join(missing_keys)}"
        )
