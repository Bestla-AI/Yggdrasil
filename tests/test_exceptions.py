"""Tests for custom exception classes."""

import pytest

from bestla.yggdrasil.exceptions import (
    ContextRequirementError,
    ContextValidationError,
    ToolkitPipelineError,
    ToolNotAvailableError,
    YggdrasilError,
)


class TestYggdrasilError:
    """Test base YggdrasilError."""

    def test_base_exception(self):
        """Test base exception can be raised."""
        with pytest.raises(YggdrasilError, match="Base error"):
            raise YggdrasilError("Base error")

    def test_inheritance(self):
        """Test YggdrasilError is an Exception."""
        error = YggdrasilError("test")
        assert isinstance(error, Exception)


class TestContextValidationError:
    """Test ContextValidationError."""

    def test_context_validation_error(self):
        """Test ContextValidationError creation and message."""
        error = ContextValidationError("Value does not match schema")

        assert isinstance(error, YggdrasilError)
        assert "Value does not match schema" in str(error)

    def test_raised_on_validation_failure(self):
        """Test error raised on validation failure."""
        from bestla.yggdrasil import Context

        context = Context(validation_enabled=True)
        context.schema.define("age", {"type": "integer"})

        with pytest.raises(ContextValidationError, match="age"):
            context.set("age", "not_an_integer")


class TestToolNotAvailableError:
    """Test ToolNotAvailableError."""

    def test_tool_not_available_error_creation(self):
        """Test ToolNotAvailableError with tool name and reason."""
        error = ToolNotAvailableError("my_tool", "state not enabled")

        assert isinstance(error, YggdrasilError)
        assert error.tool_name == "my_tool"
        assert error.reason == "state not enabled"

    def test_error_message_format(self):
        """Test error message contains tool name and reason."""
        error = ToolNotAvailableError("protected_action", "missing auth token")

        error_str = str(error)
        assert "protected_action" in error_str
        assert "missing auth token" in error_str
        assert "not available" in error_str

    def test_multiple_reasons(self):
        """Test with complex reason."""
        error = ToolNotAvailableError(
            "complex_tool", "missing states: authenticated, project_selected"
        )

        assert error.tool_name == "complex_tool"
        assert "authenticated" in error.reason
        assert "project_selected" in error.reason


class TestToolkitPipelineError:
    """Test ToolkitPipelineError."""

    def test_toolkit_pipeline_error_creation(self):
        """Test ToolkitPipelineError with partial results."""
        partial_results = [
            {"name": "step1", "success": True, "result": "ok"},
            {"name": "step2", "success": False, "error": "Failed"},
        ]

        error = ToolkitPipelineError(
            message="Pipeline failed at step 2", partial_results=partial_results, failed_at=1
        )

        assert isinstance(error, YggdrasilError)
        assert error.partial_results == partial_results
        assert error.failed_at == 1
        assert "Pipeline failed at step 2" in str(error)

    def test_empty_partial_results(self):
        """Test pipeline error with no partial results."""
        error = ToolkitPipelineError(
            message="Failed immediately", partial_results=[], failed_at=0
        )

        assert error.partial_results == []
        assert error.failed_at == 0

    def test_raised_in_pipeline(self):
        """Test error raised during toolkit pipeline execution."""
        from typing import Tuple

        from bestla.yggdrasil import Toolkit, tool

        toolkit = Toolkit()

        @tool()
        def step1() -> Tuple[str, dict]:
            return "success", {}

        @tool()
        def step2_fail() -> Tuple[str, dict]:
            raise ValueError("Step 2 intentionally fails")

        toolkit.register_tool(step1)
        toolkit.register_tool(step2_fail)

        with pytest.raises(ToolkitPipelineError) as exc_info:
            toolkit.execute_sequential(
                [{"name": "step1", "arguments": {}}, {"name": "step2_fail", "arguments": {}}]
            )

        error = exc_info.value
        assert len(error.partial_results) == 2
        assert error.partial_results[0]["success"]
        assert not error.partial_results[1]["success"]


class TestContextRequirementError:
    """Test ContextRequirementError."""

    def test_context_requirement_error_creation(self):
        """Test ContextRequirementError with tool name and missing keys."""
        error = ContextRequirementError("protected_tool", ["auth_token", "user_id"])

        assert isinstance(error, YggdrasilError)
        assert error.tool_name == "protected_tool"
        assert error.missing_keys == ["auth_token", "user_id"]

    def test_error_message_format(self):
        """Test error message contains tool name and missing keys."""
        error = ContextRequirementError("my_tool", ["key1", "key2", "key3"])

        error_str = str(error)
        assert "my_tool" in error_str
        assert "key1" in error_str
        assert "key2" in error_str
        assert "key3" in error_str
        assert "requires context keys" in error_str.lower()

    def test_single_missing_key(self):
        """Test with single missing key."""
        error = ContextRequirementError("simple_tool", ["config"])

        assert error.missing_keys == ["config"]
        assert "config" in str(error)

    def test_empty_missing_keys(self):
        """Test with empty missing keys list."""
        error = ContextRequirementError("tool", [])

        assert error.missing_keys == []
        # Should still create error even with empty list


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""

    def test_all_inherit_from_yggdrasil_error(self):
        """Test all custom exceptions inherit from YggdrasilError."""
        assert issubclass(ContextValidationError, YggdrasilError)
        assert issubclass(ToolNotAvailableError, YggdrasilError)
        assert issubclass(ToolkitPipelineError, YggdrasilError)
        assert issubclass(ContextRequirementError, YggdrasilError)

    def test_catch_base_exception(self):
        """Test catching YggdrasilError catches all custom exceptions."""
        # ContextValidationError
        try:
            raise ContextValidationError("test")
        except YggdrasilError:
            pass  # Should catch it

        # ToolNotAvailableError
        try:
            raise ToolNotAvailableError("tool", "reason")
        except YggdrasilError:
            pass  # Should catch it

        # ToolkitPipelineError
        try:
            raise ToolkitPipelineError("msg", [], 0)
        except YggdrasilError:
            pass  # Should catch it

        # ContextRequirementError
        try:
            raise ContextRequirementError("tool", ["key"])
        except YggdrasilError:
            pass  # Should catch it
