"""Tests for Context class."""

import pytest

from bestla.yggdrasil import Context, ContextValidationError


class TestContext:
    """Test Context functionality."""

    def test_create_context(self):
        """Test creating a context."""
        context = Context()
        assert context is not None
        assert context.to_dict() == {}

    def test_set_and_get(self):
        """Test setting and getting values."""
        context = Context()
        context.set("key1", "value1")
        assert context.get("key1") == "value1"
        assert context.get("missing", "default") == "default"

    def test_update(self):
        """Test updating multiple values."""
        context = Context()
        context.update({"key1": "value1", "key2": "value2"})
        assert context.get("key1") == "value1"
        assert context.get("key2") == "value2"

    def test_has_key(self):
        """Test checking if key exists."""
        context = Context()
        context.set("existing", "value")
        assert context.has("existing")
        assert not context.has("missing")

    def test_nested_access(self):
        """Test nested key access with dot notation."""
        context = Context()
        context.set("project", {"id": "p1", "name": "Test Project"})
        assert context.get("project.id") == "p1"
        assert context.get("project.name") == "Test Project"
        assert context.get("project.missing") is None

    def test_delete(self):
        """Test deleting keys."""
        context = Context()
        context.set("key", "value")
        assert context.has("key")
        context.delete("key")
        assert not context.has("key")

    def test_clear(self):
        """Test clearing all data."""
        context = Context()
        context.update({"key1": "value1", "key2": "value2"})
        context.clear()
        assert context.to_dict() == {}

    def test_copy(self):
        """Test deep copying context."""
        context = Context()
        context.set("key", {"nested": "value"})

        copy = context.copy()
        copy.set("key2", "value2")

        # Original unchanged
        assert not context.has("key2")

        # Copy has both
        assert copy.has("key")
        assert copy.has("key2")

    def test_dict_interface(self):
        """Test dictionary-like interface."""
        context = Context()
        context["key"] = "value"
        assert context["key"] == "value"
        assert "key" in context

        with pytest.raises(KeyError):
            _ = context["missing"]

    def test_validation_enabled(self):
        """Test validation with schemas."""
        context = Context(validation_enabled=True)
        context.schema.define("priority", {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        })

        # Valid value
        context.set("priority", 3)
        assert context.get("priority") == 3

        # Invalid value
        with pytest.raises(ContextValidationError):
            context.set("priority", 10)

    def test_validation_array(self):
        """Test validation with array schema."""
        context = Context(validation_enabled=True)
        context.schema.define("tags", {
            "type": "array",
            "items": {"type": "string"}
        })

        # Valid
        context.set("tags", ["tag1", "tag2"])

        # Invalid
        with pytest.raises(ContextValidationError):
            context.set("tags", [1, 2, 3])

    def test_no_validation_when_disabled(self):
        """Test that validation doesn't run when disabled."""
        context = Context(validation_enabled=False)
        context.schema.define("priority", {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        })

        # Should not raise even though invalid
        context.set("priority", 10)
        assert context.get("priority") == 10
