"""Tests for ContextSchema class."""

import pytest

from bestla.yggdrasil import ContextValidationError
from bestla.yggdrasil.context import ContextSchema


class TestContextSchema:
    """Test ContextSchema functionality."""

    def test_create_context_schema(self):
        """Test creating a ContextSchema."""
        schema = ContextSchema()
        assert schema is not None
        assert schema._schemas == {}

    def test_define_schema(self):
        """Test defining a schema for a key."""
        schema = ContextSchema()

        schema.define("username", {"type": "string", "minLength": 3})

        assert "username" in schema._schemas
        assert schema._schemas["username"]["type"] == "string"

    def test_has_schema(self):
        """Test has_schema method."""
        schema = ContextSchema()

        # No schema defined yet
        assert not schema.has_schema("key")

        # Define schema
        schema.define("key", {"type": "string"})

        # Now has schema
        assert schema.has_schema("key")

    def test_has_schema_multiple_keys(self):
        """Test has_schema with multiple keys."""
        schema = ContextSchema()

        schema.define("key1", {"type": "string"})
        schema.define("key2", {"type": "integer"})

        assert schema.has_schema("key1")
        assert schema.has_schema("key2")
        assert not schema.has_schema("key3")

    def test_validate_valid_value(self):
        """Test validation with valid value."""
        schema = ContextSchema()
        schema.define("age", {"type": "integer", "minimum": 0, "maximum": 150})

        # Should not raise
        schema.validate("age", 25)
        schema.validate("age", 0)
        schema.validate("age", 150)

    def test_validate_invalid_type(self):
        """Test validation with invalid type."""
        schema = ContextSchema()
        schema.define("age", {"type": "integer"})

        with pytest.raises(ContextValidationError, match="age"):
            schema.validate("age", "not_an_integer")

    def test_validate_invalid_range(self):
        """Test validation with value out of range."""
        schema = ContextSchema()
        schema.define("priority", {"type": "integer", "minimum": 1, "maximum": 5})

        with pytest.raises(ContextValidationError, match="priority"):
            schema.validate("priority", 10)

        with pytest.raises(ContextValidationError, match="priority"):
            schema.validate("priority", 0)

    def test_validate_complex_schema(self):
        """Test validation with complex nested schema."""
        schema = ContextSchema()
        schema.define(
            "user",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                    "email": {"type": "string", "format": "email"},
                },
                "required": ["name", "age"],
            },
        )

        # Valid
        schema.validate("user", {"name": "Alice", "age": 30, "email": "alice@example.com"})

        # Invalid - missing required field
        with pytest.raises(ContextValidationError):
            schema.validate("user", {"name": "Bob"})

    def test_validate_no_schema_defined(self):
        """Test validation when no schema is defined for key."""
        schema = ContextSchema()

        # Should not raise - validation skipped when no schema
        schema.validate("undefined_key", "any_value")
        schema.validate("undefined_key", 123)
        schema.validate("undefined_key", {"complex": "object"})

    def test_validate_array_schema(self):
        """Test validation with array schema."""
        schema = ContextSchema()
        schema.define(
            "tags",
            {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
        )

        # Valid
        schema.validate("tags", ["tag1", "tag2"])

        # Invalid - wrong item type
        with pytest.raises(ContextValidationError):
            schema.validate("tags", [1, 2, 3])

        # Invalid - too many items
        with pytest.raises(ContextValidationError):
            schema.validate("tags", ["a", "b", "c", "d", "e", "f"])

    def test_validate_string_pattern(self):
        """Test validation with string pattern."""
        schema = ContextSchema()
        schema.define("issue_id", {"type": "string", "pattern": "^[A-Z]+-[0-9]+$"})

        # Valid
        schema.validate("issue_id", "BUG-123")
        schema.validate("issue_id", "FEAT-456")

        # Invalid - doesn't match pattern
        with pytest.raises(ContextValidationError):
            schema.validate("issue_id", "invalid")

    def test_validate_enum(self):
        """Test validation with enum."""
        schema = ContextSchema()
        schema.define("status", {"type": "string", "enum": ["open", "closed", "pending"]})

        # Valid
        schema.validate("status", "open")
        schema.validate("status", "closed")

        # Invalid - not in enum
        with pytest.raises(ContextValidationError):
            schema.validate("status", "invalid")

    def test_redefine_schema(self):
        """Test redefining schema for same key."""
        schema = ContextSchema()

        # Define initial schema
        schema.define("value", {"type": "string"})
        assert schema._schemas["value"]["type"] == "string"

        # Redefine with different schema
        schema.define("value", {"type": "integer"})
        assert schema._schemas["value"]["type"] == "integer"

    def test_validation_error_message_contains_details(self):
        """Test validation error message contains useful details."""
        schema = ContextSchema()
        schema.define("email", {"type": "string", "format": "email"})

        try:
            schema.validate("email", "not_an_email")
        except ContextValidationError as e:
            error_msg = str(e)
            assert "email" in error_msg
            # Error should contain validation details


class TestContextSchemaWithContext:
    """Test ContextSchema integration with Context."""

    def test_context_uses_schema_validation(self):
        """Test Context uses ContextSchema for validation."""
        from bestla.yggdrasil import Context

        context = Context(validation_enabled=True)
        context.schema.define("count", {"type": "integer", "minimum": 0})

        # Valid
        context.set("count", 5)
        assert context.get("count") == 5

        # Invalid
        with pytest.raises(ContextValidationError):
            context.set("count", -1)

    def test_context_update_uses_schema_validation(self):
        """Test Context.update() uses schema validation."""
        from bestla.yggdrasil import Context

        context = Context(validation_enabled=True)
        context.schema.define("age", {"type": "integer"})
        context.schema.define("name", {"type": "string"})

        # Valid update
        context.update({"age": 30, "name": "Alice"})
        assert context.get("age") == 30

        # Invalid update - should fail on first invalid value
        with pytest.raises(ContextValidationError):
            context.update({"age": "invalid", "name": "Bob"})

    def test_validation_disabled_ignores_schema(self):
        """Test validation disabled ignores schema."""
        from bestla.yggdrasil import Context

        context = Context(validation_enabled=False)
        context.schema.define("age", {"type": "integer"})

        # Should not raise even though value violates schema
        context.set("age", "not_an_integer")
        assert context.get("age") == "not_an_integer"

    def test_multiple_schemas(self):
        """Test defining multiple schemas."""
        from bestla.yggdrasil import Context

        context = Context(validation_enabled=True)

        # Define multiple schemas
        context.schema.define("username", {"type": "string", "minLength": 3})
        context.schema.define("age", {"type": "integer", "minimum": 0})
        context.schema.define("active", {"type": "boolean"})

        # All should be enforced
        context.set("username", "alice")
        context.set("age", 25)
        context.set("active", True)

        # Each should validate independently
        with pytest.raises(ContextValidationError):
            context.set("username", "ab")  # Too short

        with pytest.raises(ContextValidationError):
            context.set("age", -5)  # Negative

        with pytest.raises(ContextValidationError):
            context.set("active", "yes")  # Not boolean
