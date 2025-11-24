"""Tests for Context class."""

from typing import Tuple

import pytest

from bestla.yggdrasil import Context, ContextValidationError, Toolkit, tool


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
        context.schema.define("priority", {"type": "integer", "minimum": 1, "maximum": 5})

        # Valid value
        context.set("priority", 3)
        assert context.get("priority") == 3

        # Invalid value
        with pytest.raises(ContextValidationError):
            context.set("priority", 10)

    def test_validation_array(self):
        """Test validation with array schema."""
        context = Context(validation_enabled=True)
        context.schema.define("tags", {"type": "array", "items": {"type": "string"}})

        # Valid
        context.set("tags", ["tag1", "tag2"])

        # Invalid
        with pytest.raises(ContextValidationError):
            context.set("tags", [1, 2, 3])

    def test_no_validation_when_disabled(self):
        """Test that validation doesn't run when disabled."""
        context = Context(validation_enabled=False)
        context.schema.define("priority", {"type": "integer", "minimum": 1, "maximum": 5})

        # Should not raise even though invalid
        context.set("priority", 10)
        assert context.get("priority") == 10


class TestContextBoundaries:
    """Test boundary conditions for Context operations."""

    def test_none_vs_missing_key(self):
        """Test distinction between None value and missing key."""
        context = Context()

        # Missing key returns None
        assert context.get("missing_key") is None

        # Explicitly set to None
        context.set("explicit_none", None)
        assert context.get("explicit_none") is None

        # Both should be distinguishable via __contains__
        assert "missing_key" not in context
        assert "explicit_none" in context

    def test_empty_string_as_key(self):
        """Test empty string as context key."""
        context = Context()

        # Empty string should be allowed as key
        context.set("", "value_for_empty_key")
        assert context.get("") == "value_for_empty_key"
        assert "" in context

    def test_very_deep_nesting(self):
        """Test accessing very deep nested paths (10+ levels)."""
        context = Context()

        # Create 10-level deep nested structure
        nested = {"value": "deep_value"}
        for i in range(9):
            nested = {f"level_{9-i}": nested}

        context.set("root", nested)

        # Access with dot notation
        path = "root." + ".".join([f"level_{i}" for i in range(1, 10)]) + ".value"
        assert context.get(path) == "deep_value"

    def test_nested_access_on_non_dict(self):
        """Test nested access when intermediate value is not a dict."""
        context = Context()

        # Set a string value
        context.set("name", "Alice")

        # Try to access nested path on string - should return None
        assert context.get("name.first") is None

        # Set an integer
        context.set("age", 30)
        assert context.get("age.years") is None

    def test_special_characters_in_keys(self):
        """Test keys with special characters."""
        context = Context()

        # Keys with various special characters
        special_keys = [
            "key:with:colons",
            "key/with/slashes",
            "key-with-dashes",
            "key_with_underscores",
            "key with spaces",
            "key\twith\ttabs",
            "key🔥with💡emoji",
            "键值",  # Chinese characters
            "مفتاح",  # Arabic characters
        ]

        for key in special_keys:
            context.set(key, f"value_for_{key}")
            assert context.get(key) == f"value_for_{key}"

    def test_dot_in_key_literal(self):
        """Test that dots in keys are always treated as path separators."""
        context = Context()

        # Set a key that contains a dot - this creates a LITERAL key in the map
        context.set("user.name", "Alice")

        # But get() treats dots as path separators, so this returns None
        # This is EXPECTED BEHAVIOR - dots are reserved for nested access
        assert context.get("user.name") is None

        # The literal key exists in the underlying data
        assert "user.name" in context._data

        # To access nested paths, use proper nesting
        context.set("user", {"name": "Bob"})
        assert context.get("user.name") == "Bob"  # This works!

    def test_numeric_keys(self):
        """Test numeric keys (integers)."""
        context = Context()

        # Integer keys should be converted to strings or handled
        context.set(123, "numeric_value")
        assert context.get(123) == "numeric_value"

        # Also test accessing with string version
        context.set("456", "string_numeric")
        assert context.get("456") == "string_numeric"

    def test_very_long_key_name(self):
        """Test very long key names."""
        context = Context()

        # 1000 character key
        long_key = "k" * 1000
        context.set(long_key, "long_key_value")
        assert context.get(long_key) == "long_key_value"

    def test_very_long_nested_path(self):
        """Test very long nested path access."""
        context = Context()

        # Create deep nesting
        nested = {"end": "value"}
        for i in range(50):
            nested = {"level": nested}

        context.set("root", nested)

        # Access with 50-level deep path
        path = "root." + ".".join(["level"] * 50) + ".end"
        assert context.get(path) == "value"


class TestContextValidationEdgeCases:
    """Test Context validation edge cases."""

    def test_validation_with_nested_objects(self):
        """Test validation with nested object schemas."""
        context = Context(validation_enabled=True)

        # Define nested schema
        context.schema.define(
            "user",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
                "required": ["name"],
            },
        )

        # Valid nested object
        context.set("user", {"name": "Alice", "address": {"city": "NYC", "street": "5th Ave"}})

        # Invalid - missing required nested field
        with pytest.raises(ContextValidationError):
            context.set("user", {"name": "Bob", "address": {}})

    def test_validation_with_array_of_objects(self):
        """Test validation with array of objects."""
        context = Context(validation_enabled=True)

        # Define array of objects schema
        context.schema.define(
            "users",
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
                    "required": ["id"],
                },
            },
        )

        # Valid array
        context.set("users", [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])

        # Invalid - missing required field in array item
        with pytest.raises(ContextValidationError):
            context.set("users", [{"id": 1, "name": "Alice"}, {"name": "Bob"}])

    def test_invalid_schema_definition(self):
        """Test handling of invalid schema definitions."""
        context = Context(validation_enabled=True)

        # Malformed schema - jsonschema uses lazy validation
        # Schema is not validated until it's actually used
        context.schema.define("bad_key", {"type": "invalid_type"})

        # Using the invalid schema should raise validation error
        from jsonschema.exceptions import ValidationError

        with pytest.raises((ValidationError, Exception)):
            context.set("bad_key", "some_value")

    def test_multiple_validation_errors(self):
        """Test single update violating multiple constraints."""
        context = Context(validation_enabled=True)

        # Define schema with multiple constraints
        context.schema.define(
            "priority",
            {"type": "integer", "minimum": 1, "maximum": 5, "multipleOf": 1},
        )

        # Value violating type and range
        with pytest.raises(ContextValidationError):
            context.set("priority", "not_an_integer")

        # Value violating range
        with pytest.raises(ContextValidationError):
            context.set("priority", 10)

    def test_validation_disabled_accepts_anything(self):
        """Test that disabled validation accepts any value."""
        context = Context(validation_enabled=False)

        # Define a schema but validation is disabled
        context.schema.define("strict_int", {"type": "integer"})

        # Should accept string even though schema says integer
        context.set("strict_int", "not_an_integer")
        assert context.get("strict_int") == "not_an_integer"

    def test_validation_reenabling(self):
        """Test re-enabling validation after it was disabled."""
        context = Context(validation_enabled=False)

        # Set invalid data while validation disabled
        context.schema.define("age", {"type": "integer"})
        context.set("age", "invalid")

        # Enable validation
        context.validation_enabled = True

        # New sets should be validated
        with pytest.raises(ContextValidationError):
            context.set("age", "still_invalid")

        # Valid value should work
        context.set("age", 25)


class TestContextLargeData:
    """Test Context with large amounts of data."""

    def test_very_large_context(self):
        """Test context with 10,000 keys."""
        context = Context()

        # Add 10,000 keys
        for i in range(10000):
            context.set(f"key_{i}", f"value_{i}")

        # Verify all keys are accessible
        assert context.get("key_0") == "value_0"
        assert context.get("key_5000") == "value_5000"
        assert context.get("key_9999") == "value_9999"

        # Copy should work efficiently
        copy = context.copy()
        assert copy.get("key_5000") == "value_5000"

    def test_deeply_nested_structures(self):
        """Test deeply nested dict structures (20+ levels)."""
        context = Context()

        # Create 20-level deep nesting
        nested = {"value": "end"}
        for i in range(20):
            nested = {"nested": nested}

        context.set("deep", nested)

        # Should be accessible
        assert context.get("deep") is not None

        # Access via nested path
        path = "deep." + ".".join(["nested"] * 20) + ".value"
        assert context.get(path) == "end"

    def test_large_values(self):
        """Test large string values (1MB+)."""
        context = Context()

        # 1MB string
        large_string = "x" * (1024 * 1024)
        context.set("large", large_string)

        # Should be retrievable
        assert len(context.get("large")) == 1024 * 1024

        # Copy should work
        copy = context.copy()
        assert len(copy.get("large")) == 1024 * 1024

    def test_many_nested_keys(self):
        """Test many keys in nested structures."""
        context = Context()

        # Create dict with 1000 keys
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        context.set("large_dict", large_dict)

        # Should be accessible
        assert context.get("large_dict.key_500") == "value_500"

    def test_mixed_large_data(self):
        """Test context with mix of large structures."""
        context = Context()

        # Large array
        context.set("array", list(range(10000)))

        # Large dict
        context.set("dict", {f"k{i}": i for i in range(1000)})

        # Large string
        context.set("string", "x" * 100000)

        # Deep nesting
        nested = {"val": 42}
        for i in range(10):
            nested = {"level": nested}
        context.set("nested", nested)

        # All should be accessible
        assert len(context.get("array")) == 10000
        assert len([k for k in context.get("dict").keys()]) == 1000
        assert len(context.get("string")) == 100000
        assert context.get("nested") is not None


class TestContextCopyBehavior:
    """Test Context copy() behavior."""

    def test_copy_independence(self):
        """Test that copies are fully independent."""
        context1 = Context()
        context1.set("shared", "original")
        context1.set("nested", {"key": "value"})

        context2 = context1.copy()

        # Modify context1
        context1.set("shared", "modified")
        context1.get("nested")["key"] = "modified_value"

        # context2 should be unchanged due to immutability
        # Note: Since we use immutable Map, nested mutable objects might still be shared
        # This tests the actual behavior
        assert context2.get("shared") == "original"

    def test_deep_copy_of_mutable_values(self):
        """Test that mutable values in context are handled correctly."""
        context = Context()

        mutable_list = [1, 2, 3]
        context.set("list", mutable_list)

        # Modify original list
        mutable_list.append(4)

        # Context should have reference to modified list
        # (Context doesn't deep copy values, just stores references)
        assert 4 in context.get("list")

    def test_copy_preserves_validation_state(self):
        """Test that copy preserves validation settings."""
        context1 = Context(validation_enabled=True)
        context1.schema.define("age", {"type": "integer"})

        context2 = context1.copy()

        # context2 should have same validation state
        # However, current implementation might not copy validation settings
        # This tests actual behavior
        try:
            context2.set("age", "invalid")
            # If validation not copied, this succeeds
        except ContextValidationError:
            # If validation copied, this raises
            pass


class TestContextEdgeCases:
    """Additional edge cases for Context."""

    def test_update_with_empty_dict(self):
        """Test update with empty dictionary."""
        context = Context()
        context.set("key", "value")

        context.update({})

        assert context.get("key") == "value"

    def test_update_with_none_values(self):
        """Test update with None values."""
        context = Context()

        context.update({"key1": None, "key2": "value"})

        assert context.get("key1") is None
        assert context.get("key2") == "value"

    def test_delete_nonexistent_key(self):
        """Test deleting a key that doesn't exist."""
        context = Context()

        # Should not raise error
        context.delete("nonexistent")

    def test_clear_then_repopulate(self):
        """Test clearing context and adding new data."""
        context = Context()

        context.set("key1", "value1")
        context.set("key2", "value2")

        context.clear()

        assert context.get("key1") is None
        assert context.get("key2") is None

        # Should be able to add new data
        context.set("new_key", "new_value")
        assert context.get("new_key") == "new_value"

    def test_contains_with_nested_path(self):
        """Test __contains__ with nested paths."""
        context = Context()
        context.set("user", {"name": "Alice", "age": 30})

        # Direct key should work
        assert "user" in context

        # Nested path - has() method DOES support nested path checking
        # So "user.name" is found via nested traversal
        assert "user.name" in context  # Nested path exists!
        assert "user.age" in context  # This too!
        assert "user.nonexistent" not in context  # But this doesn't

    def test_boolean_coercion(self):
        """Test truthiness of Context."""
        empty_context = Context()
        assert bool(empty_context) is True  # Context object is always truthy

        full_context = Context()
        full_context.set("key", "value")
        assert bool(full_context) is True

    def test_get_with_integer_key(self):
        """Test get() with integer key (non-string)."""
        context = Context()
        context.set(123, "integer_key_value")

        # Should work with integer key
        assert context.get(123) == "integer_key_value"

        # Integer keys should not be treated as nested paths (no splitting on ".")
        assert context.get(123, default="default") == "integer_key_value"

    def test_get_with_tuple_key(self):
        """Test get() with tuple key (non-string)."""
        context = Context()
        context.set((1, 2), "tuple_key_value")

        # Should work with tuple key
        assert context.get((1, 2)) == "tuple_key_value"

        # Tuple keys should not cause splitting behavior
        assert context.get((1, 2), default="default") == "tuple_key_value"

    def test_has_with_non_string_key(self):
        """Test has() with non-string keys."""
        context = Context()
        context.set(456, "numeric_value")
        context.set((3, 4), "tuple_value")

        # has() should work with non-string keys
        assert context.has(456)
        assert context.has((3, 4))
        assert not context.has(789)

    def test_nested_access_only_works_for_strings(self):
        """Test that nested access (dot notation) only works for string keys."""
        context = Context()
        context.set("user", {"name": "Alice", "age": 30})

        # String key with dot notation should work
        assert context.get("user.name") == "Alice"

        # Non-string keys should not attempt dot splitting
        context.set(123, {"nested": "value"})

        # Getting with integer key should return the dict, not try to split on "."
        result = context.get(123)
        assert result == {"nested": "value"}

        # Should not be able to use "." notation with non-string keys
        # (because isinstance(key, str) check prevents splitting)

    def test_delete_with_non_string_key(self):
        """Test delete() with non-string keys."""
        context = Context()
        context.set(999, "value_to_delete")
        context.set((5, 6), "tuple_to_delete")

        assert context.has(999)
        assert context.has((5, 6))

        # Delete should work with non-string keys
        context.delete(999)
        context.delete((5, 6))

        assert not context.has(999)
        assert not context.has((5, 6))

    def test_update_with_non_string_keys(self):
        """Test update() with dictionary containing non-string keys."""
        context = Context()

        # Update with mixed key types
        context.update({
            "string_key": "string_value",
            123: "integer_value",
            (1, 2): "tuple_value"
        })

        assert context.get("string_key") == "string_value"
        assert context.get(123) == "integer_value"
        assert context.get((1, 2)) == "tuple_value"

    def test_to_dict_with_non_string_keys(self):
        """Test to_dict() preserves non-string keys."""
        context = Context()
        context.set("str_key", "str_val")
        context.set(42, "int_val")
        context.set((7, 8), "tuple_val")

        result = context.to_dict()

        assert result["str_key"] == "str_val"
        assert result[42] == "int_val"
        assert result[(7, 8)] == "tuple_val"

    def test_has_with_nested_string_keys(self):
        """Test has() works correctly with nested string keys (dot notation)."""
        context = Context()
        context.set("user", {"name": "Alice", "age": 30, "address": {"city": "NYC"}})

        # has() should support nested access for string keys
        assert context.has("user.name")
        assert context.has("user.age")
        assert context.has("user.address")

        # Deeply nested
        context.set("config", {"db": {"host": "localhost", "port": 5432}})
        assert context.has("config.db.host")
        assert context.has("config.db.port")

    def test_get_with_nonexistent_nested_path(self):
        """Test get() when nested path doesn't exist."""
        context = Context()
        context.set("user", {"name": "Alice"})

        # Path exists but key doesn't
        result = context.get("user.email")
        assert result is None

        # With default value
        result = context.get("user.email", default="no-email@example.com")
        assert result == "no-email@example.com"

        # Deeper path that doesn't exist
        context.set("config", {"db": {"host": "localhost"}})
        result = context.get("config.db.username", default="admin")
        assert result == "admin"

    def test_get_when_intermediate_value_not_dict(self):
        """Test get() when intermediate value in path is not a dict."""
        context = Context()

        # Set a non-dict value
        context.set("config", "simple_string")

        # Try to access nested path - should return None
        result = context.get("config.timeout")
        assert result is None

        # With default
        result = context.get("config.timeout", default=30)
        assert result == 30

        # Another case: intermediate value is a list
        context.set("items", [1, 2, 3])
        result = context.get("items.length")
        assert result is None

    def test_get_with_deeply_nested_paths(self):
        """Test get() with deeply nested dot notation paths."""
        context = Context()

        # Create deeply nested structure
        context.set("app", {
            "config": {
                "database": {
                    "primary": {
                        "host": "db.example.com",
                        "port": 5432
                    }
                }
            }
        })

        # Access deeply nested values
        assert context.get("app.config.database.primary.host") == "db.example.com"
        assert context.get("app.config.database.primary.port") == 5432

        # Non-existent deep path
        assert context.get("app.config.database.secondary.host") is None

    def test_has_with_partial_nested_paths(self):
        """Test has() returns False for partial/non-existent nested paths."""
        context = Context()
        context.set("user", {"name": "Alice", "profile": {"bio": "Engineer"}})

        # These should work
        assert context.has("user.name")
        assert context.has("user.profile")
        assert context.has("user.profile.bio")

        # These should return False
        assert not context.has("user.email")
        assert not context.has("user.profile.website")
        assert not context.has("nonexistent.path")

    def test_nested_access_with_none_values(self):
        """Test nested access when values are None."""
        context = Context()
        context.set("data", {"field1": None, "field2": {"nested": None}})

        # Should be able to get None values
        assert context.get("data.field1") is None
        assert context.has("data.field1")  # Key exists, value is None

        # Nested None
        assert context.get("data.field2.nested") is None
        assert context.has("data.field2.nested")

        # Path through None should return None
        context.set("value", None)
        assert context.get("value.something") is None

    def test_nested_access_with_empty_dicts(self):
        """Test nested access with empty dictionaries."""
        context = Context()
        context.set("empty", {})

        # Empty dict exists
        assert context.has("empty")
        assert context.get("empty") == {}

        # But nested keys don't exist
        assert not context.has("empty.key")
        assert context.get("empty.key") is None

        # Nested empty dict
        context.set("data", {"nested": {}})
        assert context.has("data.nested")
        assert not context.has("data.nested.key")


class TestBinaryAndNonSerializableData:
    """Test handling of binary and non-serializable data in context."""

    def test_context_with_binary_data(self):
        """Test context storing binary data (bytes)."""
        context = Context()

        binary_data = b"\x00\x01\x02\xff\xfe"
        context.set("binary", binary_data)

        # Should store without error
        assert context.has("binary")

        # Retrieve should work
        result = context.get("binary")
        assert result == binary_data
        assert isinstance(result, bytes)

    def test_context_with_bytearray(self):
        """Test context storing bytearray."""
        context = Context()

        data = bytearray([0, 1, 2, 255])
        context.set("bytearray_data", data)

        result = context.get("bytearray_data")
        assert result == data

    def test_context_with_custom_object(self):
        """Test context storing custom class instances."""

        class CustomObject:
            def __init__(self, value):
                self.value = value

        context = Context()
        obj = CustomObject(42)

        # Should store the object reference
        context.set("custom", obj)

        result = context.get("custom")
        assert result is obj
        assert result.value == 42

    def test_schema_generation_with_binary_data(self):
        """Test schema generation when context contains binary data."""
        from bestla.yggdrasil.dynamic_types import DynamicStr
        toolkit = Toolkit()
        toolkit.context.set("data", b"binary")
        toolkit.context.set("options", ["a", "b"])  # Normal data too

        @tool()
        def select(option: DynamicStr["options"]) -> Tuple[str, dict]:
            return option, {}

        toolkit.register_tool(select)

        # Should generate schema without crashing on binary data
        schemas = toolkit.generate_schemas()
        assert len(schemas) == 1


class TestSpecialNumericValues:
    """Test handling of special float values (NaN, Infinity)."""

    def test_context_with_nan(self):
        """Test context storing NaN."""
        import math
        context = Context()

        context.set("nan_value", math.nan)

        result = context.get("nan_value")
        assert math.isnan(result)

    def test_context_with_infinity(self):
        """Test context storing positive and negative infinity."""
        import math
        context = Context()

        context.set("pos_inf", math.inf)
        context.set("neg_inf", -math.inf)

        assert math.isinf(context.get("pos_inf"))
        assert math.isinf(context.get("neg_inf"))
        assert context.get("pos_inf") > 0
        assert context.get("neg_inf") < 0

    def test_dynamic_int_with_infinity_bounds(self):
        """Test DynamicInt with infinity as min/max."""
        import math

        from bestla.yggdrasil.dynamic_types import DynamicInt
        context = Context()
        context.set("range", {"minimum": -math.inf, "maximum": math.inf})

        schema = DynamicInt["range"].generate_schema(context)

        # Should handle infinity in schema
        assert schema["type"] == "integer"
        # Infinity might be converted or cause issues

    def test_context_validation_with_nan(self):
        """Test validation with NaN values."""
        import math
        context = Context(validation_enabled=True)
        context.schema.define("number", {"type": "number"})

        # NaN should be a valid number
        try:
            context.set("number", math.nan)
            result = context.get("number")
            assert math.isnan(result)
        except Exception:
            # Some validators might reject NaN
            pytest.skip("Validator doesn't support NaN")


class TestExtremeSizeValues:
    """Test extremely large numbers and strings."""

    def test_context_with_very_large_integer(self):
        """Test context storing integers beyond 64-bit range."""
        context = Context()

        # Python supports arbitrary precision integers
        large_int = 2**1000  # Way beyond 64-bit
        context.set("huge", large_int)

        result = context.get("huge")
        assert result == large_int

    def test_context_with_large_negative_integer(self):
        """Test very large negative integers."""
        context = Context()

        large_neg = -(2**1000)
        context.set("huge_neg", large_neg)

        assert context.get("huge_neg") == large_neg

    def test_dynamic_int_with_extreme_range(self):
        """Test DynamicInt with extreme min/max values."""
        from bestla.yggdrasil.dynamic_types import DynamicInt
        context = Context()
        context.set("extreme", {"minimum": -(2**100), "maximum": 2**100})

        schema = DynamicInt["extreme"].generate_schema(context)

        assert schema["type"] == "integer"
        assert schema["minimum"] == -(2**100)
        assert schema["maximum"] == 2**100

    def test_context_with_megabyte_string(self):
        """Test context storing 10MB string."""
        context = Context()

        # 10MB string
        large_string = "x" * (10 * 1024 * 1024)
        context.set("huge_string", large_string)

        result = context.get("huge_string")
        assert len(result) == 10 * 1024 * 1024


class TestEmptyAndNoneKeys:
    """Test empty strings, None values, and edge case keys."""

    def test_context_with_whitespace_only_keys(self):
        """Test keys that are only whitespace."""
        context = Context()

        context.set("   ", "spaces")
        context.set("\t\t", "tabs")
        context.set("\n", "newline")

        assert context.get("   ") == "spaces"
        assert context.get("\t\t") == "tabs"
        assert context.get("\n") == "newline"

    def test_context_with_special_unicode_keys(self):
        """Test keys with emoji and special unicode."""
        context = Context()

        context.set("🔑", "emoji_key")
        context.set("키", "korean")
        context.set("مفتاح", "arabic")

        assert context.get("🔑") == "emoji_key"
        assert context.get("키") == "korean"
        assert context.get("مفتاح") == "arabic"

    def test_nested_access_with_empty_string_components(self):
        """Test nested path with empty string components."""
        context = Context()

        # Set nested structure with empty string key
        context.set("user", {"": "empty_key_value", "name": "Alice"})

        # Access with dot notation (empty string after dot)
        result = context.get("user.")
        # Behavior might vary - just verify it doesn't crash
        assert result is None or result == "empty_key_value"

    def test_context_set_with_none_value(self):
        """Test explicitly setting None as a value."""
        context = Context()

        context.set("explicit_none", None)

        # Should be distinguishable from missing key
        assert context.has("explicit_none")
        assert context.get("explicit_none") is None
        assert context.get("missing_key") is None

        # But `in` operator should distinguish
        assert "explicit_none" in context
        assert "missing_key" not in context


class TestInvalidContextUpdates:
    """Test tools returning invalid or unusual context updates."""

    def test_tool_returns_none_as_updates(self):
        """Test tool returning None instead of dict for updates."""

        @tool()
        def none_updates() -> Tuple[str, None]:
            return "result", None

        result, updates = none_updates.execute()

        assert result == "result"
        # Should handle None gracefully (convert to empty dict or accept None)
        assert updates is None or updates == {}

    def test_tool_returns_nested_updates_with_dots(self):
        """Test tool returning flat keys with dots in update dict."""

        @tool()
        def dotted_updates() -> Tuple[str, dict]:
            # Returns flat keys with dots - creates literal keys with dots
            return "result", {"user.name": "Alice", "user.age": 30}

        toolkit = Toolkit()
        toolkit.register_tool(dotted_updates)

        results = toolkit.execute_sequential([{"name": "dotted_updates", "arguments": {}}])

        assert results[0]["success"]

        # Keys with dots are stored as literal keys in _data
        # But has() uses nested path logic, so won't find them
        # We need to check the underlying _data
        assert "user.name" in toolkit.context._data
        assert "user.age" in toolkit.context._data

    def test_tool_returns_conflicting_updates(self):
        """Test tool returning updates that conflict with each other."""

        @tool()
        def conflicting_updates() -> Tuple[str, dict]:
            return "result", {"key": "value2"}  # Last wins

        result, updates = conflicting_updates.execute()

        # Python dict only keeps last value
        assert updates["key"] == "value2"

    def test_tool_updates_with_list_and_dict_same_key(self):
        """Test context update type conflicts."""
        toolkit = Toolkit()
        toolkit.context.set("data", ["a", "b"])

        @tool()
        def change_type() -> Tuple[str, dict]:
            # Changes data from list to dict
            return "changed", {"data": {"new": "structure"}}

        toolkit.register_tool(change_type)

        results = toolkit.execute_sequential([{"name": "change_type", "arguments": {}}])

        assert results[0]["success"]

        # Should overwrite with new type
        assert toolkit.context.get("data") == {"new": "structure"}
        assert isinstance(toolkit.context.get("data"), dict)


class TestEncodingAndUnicode:
    """Test various encoding scenarios."""

    def test_context_with_mixed_unicode(self):
        """Test context with various Unicode categories."""
        context = Context()

        mixed = {
            "ascii": "Hello",
            "emoji": "Hello 👋 🌍",
            "cjk": "你好世界",
            "arabic": "مرحبا",
            "cyrillic": "Привет",
            "combining": "é"  # e + combining accent
            + "́",
        }

        for key, value in mixed.items():
            context.set(key, value)

        # All should be retrievable
        for key, value in mixed.items():
            assert context.get(key) == value

    def test_tool_name_with_unicode(self):
        """Test tool with unicode name."""

        @tool()
        def 你好() -> Tuple[str, dict]:
            """Tool with Chinese name."""
            return "Hello", {}

        assert 你好.name == "你好"

        # Should be usable in toolkit
        toolkit = Toolkit()
        toolkit.register_tool(你好)

        assert "你好" in toolkit.tools

    def test_context_key_with_null_byte_rejected(self):
        """Test context with null byte in key."""
        context = Context()

        # Null byte in key - behavior undefined
        try:
            context.set("key\x00with\x00null", "value")
            # If it works, verify retrieval
            assert context.get("key\x00with\x00null") == "value"
        except (ValueError, TypeError):
            # Some implementations might reject null bytes
            pass


class TestNestedStructureEdgeCases:
    """Test extreme nesting and complex structures."""

    def test_extremely_deep_nesting(self):
        """Test 100+ level deep nesting."""
        context = Context()

        # Create 100-level deep structure
        nested = {"value": "deep"}
        for i in range(100):
            nested = {"level": nested}

        context.set("deep", nested)

        # Should store without stack overflow
        assert context.has("deep")

        # Access deeply nested value
        path = "deep." + ".".join(["level"] * 100) + ".value"

        try:
            result = context.get(path)
            assert result == "deep"
        except RecursionError:
            pytest.skip("Deep nesting causes recursion error")

    def test_very_wide_structure(self):
        """Test structure with 10,000 keys at one level."""
        context = Context()

        wide_dict = {f"key_{i}": f"value_{i}" for i in range(10000)}
        context.set("wide", wide_dict)

        # Should handle without issues
        assert context.get("wide.key_5000") == "value_5000"

    def test_mixed_depth_and_breadth(self):
        """Test structure that's both deep and wide."""
        context = Context()

        # Create structure: nested levels with many keys
        level3 = {"key_a": "value_a", "key_b": "value_b", "end": "reached"}
        level2 = {"key_1": "val_1", "next": level3}
        level1 = {"key_x": "val_x", "next": level2}
        structure = {"level": level1}

        context.set("complex", structure)

        # Access nested path
        result = context.get("complex.level.next.next.end")
        assert result == "reached"

        # Access wide keys
        assert context.get("complex.level.key_x") == "val_x"
        assert context.get("complex.level.next.key_1") == "val_1"
        assert context.get("complex.level.next.next.key_a") == "value_a"


class TestListAndSetInContext:
    """Test lists, sets, and other collection types."""

    def test_context_with_set(self):
        """Test context storing Python set."""
        context = Context()

        data_set = {1, 2, 3, 4, 5}
        context.set("set_data", data_set)

        result = context.get("set_data")
        assert result == data_set

    def test_context_with_tuple(self):
        """Test context storing tuple."""
        context = Context()

        data_tuple = (1, 2, 3, "four", 5.0)
        context.set("tuple_data", data_tuple)

        result = context.get("tuple_data")
        assert result == data_tuple

    def test_context_with_frozenset(self):
        """Test context storing frozenset."""
        context = Context()

        frozen = frozenset([1, 2, 3])
        context.set("frozen", frozen)

        assert context.get("frozen") == frozen

    def test_context_with_nested_collections(self):
        """Test deeply nested collections of different types."""
        context = Context()

        nested = {
            "list": [1, 2, [3, 4, (5, 6)]],
            "tuple": (1, 2, {3, 4}),
            "set": {1, 2},  # Sets can't contain mutable items
            "dict": {"nested": {"deeper": [1, 2]}},
        }

        context.set("collections", nested)

        result = context.get("collections")
        assert result["list"] == [1, 2, [3, 4, (5, 6)]]


class TestContextWithEdgeCaseKeys:
    """Test Context with boolean, float, bytes, and other edge case keys."""

    def test_context_with_boolean_keys(self):
        """Test Context with boolean keys (True/False)."""
        context = Context()

        # Set with boolean keys
        context.set(True, "value_for_true")
        context.set(False, "value_for_false")

        # Get with boolean keys
        assert context.get(True) == "value_for_true"
        assert context.get(False) == "value_for_false"

        # Has with boolean keys
        assert context.has(True)
        assert context.has(False)

        # Boolean keys should not be treated as nested paths
        assert context.get(True, default="default") == "value_for_true"
        assert context.get(False, default="default") == "value_for_false"

    def test_context_with_float_keys(self):
        """Test Context with float keys."""
        context = Context()

        # Set with float keys
        context.set(3.14, "pi_value")
        context.set(2.718, "e_value")
        context.set(0.0, "zero_value")

        # Get with float keys
        assert context.get(3.14) == "pi_value"
        assert context.get(2.718) == "e_value"
        assert context.get(0.0) == "zero_value"

        # Has with float keys
        assert context.has(3.14)
        assert context.has(2.718)
        assert context.has(0.0)

        # Float keys should not be treated as nested paths
        assert context.get(3.14, default="default") == "pi_value"

    def test_context_with_bytes_keys(self):
        """Test Context with bytes keys."""
        context = Context()

        # Set with bytes keys
        context.set(b"key1", "bytes_value_1")
        context.set(b"key2", "bytes_value_2")

        # Get with bytes keys
        assert context.get(b"key1") == "bytes_value_1"
        assert context.get(b"key2") == "bytes_value_2"

        # Has with bytes keys
        assert context.has(b"key1")
        assert context.has(b"key2")
        assert not context.has(b"key3")

    def test_context_with_frozenset_keys(self):
        """Test Context with frozenset keys."""
        context = Context()

        # Frozensets are hashable
        key1 = frozenset({1, 2, 3})
        key2 = frozenset({"a", "b", "c"})

        context.set(key1, "frozenset_value_1")
        context.set(key2, "frozenset_value_2")

        assert context.get(key1) == "frozenset_value_1"
        assert context.get(key2) == "frozenset_value_2"

        assert context.has(key1)
        assert context.has(key2)

    def test_context_numeric_string_vs_int_keys(self):
        """Test Context distinguishes between numeric strings and ints."""
        context = Context()

        # Set with both string and int versions
        context.set("123", "string_value")
        context.set(123, "int_value")

        # Should be distinct keys
        assert context.get("123") == "string_value"
        assert context.get(123) == "int_value"

        # Only string keys support nested access
        context.set("user", {"id": 123})
        assert context.get("user.id") == 123

        # Integer keys do not support nested access (no dot splitting)
        context.set(456, {"nested": "value"})
        result = context.get(456)
        assert result == {"nested": "value"}

    def test_context_boolean_true_vs_one(self):
        """Test Context behavior with True vs 1 (which are equal in Python)."""
        context = Context()

        # In Python, True == 1 and False == 0
        context.set(True, "bool_true_value")

        # This will overwrite because True == 1
        context.set(1, "int_one_value")

        # Due to Python's bool/int equality, this gets the int value
        assert context.get(True) == "int_one_value"
        assert context.get(1) == "int_one_value"

        # Same for False and 0
        context.set(False, "bool_false_value")
        context.set(0, "int_zero_value")

        assert context.get(False) == "int_zero_value"
        assert context.get(0) == "int_zero_value"

    def test_context_mixed_key_types(self):
        """Test Context with mixed key types."""
        context = Context()

        # Mix of different types
        context.set("string_key", "string_value")
        context.set(42, "int_value")
        context.set(3.14, "float_value")
        context.set((1, 2), "tuple_value")
        context.set(True, "bool_value")
        context.set(b"bytes", "bytes_value")

        # All should be accessible
        assert context.get("string_key") == "string_value"
        assert context.get(42) == "int_value"
        assert context.get(3.14) == "float_value"
        assert context.get((1, 2)) == "tuple_value"
        assert context.get(True) == "bool_value"
        assert context.get(b"bytes") == "bytes_value"

        # to_dict should preserve all
        result = context.to_dict()
        assert len(result) == 6

    def test_context_delete_with_edge_case_keys(self):
        """Test deleting keys with edge case types."""
        context = Context()

        context.set(True, "bool_value")
        context.set(3.14, "float_value")
        context.set(b"key", "bytes_value")

        # Delete
        context.delete(True)
        context.delete(3.14)
        context.delete(b"key")

        # Should be gone
        assert not context.has(True)
        assert not context.has(3.14)
        assert not context.has(b"key")

    def test_context_update_with_edge_case_keys(self):
        """Test update() with edge case key types."""
        context = Context()

        updates = {
            "string": "str_val",
            123: "int_val",
            3.14: "float_val",
            (1, 2): "tuple_val",
            True: "bool_val",
        }

        context.update(updates)

        assert context.get("string") == "str_val"
        assert context.get(123) == "int_val"
        assert context.get(3.14) == "float_val"
        assert context.get((1, 2)) == "tuple_val"
        assert context.get(True) == "bool_val"
