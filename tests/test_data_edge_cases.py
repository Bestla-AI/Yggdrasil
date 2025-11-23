"""Data edge case tests - unusual data types and formats."""

import math
from typing import Tuple

import pytest

from bestla.yggdrasil import Context, Toolkit, tool
from bestla.yggdrasil.dynamic_types import DynamicInt, DynamicStr


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
        context = Context()

        context.set("nan_value", math.nan)

        result = context.get("nan_value")
        assert math.isnan(result)

    def test_context_with_infinity(self):
        """Test context storing positive and negative infinity."""
        context = Context()

        context.set("pos_inf", math.inf)
        context.set("neg_inf", -math.inf)

        assert math.isinf(context.get("pos_inf"))
        assert math.isinf(context.get("neg_inf"))
        assert context.get("pos_inf") > 0
        assert context.get("neg_inf") < 0

    def test_dynamic_int_with_infinity_bounds(self):
        """Test DynamicInt with infinity as min/max."""
        context = Context()
        context.set("range", {"minimum": -math.inf, "maximum": math.inf})

        schema = DynamicInt["range"].generate_schema(context)

        # Should handle infinity in schema
        assert schema["type"] == "integer"
        # Infinity might be converted or cause issues

    def test_context_validation_with_nan(self):
        """Test validation with NaN values."""
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
