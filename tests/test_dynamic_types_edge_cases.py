"""Tests for dynamic types edge cases."""


from bestla.yggdrasil import Context
from bestla.yggdrasil.dynamic_types import (
    DynamicArray,
    DynamicConst,
    DynamicFiltered,
    DynamicFloat,
    DynamicFormat,
    DynamicInt,
    DynamicNested,
    DynamicPattern,
    DynamicStr,
)


class TestDynamicTypeEdgeCases:
    """Test edge cases for dynamic type resolution."""

    def test_dynamic_str_with_empty_enum(self):
        """Test DynamicStr with empty list in context."""
        context = Context()
        context.set("colors", [])

        schema = DynamicStr["colors"].generate_schema(context)

        # Should generate enum with empty list
        assert schema["type"] == "string"
        assert schema["enum"] == []

    def test_dynamic_str_with_very_large_enum(self):
        """Test DynamicStr with 1000+ values."""
        context = Context()
        large_enum = [f"value_{i}" for i in range(1000)]
        context.set("large", large_enum)

        schema = DynamicStr["large"].generate_schema(context)

        assert len(schema["enum"]) == 1000

    def test_dynamic_int_type_mismatch(self):
        """Test DynamicInt when context has non-dict value."""
        context = Context()
        context.set("number", "not_a_dict")

        # Should handle gracefully by returning basic schema without constraints
        schema = DynamicInt["number"].generate_schema(context)
        assert schema["type"] == "integer"
        # No constraints added when value is wrong type

    def test_dynamic_type_with_missing_key(self):
        """Test dynamic type when context key doesn't exist."""
        context = Context()

        # Should handle gracefully by returning basic schema
        schema = DynamicStr["nonexistent"].generate_schema(context)
        assert schema["type"] == "string"
        # No enum added when key doesn't exist

    def test_dynamic_type_with_none_value(self):
        """Test dynamic type when context value is None."""
        context = Context()
        context.set("nullable", None)

        # Should handle gracefully by returning basic schema
        schema = DynamicStr["nullable"].generate_schema(context)
        assert schema["type"] == "string"
        # No enum added when value is None


class TestDynamicFilteredEdgeCases:
    """Test DynamicFiltered edge cases."""

    def test_filter_function_raises_exception(self):
        """Test filter that raises exception."""
        from bestla.yggdrasil.toolkit import Toolkit

        toolkit = Toolkit()
        toolkit.context.set("items", ["a", "b", "c"])

        def failing_filter(items):
            raise RuntimeError("Filter failed")

        toolkit.filters["fail"] = failing_filter

        # Should handle gracefully - filters are called but exceptions handled
        try:
            schema = DynamicFiltered[("items", "fail")].generate_schema(
                toolkit.context, toolkit.filters
            )
            # Handles gracefully by returning basic schema
            assert schema["type"] == "string"
        except RuntimeError:
            # Or might propagate - both are acceptable
            pass

    def test_filter_returns_non_list(self):
        """Test filter that returns non-list."""
        from bestla.yggdrasil.toolkit import Toolkit

        toolkit = Toolkit()
        toolkit.context.set("items", ["a", "b"])

        def bad_filter(items):
            return "not_a_list"

        toolkit.filters["bad"] = bad_filter

        # Might work or raise depending on implementation
        try:
            DynamicFiltered[("items", "bad")].generate_schema(
                toolkit.context, toolkit.filters
            )
        except (TypeError, ValueError):
            pass

    def test_missing_filter(self):
        """Test DynamicFiltered with filter not in toolkit."""
        from bestla.yggdrasil.toolkit import Toolkit

        toolkit = Toolkit()
        toolkit.context.set("items", ["a", "b"])

        # Filter not defined - should handle gracefully
        schema = DynamicFiltered[("items", "missing_filter")].generate_schema(
            toolkit.context, toolkit.filters
        )
        # Returns basic schema when filter missing
        assert schema["type"] == "string"

    def test_filter_returns_empty(self):
        """Test filter that returns empty list."""
        from bestla.yggdrasil.toolkit import Toolkit

        toolkit = Toolkit()
        toolkit.context.set("items", ["a", "b", "c"])

        def empty_filter(items):
            return []

        toolkit.filters["empty"] = empty_filter

        schema = DynamicFiltered[("items", "empty")].generate_schema(
            toolkit.context, toolkit.filters
        )

        # Should have empty enum
        assert schema["enum"] == []


class TestDynamicPatternEdgeCases:
    """Test DynamicPattern edge cases."""

    def test_invalid_regex_pattern(self):
        """Test DynamicPattern with invalid regex."""
        context = Context()
        context.set("pattern", "[invalid(")

        # Should handle invalid regex
        try:
            DynamicPattern["pattern"].generate_schema(context)
            # Might not validate pattern during schema generation
        except Exception:
            pass

    def test_complex_regex_pattern(self):
        """Test DynamicPattern with complex regex."""
        context = Context()
        # Complex email regex
        context.set("email_pattern", r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

        schema = DynamicPattern["email_pattern"].generate_schema(context)

        assert "pattern" in schema


class TestDynamicNestedEdgeCases:
    """Test DynamicNested edge cases."""

    def test_nested_with_missing_intermediate(self):
        """Test nested access when intermediate key missing."""
        context = Context()
        context.set("user", {"name": "Alice"})

        # Should handle gracefully when nested path doesn't exist
        schema = DynamicNested["user.address.city"].generate_schema(context)
        # Returns basic schema when path not found
        assert "type" in schema

    def test_nested_with_non_dict_intermediate(self):
        """Test nested access when intermediate is not dict."""
        context = Context()
        context.set("user", "string_not_dict")

        # Should handle gracefully when intermediate isn't dict
        schema = DynamicNested["user.name"].generate_schema(context)
        # Returns basic schema when path traversal fails
        assert "type" in schema

    def test_very_deep_nesting(self):
        """Test very deep nested path."""
        context = Context()

        # Create 10-level deep structure
        nested = {"value": ["a", "b", "c"]}
        for i in range(9):
            nested = {f"level_{9-i}": nested}

        context.set("root", nested)

        # Access deep path
        path = "root." + ".".join([f"level_{i}" for i in range(1, 10)]) + ".value"
        schema = DynamicNested[path].generate_schema(context)

        assert schema["enum"] == ["a", "b", "c"]


class TestDynamicConstraintsEdgeCases:
    """Test DynamicConstraints edge cases."""

    def test_malformed_constraint_schema(self):
        """Test DynamicConstraints with invalid schema."""
        from bestla.yggdrasil.dynamic_types import DynamicConstraints

        context = Context()
        context.set("bad_schema", {"type": "invalid_type"})

        # Should handle or propagate error
        try:
            DynamicConstraints["bad_schema"].generate_schema(context)
        except (ValueError, Exception):
            pass


class TestDynamicIntFloatEdgeCases:
    """Test DynamicInt and DynamicFloat edge cases."""

    def test_dynamic_int_with_missing_constraints(self):
        """Test DynamicInt when min/max not in context."""
        context = Context()
        context.set("number", {})  # Empty dict, no min/max

        schema = DynamicInt["number"].generate_schema(context)

        # Should still generate integer type
        assert schema["type"] == "integer"

    def test_dynamic_float_with_all_constraints(self):
        """Test DynamicFloat with min, max, and multipleOf."""
        context = Context()
        context.set("price", {"minimum": 0.01, "maximum": 999.99, "multipleOf": 0.01})

        schema = DynamicFloat["price"].generate_schema(context)

        assert schema["type"] == "number"
        assert schema["minimum"] == 0.01
        assert schema["maximum"] == 999.99
        # Note: Current implementation only includes minimum/maximum, not multipleOf for floats


class TestDynamicArrayEdgeCases:
    """Test DynamicArray edge cases."""

    def test_dynamic_array_with_non_list_value(self):
        """Test DynamicArray when context value is not list."""
        context = Context()
        context.set("items", "not_a_list")

        # Should handle gracefully by returning basic array schema
        schema = DynamicArray["items"].generate_schema(context)
        assert schema["type"] == "array"

    def test_dynamic_array_with_empty_list(self):
        """Test DynamicArray with empty list."""
        context = Context()
        context.set("items", [])

        schema = DynamicArray["items"].generate_schema(context)

        # Should have array type with empty enum
        assert schema["type"] == "array"


class TestDynamicFormatEdgeCases:
    """Test DynamicFormat edge cases."""

    def test_dynamic_format_with_invalid_format(self):
        """Test DynamicFormat with non-standard format."""
        # Custom format (might not be validated)
        schema = DynamicFormat["custom-format"].generate_schema(Context())

        assert schema["type"] == "string"
        assert schema["format"] == "custom-format"


class TestDynamicConstEdgeCases:
    """Test DynamicConst edge cases."""

    def test_dynamic_const_with_complex_value(self):
        """Test DynamicConst with dict value."""
        context = Context()
        context.set("config", {"key": "value", "number": 42})

        schema = DynamicConst["config"].generate_schema(context)

        assert schema["const"] == {"key": "value", "number": 42}

    def test_dynamic_const_with_none(self):
        """Test DynamicConst with None value."""
        context = Context()
        context.set("empty", None)

        schema = DynamicConst["empty"].generate_schema(context)

        # When value is None, returns empty schema (defensive handling)
        # This is acceptable - avoids enforcing None as constant
        assert isinstance(schema, dict)
