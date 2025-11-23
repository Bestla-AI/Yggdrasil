"""Tests for advanced dynamic types: DynamicUnion and DynamicConditional."""

import pytest

from bestla.yggdrasil import Context
from bestla.yggdrasil.dynamic_types import DynamicConditional, DynamicUnion


class TestDynamicUnion:
    """Test DynamicUnion type."""

    def test_union_combines_two_lists(self):
        """Test combining two context lists into one enum."""
        context = Context()
        context.set("users", ["alice", "bob"])
        context.set("admins", ["charlie", "diana"])

        dynamic_type = DynamicUnion[("users", "admins")]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "string", "enum": ["alice", "bob", "charlie", "diana"]}

    def test_union_combines_three_lists(self):
        """Test combining three context lists."""
        context = Context()
        context.set("users", ["alice", "bob"])
        context.set("admins", ["charlie"])
        context.set("guests", ["diana", "eve"])

        dynamic_type = DynamicUnion[("users", "admins", "guests")]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "string", "enum": ["alice", "bob", "charlie", "diana", "eve"]}

    def test_union_removes_duplicates(self):
        """Test that duplicate values are removed."""
        context = Context()
        context.set("list1", ["alice", "bob", "charlie"])
        context.set("list2", ["bob", "charlie", "diana"])

        dynamic_type = DynamicUnion[("list1", "list2")]
        schema = dynamic_type.generate_schema(context)

        # Should have unique values only
        assert schema["enum"] == ["alice", "bob", "charlie", "diana"]

    def test_union_with_missing_keys(self):
        """Test union with some missing context keys."""
        context = Context()
        context.set("users", ["alice", "bob"])
        # admins key doesn't exist

        dynamic_type = DynamicUnion[("users", "admins")]
        schema = dynamic_type.generate_schema(context)

        # Should still work with just users
        assert schema == {"type": "string", "enum": ["alice", "bob"]}

    def test_union_all_keys_missing(self):
        """Test union when all keys are missing."""
        context = Context()

        dynamic_type = DynamicUnion[("users", "admins")]
        schema = dynamic_type.generate_schema(context)

        # Should return basic string type
        assert schema == {"type": "string"}

    def test_union_with_non_list_values(self):
        """Test union with non-list values."""
        context = Context()
        context.set("users", ["alice", "bob"])
        context.set("current_user", "charlie")  # Single value, not list

        dynamic_type = DynamicUnion[("users", "current_user")]
        schema = dynamic_type.generate_schema(context)

        # Should include the single value
        assert "charlie" in schema["enum"]
        assert "alice" in schema["enum"]
        assert "bob" in schema["enum"]

    def test_union_empty_lists(self):
        """Test union with empty lists."""
        context = Context()
        context.set("users", [])
        context.set("admins", [])

        dynamic_type = DynamicUnion[("users", "admins")]
        schema = dynamic_type.generate_schema(context)

        # Should return basic string type
        assert schema == {"type": "string"}


class TestDynamicConditional:
    """Test DynamicConditional type."""

    def test_conditional_true_branch(self):
        """Test conditional when condition is truthy."""
        context = Context()
        context.set("advanced_mode", True)
        context.set("simple_options", ["opt1", "opt2"])
        context.set("advanced_options", ["opt1", "opt2", "opt3", "opt4"])

        dynamic_type = DynamicConditional[("advanced_mode", "advanced_options", "simple_options")]
        schema = dynamic_type.generate_schema(context)

        # Should use advanced_options because advanced_mode is True
        assert schema == {"type": "string", "enum": ["opt1", "opt2", "opt3", "opt4"]}

    def test_conditional_false_branch(self):
        """Test conditional when condition is falsy."""
        context = Context()
        context.set("advanced_mode", False)
        context.set("simple_options", ["opt1", "opt2"])
        context.set("advanced_options", ["opt1", "opt2", "opt3", "opt4"])

        dynamic_type = DynamicConditional[("advanced_mode", "advanced_options", "simple_options")]
        schema = dynamic_type.generate_schema(context)

        # Should use simple_options because advanced_mode is False
        assert schema == {"type": "string", "enum": ["opt1", "opt2"]}

    def test_conditional_with_string_condition(self):
        """Test conditional with string condition."""
        context = Context()
        context.set("mode", "expert")  # Truthy string
        context.set("basic", ["1", "2"])
        context.set("expert", ["1", "2", "3", "4", "5"])

        dynamic_type = DynamicConditional[("mode", "expert", "basic")]
        schema = dynamic_type.generate_schema(context)

        # Non-empty string is truthy
        assert schema == {"type": "string", "enum": ["1", "2", "3", "4", "5"]}

    def test_conditional_with_none_condition(self):
        """Test conditional with None condition (falsy)."""
        context = Context()
        context.set("feature_enabled", None)
        context.set("enabled_options", ["a", "b", "c"])
        context.set("disabled_options", ["x"])

        dynamic_type = DynamicConditional[
            ("feature_enabled", "enabled_options", "disabled_options")
        ]
        schema = dynamic_type.generate_schema(context)

        # None is falsy, should use disabled_options
        assert schema == {"type": "string", "enum": ["x"]}

    def test_conditional_with_dict_schema(self):
        """Test conditional with dict schema values."""
        context = Context()
        context.set("complex_mode", True)
        context.set("simple_schema", {"type": "string", "maxLength": 10})
        context.set("complex_schema", {"type": "string", "minLength": 5, "maxLength": 100})

        dynamic_type = DynamicConditional[("complex_mode", "complex_schema", "simple_schema")]
        schema = dynamic_type.generate_schema(context)

        # Should use the full dict schema
        assert schema == {"type": "string", "minLength": 5, "maxLength": 100}

    def test_conditional_missing_condition_key(self):
        """Test conditional when condition key doesn't exist."""
        context = Context()
        context.set("option1", ["a", "b"])
        context.set("option2", ["c", "d"])
        # condition key doesn't exist

        dynamic_type = DynamicConditional[("missing", "option1", "option2")]
        schema = dynamic_type.generate_schema(context)

        # Missing key is falsy, should use option2
        assert schema == {"type": "string", "enum": ["c", "d"]}

    def test_conditional_missing_schema_key(self):
        """Test conditional when selected schema key doesn't exist."""
        context = Context()
        context.set("condition", True)
        # true_key doesn't exist
        context.set("false_options", ["a", "b"])

        dynamic_type = DynamicConditional[("condition", "true_options", "false_options")]
        schema = dynamic_type.generate_schema(context)

        # Should return basic string type when schema is missing
        assert schema == {"type": "string"}

    def test_conditional_numeric_conditions(self):
        """Test conditional with numeric conditions."""
        context = Context()
        context.set("level", 5)  # Non-zero is truthy
        context.set("high_level", ["advanced1", "advanced2"])
        context.set("low_level", ["basic1"])

        dynamic_type = DynamicConditional[("level", "high_level", "low_level")]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "string", "enum": ["advanced1", "advanced2"]}

        # Test with 0 (falsy)
        context.set("level", 0)
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "string", "enum": ["basic1"]}

    def test_conditional_requires_three_params(self):
        """Test that DynamicConditional requires exactly 3 parameters."""
        with pytest.raises(ValueError, match="exactly 3 parameters"):
            DynamicConditional[("only", "two")]

        with pytest.raises(ValueError, match="exactly 3 parameters"):
            DynamicConditional[("one", "two", "three", "four")]
