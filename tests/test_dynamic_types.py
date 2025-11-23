"""Tests for dynamic type system."""

from bestla.yggdrasil import Context
from bestla.yggdrasil.dynamic_types import (
    DynamicArray,
    DynamicConst,
    DynamicConstraints,
    DynamicFiltered,
    DynamicFloat,
    DynamicFormat,
    DynamicInt,
    DynamicNested,
    DynamicPattern,
    DynamicStr,
    generate_param_schema,
)


class TestDynamicStr:
    """Test DynamicStr type."""

    def test_enum_from_context(self):
        """Test generating enum from context list."""
        context = Context()
        context.set("issue_names", ["BUG-1", "FEAT-2", "BUG-3"])

        dynamic_type = DynamicStr["issue_names"]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "string", "enum": ["BUG-1", "FEAT-2", "BUG-3"]}

    def test_missing_context_key(self):
        """Test when context key doesn't exist."""
        context = Context()
        dynamic_type = DynamicStr["missing"]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "string"}

    def test_dict_with_enum(self):
        """Test context value as dict with enum."""
        context = Context()
        context.set("status", {"enum": ["open", "closed"]})

        dynamic_type = DynamicStr["status"]
        schema = dynamic_type.generate_schema(context)

        assert "enum" in schema


class TestDynamicInt:
    """Test DynamicInt type."""

    def test_constraints_from_dict(self):
        """Test generating constraints from dict."""
        context = Context()
        context.set("priority_range", {"minimum": 1, "maximum": 5})

        dynamic_type = DynamicInt["priority_range"]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "integer", "minimum": 1, "maximum": 5}

    def test_constraints_from_list(self):
        """Test generating constraints from [min, max] list."""
        context = Context()
        context.set("score_range", [0, 100])

        dynamic_type = DynamicInt["score_range"]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "integer", "minimum": 0, "maximum": 100}

    def test_missing_context(self):
        """Test when context key missing."""
        context = Context()
        dynamic_type = DynamicInt["missing"]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "integer"}


class TestDynamicFloat:
    """Test DynamicFloat type."""

    def test_constraints(self):
        """Test float constraints."""
        context = Context()
        context.set("confidence", {"minimum": 0.0, "maximum": 1.0})

        dynamic_type = DynamicFloat["confidence"]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "number", "minimum": 0.0, "maximum": 1.0}


class TestDynamicArray:
    """Test DynamicArray type."""

    def test_array_with_enum(self):
        """Test array with item enum."""
        context = Context()
        context.set("user_names", ["alice", "bob", "charlie"])

        dynamic_type = DynamicArray["user_names"]
        schema = dynamic_type.generate_schema(context)

        assert schema == {
            "type": "array",
            "items": {"type": "string", "enum": ["alice", "bob", "charlie"]},
        }


class TestDynamicFormat:
    """Test DynamicFormat type."""

    def test_format(self):
        """Test string format."""
        dynamic_type = DynamicFormat["date-time"]
        schema = dynamic_type.generate_schema(None)

        assert schema == {"type": "string", "format": "date-time"}


class TestDynamicPattern:
    """Test DynamicPattern type."""

    def test_pattern_from_context(self):
        """Test pattern from context."""
        context = Context()
        context.set("issue_pattern", "^[A-Z]+-[0-9]+$")

        dynamic_type = DynamicPattern["issue_pattern"]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "string", "pattern": "^[A-Z]+-[0-9]+$"}


class TestDynamicConst:
    """Test DynamicConst type."""

    def test_const_value(self):
        """Test constant value."""
        context = Context()
        context.set("selected_project", "proj-123")

        dynamic_type = DynamicConst["selected_project"]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"const": "proj-123"}


class TestDynamicNested:
    """Test DynamicNested type."""

    def test_nested_access(self):
        """Test nested context access."""
        context = Context()
        context.set("project", {"id": "p1", "statuses": ["todo", "done"]})

        dynamic_type = DynamicNested["project.statuses"]
        schema = dynamic_type.generate_schema(context)

        assert schema == {"type": "string", "enum": ["todo", "done"]}


class TestDynamicFiltered:
    """Test DynamicFiltered type."""

    def test_filtered_values(self):
        """Test filtered values."""
        context = Context()
        context.set(
            "users",
            [
                {"name": "alice", "active": True},
                {"name": "bob", "active": False},
                {"name": "charlie", "active": True},
            ],
        )

        filters = {"active_only": lambda users: [u["name"] for u in users if u.get("active")]}

        dynamic_type = DynamicFiltered[("users", "active_only")]
        schema = dynamic_type.generate_schema(context, filters)

        assert schema == {"type": "string", "enum": ["alice", "charlie"]}


class TestDynamicConstraints:
    """Test DynamicConstraints type."""

    def test_generic_schema(self):
        """Test generic schema from context."""
        context = Context()
        context.set(
            "complex_schema",
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        )

        dynamic_type = DynamicConstraints["complex_schema"]
        schema = dynamic_type.generate_schema(context)

        assert schema["type"] == "object"
        assert "properties" in schema


class TestGenerateParamSchema:
    """Test generate_param_schema helper."""

    def test_standard_types(self):
        """Test standard Python types."""
        context = Context()

        assert generate_param_schema(str, context) == {"type": "string"}
        assert generate_param_schema(int, context) == {"type": "integer"}
        assert generate_param_schema(float, context) == {"type": "number"}
        assert generate_param_schema(bool, context) == {"type": "boolean"}

    def test_dynamic_type(self):
        """Test with dynamic type."""
        context = Context()
        context.set("options", ["a", "b", "c"])

        dynamic_type = DynamicStr["options"]
        schema = generate_param_schema(dynamic_type, context)

        assert schema == {"type": "string", "enum": ["a", "b", "c"]}
