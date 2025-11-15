"""Dynamic type system for context-driven JSON schema generation.

This module provides Python type-hint-like syntax that generates JSON schemas
based on runtime context values.

Example:
    def get_issue(name: DynamicStr["issue_names"]):
        # At runtime, generates enum from context["issue_names"]
        ...
"""

from typing import Any, Dict, List, Tuple


class DynamicType:
    """Base class for all dynamic types."""

    def __init__(self, context_key: str):
        self.context_key = context_key

    def generate_schema(self, context: Any) -> dict:
        """Generate JSON schema based on context.

        Args:
            context: Context instance to read from

        Returns:
            JSON schema dictionary
        """
        raise NotImplementedError


class DynamicStr:
    """Dynamic string type that generates enum from context.

    Usage:
        def select_issue(name: DynamicStr["issue_names"]):
            ...

    Context format:
        {"issue_names": ["BUG-1", "FEAT-2"]}

    Generated schema:
        {"type": "string", "enum": ["BUG-1", "FEAT-2"]}
    """

    def __init__(self, context_key: str):
        self.context_key = context_key

    def __class_getitem__(cls, key: str) -> "DynamicStr":
        return cls(key)

    def generate_schema(self, context: Any) -> dict:
        """Generate string enum schema from context."""
        value = context.get(self.context_key)
        if value is None:
            return {"type": "string"}

        if isinstance(value, list):
            return {"type": "string", "enum": value}
        elif isinstance(value, dict) and "enum" in value:
            return {"type": "string", **value}
        else:
            return {"type": "string"}


class DynamicInt:
    """Dynamic integer type with constraints from context.

    Usage:
        def set_priority(level: DynamicInt["priority_range"]):
            ...

    Context format:
        {"priority_range": {"minimum": 1, "maximum": 5}}

    Generated schema:
        {"type": "integer", "minimum": 1, "maximum": 5}
    """

    def __init__(self, context_key: str):
        self.context_key = context_key

    def __class_getitem__(cls, key: str) -> "DynamicInt":
        return cls(key)

    def generate_schema(self, context: Any) -> dict:
        """Generate integer schema with constraints from context."""
        value = context.get(self.context_key)
        if value is None:
            return {"type": "integer"}

        schema = {"type": "integer"}
        if isinstance(value, dict):
            # Support minimum, maximum, exclusiveMinimum, exclusiveMaximum
            for constraint in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]:
                if constraint in value:
                    schema[constraint] = value[constraint]
        elif isinstance(value, list) and len(value) >= 2:
            # [min, max] shorthand
            schema["minimum"] = value[0]
            schema["maximum"] = value[1]

        return schema


class DynamicFloat:
    """Dynamic float type with constraints from context.

    Usage:
        def set_confidence(score: DynamicFloat["confidence_range"]):
            ...

    Context format:
        {"confidence_range": {"minimum": 0.0, "maximum": 1.0}}
    """

    def __init__(self, context_key: str):
        self.context_key = context_key

    def __class_getitem__(cls, key: str) -> "DynamicFloat":
        return cls(key)

    def generate_schema(self, context: Any) -> dict:
        """Generate number schema with constraints from context."""
        value = context.get(self.context_key)
        if value is None:
            return {"type": "number"}

        schema = {"type": "number"}
        if isinstance(value, dict):
            for constraint in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]:
                if constraint in value:
                    schema[constraint] = value[constraint]
        elif isinstance(value, list) and len(value) >= 2:
            schema["minimum"] = value[0]
            schema["maximum"] = value[1]

        return schema


class DynamicArray:
    """Dynamic array type with item enum from context.

    Usage:
        def assign_users(names: DynamicArray["user_names"]):
            ...

    Context format:
        {"user_names": ["alice", "bob", "charlie"]}

    Generated schema:
        {"type": "array", "items": {"type": "string", "enum": ["alice", "bob", "charlie"]}}
    """

    def __init__(self, context_key: str):
        self.context_key = context_key

    def __class_getitem__(cls, key: str) -> "DynamicArray":
        return cls(key)

    def generate_schema(self, context: Any) -> dict:
        """Generate array schema with item enum from context."""
        value = context.get(self.context_key)
        if value is None:
            return {"type": "array"}

        schema: Dict[str | List[str]] = {"type": "array"}
        if isinstance(value, list):
            schema["items"] = {"type": "string", "enum": value}
        elif isinstance(value, dict) and "items" in value:
            schema["items"] = value["items"]

        return schema


class DynamicFormat:
    """String with specific format.

    Usage:
        def schedule(time: DynamicFormat["date-time"]):
            ...

    Generated schema:
        {"type": "string", "format": "date-time"}
    """

    def __init__(self, format_type: str):
        self.format_type = format_type

    def __class_getitem__(cls, format_type: str) -> "DynamicFormat":
        return cls(format_type)

    def generate_schema(self, _: Any) -> dict:
        """Generate string schema with format."""
        return {"type": "string", "format": self.format_type}


class DynamicPattern:
    """String with regex pattern from context.

    Usage:
        def validate_input(text: DynamicPattern["input_pattern"]):
            ...

    Context format:
        {"input_pattern": "^[A-Z]{3}-[0-9]+$"}

    Generated schema:
        {"type": "string", "pattern": "^[A-Z]{3}-[0-9]+$"}
    """

    def __init__(self, context_key: str):
        self.context_key = context_key

    def __class_getitem__(cls, key: str) -> "DynamicPattern":
        return cls(key)

    def generate_schema(self, context: Any) -> dict:
        """Generate string schema with pattern from context."""
        pattern = context.get(self.context_key)
        if pattern is None:
            return {"type": "string"}
        return {"type": "string", "pattern": pattern}


class DynamicConst:
    """Constant value from context.

    Usage:
        def use_selected(project_id: DynamicConst["selected_project"]):
            ...

    Context format:
        {"selected_project": "proj-123"}

    Generated schema:
        {"const": "proj-123"}
    """

    def __init__(self, context_key: str):
        self.context_key = context_key

    def __class_getitem__(cls, key: str) -> "DynamicConst":
        return cls(key)

    def generate_schema(self, context: Any) -> dict:
        """Generate const schema from context."""
        value = context.get(self.context_key)
        if value is None:
            return {}
        return {"const": value}


class DynamicFiltered:
    """Filtered values from context using registered filter.

    Usage:
        def assign(user: DynamicFiltered[("users", "active_only")]):
            ...

    Context format:
        {"users": [{"name": "alice", "active": True}, {"name": "bob", "active": False}]}

    With filter "active_only":
        lambda users: [u["name"] for u in users if u["active"]]

    Generated schema:
        {"type": "string", "enum": ["alice"]}
    """

    def __init__(self, params: Tuple[str, str]):
        self.context_key, self.filter_name = params

    def __class_getitem__(cls, params: Tuple[str, str]) -> "DynamicFiltered":
        return cls(params)

    def generate_schema(self, context: Any, filters: dict | None = None) -> dict:
        """Generate schema with filtered values.

        Args:
            context: Context instance
            filters: Dictionary of registered filter functions

        Returns:
            JSON schema with filtered enum
        """
        value = context.get(self.context_key)
        if value is None or filters is None or self.filter_name not in filters:
            return {"type": "string"}

        filter_func = filters[self.filter_name]
        try:
            filtered = filter_func(value)
            if isinstance(filtered, list):
                return {"type": "string", "enum": filtered}
        except Exception:
            pass

        return {"type": "string"}


class DynamicNested:
    """Nested context access.

    Usage:
        def use_status(status: DynamicNested["project.statuses"]):
            ...

    Context format:
        {"project": {"id": "p1", "statuses": ["todo", "done"]}}

    Generated schema:
        {"type": "string", "enum": ["todo", "done"]}
    """

    def __init__(self, context_key: str):
        self.context_key = context_key

    def __class_getitem__(cls, key: str) -> "DynamicNested":
        return cls(key)

    def generate_schema(self, context: Any) -> dict:
        """Generate schema from nested context value."""
        value = context.get(self.context_key)  # Context.get supports dot notation
        if value is None:
            return {"type": "string"}

        if isinstance(value, list):
            return {"type": "string", "enum": value}

        return {"type": "string"}


class DynamicConstraints:
    """Generic schema application from context.

    Usage:
        def complex_param(data: DynamicConstraints["data_schema"]):
            ...

    Context format:
        {"data_schema": {"type": "object", "properties": {...}, "required": [...]}}

    Generated schema:
        <whatever is in context>
    """

    def __init__(self, context_key: str):
        self.context_key = context_key

    def __class_getitem__(cls, key: str) -> "DynamicConstraints":
        return cls(key)

    def generate_schema(self, context: Any) -> dict:
        """Generate schema from context."""
        schema = context.get(self.context_key)
        if schema is None or not isinstance(schema, dict):
            return {}
        return schema


class DynamicUnion:
    """Combine multiple context keys into a single enum.

    Usage:
        def select_person(name: DynamicUnion[("users", "admins")]):
            ...

    Context format:
        {
            "users": ["alice", "bob"],
            "admins": ["charlie", "diana"]
        }

    Generated schema:
        {"type": "string", "enum": ["alice", "bob", "charlie", "diana"]}
    """

    def __init__(self, context_keys: Tuple[str, ...]):
        self.context_keys = context_keys

    def __class_getitem__(cls, keys: Tuple[str, ...]) -> "DynamicUnion":
        return cls(keys)

    def generate_schema(self, context: Any) -> dict:
        """Generate schema by combining multiple context keys."""
        combined_values = []

        for key in self.context_keys:
            value = context.get(key)
            if value is not None:
                if isinstance(value, list):
                    combined_values.extend(value)
                else:
                    combined_values.append(value)

        if not combined_values:
            return {"type": "string"}

        # Remove duplicates while preserving order
        unique_values = []
        seen = set()
        for v in combined_values:
            # Handle unhashable types
            try:
                if v not in seen:
                    unique_values.append(v)
                    seen.add(v)
            except TypeError:
                # Unhashable type, just append
                unique_values.append(v)

        return {"type": "string", "enum": unique_values}


class DynamicConditional:
    """Conditional type selection based on context value.

    Usage:
        def configure(value: DynamicConditional[("mode", "simple_options", "advanced_options")]):
            ...

    Context format:
        {
            "mode": "simple",  # or "advanced"
            "simple_options": ["opt1", "opt2"],
            "advanced_options": ["opt1", "opt2", "opt3", "opt4"]
        }

    Generated schema:
        - If mode == "simple": {"type": "string", "enum": ["opt1", "opt2"]}
        - If mode == "advanced": {"type": "string", "enum": ["opt1", "opt2", "opt3", "opt4"]}
        - If mode is anything else: Uses simple_options as default
    """

    def __init__(self, params: Tuple[str, str, str]):
        if len(params) != 3:
            raise ValueError(
                "DynamicConditional requires exactly 3 parameters: "
                "(condition_key, true_key, false_key)"
            )
        self.condition_key, self.true_key, self.false_key = params

    def __class_getitem__(cls, params: Tuple[str, str, str]) -> "DynamicConditional":
        return cls(params)

    def generate_schema(self, context: Any) -> dict:
        """Generate schema based on condition value.

        The condition is evaluated as truthy/falsy.
        If condition is truthy, use true_key schema.
        If condition is falsy, use false_key schema.
        """
        condition_value = context.get(self.condition_key)

        # Determine which key to use based on condition
        if condition_value:
            schema_key = self.true_key
        else:
            schema_key = self.false_key

        # Get the schema values
        schema_value = context.get(schema_key)

        if schema_value is None:
            return {"type": "string"}

        # If it's a list, create enum
        if isinstance(schema_value, list):
            return {"type": "string", "enum": schema_value}

        # If it's a dict, assume it's a full schema
        if isinstance(schema_value, dict):
            return schema_value

        return {"type": "string"}


# Helper function to check if a type annotation is a dynamic type
def is_dynamic_type(annotation: Any) -> bool:
    """Check if an annotation is a dynamic type."""
    return isinstance(
        annotation,
        (
            DynamicStr,
            DynamicInt,
            DynamicFloat,
            DynamicArray,
            DynamicFormat,
            DynamicPattern,
            DynamicConst,
            DynamicFiltered,
            DynamicNested,
            DynamicConstraints,
            DynamicUnion,
            DynamicConditional,
        ),
    )


def generate_param_schema(annotation: Any, context: Any, filters: dict | None = None) -> dict:
    """Generate JSON schema for a parameter based on its type annotation.

    Args:
        annotation: Type annotation (can be dynamic type or standard type)
        context: Context instance
        filters: Optional dictionary of filter functions

    Returns:
        JSON schema dictionary
    """
    if isinstance(annotation, DynamicFiltered):
        return annotation.generate_schema(context, filters)
    elif is_dynamic_type(annotation):
        return annotation.generate_schema(context)
    elif annotation is str:
        return {"type": "string"}
    elif annotation is int:
        return {"type": "integer"}
    elif annotation is float:
        return {"type": "number"}
    elif annotation is bool:
        return {"type": "boolean"}
    elif annotation is list or annotation is List:
        return {"type": "array"}
    elif annotation is dict or annotation is Dict:
        return {"type": "object"}
    else:
        # Unknown type, default to generic
        return {}
