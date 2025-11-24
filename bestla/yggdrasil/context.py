"""Context management for stateful toolkits."""

from typing import Any, Dict

import jsonschema
from immutables import Map
from jsonschema import ValidationError

from bestla.yggdrasil.exceptions import ContextValidationError


class ContextSchema:
    """Schema validator for context values."""

    def __init__(self):
        self._schemas: Dict[str, dict] = {}

    def define(self, key: str, schema: dict) -> None:
        """Define a JSON schema for a context key.

        Args:
            key: Context key to validate
            schema: JSON schema definition
        """
        self._schemas[key] = schema

    def validate(self, key: str, value: Any) -> None:
        """Validate a value against its schema.

        Args:
            key: Context key
            value: Value to validate

        Raises:
            ContextValidationError: If validation fails
        """
        if key not in self._schemas:
            return  # No schema defined, skip validation

        try:
            jsonschema.validate(value, self._schemas[key])
        except ValidationError as e:
            raise ContextValidationError(
                f"Validation failed for context key '{key}': {e.message}"
            )

    def has_schema(self, key: str) -> bool:
        """Check if a key has a schema defined."""
        return key in self._schemas


class Context:
    """Dictionary-like state container with optional validation.

    Context stores toolkit state and can validate updates against schemas.
    Contexts are deep-copyable for sub-agent isolation.
    """

    def __init__(self, validation_enabled: bool = False):
        """Initialize context.

        Args:
            validation_enabled: Whether to validate updates against schemas
        """
        self._data = Map()
        self.validation_enabled = validation_enabled
        self.schema = ContextSchema()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value.

        Args:
            key: Context key (supports dot notation for nested access)
            default: Default value if key doesn't exist

        Returns:
            Context value or default
        """
        # Support nested access: "project.id" (only for string keys)
        if isinstance(key, str) and "." in key:
            parts = key.split(".")
            current = self._data
            for part in parts:
                # Check if current supports dict-like access (Map or dict)
                if isinstance(current, (dict, Map)) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a single context value.

        Args:
            key: Context key
            value: Value to set

        Raises:
            ContextValidationError: If validation is enabled and fails
        """
        if self.validation_enabled:
            self.schema.validate(key, value)
        self._data = self._data.set(key, value)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple context values.

        Args:
            updates: Dictionary of key-value pairs to update

        Raises:
            ContextValidationError: If validation is enabled and fails
        """
        if self.validation_enabled:
            for key, value in updates.items():
                self.schema.validate(key, value)

        # Chain set operations for immutable Map
        new_data = self._data
        for key, value in updates.items():
            new_data = new_data.set(key, value)
        self._data = new_data

    def has(self, key: str) -> bool:
        """Check if a context key exists.

        Args:
            key: Context key to check

        Returns:
            True if key exists
        """
        # Support nested access: "project.id" (only for string keys)
        if isinstance(key, str) and "." in key:
            parts = key.split(".")
            current = self._data
            for part in parts:
                # Check if current supports dict-like access (Map or dict)
                if isinstance(current, (dict, Map)) and part in current:
                    current = current[part]
                else:
                    return False
            return True
        return key in self._data

    def delete(self, key: str) -> None:
        """Delete a context key.

        Args:
            key: Context key to delete
        """
        if key in self._data:
            self._data = self._data.delete(key)

    def clear(self) -> None:
        """Clear all context data."""
        self._data = Map()

    def copy(self) -> "Context":
        """Create a shallow copy of this context.

        With immutable Map, this is O(1) as we just copy the reference.
        Each context modification creates a new Map version with structural sharing.

        Returns:
            New Context instance with shared immutable data
        """
        new_context = Context(validation_enabled=self.validation_enabled)
        new_context._data = self._data  # O(1) reference copy - Map is immutable!
        # Note: schemas are not copied, they're toolkit-level configuration
        return new_context

    def to_dict(self) -> Dict[str, Any]:
        """Get the raw context data.

        Returns:
            Dictionary of context data
        """
        return dict(self._data)

    def __repr__(self) -> str:
        return f"Context({self._data})"

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __getitem__(self, key: str) -> Any:
        if not self.has(key):
            raise KeyError(f"Context key '{key}' not found")
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)
