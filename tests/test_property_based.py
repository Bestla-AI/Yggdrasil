"""Property-based tests using Hypothesis for automatic edge case discovery."""

from typing import Tuple

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from bestla.yggdrasil import Context, Toolkit


# Custom strategies
@st.composite
def valid_context_keys(draw):
    """Generate valid context keys."""
    return draw(
        st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",),  # Exclude surrogates
                blacklist_characters="\x00.",  # Exclude null and dots (reserved for nesting)
            ),
            min_size=1,
            max_size=100,
        )
    )


@st.composite
def context_values(draw):
    """Generate valid context values."""
    return draw(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(min_value=-1000000, max_value=1000000),
            st.floats(allow_nan=False, allow_infinity=False, width=32),
            st.text(max_size=1000),
            st.lists(st.integers(), max_size=100),
            st.dictionaries(st.text(max_size=50), st.integers(), max_size=20),
        )
    )


class TestContextProperties:
    """Property-based tests for Context."""

    @given(key=valid_context_keys(), value=context_values())
    @settings(max_examples=50, deadline=1000)
    def test_set_then_get_returns_same_value(self, key, value):
        """Property: set(k, v) then get(k) returns v."""
        context = Context()
        context.set(key, value)

        result = context.get(key)
        assert result == value

    @given(data=st.dictionaries(valid_context_keys(), context_values(), min_size=0, max_size=50))
    @settings(max_examples=30, deadline=1000)
    def test_context_copy_is_independent(self, data):
        """Property: Copied context is independent."""
        context1 = Context()
        for k, v in data.items():
            context1.set(k, v)

        context2 = context1.copy()

        # Modify context1
        if data:
            first_key = list(data.keys())[0]
            context1.set(first_key, "modified")

            # context2 should have original value
            assert context2.get(first_key) == data[first_key]

    @given(
        keys=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20
            ),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    @settings(max_examples=30, deadline=1000)
    def test_nested_access_consistency(self, keys):
        """Property: Nested access is consistent with dict structure."""
        context = Context()

        # Build nested dict
        nested_dict = {}
        current = nested_dict
        for i, key in enumerate(keys[:-1]):
            current[key] = {}
            current = current[key]
        current[keys[-1]] = "test_value"

        # Set in context
        context.set(keys[0], nested_dict[keys[0]])

        # Access via nested path
        path = ".".join(keys)
        result = context.get(path)

        assert result == "test_value"

    @given(updates=st.dictionaries(valid_context_keys(), context_values(), min_size=1, max_size=20))
    @settings(max_examples=30, deadline=1000)
    def test_update_applies_all_changes(self, updates):
        """Property: update() applies all key-value pairs."""
        context = Context()
        context.update(updates)

        for key, value in updates.items():
            assert context.get(key) == value

    @given(key=valid_context_keys(), value=context_values())
    @settings(max_examples=50, deadline=1000)
    def test_contains_after_set(self, key, value):
        """Property: key in context after set(key, value)."""
        context = Context()
        context.set(key, value)

        assert key in context

    @given(key=valid_context_keys())
    @settings(max_examples=50, deadline=1000)
    def test_delete_removes_key(self, key):
        """Property: delete(key) removes key from context."""
        context = Context()
        context.set(key, "value")

        assert key in context

        context.delete(key)

        assert key not in context
        assert context.get(key) is None


class TestToolkitProperties:
    """Property-based tests for Toolkit."""

    @given(state_name=st.text(min_size=1, max_size=50))
    @settings(max_examples=50, deadline=1000)
    def test_state_enabling_idempotent(self, state_name):
        """Property: Enabling a state multiple times is idempotent."""
        toolkit = Toolkit()

        # Enable once
        toolkit.set_unlocked_states({state_name})
        states1 = toolkit.unlocked_states

        # Enable again
        toolkit.set_unlocked_states({state_name})
        states2 = toolkit.unlocked_states

        assert states1 == states2 == {state_name}

    @given(states=st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=10, unique=True))
    @settings(max_examples=30, deadline=1000)
    def test_multiple_states_all_enabled(self, states):
        """Property: All states in set are enabled."""
        toolkit = Toolkit()

        state_set = set(states)
        toolkit.set_unlocked_states(state_set)

        enabled = toolkit.unlocked_states

        assert enabled == state_set


class TestDynamicTypeProperties:
    """Property-based tests for dynamic types."""

    @given(
        enum_values=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=50, unique=True)
    )
    @settings(max_examples=30, deadline=1000)
    def test_dynamic_str_generates_valid_schema(self, enum_values):
        """Property: DynamicStr always generates valid JSON schema."""
        from bestla.yggdrasil.dynamic_types import DynamicStr

        context = Context()
        context.set("test_key", enum_values)

        schema = DynamicStr["test_key"].generate_schema(context)

        assert schema["type"] == "string"
        assert "enum" in schema
        assert set(schema["enum"]) == set(enum_values)

    @given(
        min_val=st.integers(min_value=0, max_value=100),
        max_val=st.integers(min_value=101, max_value=1000),
    )
    @settings(max_examples=30, deadline=1000)
    def test_dynamic_int_generates_valid_constraints(self, min_val, max_val):
        """Property: DynamicInt generates valid constraints."""
        from bestla.yggdrasil.dynamic_types import DynamicInt

        context = Context()
        context.set("number", {"minimum": min_val, "maximum": max_val})

        schema = DynamicInt["number"].generate_schema(context)

        assert schema["type"] == "integer"
        assert schema["minimum"] == min_val
        assert schema["maximum"] == max_val


class TestAgentProperties:
    """Property-based tests for Agent."""

    @given(
        num_toolkits=st.integers(min_value=0, max_value=5),
        num_tools_per_toolkit=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=20, deadline=2000)
    def test_schema_generation_always_valid(self, num_toolkits, num_tools_per_toolkit):
        """Property: Generated schemas are always valid OpenAI format."""
        from unittest.mock import Mock

        from bestla.yggdrasil import Agent

        agent = Agent(provider=Mock(), model="gpt-4")

        for i in range(num_toolkits):
            toolkit = Toolkit()

            for j in range(num_tools_per_toolkit):

                def test_tool() -> Tuple[str, dict]:
                    return "result", {}

                toolkit.add_tool(f"tool_{j}", test_tool)

            agent.add_toolkit(f"tk_{i}", toolkit)

        # Verify toolkits registered correctly
        assert len(agent.toolkits) == num_toolkits

        # Verify each toolkit has correct number of tools
        for toolkit_name, toolkit in agent.toolkits.items():
            assert len(toolkit.tools) == num_tools_per_toolkit


class TestInvariantProperties:
    """Test invariants that should always hold."""

    @given(key=valid_context_keys(), value=context_values())
    @settings(max_examples=50, deadline=1000)
    def test_context_size_non_decreasing_on_set(self, key, value):
        """Property: Context size doesn't decrease when adding keys."""
        context = Context()

        initial_size = len([k for k in context._data.keys()])

        context.set(key, value)

        final_size = len([k for k in context._data.keys()])

        assert final_size >= initial_size

    @given(data=st.dictionaries(valid_context_keys(), context_values(), min_size=1, max_size=20))
    @settings(max_examples=20, deadline=1000)
    def test_copy_preserves_all_data(self, data):
        """Property: Copy contains all original data."""
        context = Context()

        for k, v in data.items():
            context.set(k, v)

        copy = context.copy()

        # All keys should be in copy
        for k in data.keys():
            assert k in copy
            assert copy.get(k) == data[k]


class TestCommutativity:
    """Test commutative operations."""

    @given(
        key1=valid_context_keys(),
        value1=context_values(),
        key2=valid_context_keys(),
        value2=context_values(),
    )
    @settings(max_examples=30, deadline=1000)
    def test_set_order_independence(self, key1, value1, key2, value2):
        """Property: Setting different keys is order-independent."""
        assume(key1 != key2)  # Keys must be different

        context1 = Context()
        context1.set(key1, value1)
        context1.set(key2, value2)

        context2 = Context()
        context2.set(key2, value2)
        context2.set(key1, value1)

        # Both should have same values
        assert context1.get(key1) == context2.get(key1) == value1
        assert context1.get(key2) == context2.get(key2) == value2


class TestIdempotence:
    """Test idempotent operations."""

    @given(key=valid_context_keys(), value=context_values())
    @settings(max_examples=50, deadline=1000)
    def test_set_idempotent(self, key, value):
        """Property: Setting same key twice is idempotent."""
        context = Context()

        context.set(key, value)
        result1 = context.get(key)

        context.set(key, value)
        result2 = context.get(key)

        assert result1 == result2 == value

    @given(key=valid_context_keys())
    @settings(max_examples=50, deadline=1000)
    def test_delete_idempotent(self, key):
        """Property: Deleting non-existent key is idempotent."""
        context = Context()

        # Delete non-existent key (should not raise)
        context.delete(key)
        context.delete(key)

        assert key not in context


class TestRoundtripProperties:
    """Test roundtrip properties."""

    @given(data=st.dictionaries(valid_context_keys(), context_values(), min_size=1, max_size=20))
    @settings(max_examples=20, deadline=1000)
    def test_update_then_copy_preserves_data(self, data):
        """Property: update() then copy() preserves all data."""
        context = Context()
        context.update(data)

        copy = context.copy()

        for key, value in data.items():
            assert copy.get(key) == value
