"""Tests for FSM (Finite State Machine) edge cases."""

from typing import Tuple

from bestla.yggdrasil import Toolkit, tool


class TestStateConflicts:
    """Test FSM state conflicts and edge cases."""

    def test_circular_state_dependencies(self):
        """Test circular dependencies: A enables B, B enables A."""
        toolkit = Toolkit()

        def tool_a() -> Tuple[str, dict]:
            return "a", {}

        def tool_b() -> Tuple[str, dict]:
            return "b", {}

        def tool_c() -> Tuple[str, dict]:
            return "c", {}

        toolkit.add_tool("tool_a", tool_a, enables_states=["state_b"])
        toolkit.add_tool("tool_b", tool_b, required_states=["state_a"], enables_states=["state_c"])
        toolkit.add_tool("tool_c", tool_c, required_states=["state_b"], enables_states=["state_a"])

        # tool_a should be available (no requirements)
        assert toolkit.is_tool_available("tool_a")

        # tool_b and tool_c form circular dependency
        assert not toolkit.is_tool_available("tool_b")  # Needs state_a
        assert not toolkit.is_tool_available("tool_c")  # Needs state_b

    def test_conflicting_state_requirements(self):
        """Test tool requiring and forbidding same state."""

        def impossible_tool() -> Tuple[str, dict]:
            return "never_available", {}

        toolkit = Toolkit()
        toolkit.add_tool(
            "impossible",
            impossible_tool,
            required_states=["auth"],
            forbidden_states=["auth"],
        )

        # Enable the state
        toolkit.set_unlocked_states({"auth"})

        # Tool should never be available (conflict)
        available = toolkit.is_tool_available("impossible")
        reason = toolkit.get_availability_reason("impossible")
        # Depending on implementation, might be false or raise
        # Test actual behavior
        assert not available or "conflict" in reason.lower()

    def test_enable_and_disable_same_state(self):
        """Test tool with same state in enables and disables."""

        @tool(enables_states=["state_x"], disables_states=["state_x"])
        def conflicting() -> Tuple[str, dict]:
            return "ok", {}

        # Should be created without error
        assert "state_x" in conflicting.enables_states
        assert "state_x" in conflicting.disables_states

    def test_state_that_nothing_enables(self):
        """Test requiring state that no tool enables."""
        toolkit = Toolkit()

        def unreachable() -> Tuple[str, dict]:
            return "never", {}

        toolkit.add_tool("unreachable", unreachable, required_states=["impossible_state"])

        # Should not be available
        assert not toolkit.is_tool_available("unreachable")

        # Even if we manually enable it
        toolkit.set_unlocked_states({"impossible_state"})
        assert toolkit.is_tool_available("unreachable")


class TestStateTransitionOrder:
    """Test state transition order dependencies."""

    def test_state_transition_chain(self):
        """Test chain of state transitions A->B->C."""
        toolkit = Toolkit()

        def login() -> Tuple[str, dict]:
            return "logged in", {}

        def select_project() -> Tuple[str, dict]:
            return "project selected", {}

        def enable_editing() -> Tuple[str, dict]:
            return "editing enabled", {}

        toolkit.add_tool("login", login, enables_states=["logged_in"])
        toolkit.add_tool(
            "select_project",
            select_project,
            required_states=["logged_in"],
            enables_states=["project_selected"],
        )
        toolkit.add_tool(
            "enable_editing",
            enable_editing,
            required_states=["project_selected"],
            enables_states=["can_edit"],
        )

        # Initially, only login available
        assert toolkit.is_tool_available("login")
        assert not toolkit.is_tool_available("select_project")
        assert not toolkit.is_tool_available("enable_editing")

        # After login
        toolkit.execute_sequential([{"id": "1", "name": "login", "arguments": {}}])

        assert toolkit.is_tool_available("select_project")
        assert not toolkit.is_tool_available("enable_editing")

    def test_state_requirement_check_timing(self):
        """Test when state requirements are checked."""
        toolkit = Toolkit()

        def prepare() -> Tuple[str, dict]:
            return "prepared", {}

        def execute() -> Tuple[str, dict]:
            return "executed", {}

        toolkit.add_tool("prepare", prepare, enables_states=["ready"])
        toolkit.add_tool("execute", execute, required_states=["ready"])

        # Check availability before prepare
        assert not toolkit.is_tool_available("execute")

        # Execute prepare
        toolkit.execute_sequential([{"id": "1", "name": "prepare", "arguments": {}}])

        # Now execute should be available
        assert toolkit.is_tool_available("execute")


class TestStateValidation:
    """Test state name validation and edge cases."""

    def test_empty_state_name(self):
        """Test state with empty string name."""

        @tool(enables_states=[""])
        def empty_state() -> Tuple[str, dict]:
            return "ok", {}

        # Should be created (no validation on creation)
        assert "" in empty_state.enables_states

    def test_special_characters_in_state_names(self):
        """Test state names with special characters."""

        @tool(enables_states=["state:with:colons", "state-with-dashes", "state with spaces"])
        def special_states() -> Tuple[str, dict]:
            return "ok", {}

        # Should be created
        assert len(special_states.enables_states) == 3

    def test_very_large_state_set(self):
        """Test toolkit with many states."""
        toolkit = Toolkit()

        # Create 100 states
        large_state_set = {f"state_{i}" for i in range(100)}
        toolkit.set_unlocked_states(large_state_set)

        # Should handle large sets
        assert len(toolkit.unlocked_states) == 100

    def test_unicode_in_state_names(self):
        """Test state names with Unicode characters."""

        @tool(enables_states=["状态", "حالة", "état"])
        def unicode_states() -> Tuple[str, dict]:
            return "ok", {}

        # Should be created
        assert "状态" in unicode_states.enables_states


class TestForbiddenStates:
    """Test forbidden states functionality."""

    def test_tool_with_forbidden_state(self):
        """Test tool becomes unavailable when forbidden state enabled."""
        toolkit = Toolkit()

        def normal_operation() -> Tuple[str, dict]:
            return "operating", {}

        toolkit.add_tool("normal", normal_operation, forbidden_states=["maintenance_mode"])

        # Available initially
        assert toolkit.is_tool_available("normal")

        # Enable forbidden state
        toolkit.set_unlocked_states({"maintenance_mode"})

        # Should no longer be available
        available = toolkit.is_tool_available("normal")
        reason = toolkit.get_availability_reason("normal")
        assert not available
        assert "maintenance_mode" in reason

    def test_multiple_forbidden_states(self):
        """Test tool with multiple forbidden states."""
        toolkit = Toolkit()

        def restricted() -> Tuple[str, dict]:
            return "ok", {}

        toolkit.add_tool(
            "restricted",
            restricted,
            forbidden_states=["banned", "suspended", "locked"],
        )

        # Enable one forbidden state
        toolkit.set_unlocked_states({"banned"})

        # Should be unavailable
        assert not toolkit.is_tool_available("restricted")

    def test_required_and_forbidden_different_states(self):
        """Test tool with both required and forbidden states."""
        toolkit = Toolkit()

        def write_operation() -> Tuple[str, dict]:
            return "written", {}

        toolkit.add_tool(
            "write",
            write_operation,
            required_states=["auth"],
            forbidden_states=["readonly"],
        )

        # Only auth enabled - should be available
        toolkit.set_unlocked_states({"auth"})
        assert toolkit.is_tool_available("write")

        # Both auth and readonly enabled - should not be available
        toolkit.set_unlocked_states({"auth", "readonly"})
        assert not toolkit.is_tool_available("write")


class TestStateManipulation:
    """Test state manipulation methods."""

    def test_set_unlocked_states_replaces(self):
        """Test set_unlocked_states replaces current states."""
        toolkit = Toolkit()

        toolkit.set_unlocked_states({"state_a", "state_b"})
        assert toolkit.unlocked_states == {"state_a", "state_b"}

        # Replace with new set
        toolkit.set_unlocked_states({"state_c"})
        assert toolkit.unlocked_states == {"state_c"}
        # Old states should be gone
        assert "state_a" not in toolkit.unlocked_states

    def test_state_persistence_across_copies(self):
        """Test state preservation when copying toolkit."""
        toolkit1 = Toolkit()
        toolkit1.set_unlocked_states({"state_a", "state_b"})

        toolkit2 = toolkit1.copy()

        # States should be copied
        assert toolkit2.unlocked_states == {"state_a", "state_b"}

        # Modifying copy shouldn't affect original
        toolkit2.set_unlocked_states({"state_c"})
        assert toolkit1.unlocked_states == {"state_a", "state_b"}

    def test_empty_state_set(self):
        """Test setting empty state set."""
        toolkit = Toolkit()

        toolkit.set_unlocked_states({"state_a"})
        toolkit.set_unlocked_states(set())

        # Should have empty set
        assert toolkit.unlocked_states == set()


class TestStateBasedAvailability:
    """Test comprehensive state-based tool availability."""

    def test_complex_state_scenario(self):
        """Test complex scenario with multiple tools and states."""
        toolkit = Toolkit()

        def login() -> Tuple[str, dict]:
            return "ok", {}

        def verify_email() -> Tuple[str, dict]:
            return "ok", {}

        def premium_feature() -> Tuple[str, dict]:
            return "ok", {}

        def guest_only() -> Tuple[str, dict]:
            return "ok", {}

        toolkit.add_tool("login", login, enables_states=["logged_in"])
        toolkit.add_tool(
            "verify",
            verify_email,
            required_states=["logged_in"],
            enables_states=["verified"],
            disables_states=["guest"],
        )
        toolkit.add_tool(
            "premium",
            premium_feature,
            required_states=["logged_in", "verified"],
            forbidden_states=["suspended"],
        )
        toolkit.add_tool("guest", guest_only, required_states=["guest"])

        # Initial state
        toolkit.set_unlocked_states({"guest"})

        assert toolkit.is_tool_available("login")
        assert not toolkit.is_tool_available("verify")  # Needs logged_in
        assert not toolkit.is_tool_available("premium")  # Needs logged_in + verified
        assert toolkit.is_tool_available("guest")  # Guest mode active

        # Simulate login
        toolkit.set_unlocked_states({"logged_in", "guest"})

        assert toolkit.is_tool_available("verify")
        assert not toolkit.is_tool_available("premium")  # Still needs verified

        # Simulate verification (disables guest)
        toolkit.set_unlocked_states({"logged_in", "verified"})

        assert toolkit.is_tool_available("premium")
        assert not toolkit.is_tool_available("guest")  # Guest disabled
