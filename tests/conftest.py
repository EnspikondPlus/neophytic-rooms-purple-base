import pytest
from src.agent import BaselinePurpleAgent


@pytest.fixture
def agent():
    """Fresh purple agent instance per test."""
    return BaselinePurpleAgent()


@pytest.fixture
def make_prompt():
    """Factory fixture that builds a prompt string matching the green agent's
    exact template layout.  All fields are required so tests can't accidentally
    omit one and silently get a broken prompt."""

    def _make(
        *,
        current_room: int,
        phase: str,                          # "Observation" or "Execution"
        keys_held: int,
        steps_remaining: int,
        room_visited: list[int],             # 8-element list
        room_inspected: list[int],           # 8-element list
        room_locked: list[int],              # 8-element list, -1 = unknown
        room_haskey: list[int],              # 8-element list, -1 = unknown
        room_exit: list[int],                # 8-element list, -1 = unknown
        move: int = 0,
    ) -> str:
        return (
            f"You are solving a Rooms navigation puzzle. Current state (Move {move}):\n"
            f"\n"
            f"Current Room: {current_room}\n"
            f"Phase: {phase}\n"
            f"Keys Held: {keys_held}\n"
            f"Steps Remaining: {steps_remaining}\n"
            f"\n"
            f"Room Status (1 means true and 0 means false):\n"
            f"Rooms Visited: {room_visited}\n"
            f"Rooms Inspected: {room_inspected}\n"
            f"\n"
            f"Room Properties (1 means true and 0 means false for inspected rooms; -1 means unknown):\n"
            f"- Locked: {room_locked}\n"
            f"- Has Key: {room_haskey}\n"
            f"- Is Exit: {room_exit}\n"
        )

    return _make