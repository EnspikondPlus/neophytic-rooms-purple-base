"""Unit tests for BaselinePurpleAgent.

All tests exercise select_action() directly.  No A2A server is needed — the purple
agent is a plain stateful class; the A2A transport layer is the *green* agent's
responsibility.

Graph notation used in comments:
    0 ↔ 1 ↔ 2   means rooms 0,1,2 exist; 0-1 and 1-2 are adjacent.
    [exit]       marks the exit room.
"""

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unknown():
    """8-element list of unknowns — the default for unvisited rooms."""
    return [-1] * 8


def _zeros():
    return [0] * 8


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Verify _sync_state correctly extracts every field from a prompt."""

    def test_parses_current_room(self, agent, make_prompt):
        prompt = make_prompt(
            current_room=3, phase="Observation", keys_held=0,
            steps_remaining=30,
            room_visited=[1, 1, 1, 1, 0, 0, 0, 0],
            room_inspected=[1, 1, 1, 1, 0, 0, 0, 0],
            room_locked=_unknown(), room_haskey=_unknown(), room_exit=_unknown(),
        )
        agent._sync_state(prompt)
        assert agent.current_room == 3

    def test_parses_phase_observation(self, agent, make_prompt):
        prompt = make_prompt(
            current_room=0, phase="Observation", keys_held=0,
            steps_remaining=30,
            room_visited=[1, 0, 0, 0, 0, 0, 0, 0],
            room_inspected=[1, 0, 0, 0, 0, 0, 0, 0],
            room_locked=_unknown(), room_haskey=_unknown(), room_exit=_unknown(),
        )
        agent._sync_state(prompt)
        assert agent.has_committed is False

    def test_parses_phase_execution(self, agent, make_prompt):
        prompt = make_prompt(
            current_room=0, phase="Execution", keys_held=0,
            steps_remaining=30,
            room_visited=[1, 1, 0, 0, 0, 0, 0, 0],
            room_inspected=[1, 1, 0, 0, 0, 0, 0, 0],
            room_locked=_unknown(), room_haskey=_unknown(), room_exit=_unknown(),
        )
        agent._sync_state(prompt)
        assert agent.has_committed is True

    def test_parses_exit_room(self, agent, make_prompt):
        exits = [0, 0, 1, 0, 0, 0, 0, 0]
        prompt = make_prompt(
            current_room=2, phase="Observation", keys_held=0,
            steps_remaining=30,
            room_visited=[1, 1, 1, 0, 0, 0, 0, 0],
            room_inspected=[1, 1, 1, 0, 0, 0, 0, 0],
            room_locked=[0, 0, 0, -1, -1, -1, -1, -1],
            room_haskey=[0, 0, 0, -1, -1, -1, -1, -1],
            room_exit=exits,
        )
        agent._sync_state(prompt)
        assert agent.exit_room == 2

    def test_parses_locked_and_haskey(self, agent, make_prompt):
        locked  = [0, 1, -1, -1, -1, -1, -1, -1]
        haskey  = [1, 0, -1, -1, -1, -1, -1, -1]
        prompt = make_prompt(
            current_room=0, phase="Observation", keys_held=0,
            steps_remaining=30,
            room_visited=[1, 1, 0, 0, 0, 0, 0, 0],
            room_inspected=[1, 1, 0, 0, 0, 0, 0, 0],
            room_locked=locked, room_haskey=haskey, room_exit=_unknown(),
        )
        agent._sync_state(prompt)
        assert agent.room_locked == locked
        assert agent.room_haskey == haskey


# ---------------------------------------------------------------------------
# Observation phase
# ---------------------------------------------------------------------------


class TestObservationPhase:
    """Verify exploration behaviour in the observation phase."""

    def test_first_action_is_move(self, agent, make_prompt):
        """On the very first prompt the agent should try to MOVE somewhere."""
        prompt = make_prompt(
            current_room=0, phase="Observation", keys_held=0,
            steps_remaining=30,
            room_visited=[1, 0, 0, 0, 0, 0, 0, 0],
            room_inspected=[1, 0, 0, 0, 0, 0, 0, 0],
            room_locked=_unknown(), room_haskey=_unknown(), room_exit=_unknown(),
        )
        action = agent.select_action(prompt)
        assert action["command"] == "MOVE"
        assert "target_room" in action
        assert action["target_room"] != 0  # shouldn't try to move to itself

    def test_commits_immediately_when_exit_found(self, agent, make_prompt):
        """If the exit is visible, the agent should COMMIT without further exploring."""
        # Simulate: agent already moved to room 1, which revealed exit at room 1.
        # Feed two prompts: first to establish room 0 state, then room 1 with exit.
        p0 = make_prompt(
            current_room=0, phase="Observation", keys_held=0,
            steps_remaining=30, move=0,
            room_visited=[1, 0, 0, 0, 0, 0, 0, 0],
            room_inspected=[1, 0, 0, 0, 0, 0, 0, 0],
            room_locked=_unknown(), room_haskey=_unknown(), room_exit=_unknown(),
        )
        agent.select_action(p0)  # seeds frontier, returns MOVE

        p1 = make_prompt(
            current_room=1, phase="Observation", keys_held=0,
            steps_remaining=30, move=1,
            room_visited=[1, 1, 0, 0, 0, 0, 0, 0],
            room_inspected=[1, 1, 0, 0, 0, 0, 0, 0],
            room_locked=[0, 0, -1, -1, -1, -1, -1, -1],
            room_haskey=[0, 0, -1, -1, -1, -1, -1, -1],
            room_exit=[0, 1, -1, -1, -1, -1, -1, -1],  # exit at room 1
        )
        action = agent.select_action(p1)
        assert action["command"] == "COMMIT"

    def test_commits_when_frontier_exhausted(self, agent, make_prompt):
        """If all reachable rooms are explored and no exit found, still COMMIT
        (agent will fail in execution, but must not loop forever)."""
        # 2-room graph, no exit (degenerate but possible in test).
        p0 = make_prompt(
            current_room=0, phase="Observation", keys_held=0,
            steps_remaining=30, move=0,
            room_visited=[1, 0, 0, 0, 0, 0, 0, 0],
            room_inspected=[1, 0, 0, 0, 0, 0, 0, 0],
            room_locked=_unknown(), room_haskey=_unknown(), room_exit=_unknown(),
        )
        agent.select_action(p0)  # MOVE

        # Room 1 visited, no exit anywhere known.  Drain the frontier by
        # repeatedly feeding "no new rooms" until agent commits.
        p1 = make_prompt(
            current_room=1, phase="Observation", keys_held=0,
            steps_remaining=30, move=1,
            room_visited=[1, 1, 0, 0, 0, 0, 0, 0],
            room_inspected=[1, 1, 0, 0, 0, 0, 0, 0],
            room_locked=[0, 0, -1, -1, -1, -1, -1, -1],
            room_haskey=[0, 0, -1, -1, -1, -1, -1, -1],
            room_exit=[0, 0, -1, -1, -1, -1, -1, -1],
        )
        # Keep feeding the same state (failed moves don't change visited).
        # The agent will keep popping candidates from the frontier and issuing
        # MOVEs that "fail" (visited doesn't change).  Eventually the frontier
        # empties and it COMMITs.  Cap at 20 iterations to avoid infinite loop in
        # case of a bug.
        committed = False
        for _ in range(20):
            action = agent.select_action(p1)
            if action["command"] == "COMMIT":
                committed = True
                break
        assert committed, "Agent never committed after frontier exhausted"


# ---------------------------------------------------------------------------
# Execution phase
# ---------------------------------------------------------------------------


class TestExecutionPhase:
    """Verify path-replay behaviour after COMMIT."""

    def _run_observation_phase(self, agent, make_prompt):
        """Drive the agent through a tutorial observation phase for graph 0↔1↔2[exit].
        Returns the agent with adj and path fully populated, phase still Observation
        (caller sends the Execution prompt next)."""
        # Move 0: in room 0, nothing known yet.
        agent.select_action(make_prompt(
            current_room=0, phase="Observation", keys_held=0, steps_remaining=30, move=0,
            room_visited   =[1, 0, 0, 0, 0, 0, 0, 0],
            room_inspected =[1, 0, 0, 0, 0, 0, 0, 0],
            room_locked    =_unknown(), room_haskey=_unknown(), room_exit=_unknown(),
        ))
        # Move 1: successfully moved to room 1.
        agent.select_action(make_prompt(
            current_room=1, phase="Observation", keys_held=0, steps_remaining=30, move=1,
            room_visited   =[1, 1, 0, 0, 0, 0, 0, 0],
            room_inspected =[1, 1, 0, 0, 0, 0, 0, 0],
            room_locked    =[0, 0, -1, -1, -1, -1, -1, -1],
            room_haskey    =[0, 0, -1, -1, -1, -1, -1, -1],
            room_exit      =[0, 0, -1, -1, -1, -1, -1, -1],
        ))
        # Move 2: successfully moved to room 2, which is the exit → COMMIT returned.
        agent.select_action(make_prompt(
            current_room=2, phase="Observation", keys_held=0, steps_remaining=30, move=2,
            room_visited   =[1, 1, 1, 0, 0, 0, 0, 0],
            room_inspected =[1, 1, 1, 0, 0, 0, 0, 0],
            room_locked    =[0, 0, 0, -1, -1, -1, -1, -1],
            room_haskey    =[0, 0, 0, -1, -1, -1, -1, -1],
            room_exit      =[0, 0, 1, -1, -1, -1, -1, -1],
        ))

    def test_execution_walks_path_to_exit(self, agent, make_prompt):
        """After COMMIT, the agent replays the shortest path: 0 → 1 → 2."""
        self._run_observation_phase(agent, make_prompt)

        # Green agent resets position to 0 on COMMIT.
        exec_base = dict(
            phase="Execution", keys_held=0, steps_remaining=30,
            room_visited   =[1, 1, 1, 0, 0, 0, 0, 0],
            room_inspected =[1, 1, 1, 0, 0, 0, 0, 0],
            room_locked    =[0, 0, 0, -1, -1, -1, -1, -1],
            room_haskey    =[0, 0, 0, -1, -1, -1, -1, -1],
            room_exit      =[0, 0, 1, -1, -1, -1, -1, -1],
        )

        # First exec action: move to room 1.
        a1 = agent.select_action(make_prompt(current_room=0, move=3, **exec_base))
        assert a1 == {"command": "MOVE", "target_room": 1}

        # Second exec action: move to room 2 (exit).
        a2 = agent.select_action(make_prompt(current_room=1, move=4, **exec_base))
        assert a2 == {"command": "MOVE", "target_room": 2}

    def test_execution_inspects_when_path_exhausted(self, agent, make_prompt):
        """If path runs out before reaching exit (shouldn't happen normally),
        the fallback is INSPECT — not a crash."""
        # Manually set up state as if path was empty but we're not at exit.
        agent.has_committed = True
        agent.exit_room = 5
        agent.path_to_exit = []  # empty — simulates the edge case
        agent.path_index = 0
        agent.current_room = 0
        agent.adj = {0: {1}, 1: {0}}

        prompt = make_prompt(
            current_room=0, phase="Execution", keys_held=0, steps_remaining=30,
            room_visited=[1, 1, 0, 0, 0, 0, 0, 0],
            room_inspected=[1, 0, 0, 0, 0, 0, 0, 0],
            room_locked=_zeros(), room_haskey=_zeros(),
            room_exit=[0, 0, 0, 0, 0, 1, 0, 0],
        )
        action = agent.select_action(prompt)
        assert action["command"] == "INSPECT"


# ---------------------------------------------------------------------------
# Full tutorial solve (end-to-end)
# ---------------------------------------------------------------------------


class TestTutorialSolve:
    """Drive the agent through a complete tutorial scenario and verify it produces
    the correct action sequence from start to finish.

    Graph: 0 ↔ 1 ↔ 2[exit].  No locks, no keys.
    Expected action sequence:
        Obs  Move 0 → MOVE 1
        Obs  Move 1 → MOVE 2          (exit discovered)
        Obs  Move 2 → COMMIT
        Exec Move 3 → MOVE 1          (path replay step 1)
        Exec Move 4 → MOVE 2          (path replay step 2 — at exit, done)
    """

    def test_full_tutorial_action_sequence(self, agent, make_prompt):
        actions = []

        # --- Observation phase ---
        actions.append(agent.select_action(make_prompt(
            current_room=0, phase="Observation", keys_held=0, steps_remaining=30, move=0,
            room_visited   =[1, 0, 0, 0, 0, 0, 0, 0],
            room_inspected =[1, 0, 0, 0, 0, 0, 0, 0],
            room_locked    =_unknown(), room_haskey=_unknown(), room_exit=_unknown(),
        )))

        actions.append(agent.select_action(make_prompt(
            current_room=1, phase="Observation", keys_held=0, steps_remaining=30, move=1,
            room_visited   =[1, 1, 0, 0, 0, 0, 0, 0],
            room_inspected =[1, 1, 0, 0, 0, 0, 0, 0],
            room_locked    =[0, 0, -1, -1, -1, -1, -1, -1],
            room_haskey    =[0, 0, -1, -1, -1, -1, -1, -1],
            room_exit      =[0, 0, -1, -1, -1, -1, -1, -1],
        )))

        actions.append(agent.select_action(make_prompt(
            current_room=2, phase="Observation", keys_held=0, steps_remaining=30, move=2,
            room_visited   =[1, 1, 1, 0, 0, 0, 0, 0],
            room_inspected =[1, 1, 1, 0, 0, 0, 0, 0],
            room_locked    =[0, 0, 0, -1, -1, -1, -1, -1],
            room_haskey    =[0, 0, 0, -1, -1, -1, -1, -1],
            room_exit      =[0, 0, 1, -1, -1, -1, -1, -1],
        )))

        # --- Execution phase (green agent resets us to room 0) ---
        exec_base = dict(
            phase="Execution", keys_held=0, steps_remaining=30,
            room_visited   =[1, 1, 1, 0, 0, 0, 0, 0],
            room_inspected =[1, 1, 1, 0, 0, 0, 0, 0],
            room_locked    =[0, 0, 0, -1, -1, -1, -1, -1],
            room_haskey    =[0, 0, 0, -1, -1, -1, -1, -1],
            room_exit      =[0, 0, 1, -1, -1, -1, -1, -1],
        )

        actions.append(agent.select_action(make_prompt(current_room=0, move=3, **exec_base)))
        actions.append(agent.select_action(make_prompt(current_room=1, move=4, **exec_base)))

        # Verify the full sequence.
        assert actions == [
            {"command": "MOVE", "target_room": 1},
            {"command": "MOVE", "target_room": 2},
            {"command": "COMMIT"},
            {"command": "MOVE", "target_room": 1},
            {"command": "MOVE", "target_room": 2},
        ]


# ---------------------------------------------------------------------------
# Multi-run reset
# ---------------------------------------------------------------------------


class TestMultiRun:
    """Verify that reset() fully wipes state so consecutive runs don't bleed."""

    def test_reset_clears_all_state(self, agent):
        # Pollute every piece of state.
        agent.adj = {0: {1}, 1: {0, 2}, 2: {1}}
        agent.exit_room = 2
        agent.has_committed = True
        agent.path_to_exit = [1, 2]
        agent.path_index = 2
        agent.obs_visited = {0, 1, 2}
        agent.room_locked = [0, 0, 0, 0, 0, 0, 0, 0]
        agent.room_haskey = [1, 0, 0, 0, 0, 0, 0, 0]
        agent.room_exit   = [0, 0, 1, 0, 0, 0, 0, 0]
        agent.pending_getkey = True
        agent.pending_usekey = True

        agent.reset()

        assert agent.adj == {}
        assert agent.exit_room is None
        assert agent.has_committed is False
        assert agent.path_to_exit == []
        assert agent.path_index == 0
        assert agent.obs_visited == {0}
        assert agent.current_room == 0
        assert agent.room_locked == [-1] * 8
        assert agent.room_haskey == [-1] * 8
        assert agent.room_exit  == [-1] * 8
        assert agent.pending_getkey is False
        assert agent.pending_usekey is False

    def test_second_run_solves_independently(self, agent, make_prompt):
        """Run a full tutorial solve, reset, then run again.  Second run must
        produce the same correct sequence — no leftover state from the first."""

        def run_tutorial():
            obs_prompts = [
                make_prompt(
                    current_room=0, phase="Observation", keys_held=0, steps_remaining=30, move=0,
                    room_visited   =[1, 0, 0, 0, 0, 0, 0, 0],
                    room_inspected =[1, 0, 0, 0, 0, 0, 0, 0],
                    room_locked    =_unknown(), room_haskey=_unknown(), room_exit=_unknown(),
                ),
                make_prompt(
                    current_room=1, phase="Observation", keys_held=0, steps_remaining=30, move=1,
                    room_visited   =[1, 1, 0, 0, 0, 0, 0, 0],
                    room_inspected =[1, 1, 0, 0, 0, 0, 0, 0],
                    room_locked    =[0, 0, -1, -1, -1, -1, -1, -1],
                    room_haskey    =[0, 0, -1, -1, -1, -1, -1, -1],
                    room_exit      =[0, 0, -1, -1, -1, -1, -1, -1],
                ),
                make_prompt(
                    current_room=2, phase="Observation", keys_held=0, steps_remaining=30, move=2,
                    room_visited   =[1, 1, 1, 0, 0, 0, 0, 0],
                    room_inspected =[1, 1, 1, 0, 0, 0, 0, 0],
                    room_locked    =[0, 0, 0, -1, -1, -1, -1, -1],
                    room_haskey    =[0, 0, 0, -1, -1, -1, -1, -1],
                    room_exit      =[0, 0, 1, -1, -1, -1, -1, -1],
                ),
            ]
            exec_base = dict(
                phase="Execution", keys_held=0, steps_remaining=30,
                room_visited   =[1, 1, 1, 0, 0, 0, 0, 0],
                room_inspected =[1, 1, 1, 0, 0, 0, 0, 0],
                room_locked    =[0, 0, 0, -1, -1, -1, -1, -1],
                room_haskey    =[0, 0, 0, -1, -1, -1, -1, -1],
                room_exit      =[0, 0, 1, -1, -1, -1, -1, -1],
            )
            actions = [agent.select_action(p) for p in obs_prompts]
            actions.append(agent.select_action(make_prompt(current_room=0, move=3, **exec_base)))
            actions.append(agent.select_action(make_prompt(current_room=1, move=4, **exec_base)))
            return actions

        expected = [
            {"command": "MOVE", "target_room": 1},
            {"command": "MOVE", "target_room": 2},
            {"command": "COMMIT"},
            {"command": "MOVE", "target_room": 1},
            {"command": "MOVE", "target_room": 2},
        ]

        assert run_tutorial() == expected

        agent.reset()  # ← this is what the green agent does between runs

        assert run_tutorial() == expected


# ---------------------------------------------------------------------------
# Key handling (easy-difficulty edge case)
# ---------------------------------------------------------------------------


class TestKeyHandling:
    """The agent picks up a key if it finds one on its path and uses it if the
    next room is locked.  This only covers the trivial single-key-on-path case."""

    def test_getkey_when_key_in_current_room(self, agent, make_prompt):
        """In execution phase, if the current room has a key, agent should GETKEY
        before continuing along the path."""
        # Set up state directly: committed, path = [1, 2], currently at room 0,
        # room 0 has a key.
        agent.has_committed = True
        agent.exit_room = 2
        agent.path_to_exit = [1, 2]
        agent.path_index = 0
        agent.adj = {0: {1}, 1: {0, 2}, 2: {1}}
        agent.current_room = 0

        prompt = make_prompt(
            current_room=0, phase="Execution", keys_held=0, steps_remaining=30,
            room_visited   =[1, 1, 1, 0, 0, 0, 0, 0],
            room_inspected =[1, 1, 1, 0, 0, 0, 0, 0],
            room_locked    =[0, 1, 0, -1, -1, -1, -1, -1],   # room 1 locked
            room_haskey    =[1, 0, 0, -1, -1, -1, -1, -1],   # key in room 0
            room_exit      =[0, 0, 1, -1, -1, -1, -1, -1],
        )
        action = agent.select_action(prompt)
        assert action == {"command": "GETKEY"}