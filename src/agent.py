from typing import Optional
import json
import ast
import re
from collections import deque


class BaselinePurpleAgent:
    """
    Baseline purple agent for Neophytic Rooms that solves tutorial-difficulty puzzles
    and particular easy ones.

    Strategy:
        Observation phase: BFS explore from room 0, recording real adjacency by
        watching which rooms become visited after each MOVE. Stop exploring once
        the exit is found (or all reachable rooms have been visited). Then COMMIT.

        Execution phase: Replay the shortest path (already computed by BFS) from
        room 0 to the exit. Along the way, pick up any key found in a room that is
        locked and sits on the path (handles the trivial single-key case).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Wipe states, called once per run.
        self.adj: dict[int, set[int]] = {}
        self.room_locked: list[int] = [-1] * 8
        self.room_haskey: list[int] = [-1] * 8
        self.room_exit: list[int] = [-1] * 8
        self.room_visited: list[int] = [0] * 8

        self.current_room: int = 0
        self.obs_visited: set[int] = {0}
        self.obs_frontier: deque[int] = deque()
        self.obs_parent: dict[int, Optional[int]] = {0: None}

        self.exit_room: Optional[int] = None
        self.path_to_exit: list[int] = []
        self.path_index: int = 0

        self.has_committed: bool = False
        self.pending_getkey: bool = False
        self.pending_usekey: bool = False

    def _parse_list(self, prompt: str, label: str) -> Optional[list[int]]:
        """Extract a labeled list field like 'Rooms Visited: [1, 0, 0, ...]'."""
        pattern = rf"{label}:\s*(\[.*?\])"
        match = re.search(pattern, prompt)
        if match:
            try:
                return ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                return None
        return None

    def _parse_int(self, prompt: str, label: str) -> Optional[int]:
        pattern = rf"{label}:\s*(\d+)"
        match = re.search(pattern, prompt)
        return int(match.group(1)) if match else None

    def _sync_state(self, prompt: str):
        current = self._parse_int(prompt, "Current Room")
        if current is not None:
            self.current_room = current

        visited = self._parse_list(prompt, "Rooms Visited")
        if visited is not None:
            self.room_visited = visited

        inspected = self._parse_list(prompt, "Rooms Inspected")

        locked = self._parse_list(prompt, "Locked")
        if locked is not None:
            self.room_locked = locked

        haskey = self._parse_list(prompt, "Has Key")
        if haskey is not None:
            self.room_haskey = haskey

        exit_list = self._parse_list(prompt, "Is Exit")
        if exit_list is not None:
            self.room_exit = exit_list
            for i, val in enumerate(exit_list):
                if val == 1:
                    self.exit_room = i

        phase_match = re.search(r"Phase:\s*(\w+)", prompt)
        if phase_match and "Execution" in phase_match.group(1):
            self.has_committed = True

    def _obs_action(self) -> dict:
        if self.exit_room is not None:
            self._compute_path()
            return {"command": "COMMIT"}

        if not self.obs_frontier:
            self._compute_path()
            return {"command": "COMMIT"}

        target = self.obs_frontier.popleft()
        return {"command": "MOVE", "target_room": target}

    def _record_new_neighbors(self, prev_room: int, prev_visited: list[int]):
        """After a MOVE in obs phase, figure out which rooms are newly visited.

        In observation phase, MOVE auto-inspects.  The only way a room flips from
        unvisited to visited is if we successfully moved into it.  So the set
        difference between old and new room_visited tells us exactly which room
        we landed in.  That room is a neighbor of prev_room.
        """
        for i in range(8):
            if self.room_visited[i] == 1 and prev_visited[i] == 0:
                self.adj.setdefault(prev_room, set()).add(i)
                self.adj.setdefault(i, set()).add(prev_room)

                if i not in self.obs_visited:
                    self.obs_visited.add(i)
                    self.obs_parent[i] = prev_room
                    for candidate in range(8):
                        if candidate not in self.obs_visited:
                            self.obs_frontier.append(candidate)


    def _compute_path(self):
        if self.exit_room is None:
            self.path_to_exit = []
            return

        visited = {0}
        queue: deque[tuple[int, list[int]]] = deque([(0, [])])

        while queue:
            node, path = queue.popleft()
            if node == self.exit_room:
                self.path_to_exit = path
                return
            for neighbor in self.adj.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        self.path_to_exit = []


    def _exec_action(self) -> dict:
        if self.pending_getkey:
            self.pending_getkey = False
            return {"command": "GETKEY"}

        if self.pending_usekey:
            self.pending_usekey = False
            return {"command": "USEKEY"}

        if self.current_room == self.exit_room:
            if self.room_locked[self.exit_room] == 1:
                keys_held = self._current_keys_from_state()
                if keys_held > 0:
                    return {"command": "USEKEY"}
            for neighbor in self.adj.get(self.exit_room, set()):
                return {"command": "MOVE", "target_room": neighbor}
            return {"command": "MOVE", "target_room": self.exit_room}

        if self.path_index < len(self.path_to_exit):
            next_room = self.path_to_exit[self.path_index]
            self.path_index += 1

            if self.room_haskey[self.current_room] == 1:
                self.pending_getkey = False
                self.path_index -= 1 
                return {"command": "GETKEY"}

            return {"command": "MOVE", "target_room": next_room}

        return {"command": "INSPECT"}

    def _current_keys_from_state(self) -> int:
        return 0

    def select_action(self, prompt: str) -> dict:
        prev_visited = list(self.room_visited)
        prev_room = self.current_room

        self._sync_state(prompt)

        if not self.has_committed:
            if self.current_room != prev_room or any(
                self.room_visited[i] != prev_visited[i] for i in range(8)
            ):
                self._record_new_neighbors(prev_room, prev_visited)

            if not self.obs_frontier and len(self.obs_visited) == 1:
                for candidate in range(8):
                    if candidate != 0:
                        self.obs_frontier.append(candidate)

            return self._obs_action()

        else:
            if not self.path_to_exit and self.exit_room is not None:
                self._compute_path()

            if self.room_locked[self.current_room] == 1:
                keys = self._current_keys_from_state()
                if keys > 0:
                    return {"command": "USEKEY"}

            return self._exec_action()

    def format_action(self, action: dict) -> str:
        return json.dumps(action)