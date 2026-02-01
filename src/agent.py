from typing import Optional, Any
import json
import ast
import re
from collections import deque
import traceback

from a2a.utils import new_agent_text_message, get_message_text
from a2a.server.tasks import TaskUpdater

class BaselinePurpleAgent:
    def __init__(self):
        self.reset()

    def reset(self):
        print("🟣 [Purple] Resetting state...")
        self.adj: dict[int, set[int]] = {}
        self.room_locked: list[int] = [-1] * 8
        self.room_haskey: list[int] = [-1] * 8
        self.room_exit: list[int] = [-1] * 8
        self.room_visited: list[int] = [0] * 8
        self.room_inspected: list[int] = [0] * 8

        self.current_room: int = 0
        self.obs_visited: set[int] = {0}
        self.obs_frontier: deque[int] = deque()
        self.obs_parent: dict[int, Optional[int]] = {0: None}

        self.exit_room: Optional[int] = None
        self.known_key_room: Optional[int] = None
        
        self.path_to_exit: list[int] = []
        self.path_index: int = 0

        self.has_committed: bool = False
        self.pending_getkey: bool = False
        self.pending_usekey: bool = False

    async def run(self, request: Any, updater: TaskUpdater) -> None:
        prompt = ""
        
        try:
            prompt = get_message_text(request)
        except Exception:
            try:
                if isinstance(request, str):
                    prompt = request
                elif isinstance(request, dict):
                    if "message" in request:
                        msg = request["message"]
                        if isinstance(msg, dict) and "parts" in msg:
                            prompt = "".join([p.get("text", "") for p in msg["parts"]])
                        else:
                            prompt = str(msg)
                    else:
                        prompt = str(request)
                else:
                    prompt = str(request)
            except Exception:
                prompt = str(request)

        if "(Move 0)" in prompt:
            self.reset()

        print(f"🟣 [Purple] Step Prompt ({len(prompt)} chars). Current Room: {self.current_room}")

        try:
            action = self.select_action(prompt)
            print(f"🟣 [Purple] Selected Action: {action}")
            action_json = json.dumps(action)
        except Exception as e:
            print(f"🟣 [Fatal Error] Crash in select_action: {e}")
            traceback.print_exc()
            action_json = json.dumps({"command": "INSPECT"})

        try:
            response_msg = new_agent_text_message(action_json)
            await updater.complete(message=response_msg)
        except Exception as e:
            print(f"🟣 [Error] Failed to send message via updater: {e}")
            traceback.print_exc()

    def _parse_list(self, prompt: str, label: str) -> Optional[list[int]]:
        pattern = rf"{label}:\s*(\[.*?\])"
        match = re.search(pattern, prompt, re.DOTALL)
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
        if inspected is not None:
            self.room_inspected = inspected

        locked = self._parse_list(prompt, "Locked")
        if locked is not None:
            self.room_locked = locked

        haskey = self._parse_list(prompt, "Has Key")
        if haskey is not None:
            self.room_haskey = haskey
            for i, val in enumerate(haskey):
                if val == 1:
                    self.known_key_room = i

        exit_list = self._parse_list(prompt, "Is Exit")
        if exit_list is not None:
            self.room_exit = exit_list
            for i, val in enumerate(exit_list):
                if val == 1:
                    self.exit_room = i

        phase_match = re.search(r"Phase:\s*(\w+)", prompt)
        if phase_match and "Execution" in phase_match.group(1):
            self.has_committed = True

    def _bfs_path(self, start: int, target: int) -> list[int]:
        """Simple BFS to find path from start to target."""
        if start == target:
            return [start]
        
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            
            for neighbor in self.adj.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    def _obs_action(self) -> dict:
        can_commit = False
        
        if self.exit_room is not None:
            self._plan_solution()
            path_is_locked = any(self.room_locked[r] == 1 for r in self.path_to_exit)
            
            if not path_is_locked:
                can_commit = True
            elif self.known_key_room is not None:
                can_commit = True
            else:
                can_commit = False
                if not self.obs_frontier:
                    can_commit = True

        if can_commit:
            return {"command": "COMMIT"}
        if not self.obs_frontier:
            self._plan_solution()
            return {"command": "COMMIT"}

        target = self.obs_frontier.popleft()
        return {"command": "MOVE", "target_room": int(target)}

    def _record_new_neighbors(self, prev_room: int, prev_visited: list[int]):
        """Update adjacency graph based on movement."""
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

    def _plan_solution(self):
        if self.exit_room is None:
            self.path_to_exit = []
            return

        path_direct = self._bfs_path(0, self.exit_room)
        if not path_direct:
            self.path_to_exit = []
            return

        locks_on_path = [r for r in path_direct if self.room_locked[r] == 1]
        
        key_needed = len(locks_on_path) > 0
        have_key_on_path = False
        
        if self.known_key_room is not None and self.known_key_room in path_direct:
            key_idx = path_direct.index(self.known_key_room)
            first_lock_idx = 999
            if locks_on_path:
                for r in locks_on_path:
                    idx = path_direct.index(r)
                    if idx < first_lock_idx:
                        first_lock_idx = idx
            
            if key_idx < first_lock_idx:
                have_key_on_path = True

        if key_needed and not have_key_on_path and self.known_key_room is not None:
            print(f"🟣 [Purple] Planning Detour for Key at {self.known_key_room}")
            path_to_key = self._bfs_path(0, self.known_key_room)
            path_from_key = self._bfs_path(self.known_key_room, self.exit_room)
            
            if path_to_key and path_from_key:
                self.path_to_exit = path_to_key + path_from_key[1:]
                return

        self.path_to_exit = path_direct

    def _exec_action(self) -> dict:
        if self.pending_getkey:
            self.pending_getkey = False
            return {"command": "GETKEY"}

        if self.pending_usekey:
            self.pending_usekey = False
            return {"command": "USEKEY"}
        
        if self.room_haskey[self.current_room] == 1:
            return {"command": "GETKEY"}

        if self.room_locked[self.current_room] == 1:
             return {"command": "USEKEY"}

        if self.path_index < len(self.path_to_exit):
            try:
                curr_idx_in_path = self.path_to_exit.index(self.current_room)
                if curr_idx_in_path + 1 < len(self.path_to_exit):
                    next_room = self.path_to_exit[curr_idx_in_path + 1]
                    return {"command": "MOVE", "target_room": int(next_room)}
            except ValueError:
                pass

            if self.path_index + 1 < len(self.path_to_exit):
                 next_room = self.path_to_exit[self.path_index + 1]
                 return {"command": "MOVE", "target_room": int(next_room)}
        if self.current_room == self.exit_room:
             if self.room_locked[self.current_room] == 1:
                 return {"command": "USEKEY"}
             return {"command": "INSPECT"} # Done

        return {"command": "INSPECT"}

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
                self._plan_solution()

            return self._exec_action()

    def format_action(self, action: dict) -> str:
        return json.dumps(action)