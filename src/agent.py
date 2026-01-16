from typing import Optional
import json
import random


class BaselinePurpleAgent:
    """Baseline purple agent that attempts to solve the Rooms puzzle.
    
    This is a simple random/heuristic baseline that:
    1. Explores randomly in observation phase
    2. Commits when it thinks it has enough info
    3. Tries to navigate to exit in execution phase
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset agent state for new episode."""
        self.observation_steps = 0
        self.has_committed = False
        self.known_exit = None
        self.current_keys = 0
    
    def parse_observation(self, prompt: str) -> dict:
        """Parse observation from green agent's prompt."""
        obs = {}
        
        # Extract key information from prompt
        lines = prompt.split('\n')
        for line in lines:
            if "Current Room:" in line:
                obs['current_room'] = int(line.split(':')[1].strip())
            elif "Phase:" in line:
                obs['phase'] = line.split(':')[1].strip()
                self.has_committed = "Execution" in obs['phase']
            elif "Keys Held:" in line:
                obs['keys_held'] = int(line.split(':')[1].strip())
                self.current_keys = obs['keys_held']
            elif "Rooms Visited:" in line:
                obs['rooms_visited'] = eval(line.split(':', 1)[1].strip())
            elif "Rooms Inspected:" in line:
                obs['rooms_inspected'] = eval(line.split(':', 1)[1].strip())
            elif "Is Exit:" in line:
                exit_list = eval(line.split(':', 1)[1].strip())
                # Find exit room
                for i, val in enumerate(exit_list):
                    if val == 1:
                        self.known_exit = i
                        obs['exit_room'] = i
        
        return obs
    
    def select_action(self, prompt: str) -> dict:
        """Select next action based on observation.
        
        Strategy:
        - Observation phase: Explore randomly, commit after a few moves
        - Execution phase: Navigate towards exit if known
        """
        obs = self.parse_observation(prompt)
        current_room = obs.get('current_room', 0)
        
        if not self.has_committed:
            # Observation phase: explore then commit
            self.observation_steps += 1
            
            if self.observation_steps > 3:  # Commit after 3 exploration steps
                return {"command": "COMMIT"}
            else:
                # Try to move to a random adjacent room
                target = random.randint(0, 7)
                if target != current_room:
                    return {"command": "MOVE", "target_room": target}
                return {"command": "COMMIT"}
        
        else:
            # Execution phase: try to reach exit
            if self.known_exit is not None:
                if current_room == self.known_exit:
                    # At exit, try to inspect then unlock if needed
                    return {"command": "INSPECT"}
                else:
                    # Try to move towards exit
                    return {"command": "MOVE", "target_room": self.known_exit}
            else:
                # Don't know exit, explore
                inspected = obs.get('rooms_inspected', [0]*8)
                if inspected[current_room] == 0:
                    return {"command": "INSPECT"}
                else:
                    # Move to adjacent room
                    target = (current_room + 1) % 8
                    return {"command": "MOVE", "target_room": target}
    
    def format_action(self, action: dict) -> str:
        """Format action as JSON string for response."""
        return json.dumps(action)