import numpy as np
from gymnasium import spaces

class MultiAgentEnvironment:
    def __init__(self, grid_size=5):
        """
        A PettingZoo-like multi-agent environment where two agents compete to capture points.
        """
        self.grid_size = grid_size
        self.agents = ["agent_1", "agent_2"]
        self.action_space = spaces.Discrete(5)  # 5 actions: Up, Down, Left, Right, Stay
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(6,), dtype=np.int32)  # [agent_pos, opponent_pos, point_pos]
        self.reset()

    def reset(self):
        """
        Resets the environment to its initial state and returns observations for both agents.
        """
        # Randomly place agents and the point
        self.agent_positions = {
            "agent_1": np.random.randint(0, self.grid_size, size=2),
            "agent_2": np.random.randint(0, self.grid_size, size=2),
        }
        self.point_position = np.random.randint(0, self.grid_size, size=2)
        self.done = False
        self.current_agent_index = 0
        return self._get_observation()

    def _get_observation(self):
        """
        Returns the observation for the current agent.
        """
        current_agent = self.agents[self.current_agent_index]
        opponent_agent = self.agents[1 - self.current_agent_index]
        return np.concatenate([
            self.agent_positions[current_agent],  # Current agent position
            self.agent_positions[opponent_agent],  # Opponent position
            self.point_position,  # Point position
        ])

    def step(self, action):
        """
        Executes the action for the current agent and returns:
        - Next observation
        - Reward
        - Done flag
        - Info dict
        """
        directions = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]),   # Down
            2: np.array([0, -1]),  # Left
            3: np.array([0, 1]),   # Right
            4: np.array([0, 0]),   # Stay
        }

        # Get current agent
        current_agent = self.agents[self.current_agent_index]
        move = directions.get(action, np.array([0, 0]))
        self.agent_positions[current_agent] = np.clip(
            self.agent_positions[current_agent] + move, 0, self.grid_size - 1
        )

        # Check if the current agent captures the point
        reward = 0
        if np.array_equal(self.agent_positions[current_agent], self.point_position):
            reward = 1
            self.done = True

        # Switch to the next agent
        self.current_agent_index = 1 - self.current_agent_index
        next_observation = self._get_observation()
        info = {}

        return next_observation, reward, self.done, info

    def render(self):
        """
        Renders the current state of the environment as a grid.
        """
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)

        # Place agents
        grid[tuple(self.agent_positions["agent_1"])] = '1'
        grid[tuple(self.agent_positions["agent_2"])] = '2'

        # Place point
        grid[tuple(self.point_position)] = 'P'

        # Print the grid
        print("\n".join(["".join(row) for row in grid]))
        print()
