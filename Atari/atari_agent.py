from agent import Agent
from Atari.deepqn import DeepQN
import torch
import numpy as np
import torch.optim as optim


class AtariAgent(Agent):
    def __init__(self, env, args):
        """
        Initialize an Atari agent.

        Args:
            input_channels (int): Number of input channels (e.g., for stacked frames).
            n_actions (int): Number of possible actions in the Atari environment.
            args: Configuration arguments (precision, etc.).
        """
        # Use DeepQN model for Atari-specific architectures
        # the two agents have the same observation space
        self.input_channels=env.observation_space(env.agents[0]).shape[-1]
        self.n_actions=env.action_space(env.agents[0]).n 
        self.model = DeepQN(self.input_channels, self.n_actions, args.precision)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        super().__init__(self.model, self.optimizer, args)
        

    def clone(self, env, args):
        # Clone the model weights
        clone = AtariAgent(env, args)
        clone.model.load_state_dict(self.model.state_dict())
        return clone
