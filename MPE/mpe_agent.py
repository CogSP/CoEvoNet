from agent import Agent
from MPE.fcnetwork import FCNetwork
import torch
import torch.optim as optim
import numpy as np

class MPEAgent(Agent):

    def __init__(self, env, args):
        """
        Initialize an MPEAgent for a specific agent in the MPE environment.
        """
        # we will train only Good Agents, so we specify their obs and act space
        # agent_0 and agent_1 are equal, while adversart_0 has a different state and action space
        self.input_channels=env.observation_space("agent_0").shape[-1]
        self.n_actions=env.action_space("agent_0").n 
        self.model = FCNetwork(self.input_channels, self.n_actions, args.precision)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        super().__init__(self.model, self.optimizer, args)
        

    def clone(self, env, args):
        # Clone the model weights
        clone = MPEAgent(env, args)
        clone.model.load_state_dict(self.model.state_dict())
        return clone
