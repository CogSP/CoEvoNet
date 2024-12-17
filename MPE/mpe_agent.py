from agent import Agent
from MPE.fcnetwork import FCNetwork
import torch
import torch.optim as optim
import numpy as np



class MPEAgent(Agent):

    def __init__(self, env, args, role):
        """
        Initialize an MPEAgent for a specific agent in the MPE environment.
        """
        # we will train only Good Agents, so we specify their obs and act space
        # agent_0 and agent_1 are equal, while adversart_0 has a different state and action space
        self.input_channels=env.observation_space(role).shape[-1]
        self.n_actions=env.action_space(role).n 
        self.model = FCNetwork(self.input_channels, self.n_actions, args.precision)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        super().__init__(self.model, self.optimizer, args)
        

    def clone(self, env, args, role):
        # Clone the model weights
        clone = MPEAgent(env, args, role)
        clone.model.load_state_dict(self.model.state_dict())
        return clone

    def log_weight_statistics(self, step, weights_logging_agent_0=None, weights_logging_agent_1=None, weights_logging_adversary=None, role=None):
        """
        Logs summary statistics of the perturbable weights.
        
        Args:
            step (int, optional): The current step or mutation number for reference.
        """
        weights = self.model.get_perturbable_weights()
        mean = weights.mean()
        std = weights.std()
        min_val = weights.min()
        max_val = weights.max()
        
        weights = {"step": step, "mean": mean, "min": min_val, "max": max_val, "std": std}

        if role == "agent_0":
            weights_logging_agent_0.append(weights)
        if role == "agent_1":
            weights_logging_agent_1.append(weights)
        if role == "adversary_0":
            weights_logging_adversary.append(weights)
        