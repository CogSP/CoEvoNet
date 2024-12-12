from random import randint
import numpy as np

class RandomPolicy(object):
    """ Policy that samples a random move from the number of actions that
    are available."""

    def __init__(self, number_actions):
        self.number_actions = number_actions

    def determine_action(self, input):
        return randint(0, self.number_actions - 1)



# used during debug
class PeriodicPolicy(object):
    """ Policy that iterates over an array of all the possible moves"""
    
    def __init__(self, number_actions):
        self.number_actions = number_actions
        self.actions = np.arange(self.number_actions)
        self.counter = 0
        
    def determine_action(self, input):
        chosen_action = self.actions[self.counter]
        self.counter = (self.counter + 1) % self.number_actions 
        return chosen_action


# used during debug (NOTE: has a meaning only for Pong)
class AlwaysFirePolicy(object):
    """ Policy that always shoot """
    
    def __init__(self, number_actions):
        return
        
    def determine_action(self, input):
        return 1
