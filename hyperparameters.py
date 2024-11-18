import numpy as np

# Define hyperparameters
class Hyperparameters:
    def __init__(self, algo="GA"):
        if algo == "GA":
            self.population_size = 200
            self.noise_std = 0.005
            self.hof_evaluations = 3
        elif algo == "ES":
            self.population_size = 200
            self.noise_std = 0.05
            self.hof_evaluations = 1
        else:
            raise ValueError("Algorithm must be 'GA' or 'ES'.")
