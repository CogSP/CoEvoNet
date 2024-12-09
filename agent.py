from deepqn import DeepQN
import torch
import torch.optim as optim
import numpy as np


# Agent class to hold the neural network and training logic
class Agent:
    def __init__(self, input_channels, n_actions, precision):
        self.model = DeepQN(input_channels, n_actions, precision)
        self.precision = precision
        self.apply_precision
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
    
        
    def apply_precision(model, precision):
        """Apply the specified precision to the model and default tensor type."""
        if self.precision == "float16":
                    self.model = self.model.half()
        elif self.precision == "float32":
            self.model = self.model.float()
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")


    def clone(self, args):
        # Clone the model weights
        clone = Agent(input_channels=self.model.conv1.in_channels, 
                      n_actions=self.model.output.out_features, precision=args.precision)
        clone.model.load_state_dict(self.model.state_dict())
        return clone
    
    def mutate(self, noise_std):
        # Add Gaussian noise to the model weights (for ES or GA mutation)
        for param in self.model.parameters():
            noise = torch.normal(0, noise_std, size=param.size()).to(param.device)
            param.data += noise

    def mutate_ES(self, args):
        """
        Mutates the weights of the model using a normal distribution.

        Args:
            args: Arguments containing the mutation power (scale of the mutation).

        Returns:
            dict: A dictionary of perturbations applied to the weights.
        """
        weights = self.model.get_perturbable_weights()
        noise = np.random.normal(loc=0.0, scale=args.mutation_power, size=len(weights))
        self.model.set_perturbable_weights(weights + noise, args)
        

        """for key, value in weights.items():
            # Generate noise with the same shape as the weight tensor
            noise = np.random.normal(loc=0.0, scale=args.mutation_power, size=value.shape).astype(args.precision)
            perturbations[key] = torch.tensor(noise, dtype=value.dtype, device=value.device)
            # Add the noise to the weights
            weights[key] += perturbations[key]

        # Set the mutated weights back to the model
        #print(f"weights = {weights.keys()}")
        self.model.set_perturbable_weights(weights)"""

        return noise


    def set_weights(self, weights):
        # Set specific weights for the model (useful for ES)
        self.model.load_state_dict(weights)
    
    def get_weights(self):
        # Return model weights (useful for ES)
        return self.model.state_dict()
