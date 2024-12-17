import torch
import torch.optim as optim
import numpy as np



class Agent:

    def __init__(self, model, optimizer, args):

        self.model = model
        self.optimizer = optimizer 
        self.precision = args.precision
        

    def apply_precision(self, model, precision):
        """Apply the specified precision to the model and default tensor type."""
        if self.precision == "float16":
            self.model = self.model.half()
        elif self.precision == "float32":
            self.model = self.model.float()
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")

    def mutate(self, noise_std):
        # Add Gaussian noise to the model weights (for ES or GA mutation)
        for param in self.model.parameters():
            noise = torch.normal(0, noise_std, size=param.size()).to(param.device)
            param.data += noise

    def mutate_ES(self, args, role):
        """
        Mutates the weights of the model using a normal distribution.

        Args:
            args: Arguments containing the mutation power (scale of the mutation).

        Returns:
            dict: A dictionary of perturbations applied to the weights.
        """

        if role == "agent_0":
            mutation_power = args.mutation_power_agent_0

        if role == "agent_1":
            mutation_power = args.mutation_power_agent_1

        if role == "adversary_0":
            mutation_power = args.mutation_power_adversary

        weights = self.model.get_perturbable_weights()
        noise = np.random.normal(loc=0.0, scale=mutation_power, size=len(weights))
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


    def clone(self, args):
        """
        Delegates the clone method to the specific subclass instance.
        """
        raise NotImplementedError("The clone method should be implemented by the specific agent type.")

