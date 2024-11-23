import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepQN(nn.Module):
    def __init__(self, input_channels, n_actions):
        
        super(DeepQN, self).__init__()

        self.layers = []
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.layers.append(self.conv1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.layers.append(self.conv2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.layers.append(self.conv3)
        
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 22 * 16, 512)  # Adjust this based on input size
        self.layers.append(self.fc1)
        self.output = nn.Linear(512, n_actions)
        self.layers.append(self.output)
        
        # Virtual Batch Normalization layers
        self.vbn1 = nn.BatchNorm2d(32)
        self.layers.append(self.vbn1)
        self.vbn2 = nn.BatchNorm2d(64)
        self.layers.append(self.vbn2)
        self.vbn3 = nn.BatchNorm2d(64)
        self.layers.append(self.vbn3)
        

    def forward(self, x):

        # Convolutional layers with VBN and ReLU
        #print(f"INPUT SIZE: { x.shape}")
        x = F.relu(self.vbn1(self.conv1(x)))
        #print(f"AFTER VBN1(CONV1): { x.shape}")
        x = F.relu(self.vbn2(self.conv2(x)))
        #print(f"AFTER VBN2(CONV2): { x.shape}")
        x = F.relu(self.vbn3(self.conv3(x)))
        #print(f"AFTER VBN3(CONV3): { x.shape}")

        # Flatten
        x = x.reshape(x.size(0), -1)  # Replace .view() with .reshape()
        #print(f"AFTER RESHAPING: { x.shape}")


        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        #print(f"AFTER FC1: { x.shape}")
        x = self.output(x)
        return x
    
    def determine_action(self, inputs):
        """ Choose an action based on the observation. We do this by simply
        selecting the action with the highest outputted value. """
        actions = self.forward(inputs)
        return [np.argmax(action_set.detach().numpy()) for action_set in actions][0]



    def get_weights(self):
        return {k: v.clone() for k, v in self.state_dict().items()}
    
    def set_weights(self, new_weights, layers=None):
        """
        Sets the weights of the network.

        Args:
            new_weights (dict): A dictionary containing new weights for the model, 
                                where keys match the state_dict keys.
            layers (list, optional): A list of layer names (keys in state_dict) whose 
                                    parameters will be updated. If None, all layers 
                                    will be updated.
        """
        # Get the current state_dict of the model
        current_state_dict = self.state_dict()

        if layers is not None:
            # Filter the keys to only include the specified layers
            filtered_state_dict = {key: current_state_dict[key] for key in layers}
        else:
            # Use the full state_dict if no layers are specified
            filtered_state_dict = current_state_dict

        # Validate the new_weights for the filtered keys
        for key in filtered_state_dict.keys():
            if key not in new_weights:
                raise ValueError(f"Missing key in new_weights: {key}")
            if new_weights[key].shape != filtered_state_dict[key].shape:
                raise ValueError(f"Shape mismatch for key '{key}': expected {filtered_state_dict[key].shape}, got {new_weights[key].shape}")

        # Update the filtered state_dict with the new weights
        filtered_state_dict.update({key: new_weights[key] for key in filtered_state_dict.keys()})
        
        # Load the updated weights into the model
        self.load_state_dict(filtered_state_dict, strict=False)

    def get_perturbable_layers(self):
        """ Get all the perturbable layers of the network. This excludes the
        BatchNorm layers. """
        return [layer for layer in self.layers if
                not isinstance(layer, nn.BatchNorm2d)]

    def get_perturbable_weights(self):
        """ Get all the perturbable weights of the network. This excludes the
        BatchNorm weights. """
        return self.get_weights(self.get_perturbable_layers())
    

    def set_perturbable_weights(self, weights_to_set):
        """ Set all the perturbable weights of the network. This excludes setting
        the BatchNorm weights. """
        self.set_weights(weights_to_set, self.get_perturbable_layers())


    def mutate(self, mutation_power):
        """ Mutate the current weights by adding a normally distributed vector of
        noise to the current weights. """
        weights = self.get_perturbable_weights()
        noise = np.random.normal(loc=0.0, scale=mutation_power, size=weights.shape)
        self.set_perturbable_weights(weights + noise)
        return noise

        

