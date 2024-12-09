import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain

class DeepQN(nn.Module):
    def __init__(self, input_channels, n_actions, precision):
        
        super(DeepQN, self).__init__()

        self.dtype = torch.float16 if precision == "float16" else torch.float32

        self.layers = []
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4).to(dtype=self.dtype)
        self.layers.append(self.conv1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(dtype=self.dtype)
        self.layers.append(self.conv2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(dtype=self.dtype)
        self.layers.append(self.conv3)
        
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 512).to(dtype=self.dtype)  # Adjust this based on input size
        self.layers.append(self.fc1)
        self.output = nn.Linear(512, n_actions).to(dtype=self.dtype)
        self.layers.append(self.output)
        
        # Virtual Batch Normalization layers
        self.vbn1 = nn.BatchNorm2d(32).to(dtype=self.dtype)
        self.layers.append(self.vbn1)
        self.vbn2 = nn.BatchNorm2d(64).to(dtype=self.dtype)
        self.layers.append(self.vbn2)
        self.vbn3 = nn.BatchNorm2d(64).to(dtype=self.dtype)
        self.layers.append(self.vbn3)
        

    def forward(self, x):
        x = x.to(dtype=self.dtype)
        x = x / 255
        x = F.relu(self.vbn1(self.conv1(x)))
        x = F.relu(self.vbn2(self.conv2(x)))
        x = F.relu(self.vbn3(self.conv3(x)))
        x = x.reshape(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x
    
    def determine_action(self, inputs):
        """ Choose an action based on the observation. We do this by simply
        selecting the action with the highest outputted value. """
        actions = self.forward(inputs)
        return [np.argmax(action_set.detach().numpy()) for action_set in actions][0]


    def get_weights(self, layers=None):
        """
        Retrieves the weights of the network. (For GA)

        Args:
            layers (list, optional): A names list of layers from which to get the weights.
                                     If None, retrieves weights from all layers.

        Returns:
            dict: A dictionary of weights for the specified layers (or all layers if None).
        """
        if layers is None:
            # If no layers specified, return weights for all layers
            return {k: v.clone() for k, v in self.state_dict().items()}
        else:
            # Filter the weights for the specified layer names
            state_dict = self.state_dict()
            selected_weights = {}
            for key in state_dict:
                # Match keys based on layer name prefixes
                if any(key.startswith(layer) for layer in layers):
                    selected_weights[key] = state_dict[key].clone()
            
            #print(f"selected_weights = {selected_weights.keys()}")
            return selected_weights

    
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
            filtered_state_dict = {key: current_state_dict[key] for key in new_weights.keys()}
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


    def get_perturbable_weights(self):
        return self.get_weights_ES(self.get_perturbable_layers())


    def get_weights_ES(self, layers=None):
        """ Retrieve all the weights of the network. (For ES)"""
        layers = layers if layers else self.layers
        #print(f"layers = {layers}")

        ws_and_bs = []
        for layer in layers:
            #print(f"weight = {layer.weight.detach().cpu().numpy()}")
            weight =  layer.weight.detach().cpu().numpy()
            bias = []
            if layer.bias is not None:
                #print(f"bias = {layer.bias.detach().cpu().numpy()}")
                bias = layer.bias.detach().cpu().numpy()
            tup = (weight, bias)
            ws_and_bs.append(tup) 

        layer_weights = chain(*ws_and_bs)
        
        flat_weights = []
        for weights in layer_weights:
            weights = weights.flatten()
            flat_weights.append(weights)
        
        return np.concatenate(flat_weights)


    def get_perturbable_layers(self):
        """
        Get the names of all perturbable layers in the network.

        Returns:
            list: A list of layer whose weights are perturbable.
        """
        layers = []
        for name, layer in self.named_modules():
            if name == "":
                continue # skip the root module
            if not isinstance(layer, nn.BatchNorm2d):
                layers.append(layer)
        #print(f"names = {names}")
        return layers
        

    def set_weights_ES(self, flat_weights, args, layers=None):
        """ Set all the weights of the network. (for ES)"""
        # Variables to keep track of the position in flat_weights
        i = 0

        if args.precision == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        if layers is None:
            layers = self.get_perturbable_layers()

        for layer in layers:
            # Initialize a list to store reshaped weights and biases
            new_weights = []
            
            # Get the weights (and reshape)
            weight_size = layer.weight.numel()  # Total number of elements in the weight tensor
            reshaped_weight = flat_weights[i: i + weight_size].reshape(layer.weight.shape)

            reshaped_weight = torch.tensor(reshaped_weight, dtype=dtype)
            new_weights.append(reshaped_weight)
            i += weight_size  # Move the index
            
            # Check if the layer has biases and process them
            if layer.bias is not None:
                bias_size = layer.bias.numel()  # Total number of elements in the bias tensor
                reshaped_bias = flat_weights[i: i + bias_size].reshape(layer.bias.shape)
                reshaped_bias = torch.tensor(reshaped_bias, dtype=dtype)
                new_weights.append(reshaped_bias)
                i += bias_size  # Move the index
            
            # Set the weights and biases in the layer
            layer.weight.data.copy_(new_weights[0])  # Set weight
            if layer.bias is not None:
                layer.bias.data.copy_(new_weights[1])  # Set bias



    def set_perturbable_weights(self, weights_to_set, args):
        """ Set all the perturbable weights of the network. This excludes setting
        the BatchNorm weights. """
        self.set_weights_ES(weights_to_set, args, self.get_perturbable_layers())


    def mutate(self, mutation_power):
        """ Mutate the current weights by adding a normally distributed vector of
        noise to the current weights. """
        weights = self.get_perturbable_weights()
        noise = np.random.normal(loc=0.0, scale=mutation_power, size=weights.shape).astype(np.float16 if self.precision=="float16" else np.float32)

        self.set_perturbable_weights(weights + noise)
        return noise

        

