import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQN(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(DeepQN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Adjust this based on input size
        self.output = nn.Linear(512, n_actions)

        # Virtual Batch Normalization layers
        self.vbn1 = nn.BatchNorm2d(32)
        self.vbn2 = nn.BatchNorm2d(64)
        self.vbn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        # Convolutional layers with VBN and ReLU
        x = F.relu(self.vbn1(self.conv1(x)))
        x = F.relu(self.vbn2(self.conv2(x)))
        x = F.relu(self.vbn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

# Example usage:
if __name__ == "__main__":
    # Define input dimensions and number of actions
    input_channels = 4  # For example, a stack of 4 frames
    n_actions = 6       # Number of possible actions in the environment

    # Initialize the network
    dqn = DeepQN(input_channels, n_actions)

    # Example input: batch of 32, 4-channel, 84x84 images
    input_tensor = torch.randn(32, input_channels, 84, 84)
    output = dqn(input_tensor)

    print("Output shape:", output.shape)
