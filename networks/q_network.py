import torch.nn as nn
import torch.nn.functional as F
from torch import manual_seed

class DQN(nn.Module):
    """
    DQN extension of torch.nn creates neural network for incoming gym
    environment where input_nodes=state_size and output_nodes=action_size\n
    Authors:
        Mathew Belmont, Nazarii Revitskyi
    Date: Apr 23, 2025.
    """
    def __init__(self, state_size: int, action_size: int, seed: int)->None:
        """
        Extending the nn to initialize custom neural network with custom layers.
        :param state_size: number of input nodes
        :param action_size: number of output nodes
        :param seed: seed to reproduce random state
        """
        super(DQN, self).__init__()
        self.seed = manual_seed(seed)
        # Define network layers
        self.fc1 = nn.Linear(state_size, 64)   # first fully connected layer
        self.fc2 = nn.Linear(64, 64)  # second fully connected layer
        self.out = nn.Linear(64, action_size) # output layer

    def forward(self, state: any) -> any:
        """
        Overridden method applies non_linear computation on layer advance.
        :param state:  state of network.
        :return: q_values
        """
        x = F.relu(self.fc1(state))  # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        q_vals = self.out(x)
        return q_vals  # Calculate output