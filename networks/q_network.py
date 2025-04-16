import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        # 8 in-states, probably 8 hidden nodes, 4 out actions
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # ouptut layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply rectified linear unit (ReLU) activation
        return self.out(x)  # Calculate output