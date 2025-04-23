import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_states, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, 32)   # first fully connected layer
        self.layer2 = nn.Linear(32, 32) # second fully connected layer
        self.out = nn.Linear(32, out_actions) # ouptut layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.layer2(x))
        return self.out(x)  # Calculate output