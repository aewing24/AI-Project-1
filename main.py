# Create agent
import agents.dqn_agent
from networks.q_network import LunarLandarDQN

# Train agent
# Test agent
# Create plots

#Sources:
# https://gymnasium.farama.org/introduction/basic_usage/

# Necessary downloads
# pip install swig
# pip install "gymnasium[box2d]"

#pip install gym torch numpy matplotlib

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

if __name__ == '__main__':
    lunar_lander = LunarLandarDQN()
    continuous = True
    lunar_lander.train(1000, continuous = continuous)
    lunar_lander.test(10, continuous = continuous)