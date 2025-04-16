import random

class ReplayBuffer:
    def __init__(self):
        self.memory = []

    # transitions are in the form (state, action, new_state, reward, terminated)
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)