import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'new_state', 'reward', 'terminated'))
class ReplayBuffer:
    def __init__(self, capacity):
        """
        Replay Buffer defined by named tuples representing a single
        transition in our environment mapping (state, action) to
        (next_state, reward)
        :param capacity: limit of number of transitions in the replay buffer
        """
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        """
        Push the transition that the agent observes
        :param args: state, action, next_state, reward
        """
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        """
        Return a list of 'batch_size' length of random transitions chosen
        from ReplayBuffer.
        :param batch_size: size of list of random transitions to return
        :return: random list of transitions of 'batch_size' length
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Returns length of ReplayBuffer
        :return: number of transitions in ReplayBuffer
        """
        return len(self.buffer)