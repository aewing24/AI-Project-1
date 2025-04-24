import random
import torch
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:
    """
    Utility class to store the incoming transition tuples from gym environment
    and sample random batches for stabilizing neural network\n
    Authors:
        name, Nazarii Revitskyi
    Date: Apr 23, 2025
    """
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward', 'done'))
    def __init__(self, device, capacity, seed)->None:
        """
        Create a buffer of certain capacity and input seed
        :param device: specify device that is
        :param capacity: size of buffer
        :param seed: seed for random batch
        """
        self.device = device if device is not None else 'cpu'
        self.buffer = deque(maxlen=capacity)
        self.seed = random.seed(seed)
        self.transition = namedtuple("Transition",
                                     field_names=["state", "action", "next_state", "reward", "done"])

    def push(self, state, action, next_state, reward, done)->None:
        """
        Push a new Transition onto the Replay buffer deque.
        :param state: current state
        :param action: current action
        :param next_state: mapping from this to next state
        :param reward: mapping of reward from action
        :param done: for q-target if terminal or non-terminal
        :return:none
        """
        self.buffer.append(self.transition(state,action,next_state,reward,done))

    def sample(self,batch_size)->torch.tensor:
        """
        Collects a batch of transitions (s, a) -> (s',r) and converts them to torch tensors
        using numpy ndarray by sequentially adding each individual value from transition vertically
        such that they are aligned for transformation into tensor.
        :param batch_size: size of batch to sample from the buffer
        :return: torch tensor for policy q value computation.
        """
        transitions = random.sample(self.buffer, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in transitions if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in transitions if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states,actions,next_states,rewards,dones)

    def __len__(self)->int:
        """
        This method returns the size of buffer.
        :return: size of the buffer
        """
        return len(self.buffer)