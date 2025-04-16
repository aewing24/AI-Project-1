import gymnasium as gym
import torch
from ..networks import q_network as DQN
from ..utils import replay_buffer as rb
from ..utils import train_logger as tl

class DQNAgent():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001  # learning rate (alpha)
    discount_factor_g = 0.9  # discount rate (gamma)
    network_sync_rate = 10  # number of steps the agent takes before syncing the policy and target network
    mini_batch_size = 32  # size of the training data set sampled from the replay memory

    def train(self, episodes):
        # Create Lunar Lander instance
        env = gym.make('LunarLander-v3')
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        epsilon = 1  # 1 = 100% random actions
        memory = rb.ReplayBuffer()
        logger = tl.TrainingLogger()

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN.Net(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN.Net(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0

        for i in range(episodes):
            #implement training algorithm

    # Run the Lunar Laner environment using an already learned policy
    def test(self, episodes):
        # Create Lunar Lander instance
        env = gym.make('LunarLander')
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        # Load learned policy
        policy_dqn = DQN.Net(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("lunar_lander_dql.pt"))
        policy_dqn.eval()  # switch model to evaluation mode

        for i in range(episodes):
            # use the policy to navigate the environment

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # update the policy network