import gymnasium as gym
import torch
import random
import numpy as np
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

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        for i in range(episodes):
            #implement training algorithm
            state = env.reset()[0]  # Initialize to state 0
            terminated = False  # True when agent crashes, leaves viewport, or reached goal
            truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while (not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()  # actions: 0=nothing,1=left engine,2=main engine,3=right engine
                else:
                    # select best action, we use no_grad because we are just evaluating
                    # To use the model, we pass it the input data. This executes the model’s forward, along with some background operations.
                    # Do not call model.forward() directly!
                    # We must also convert the environment's state to a tensor representation for the nn to use
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state, reward, terminated, truncated, _ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count += 1

            # Keep track of the rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

            # Close environment
            env.close()

            # Save policy
            torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")

    # Run the Lunar Laner environment using an already learned policy
    def test(self, episodes):
        # Create Lunar Lander instance
        env = gym.make('LunarLander-v3')
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

    '''
    The state is an 8-dimensional vector: the coordinates of the lander in x & y,
    its linear velocities in x & y, its angle, its angular velocity,
    and two booleans that represent whether each leg is in contact with the ground or not.
    '''
    def state_to_dqn_input(self, state: int, num_states: int) -> torch.Tensor:
        input_tensor = torch.zeros(num_states)
        # convert here
        return input_tensor