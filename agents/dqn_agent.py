import gymnasium as gym
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

from networks.q_network import Net
from utils.replay_buffer import ReplayBuffer
from utils.train_logger import TrainingLogger

class DQNAgent():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.01  # learning rate (alpha)
    discount_factor_g = 0.95  # discount rate (gamma)
    network_sync_rate = 10  # number of steps the agent takes before soft updating the target network
    mini_batch_size = 64  # size of the training data set sampled from the replay memory
    tau = 0.001 # soft update amount

    loss_fn = nn.MSELoss()  # NN loss function for MSE
    optimizer = None  # optimizer

    def train(self, episodes):
        # Create Lunar Lander instance
        env = gym.make('LunarLander-v3')
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        epsilon = 1  # 1 = 100% random actions
        memory = ReplayBuffer(episodes * 350)
        logger = TrainingLogger()

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = Net(in_states=num_states, out_actions=num_actions)
        target_dqn = Net(in_states=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Network optimizer, Adam
        # just to start with, may have to change
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        for i in range(episodes):
            #implement training algorithm
            state = env.reset()[0]  # Initialize to state 0
            terminated = False  # True when agent crashes, leaves viewport, or reached goal
            truncated = False  # True when agent takes more than 200 actions
            cuml_reward = 0

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
                        action = policy_dqn(torch.tensor(state, dtype=torch.float32)).argmax().item()

                # Execute action
                new_state, reward, terminated, truncated, _ = env.step(action)

                # Save experience into memory
                memory.push(state, action, new_state, reward, terminated)

                # cumulate reward
                cuml_reward += reward

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count += 1

                # Check if enough experience has been collected
                if len(memory) > self.mini_batch_size and step_count > self.network_sync_rate:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # soft update policy network to target network after a certain number of steps
                    for target_param, local_param in zip(target_dqn.parameters(), policy_dqn.parameters()):
                        target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

                    step_count = 0

            # Keep track of the rewards collected per episode.
            # An episode is considered a solution if it scores at least 200 points.
            rewards_per_episode[i] = cuml_reward

            # Decay epsilon
            epsilon = max(epsilon - 1 / episodes, 0.01)

        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "lunar_lander_dql.pt")

        plt.plot(rewards_per_episode)

        # Save plots
        plt.savefig('frozen_lake_dql.png')

        print(rewards_per_episode.max())

    # Run the Lunar Laner environment using an already learned policy
    def test(self, episodes):
        # Create Lunar Lander instance
        env = gym.make('LunarLander-v3')
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = Net(in_states=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("lunar_lander_dql.pt"))
        policy_dqn.eval()  # switch model to evaluation mode

        for i in range(episodes):
            # use the policy to navigate the environment
            state = env.reset()[0]  # Initialize to state 0
            terminated = False  # True when agent crashes, moves out of viewport, or reached goal
            truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it crashes (terminated), leaves viewport (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while (not terminated and not truncated):
                # Select best action
                with torch.no_grad():
                    action = policy_dqn(torch.tensor(state, dtype=torch.float32)).argmax().item()

                # Execute action
                state, reward, terminated, truncated, _ = env.step(action)

        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # update the policy network
        states = torch.tensor(np.array([s for s, _, _, _, _ in mini_batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([a for _, a, _, _, _ in mini_batch]), dtype=torch.int64)
        rewards = torch.tensor(np.array([r for _, _, _, r, _ in mini_batch]), dtype=torch.float32)
        next_states = torch.tensor(np.array([ns for _, _, ns, _, _ in mini_batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([done for _, _, _, _, done in mini_batch]), dtype=torch.float32)

        # Get current Q values for taken actions
        q_values = policy_dqn(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q values
        with torch.no_grad():
            max_next_q = target_dqn(next_states).max(1)[0]
            # (1-dones) handles the terminal cases
            target_q = rewards + (1 - dones) * self.discount_factor_g * max_next_q

        # Loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()