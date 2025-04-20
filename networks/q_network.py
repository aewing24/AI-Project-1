import random
import gymnasium as gym
import gym as gm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


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

#Define memory for replay
class ReplayMemory():
    # Initialize a dequeu to memorize the experience and also replay it
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    # Holds state, action, new state, reward, and termination
    def append(self, transition):
        self.memory.append(transition)

    # Return random sample of whatever size we want, sample size is the length of samples
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    # Return length of memory
    def __len__(self):
        return len(self.memory)

class LunarLandarDQN():
    # Hyperparameters
    learning_rate = 0.001   # learning ratem(alpha)
    gamma = 0.99    # discount rate (gamma)
    network_sync_rate = 10 # number of steps agent takes before synchronizing target + policy
    #epsilon_min = 0.01
    epsilon_decay = 0.995
    #batch_size = 32 # batch size
    target_update_freq = 1000
    memory_size = 10000 # replay memory size
    #episodes = 1000

    #Neural Network
    loss_function = nn.MSELoss() # NN loss function for MSE
    optimizer = None # optimizer

    # Four discrete actions: do nothing, fire left engine, fire right engine, or fire main engine
    ACTIONS = ['N', 'L', 'R', 'M']

    # Train lunar landar enviroment
    def train(self, episodes, render = False):
        # Create instance of lunar landar enviroment
        env = gym.make("LunarLander-v3", render_mode="human" if render else None)

        #may need an option for continuous

        #Ensure space is consistent across testing (seed)
        env.action_space.seed(42)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # epsilon = 1.0 meanas 100% random action for exploration
        epsilon = 1.0

        memory = ReplayMemory(self.memory_size)

        # Create policy and target network, number of nodes can be adjusted in the hidden layer
        policy_net = Net(in_states = num_states, h1_nodes= num_actions, out_actions = num_actions)
        target_net = Net(in_states=num_states, h1_nodes=num_actions, out_actions=num_actions)

        # Initialize target as the same as policy
        target_net.load_state_dict(policy_net.state_dict())

        print("Policy network is loaded, and is random for training")
        #fix this

        self.print_dqn(policy_net)

        # Network optimizer, Adam
        self.optimizer = nn.optim.Adam(policy_net.parameters(), lr=self.learning_rate)

        # List of tracked rewards, init to 0
        rewards_per_episode = np.zeros(episodes)

        # List to track epsilon decay
        epsilon_history = []

        # Track number of steps taken
        step_count = 0

        for i in range(episodes):
            state = env.reset()[0] # Init to 0
            terminated = False # True if agent falls off-screen, crashes, or lands properly
            truncated = False #True if agent takes more than 200 actions
            episode_reward = 0
            done = False

            # Agent pilots lander until it reaches goal (doesn't crash and lands)
            while not terminated and not truncated:

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # Choose a complete random action - exploration
                    return env.action_space.sample() #Actions = 0 - nothing, 1 - left, 2 - right, 3 - main
                else:
                    with nn.no_grad():
                        action = policy_net(self.state_to_net_input(state, num_states)).argmax().item()

                    # Otherwise, select the most optimal action - exploitation
                    # state = nn.FloatTensor(state).unsqueeze(0)
                    # q_values = policy_net(state)
                    #return nn.argmax(q_values).item()  # Exploit

                # Select action based on epsilon-greedy
                new_state, reward, terminated, truncated = env.step(action)

                # Store transition in memory
                memory.append((state, action, new_state, reward, terminated))

                # Update state
                state = new_state

                # Increment step
                step_count += 1

            #Track of rewards
            if reward == 1:
                rewards_per_episode[i] += 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_net, target_net)

                # Decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_net.load_state_dict(policy_net.state_dict())
                    step_count = 0

        # Close environment
        env.close()

        # Save policy
        nn.save(policy_net.state_dict(), "lunar_landing.pt")

        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x - 100):(x + 1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)

        # Save plots
        plt.savefig('lunar_landing_dql.png')

     # Optimize policy network
    def optimize(self, mini_batch, policy_net, target_net):

        # Get number of input nodes
        num_states = policy_net.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = nn.FloatTensor([reward])
            else:
                # Calculate target q value
                with nn.no_grad():
                    target = nn.FloatTensor(
                        reward + self.discount_factor_g * target_net(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_net(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_net(self.state_to_dqn_input(state, num_states))
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(nn.stack(current_q_list), nn.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #This needs to be formatted according to dqn
        '''
        Converts an state (int) to a tensor representation.
        For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

        Parameters: state=1, num_states=16
        Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        '''

    def state_to_dqn_input(self, state: int, num_states: int) -> nn.Tensor:
        input_tensor = nn.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

        # Run the Lunar environment with the learned policy
    def test(self, episodes, continuous = False):
        # Create Lunar instance
        env = gym.make('LunarLander-v3', render_mode='human')

        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = Net(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(nn.load("frozen_lake_dql.pt"))
        policy_dqn.eval()  # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False  # True when agent falls in hole or reached goal
            truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while (not terminated and not truncated):
                # Select best action
                with nn.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                    # Execute action
                state, reward, terminated, truncated, _ = env.step(action)

        env.close()

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                 q_values += "{:+.2f}".format(q) + ' '  # Concatenate q values, format to 2 decimals
            q_values = q_values.rstrip()  # Remove space at the end

            # Map the best action to N L R M
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')
            if (s + 1) % 4 == 0:
                print()  # Print a newline every 4 states

if __name__ == '__main__':
    lunar_lander = LunarLandarDQN()
    continuous = True
    lunar_lander.train(1000, continuous = continuous)
    lunar_lander.test(10, continuous = continuous)


# Source:
# https://www.youtube.com/watch?v=EUrWGTCGzlA