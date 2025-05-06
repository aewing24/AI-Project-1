import torch
import random
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn

import gymnasium as gym

from networks.q_network import DQN
from utils.replay_buffer import ReplayBuffer
from utils.train_logger import TrainingLogger


class DQNAgent:
    """
    This is the DQN type agent that interacts with custom environment
    and utilizes target policy eps-greedy networks that operate using
    DQN algorithm.\n
    Authors:
        Mathew Belmont, Alexander Ewing, Lucas Jeong, Nathan Wanjongkhum, Nazarii Revitskyi
    Date: May 7, 2025
    """

    # if gpu is available, otherwise utilize CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current device: ", device)

    def __init__(
        self,
        state_size: int,
        action_size: int,
        batch_size: int,
        gamma: float,
        sync_steps: int,
        capacity: int,
        alpha: float,
        seed: int,
        enable_double_ddqn=False,
    ) -> None:
        """
        Initialize DQN Agent with replay buffer and adam optimizer.
        :param state_size: size of env state to consider in computation
        :param action_size: size of env actions available in computation
        :param batch_size: size of replay buffer to feed into network
        :param gamma: constant discount factor to optimize behavior and converge rewards
        :param capacity: how large of a replay buffer we want
        :param alpha: learning rate constant, determines to what extent new info overrides old.
        :param seed: seed to use for random
        """
        # net param
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # hyper
        self.batch_size = batch_size
        self.gamma = gamma
        # Epsilon: The probability of taking a random action
        self.eps = 1.0  # 1.0 = purely random, 0.0 = purely greedy
        # Epsilon decay: The rate of decay of epsilon, increasing exploitation over time
        self.eps_decay = 0.0025
        # Epsilon end: The minimum epsilon value to reach
        self.eps_end = 0.01
        self.tau = 0.001  # soft update constant
        self.sync_steps = sync_steps # How many steps before policy and target network are synched.

        # Enable double DQN: Toggle DDQN
        self.enable_double_dqn = enable_double_ddqn

        # Replay buffer
        self.buffer_replay = ReplayBuffer(self.device, capacity, seed)
        # logger
        self.training_logger = TrainingLogger()

        # Eval network: The network that is used to evaluate the Q values
        self.net_eval = DQN(state_size, action_size, seed).to(self.device)
        # Target network: The network that is used to calculate the target Q values
        self.net_target = DQN(state_size, action_size, seed).to(self.device)
        # Optimizer: The optimizer used to update the network parameters
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=alpha)
        # Loss function: The loss function used to calculate the error between the predicted Q values and the expected Q values
        self.loss = nn.MSELoss()
        # update counter
        self.steps = 0

    def take_action(self, state: any, eps=0.0) -> int:
        """
        Converts numpy array to tensor and moves to specified device
        then gets the action values from network and based on epsilon chooses
        either optimized action or random direction.
        :param state: state to decide action for
        :param eps: epsilon value to introduce exploration vs exploitation
        :return: action to take.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.net_eval.eval()
        with torch.no_grad():
            action_values = self.net_eval(state)
        self.net_eval.train()
        # select action by epsilon
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def prepare_batch(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool) -> None:
        """
        Adds transition to replay buffer and checks if enough steps were taken
        before syncing policy and target networks
        :param state: state from environment
        :param action: action from environment
        :param next_state: next state from environment
        :param reward:  reward from environment
        :param done: current step performance from environment
        :return: none
        """
        self.buffer_replay.push(state, action, next_state, reward, done)

        self.steps += 1
        if self.steps % self.sync_steps == 0:
            if len(self.buffer_replay) >= self.batch_size:
                transitions = self.buffer_replay.sample(self.batch_size)
                self.update_params(transitions)

    def update_params(self, transitions: any) -> None:
        """
        If buffer size is sufficient we take the named tuple and convert it to
        tensors to calculate the q values in value network then compute error
        and backpropagate calculating gradients
        :return: none
        """
        if len(self.buffer_replay) < self.batch_size:
            return
        # get batch
        states, actions, next_states, rewards, dones = transitions

        if self.enable_double_dqn:
            # Step 1: Use target network to SELECT the best action for the NEXT state
            best_action_indices = self.net_target(next_states).max(1)[1].unsqueeze(1)

            # Step 2: Use eval network to EVALUATE the value of that action
            next_state_values = (
                self.net_eval(next_states).gather(1, best_action_indices).detach()
            )
        else:
            # compute V(s_{t+1})
            next_state_values = (
                self.net_target(next_states).detach().max(axis=1)[0].unsqueeze(1)
            )

        # compute expected Q
        expected_state_action_values = rewards + self.gamma * next_state_values * (
            1 - dones
        )

        # compute Q(s_t, a)
        state_action_values = self.net_eval(states).gather(1, actions)

        # loss
        loss = self.loss(state_action_values, expected_state_action_values)
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # soft update
        self.soft_update()

    def soft_update(self) -> None:
        """
        updates target network by copying over the data from policy network
        :return: none
        """
        for eval_param, target_param in zip(
            self.net_eval.parameters(), self.net_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, fname: str) -> None:
        """
        save network state named as fname
        :param fname: file name to save the state of network.
        :return: none
        """
        torch.save(self.net_eval.state_dict(), fname)

    def load(self, fname: str) -> None:
        """
        Load saved state of network in fname file and map tensors from gpu to cpu
        :param fname:
        :return:none
        """
        self.net_eval.load_state_dict(torch.load(fname))

    def train(
        self, env: gym.Env, episodes: int, agent_name: str, max_steps=1000, target=215, terminate_on_target=False
    ) -> TrainingLogger:
        """
        This method is used to train DQN agent using gym LLv3 env over
        policy and target Deep Q Network.
        :param env: environment to train in
        :param episodes: number of episodes to train over
        :param max_steps: max env steps per episode if exceeded will go to next episode
        :param target: target average reward to reach
        :param terminate_on_target: whether to terminate training on reaching target
        :return: TrainingLogger
        """
        reward_list = []
        reward_avg = 0
        reward_avg_list = []

        fails = 0

        print("\n Agent: ", agent_name, "\n")
        # train loop
        for i_ep in range(1, episodes + 1):
            state = env.reset()[0]
            cumulative_reward = 0
            g = 0
            for i_step in range(1, max_steps + 1):
                action = self.take_action(state, self.eps)
                next_state, reward, done, truncated, _ = env.step(action)
                self.prepare_batch(state, action, next_state, reward, done)
                state = next_state
                cumulative_reward += reward
                g += reward * (self.gamma ** i_step)
                if done or truncated:
                    break

            # after each episode
            if cumulative_reward >= 200:
                self.training_logger.push(cumulative_reward, g, True)
            else:
                self.training_logger.push(cumulative_reward, g, False)
                fails += 1
            reward_list.append(cumulative_reward)  # save new score
            reward_avg = np.mean(reward_list[-100:])  # recent 100
            reward_avg_list.append(reward_avg)
            self.eps = max(
                self.eps * (1 - self.eps_decay), self.eps_end
            )  # decrease eps

            print(
                "\rEpisode {}\tAverage Reward: {:.2f}\tTotal Episodic Reward: {:.2f}\tReturn G: {:.2f}\tEpsilon: {:.2f}\tSuccesses: {}/{}".format(
                    i_ep, reward_avg, sum(reward_list), g, self.eps, i_ep - fails, i_ep
                ),
                end="",
            )
            if i_ep == 50:
                self.save("checkpoint_mid_train1.pt")
            if i_ep == 100:
                self.save("checkpoint_mid_train2.pt")
            if i_ep == 200:
                self.save("checkpoint_mid_train3.pt")
            if i_ep == 500:
                self.save("checkpoint_mid_train4.pt")
            if i_ep == 1000:
                self.save("checkpoint_mid_train5.pt")

            if i_ep % 25 == 0:
                print(
                    "\rEpisode {}\tAverage Reward: {:.2f}\tTotal Episodic Reward: {:.2f}\tReturn G: {:.2f}\tEpsilon: {:.2f}\tSuccesses: {}/{}".format(
                        i_ep, reward_avg, sum(reward_list), g, self.eps, i_ep - fails, i_ep
                    )
                )
            if np.mean(reward_avg) >= target and terminate_on_target:
                print(
                    "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                        i_ep, reward_avg
                    )
                )
                self.save("final_checkpoint.pt")
                break

        # Extra plots for comparison if multiple agents are run
        if hasattr(self, 'comparison_logger'):  # Hacky but avoids changing other classes
            agent_names = ['Agent A', 'Agent B']
            loggers = [self.training_logger, self.comparison_logger]

            # Plot 1: Episodic Reward
            plt.figure()
            for i, logger in enumerate(loggers):
                rewards = [log.episodic_reward for log in logger.memory]
                plt.plot(rewards, label=agent_names[i])
            plt.xlabel("Episode")
            plt.ylabel("Episodic Reward")
            plt.title("Episodic Reward vs. Episode")
            plt.legend()
            plt.savefig("Comparison_Reward_vs_Episode.png")

            # Plot 2: Return G
            plt.figure()
            for i, logger in enumerate(loggers):
                returns = [log.return_g for log in logger.memory]
                plt.plot(returns, label=agent_names[i])
            plt.xlabel("Episode")
            plt.ylabel("Return Sum of Discounted Rewards")
            plt.title("Episodic Return vs. Episode")
            plt.legend()
            plt.savefig("Comparison_Return_vs_Episode.png")

            # Plot 3: Success Rate (binary success per episode)
            plt.figure()
            for i, logger in enumerate(loggers):
                successes = [int(log.success) for log in logger.memory]
                plt.plot(successes, label=agent_names[i])
            plt.xlabel("Episode")
            plt.ylabel("Success (1 = landed)")
            plt.title("Success Rate over Time")
            plt.legend()
            plt.savefig("Comparison_SuccessRate_vs_Episode.png")
        return self.training_logger

    def test(self, env: gym.Env, episodes: int) -> None:
        """
        Loads the saved network parameters and attempts to solve the environment.
        :param env: gym environment to play
        :param episodes: number of episodes to play
        :return: none
        """
        self.net_eval.load_state_dict(torch.load("final_checkpoint.pt"))
        for i_ep in range(episodes + 1):
            state = env.reset()[0]
            for step in range(500):
                action = self.take_action(state, eps=0)
                state, reward, done, truncated, _ = env.step(action)
                if done:
                    break
        env.close()
