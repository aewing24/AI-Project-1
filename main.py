import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from agents.dqn_agent import DQNAgent
from utils.train_logger import Log, TrainingLogger

"""
Code entry, establishes gym environments dqn agents and neural networks to
learn to play Lunar Lander-v3 gym game.\n
Authors: Nathan Wanjongkhum, Lucas Jeong, Mathew Belmont, Alexander Ewing, Nazarii Revitskyi
Date: May 7, 2025.
"""

if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    state_size: int = env.observation_space.shape[0]
    action_size: int = env.action_space.n
    training_length = 1000
    print("Env state size:", state_size, "Env action size:", action_size)
    print("Training length:", training_length)

    agent_config = {
        'state_size': state_size,
        'action_size': action_size,
        'batch_size': 128,
        'gamma': 0.99,
        'sync_steps': 10,
        'capacity': 25000,
        'alpha':0.001,
        'seed': 0
    }

    # First agent (baseline DQN)
    agent_a = DQNAgent(**agent_config, enable_double_ddqn=False)
    logger_a = agent_a.train(env, episodes=training_length, agent_name= "DQN", terminate_on_target=True)

    # Second agent (DDQN exstentsion)
    agent_b = DQNAgent(**agent_config, enable_double_ddqn=True)
    agent_b.comparison_logger = logger_a
    logger_b = agent_b.train(env, episodes=training_length, agent_name="Double - DQN", terminate_on_target=True)

    # Plots using logger_a and logger_b

    def plot_metric(metric_name: str, logger_a: TrainingLogger, logger_b: TrainingLogger, label_a="DQN", label_b="DDQN"):
        # Dictionary which contains the metric name associated with the index in the logger.
        index = {"episodic_reward": 0, "return_g": 1, "success": 2}[metric_name]
        
        # Gets a list representation of the Log namedtuple (episodic_reward: float, return_g: float, success: bool)
        # bool can be interpreted as a float 0 or 1 since we are taking the average of the success rate
        data_a: list[float] = [entry[index] for entry in logger_a.memory]
        data_b: list[float] = [entry[index] for entry in logger_b.memory]

        # Change success as a binary graph
        if metric_name == "success":
            # Calculate the average success rate
            data_a_array = np.array(data_a)
            running_averages = np.cumsum(data_a_array) / np.arange(1, len(data_a_array) + 1)
            data_a = running_averages.tolist()
            
            data_b_array = np.array(data_b)
            running_averages = np.cumsum(data_b_array) / np.arange(1, len(data_b_array) + 1)
            data_b = running_averages.tolist()

        # Create figure and plot it with the data from the loggers
        plt.figure(figsize=(10, 5))
        plt.plot(data_a, label=label_a)
        plt.plot(data_b, label=label_b)

        # Title and labels
        plt.title(f"{metric_name.replace('_', ' ').title()} over Episodes")
        plt.xlabel("Episode")
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Create plots using a function
    plot_metric("episodic_reward", logger_a, logger_b)
    plot_metric("return_g", logger_a, logger_b)
    plot_metric("success", logger_a, logger_b)

    # Retrieve last 100 episodes of training for both agents via the logger
    a_end = logger_a.memory[-100:]
    b_end = logger_b.memory[-100:]

    # Tables using logger_a and logger_b
    data = {
        "DQN (Vanilla)": [
            sum(entry.episodic_reward for entry in a_end) / 100,
            sum(entry.return_g for entry in a_end) / 100,
            sum(1 for entry in a_end if entry.success),
        ],
        "Double DQN": [
            sum(entry.episodic_reward for entry in b_end) / 100,
            sum(entry.return_g for entry in b_end) / 100,
            sum(1 for entry in b_end if entry.success),
        ],
    }
    index = ["Average Episode Reward", "Average Return", "Success Rate (%)"]
    df = pd.DataFrame(data, index=index)
    print(df)