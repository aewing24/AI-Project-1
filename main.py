import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from agents.dqn_agent import DQNAgent
from utils.train_logger import TrainingLogger

"""
Code entry, establishes gym environments dqn agents and neural networks to
learn to play Lunar Lander-v3 gym game.\n
Authors: Lucas Jeong, name, name, name, Nazarii Revitskyi
Date: Apr 23, 2025.
"""
# Sources:
# https://gymnasium.farama.org/introduction/basic_usage/

if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    training_length = 100
    print("Env state size:", state_size, "Env action size:", action_size)
    print("Training length:", training_length)

    agent_config = {
        'state_size': state_size,
        'action_size': action_size,
        'batch_size': 128,
        'gamma': 0.99,
        'sync_steps': 10,
        'capacity': 25000,
        'alpha':0.0005,
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

    def plot_metric(metric_name: str, logger_a: TrainingLogger, logger_b: TrainingLogger, label_a="Agent A", label_b="Agent B"):
        index = {"episodic_reward": 0, "return_g": 1, "success": 2}[metric_name]
        data_a = [entry[index] for entry in logger_a.memory]
        data_b = [entry[index] for entry in logger_b.memory]

        plt.figure(figsize=(10, 5))
        plt.plot(data_a, label=label_a)
        plt.plot(data_b, label=label_b)
        plt.title(f"{metric_name.replace('_', ' ').title()} over Episodes")
        plt.xlabel("Episode")
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    plot_metric("episodic_reward", logger_a, logger_b)
    plot_metric("return_g", logger_a, logger_b)
    plot_metric("success", logger_a, logger_b)

    # Tables using logger_a and logger_b
    data = {
        "DQN (Vanilla)": [
            sum(entry.episodic_reward for entry in logger_a.memory) / len(logger_a.memory),
            sum(entry.return_g for entry in logger_a.memory) / len(logger_a.memory),
            sum(1 for entry in logger_a.memory if entry.success) / len(logger_a.memory) * 100,
        ],
        "Double DQN": [
            sum(entry.episodic_reward for entry in logger_b.memory) / len(logger_b.memory),
            sum(entry.return_g for entry in logger_b.memory) / len(logger_b.memory),
            sum(1 for entry in logger_b.memory if entry.success) / len(logger_b.memory) * 100,
        ],
    }

    index = ["Average Episode Reward", "Average Return", "Success Rate (%)"]

    df = pd.DataFrame(data, index=index)

    print(df)