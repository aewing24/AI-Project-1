import gymnasium as gym
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent

"""
Code entry, establishes gym environments dqn agents and neural networks to
learn to play Lunar Lander-v3 gym game.\n
Authors: name, name, name, name, Nazarii Revitskyi
Date: Apr 23, 2025.
"""
# Sources:
# https://gymnasium.farama.org/introduction/basic_usage/

# Necessary downloads
# pip install swig
# pip install "gymnasium[box2d]"

if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("Env state size:", state_size, "Env action size:", action_size)

    # First agent (baseline DDQN)
    agent_a = DQNAgent(
        state_size,
        action_size,
        batch_size=128,
        gamma=0.99,
        sync_steps=10,
        capacity=25000,
        alpha=0.0005,
        seed=0,
        enable_double_dqn=True,
    )
    logger_a = agent_a.train(env, episodes=1000, terminate_on_target=True)

    # Second agent (variant to compare, e.g., more episodes or no early stop)
    agent_b = DQNAgent(
        state_size,
        action_size,
        batch_size=128,
        gamma=0.99,
        sync_steps=10,
        capacity=25000,
        alpha=0.0005,
        seed=0,
        enable_double_dqn=True,
    )
    agent_b.comparison_logger = logger_a
    logger_b = agent_b.train(env, episodes=1000, terminate_on_target=True)

    # Plots using logger_a and logger_b

    def plot_metric(metric_name, logger_a, logger_b, label_a="Agent A", label_b="Agent B"):
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