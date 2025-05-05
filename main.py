import gymnasium as gym
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
    # env
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("Env state size: ", state_size, " Env action size: ", action_size)
    dqn_agent = DQNAgent(
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

    ddqn = dqn_agent.train(env, 2500, terminate_on_target=True)
    vdqn = dqn_agent.train(env, 5000, terminate_on_target=False)
    #env = gym.make("LunarLander-v3", render_mode="human")
    #dqn_agent.test(env, 10)
    #env.close()

    # make plots here

    # make table here
