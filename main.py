# Create agent
from agents.dqn_agent import DQNAgent
# Train agent
# Test agent
# Create plots

#Sources:
# https://gymnasium.farama.org/introduction/basic_usage/

# Necessary downloads
# pip install swig
# pip install "gymnasium[box2d]"

if __name__ == '__main__':
    lunar_lander = DQNAgent()
    lunar_lander.train(1000)
    lunar_lander.test(10)