'''
For each training run, record and log the following metrics per episode:
- Total episodic reward (sum of all rewards in the episode)
- Return G
- Landing success: whether the lander successfully landed
'''

class TrainingLogger:
    def __init__(self):
        self.memory = []

    # episodes are in the form (total reward, Return G, Landing Success)
    def append(self, episode):
        self.memory.append(episode)

    def __len__(self):
        return len(self.memory)