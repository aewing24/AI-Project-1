from collections import namedtuple

Log = namedtuple('Log',
                            ('episodic_reward', 'return_g', 'success'))
class TrainingLogger:
    """
    Utility class to store important values from training as tuples from gym environment
    and plot them\n
    Authors:
        name, Mat belmont
    Date: May 5, 2025
    """
    def __init__(self)->None:
        self.memory = []
        self.log = namedtuple("Log",
                                     field_names=['episodic_reward', 'return_g', 'success'])
    # episodes are in the form (total reward, Return G, Landing Success)
    def push(self, episodic_reward: float, return_g: float, success: bool)->None:
        """
        Push a new Log onto the memory array\n
        :param episodic_reward: sum of all rewards in the episode
        :param return_g: sum of the discounted rewards
        :param success: True if the lander successfully landed, False otherwise
        :return:none
        """
        self.memory.append(self.log(episodic_reward, return_g, success))

    def __len__(self):
        return len(self.memory)