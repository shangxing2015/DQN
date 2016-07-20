import random

import numpy as np

from config_2 import *

"""generate a random policy for benchmark"""


class RandomPolciy:
    def __init__(self):
        self.action_size = ACTION_SIZE

    def getAction(self):
        """return: a random action with one-hot encoding"""

        action = np.zeros(int(self.action_size))
        action_index = random.randint(0, self.action_size - 1)
        action[action_index] = 1
        return action
