from Async_DQN_3 import Async_DQN
from config_2 import *

temp_prob = [(random.random(), random.random())]
p_matrix = [[(x, 1 - x), (y, 1 - y)] for x, y in temp_prob] * 3

print p_matrix