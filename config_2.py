# basic settings
"""

Denote s = N_NODES, c = N_CHANNELS, w = AGENT_STATE_WINDOWS_SIZE.

complexity:
  #(observation): pow(2, s)
  #(state): pow(2, ws)
  #(action): pow(c + 1, s)


"""


import math
import itertools
import random


def combination(n,k):
    numerator=math.factorial(n)
    denominator=(math.factorial(k)*math.factorial(n-k))
    answer=numerator/denominator
    return answer


N_NODES = 1

N_CHANNELS = 3


# q-learning settings
AGENT_STATE_WINDOWS_SIZE = 20





DEBUG_NO_CONFLICT_GRAPH = True

#transition matrix
P_MATRIX = [(0.6, 0.4), (0.4, 0.6)]
GOOD_CHANNEL = True
N_SENSING = 2
ACTION_SIZE = combination(N_CHANNELS, N_SENSING)

#def action space
ACTION_LIST = [i for i in range(N_CHANNELS)]
ACTION_SPACE = list(itertools.combinations(ACTION_LIST, N_SENSING))


#DISCOUNT / AVERAGE REWARD
DISCOUNT = False
temp_prob = [(random.random(), random.random()) for i in range(int(N_CHANNELS))]
P_DISTINCT_MATRIX = [[(x, 1 - x), (y, 1 - y)] for x, y in temp_prob]
B = [1 for i in range(N_CHANNELS)]
