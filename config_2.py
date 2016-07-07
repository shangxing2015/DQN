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
P_MATRIX = [(0.6, 0.4), (0.49, 0.51)]
GOOD_CHANNEL = False
N_SENSING = 1
ACTION_SIZE = combination(N_CHANNELS, N_SENSING)

#def action space
ACTION_LIST =  [i for i in range(N_CHANNELS)]
ACTION_SPACE = list(itertools.combinations(ACTION_LIST, N_SENSING))
