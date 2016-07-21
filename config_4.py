# basic settings
"""

Denote s = N_NODES, c = N_CHANNELS, w = AGENT_STATE_WINDOWS_SIZE.

complexity:
  #(observation): pow(2, s)
  #(state): pow(2, ws)
  #(action): pow(c + 1, s)


"""

import itertools
import math


def combination(n, k):
    numerator = math.factorial(n)
    denominator = (math.factorial(k) * math.factorial(n - k))
    answer = numerator / denominator
    return answer


N_NODES = 1

N_CHANNELS = 2

# q-learning settings
AGENT_STATE_WINDOWS_SIZE = 1

DEBUG_NO_CONFLICT_GRAPH = True

# transition matrix
P_MATRIX = [[(0.2, 0.8), (0.6, 0.4)]] * (N_CHANNELS)
GOOD_CHANNEL = True
N_SENSING = 1
ACTION_SIZE = combination(N_CHANNELS, N_SENSING)

# def action space
ACTION_LIST = [i for i in range(N_CHANNELS)]
ACTION_SPACE = list(itertools.combinations(ACTION_LIST, N_SENSING))

# DISCOUNT / AVERAGE REWARD
DISCOUNT = False
# temp_prob = [(random.random(), random.random()) for i in range(int(N_CHANNELS))]
# P_DISTINCT_MATRIX = [[(x, 1 - x), (y, 1 - y)] for x, y in temp_prob]

P_DISTINCT_MATRIX = [[(0.4102123599894729, 0.5897876400105271), (0.3398750548636893, 0.6601249451363107)],
                     [(0.34121492090001593, 0.6587850790999841), (0.7583916471527934, 0.2416083528472066)],
                     [(0.7393363711238706, 0.26066362887612937), (0.32311980684897756, 0.6768801931510224)]]

B = [1 for i in range(N_CHANNELS)]

# for writing to the file
PERIOD = 100  # for writing to the file
T_THRESHOLD = 3000000  # 5000000# num of plays; 80000000
T_EVAL = 50000
T_CVG = 15000
