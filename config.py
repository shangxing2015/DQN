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

N_NODES = 1
N_NOISES = 0.5
N_CHANNELS = 4

# q-learning settings
AGENT_STATE_WINDOWS_SIZE = 2
ACTION_SIZE = math.pow(N_CHANNELS, N_NODES)

# def action space
ACTION_LIST = [-1] + [i for i in range(N_CHANNELS)]
ACTION_SPACE = list(itertools.product(ACTION_LIST, repeat=N_NODES))

# test settings
STAGES = 20
ROUNDS = 100

# debug settings
OUTPUT_F = None
DEBUG_NO_NOISE = False
DEBUG_NO_CONFLICT_GRAPH = True

# transition matrix
P_MATRIX = [(0.8, 0.2), (0.6, 0.4)]
GOOD_CHANNEL = False
N_SENSING = 1
