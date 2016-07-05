import tensorflow as tf
import numpy as np
import random
import math
from config import *

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

nominator = float(ACTION_SIZE)

prob = []

for i in range(N_NODES+1):

    temp = nCr(N_NODES, i) * math.pow(N_CHANNELS, i)
    prob.append(temp/nominator)

total  = 0

for i in range(len(prob)):
    total += (i-(4-i)) * prob[i]

total_dqn = 0
prob_dqn = [0.0, 0.008991008991008992, 0.08391608391608392, 0.43156843156843155, 0.4755244755244755]

for i in range(len(prob_dqn)):
    total_dqn += (i-(4-i)) * prob_dqn[i]

print total
print prob
print total_dqn
print prob_dqn