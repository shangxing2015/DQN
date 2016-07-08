from whittleIndex import WhittleIndex
from environment_markov_channel import Environment
from config_2 import *
import time


T_THRESHOLD = 500000
PERIOD = 100
gamma = 1

env = Environment()

brain = WhittleIndex(B, gamma)

action = [i for i in range(N_SENSING)]

total = 0

fileName = 'log_whittle'

f = open(fileName,'w')

start_time = time.time()

for i in range(T_THRESHOLD):

    observation, reward, terminal = env.step(action)
    total += reward

    action = brain.getAction(action, observation)

    count = i+1

    if count % PERIOD == 0:
        accum_reward = total / float(count)
        duration = time.time() - start_time
        f.write('Index %d: accu_reward is %f, action is: %s and time duration is %f' % (count, accum_reward, str(action), duration))
        f.write('\n')



f.close()



