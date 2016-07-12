from whittleIndex import WhittleIndex
from env_markov_distinct_channel import Environment
from config_2 import *
import time

gamma = 1

p_matrix = [[(0.6, 0.4), (0.2, 0.8)]] * N_CHANNELS

env = Environment(p_matrix)

brain = WhittleIndex(B, gamma, p_matrix)

action = [i for i in range(N_SENSING)]

total = 0

fileName = 'log_whittle_temp'

f = open(fileName,'w')

start_time = time.time()

for i in range(5000):
    count = i + 1

    observation, reward, terminal = env.step(action)
    total += reward

    action = brain.getAction(action, observation, count)

    #if count % 100 ==0:
        #print observation
        #print action



    if count % 10 == 0:
        accum_reward = total / float(count)
        duration = time.time() - start_time
        f.write('Index %d: accu_reward is %f, action is: %s and time duration is %f' % (count, accum_reward, str(action), duration))
        f.write('\n')



f.close()


