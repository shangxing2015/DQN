from environment_markov_channel import Environment
from MyopicPolicy import MyopicPolicy
from config_2 import *
import time

T_THRESHOLD = 500000
PERIOD = 100

env = Environment()

brain = MyopicPolicy()

action = [i for i in range(N_SENSING)]

total = 0

fileName = 'log_myopic'

f = open(fileName,'w')

start_time = time.time()

for i in range(T_THRESHOLD):

    observation, reward, terminal = env.step(action)
    total += reward

    if GOOD_CHANNEL:
        action = brain.getAction_good(observation)
    else:
        action = brain.getAction_bad(observation)

    count = i+1

    if count % PERIOD == 0:
        accum_reward = total / float(count)
        duration = time.time() - start_time
        f.write('Index %d: accu_reward is %f, action is: %s and time duration is %f' % (count, accum_reward, str(action), duration))
        f.write('\n')



f.close()



