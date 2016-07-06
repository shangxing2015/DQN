from RadomPolicy import RandomPolciy
import numpy as np
from environment_markov_channel import Environment
from config_2 import *
import time

#def preprocess(observation)

#change one-hot action vector to action action in the environment
def process(action):

    action_id = np.nonzero(action)[0]

    action_evn = list(ACTION_SPACE[int(action_id)])

    return action_evn



#step 1: init BrainDQN
env = Environment()
brain = RandomPolciy()

fileName = 'log_random'

action = np.zeros(int(ACTION_SIZE))
action[0] = 1
action_env = process(action)

observation, reward, terminal = env.step(action_env)



T_THRESHOLD = 500000
PERIOD = 10

count = 0
total = 0
total += reward



start_time = time.time()



f = open(fileName, 'w')
for i in range(T_THRESHOLD):

    count = i+1

    action = brain.getAction()
    action_env = process(action)

    observation, reward, terminal = env.step(action_env)


    total += reward
    duration = time.time()-start_time

    if count % PERIOD == 0:
        accum_reward = total / float(count)
        duration = time.time() - start_time
        f.write('Index %d: accu_reward is %f, action is: %s and time duration is %f' % (
        count, accum_reward, str(action_env), duration))
        f.write('\n')


f.close()