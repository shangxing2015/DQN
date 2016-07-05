from RadomPolicy import RandomPolciy
import numpy as np
from environment import Environment
from config import *
import time

#def preprocess(observation)

#change one-hot action vector to action action in the environment
def process(action):

    action_id = np.nonzero(action)[0]

    action_evn = list(ACTION_SPACE[action_id])

    return action_evn



#step 1: init BrainDQN
env = Environment(N_NODES, N_CHANNELS, N_NOISES)
brain = RandomPolciy()

fileName = 'log_bench'

action = np.array([1] + [0 for i in range(N_NODES-1)])
action_env = process(action)
observation, reward, terminal = env.step(action_env)




index = 0
total = 0

succ_user_prob = [0 for i in range(N_NODES+1)]

start_time = time.time()

period = 1

f = open(fileName, 'w')
while index <= 5000000:

    index += 1

    action = brain.getAction()
    action_env = process(action)

    observation, reward, terminal = env.step(action_env)

    user_num = N_NODES - action_env.count(-1)
    succ_user_prob[user_num] += 1

    total += reward
    duration = time.time()-start_time

    if index % period == 0:
        accum_reward = total / float(index)
        #sum = 0
        f.write('Index %d: accu_reward is %.2f, action is: %s and time duration is %.2f' % (index, accum_reward, str(action_env), duration))
        f.write('\n')

succ_user_prob = [float(i) / sum(succ_user_prob) for i in succ_user_prob]
f.write('succ_user_prob is : ' + str(succ_user_prob))

f.close()