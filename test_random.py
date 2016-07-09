from RadomPolicy import RandomPolciy
import numpy as np
from env_markov_distinct_channel import Environment
from config_2 import *
import time

#def preprocess(observation)

#change one-hot action vector to action action in the environment
def _process(action):

    action_id = np.nonzero(action)[0]

    action_evn = list(ACTION_SPACE[int(action_id)])

    return action_evn

def run_random(f_result, p_matrix = P_DISTINCT_MATRIX, fileName = 'log_random'):

    #step 1: init BrainDQN
    env = Environment(p_matrix)
    brain = RandomPolciy()

    fileName = fileName

    action = np.zeros(int(ACTION_SIZE))
    action[0] = 1
    action_env = _process(action)

    observation, reward, terminal = env.step(action_env)

    count = 0
    total = 0
    total += reward



    start_time = time.time()



    f = open(fileName, 'w')
    for i in range(T_THRESHOLD):

        count = i+1

        action = brain.getAction()
        action_env = _process(action)

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

    duration = time.time() - start_time
    count = i + 1
    accum_reward = total / float(count)
    duration = time.time() - start_time
    f_result.write('Random final accu_reward is %f and time duration is %f\n' % (accum_reward, duration))