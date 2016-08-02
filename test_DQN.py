__author__ = 'shangxing'

import sys

from BrainDQN_Goolge_Nature import BrainDQN

sys.path.append("game/")
import numpy as np
from env_markov_distinct_channel import Environment
from config_2 import *
import time


# def preprocess(observation)

# change one-hot action vector to action action in the environment
def process(action):
    action_id = np.nonzero(action)[0]

    action_evn = list(ACTION_SPACE[int(action_id)])

    return action_evn


p_matrix = [[(0.6, 0.4), (0.2, 0.8)]] * N_CHANNELS

# step 1: init BrainDQN
env = Environment(p_matrix)

brain = BrainDQN()

fileName = 'log_DQN_temp'

action = np.zeros(int(ACTION_SIZE))
action[0] = 1
action_env = process(action)
observation, reward, terminal = env.step(action_env)

brain.setInitState(observation)

index = 0
total = 0

start_time = time.time()

f = open(fileName, 'w')

# step 2: play the game while learning
while index <= T_THRESHOLD:

    index += 1

    action = brain.getAction()
    action_env = process(action)
    observation, reward, terminal = env.step(action_env)

    brain.setPerception(observation, action, reward, terminal)

    # for writing to the file



    total += reward
    duration = time.time() - start_time

    if index % PERIOD == 0:
        accum_reward = total / float(index)
        f.write('Index %d: accu_reward is %.2f, action is: %s and time duration is %.2f' % (
            index, accum_reward, str(action_env), duration))
        f.write('\n')

f.close()

# final evaluation


total = 0

index = 0

while index <= 50:
    index = index + 1
    action = brain.target_get_action()

    action_env = process(action)

    if index >= 0:
        print('observation')
        print(observation)

        print('action')
        print(action_env)

    observation, reward, terminal = env.step(action_env)

    brain.setPerception(observation, action, reward, terminal)
