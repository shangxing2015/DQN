from config_2 import *
from itertools import product
import numpy as np

from qtable_learning import QAgent
from environment_markov_channel import Environment

T_threshold = 500000
PERIOD = 100

all_states_list = tuple(product(range(-1, 2), repeat=N_CHANNELS))
all_actions_list = tuple(product(range(0, N_CHANNELS), repeat=N_SENSING))
all_observations_list = tuple(product(all_states_list,
                                      repeat=AGENT_STATE_WINDOWS_SIZE))



"""
Explanation on parameters:

  state: a tuple of observations of length AGENT_STATE_WINDOWS_SIZE
  observation: a tuple of length N_CHANNELS, each entry is -1 (unknown), 0 (bad) or 1 (good)
  action: a tuple of length N_SENSING, each entry is a channel index and the first entry is the channel used to transmit data

Example: ( N_CHANNELS = 3, AGENT_STATE_WINDOWS_SIZE = 3, N_SENSING = 2):
  one observation may be like (-1,0,1)
  one state may be like ((0, 1, -1), (1, 1, -1), (-1, 0, 1))
  one action may be like (2, 1)
"""


def state_transition_function(state, action, observation):
  return state[1:] + (observation, )


def reward_function(state, action, observation):
  return observation[0]


def run_test(epsilon):
  env = Environment()

  init_state = tuple([tuple([-1 for i in xrange(N_CHANNELS)]) for j in
                      xrange(AGENT_STATE_WINDOWS_SIZE)])
  q_agent = QAgent(state_transition_function, init_state, all_actions_list, epsilon)


  total  = 0
  count = 0

  action_evn = [i for i in range(N_SENSING)] #inital action

  observation, reward, terminal = env.step(action_evn)
  total += reward

  #fileName = 'log_q_table' + round
  #f = open(fileName, 'w')


  while count <= T_threshold:
    count += 1
    observation = tuple(observation.tolist())
    action = q_agent.observe_and_act(observation, reward, count)

    action_evn = list(action)


    observation, reward, terminal = env.step(action_evn)
    total += reward

    #if (count+1) % PERIOD == 0:
      #accu_reward = total / float(count+1)

      #info_message = str(count+1) + '\t' + str(accu_reward) + '\t' + str(action)

      #f.write(info_message + '\n')


  return total/float(count+1)


  #f.close()

epsilon_list = np.arange(0.001, 1, 0.01)
epsilon_list = epsilon_list.tolist()
max_avg_reward = float('-infinity')
epsilon_max = -1
for i in epsilon_list:

   cur_avg_reward = run_test(i)
   if cur_avg_reward >= max_avg_reward:
     max_avg_reward = cur_avg_reward
     epsilon_max = i

print max_avg_reward
print i
