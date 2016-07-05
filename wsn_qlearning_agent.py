from config import *
from itertools import product

from qtable_learning import QAgent
from environment import Environment

all_states_list = tuple(product(range(0, 2), repeat=N_NODES))
all_actions_list = tuple(product(range(-1, N_CHANNELS), repeat=N_NODES))
all_observations_list = tuple(product(all_states_list,
                                      repeat=AGENT_STATE_WINDOWS_SIZE))

env = Environment(N_NODES, N_CHANNELS, N_NOISES)

"""
Explanation on parameters:

  state: a tuple of observations of length AGENT_STATE_WINDOWS_SIZE
  observation: a tuple of length N_NODES, each entry is 1 or 0 which indicates
               whether the node's transition success or not
  action: a tuple of length N_NODES, each entry is either -1 or some int
               which refers to channel index

Example: (N_NODES = 2, N_CHANNELS = 3, AGENT_STATE_WINDOWS_SIZE = 3):
  one observation may be like (0, 0)
  one state may be like ((0, 1), (1, 1), (0, 1))
  one action may be like (2, -1)
"""


def state_transition_function(state, action, observation):
  return state[1:] + (observation, )


def reward_function(state, action, observation):
  return reduce(lambda x, y: x + y, observation)


def run_test(total_stages=STAGES, stage_rounds=ROUNDS, output_f=OUTPUT_F):
  init_state = tuple([tuple([0 for i in xrange(N_NODES)]) for j in
                      xrange(AGENT_STATE_WINDOWS_SIZE)])
  q_agent = QAgent(state_transition_function, reward_function,
                   init_state, all_actions_list)
  init_observation = tuple([0 for i in xrange(N_NODES)])
  observation = init_observation

  f = None
  if output_f is not None:
    f = open(output_f, 'w')

  for i in xrange(total_stages):
    count = 0
    for j in xrange(stage_rounds):
      action, reward = q_agent.observe_and_act(observation)
      observation = env.process(action)
      # print len(q_agent.Q), action, observation, reward
      count += reward

    tx_rate = count / float(stage_rounds * N_NODES)
    info_message = str(i) + '\t' + str(tx_rate)
    print '[INFO]', info_message
    if f is not None:
      f.write(info_message + '\n')

  if f is not None:
    f.close()


def reset_env():
  global env
  env = Environment(N_NODES, N_CHANNELS, N_NOISES)


if __name__ == '__main__':
  run_test()
