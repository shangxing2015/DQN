import time
from itertools import product

from config_4 import *
from env_markov_distinct_channel import Environment
from qtable_learning_cvg import QAgent

# all_states_list = tuple(product(range(-1, 2), repeat=N_CHANNELS))
# all_actions_list = tuple(product(range(0, N_CHANNELS), repeat=N_SENSING))
# all_observations_list = tuple(product(all_states_list, repeat=AGENT_STATE_WINDOWS_SIZE))



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


# define the state transition of q learning
def state_transition_function(state, action, observation):
    return state[1:] + (observation,)


# reward function
def reward_function(state, action, observation):
    return observation[0]


# main test: training + evaluation
def run_test(f_result, p_matrix=P_MATRIX, fileName='log_q_table', history=AGENT_STATE_WINDOWS_SIZE):
    all_actions_list = tuple(product(range(0, N_CHANNELS), repeat=N_SENSING))

    env = Environment(p_matrix)

    # init_state = tuple([tuple([-1 for i in xrange(N_CHANNELS)]) for j in xrange(history)])


    init_state = tuple([tuple([0, -1])])
    q_agent = QAgent(state_transition_function, init_state, all_actions_list)

    total = 0

    action_evn = [i for i in range(N_SENSING)]  # inital action

    observation, reward, terminal = env.step(action_evn)
    total += reward

    fileName = fileName
    f = open(fileName, 'w')

    start_time = time.time()

    prev_value_dict = {}
    count_cvg = 0

    # training
    for i in range(T_THRESHOLD):
        count = i + 1
        observation = tuple(observation.tolist())
        action, prev_value_dict, count_cvg = q_agent.observe_and_act(observation, reward, count, prev_value_dict,
                                                                     count_cvg)

        # if count_cvg == T_CVG:
        #     print 'policy converged, and round of training %d' % i
        #
        #     break


        action_evn = list(action)

        observation, reward, terminal = env.step(action_evn)
        total += reward

        if (count) % PERIOD == 0:
            accum_reward = total / float(count)

            duration = time.time() - start_time
            f.write('Index %d: accu_reward is %f, action is: %s and time duration is %f' % (
                count, accum_reward, str(action), duration))
            f.write('\n')
    f.close()

    f_result.write('count_cvg is %d' % count_cvg)
    f_result.write('\n')
    f_result.write(str(prev_value_dict))
    f_result.write('\n')

    # evaluation
    total = 0

    fileName = fileName + '_target'
    f = open(fileName, 'w')

    start_time = time.time()

    for i in range(T_EVAL):
        count = i + 1

        if type(observation).__module__ == 'numpy':
            observation = tuple(observation.tolist())

        else:
            print 'train finished'
        action = q_agent.target_observe_and_act(observation, reward, count)

        action_evn = list(action)

        # if count <= 50:
        #
        #   print('observation')
        #   print(observation)
        #
        #   print('action')
        #   print(action_evn)

        observation, reward, terminal = env.step(action_evn)
        total += reward

        if (count) % PERIOD == 0:
            accum_reward = total / float(count)

            duration = time.time() - start_time
            f.write('Index %d: accu_reward is %f, action is: %s and time duration is %f' % (
                count, accum_reward, str(action), duration))
            f.write('\n')
    f.close()

    count = i + 1
    accum_reward = total / float(count)
    duration = time.time() - start_time
    f_result.write('Q table final accu_reward is %f and time duration is %f\n' % (accum_reward, duration))
