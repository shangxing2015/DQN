import time

from config_4 import *
from env_markov_distinct_channel import Environment
from whittleIndex import WhittleIndex


def run_whittleIndex(f_result, p_matrix=P_DISTINCT_MATRIX, fileName='log_whittle'):

    #fileName = fileName

    # f = open(fileName, 'w')
    #
    start_time = time.time()
    accum_reward = 0

    for j in range(T_TIMES):

        total = 0

        gamma = GAMMA

        env = Environment(p_matrix)

        print p_matrix

        brain = WhittleIndex(B, gamma, p_matrix)

        action = [i for i in range(N_SENSING)]



        for i in range(T_EVAL):

            observation, reward, terminal = env.step(action)
            total += reward*(gamma**i)

            action = brain.getAction(action, observation, 0)

            count = i + 1

            # if count % PERIOD == 0:
            #     accum_reward = total
            #     duration = time.time() - start_time
            #     f.write('Index %d: accu_reward is %f, action is: %s and time duration is %f' % (
            #         count, accum_reward, str(action), duration))
            #     f.write('\n')

        # f.close()

        duration = time.time() - start_time
        count = i + 1
        accum_reward += total
        duration = time.time() - start_time
        f_result.write('Whittle Index final accu_reward is %f and time duration is %f\n' % (total, duration))

    return accum_reward
