from MyopicPolicy import MyopicPolicy
from env_markov_distinct_channel import Environment
from config_2 import *
import time

def run_myopic(f_result, p_matrix = P_DISTINCT_MATRIX, fileName = 'log_myopic', good_channle=GOOD_CHANNEL):


    env = Environment(p_matrix)

    brain = MyopicPolicy()

    action = [i for i in range(N_SENSING)]

    total = 0

    fileName = fileName

    f = open(fileName,'w')

    start_time = time.time()

    for i in range(T_THRESHOLD):

        observation, reward, terminal = env.step(action)
        total += reward

        if good_channle:

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

    duration = time.time() - start_time
    count = i + 1
    accum_reward = total / float(count)
    duration = time.time() - start_time
    f_result.write('Myopic final accu_reward is %f and time duration is %f\n' % (accum_reward, duration))



