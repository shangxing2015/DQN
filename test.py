from config_2 import *
from test_random import *
from test_whittleIndex import *
from test_async_learner import *
from test_myopic import run_myopic
import random
from test_LZ import *


finalResult = 'final_result_test_myopic_whittle'
f_result = open(finalResult, 'w')

for i in range(1):

    # CASE: same channels
    temp_prob = [(random.random(), random.random())]
    p_matrix = [[(0.0755316539634392, 0.9244683460365608), (0.15494970216977244, 0.8450502978302276)], [(0.0755316539634392, 0.9244683460365608), (0.15494970216977244, 0.8450502978302276)], [(0.0755316539634392, 0.9244683460365608), (0.15494970216977244, 0.8450502978302276)], [(0.0755316539634392, 0.9244683460365608), (0.15494970216977244, 0.8450502978302276)], [(0.0755316539634392, 0.9244683460365608), (0.15494970216977244, 0.8450502978302276)], [(0.0755316539634392, 0.9244683460365608), (0.15494970216977244, 0.8450502978302276)]]

    temp_prob = p_matrix[0]


    if temp_prob[1][1] >= temp_prob[0][1]:
        good_channel = True
    else:
        good_channel = False




    # CASE: distinct channels
    # temp_prob = [(random.random(), random.random()) for j in range(int(N_CHANNELS))]
    # p_matrix = [[(x, 1 - x), (y, 1 - y)] for x, y in temp_prob]
    # good_channel = True

    f_result.write('test case ' + str(i) + '\n' + '\n')

    f_result.write('P Matrix is: \n')
    f_result.write(str(p_matrix))
    f_result.write('\n')
    f_result.write('\n')


    file_myopic = 'log_myopic_temp_' + str(i)
    file_whittle = 'log_whittle_'+str(i)
    file_random = 'log_random_'+str(i)
    file_async_qlearning = 'log_async_qlearning_'+str(i)+'_'



    run_myopic(f_result, p_matrix, file_myopic, good_channel)
    f_result.write('\n')

    run_whittleIndex(f_result, p_matrix, file_whittle)
    f_result.write('\n')



#f_result.close()



