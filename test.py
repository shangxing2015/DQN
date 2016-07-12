from config_2 import *
from test_random import *
from test_whittleIndex import *
from test_async_learner import *
from test_myopic import run_myopic
import random
from test_LZ import *


finalResult = 'final_result_identical_channel_testing'
f_result = open(finalResult, 'w')

for i in range(1):

    # # CASE: same channels
    #temp_prob = [(random.random(), random.random())]
    p_matrix = [[(0.6, 0.4), (0.2, 0.8)] ] * N_CHANNELS

    good_channel = True


    f_result.write('test case ' + str(i) + '\n' + '\n')

    f_result.write('P Matrix is: \n')
    f_result.write(str(p_matrix))
    f_result.write('\n')
    f_result.write('\n')


    file_myopic = 'log_myopic_identical_' + str(i)
    file_whittle = 'log_whittle_identical_target_'+str(i)
    file_random = 'log_random_identical_'+str(i)
    file_async_qlearning = 'log_async_qlearning_identical_target_'+str(i)+'_'

    # run_random(f_result, p_matrix, file_random)
    #
    # f_result.write('\n')

    #run_myopic(f_result, p_matrix, file_myopic, good_channel)
    #f_result.write('\n')

    run_whittleIndex(f_result, p_matrix, file_whittle)
    f_result.write('\n')

    run_async_qlearning(f_result, p_matrix, file_async_qlearning)

    f_result.write('\n')

#f_result.close()



