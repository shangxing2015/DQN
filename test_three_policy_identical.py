import random

from test_LZ import *
from test_async_learner import *
from test_myopic import run_myopic
from test_random import *
from test_whittleIndex import *

finalResult = 'final_result_identical_channel_1_sensing'
f_result = open(finalResult, 'w')

for i in range(10):

    # # CASE: same channels
    temp_prob = [(random.random(), random.random())]
    p_matrix = [[(x, 1 - x), (y, 1 - y)] for x, y in temp_prob] * N_CHANNELS

    if temp_prob[0][0] >= temp_prob[0][1]:
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

    lz = list()

    for j in range(N_CHANNELS):
        p = p_matrix[j]
        lz.append(lz_complexity_markov(p))

    f_result.write('the LZ complexity of each channel is: %s, and the the average LZ complexity is %f \n' % (
    str(lz), sum(lz) / float(len(lz))))
    f_result.write('\n')

    file_myopic = 'log_myopic_identical_' + str(i)
    file_whittle = 'log_whittle_identical_' + str(i)
    file_random = 'log_random_identical_' + str(i)
    file_async_qlearning = 'log_async_qlearning_identical_' + str(i) + '_'

    run_random(f_result, p_matrix, file_random)

    f_result.write('\n')

    run_myopic(f_result, p_matrix, file_myopic, good_channel)
    f_result.write('\n')

    run_whittleIndex(f_result, p_matrix, file_whittle)
    f_result.write('\n')

    run_async_qlearning(f_result, p_matrix, file_async_qlearning)

    f_result.write('\n')

# f_result.close()
