__author__ = 'shangxing'

from Async_DQN_2 import Async_DQN
from test_whittleIndex import *


def run_async_qlearning(f_result, p_matrix=P_DISTINCT_MATRIX, fileName='log_asyn_qlearning_'):
    brain = Async_DQN()

    brain.create_thread(p_matrix, fileName, f_result)


p_matrix = [[(0.5, 0.5), (0.45, 0.55)]] * N_CHANNELS

file_async_qlearning = 'log_async_qlearning_identical_2_near' + '_'

finalResult = 'final_result_identical_channel_2_near'

file_whittle = 'log_whittle_identical_2_near'

f_result = open(finalResult, 'w')

f_result.write('test case ' + str(i) + '\n' + '\n')

f_result.write('P Matrix is: \n')
f_result.write(str(p_matrix))
f_result.write('\n')
f_result.write('\n')

run_async_qlearning(f_result, p_matrix, file_async_qlearning)

f_result.write('\n')


# run_whittleIndex(f_result, p_matrix, file_whittle)
# f_result.write('\n')
