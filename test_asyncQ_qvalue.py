__author__ = 'shangxing'

from async_3 import Async_DQN
from config_2 import *
from test_whittleIndex import *

def run_async_qlearning(f_result, p_matrix = P_DISTINCT_MATRIX, fileName='log_asyn_qlearning_'):

    brain = Async_DQN()

    brain.create_thread(p_matrix, fileName, f_result)



p_matrix = [[(0.6, 0.4), (0.2, 0.8)]] * N_CHANNELS





file_async_qlearning = 'log_async_qlearning_qvalue'+'_'

finalResult = 'final_result_qvalue'



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


