__author__ = 'shangxing'

from Async_DQN_2 import Async_DQN
from config_2 import *

def run_async_qlearning(f_result, p_matrix = P_DISTINCT_MATRIX, fileName='log_asyn_qlearning_'):

    brain = Async_DQN()

    brain.create_thread(p_matrix, fileName, f_result)



p_matrix = [[(0.6, 0.4), (0.2, 0.8)]] * N_CHANNELS





file_async_qlearning = 'log_async_qlearning_identical_3'+'_'

finalResult = 'final_result_identical_channel_3'
f_result = open(finalResult, 'w')

# run_whittleIndex(f_result, p_matrix, file_whittle)
# f_result.write('\n')

run_async_qlearning(f_result, p_matrix, file_async_qlearning)

f_result.write('\n')