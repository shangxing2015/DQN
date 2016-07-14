__author__ = 'shangxing'

from testNerualNetwork import Async_DQN
from config_4 import *

def run_async_qlearning(f_result, p_matrix = P_DISTINCT_MATRIX, fileName='log_asyn_qlearning_'):

    brain = Async_DQN()

    brain.create_thread(p_matrix, fileName, f_result)



p_matrix = [[(0.6, 0.4), (0.2, 0.8)]] * N_CHANNELS




file_myopic = 'log_myopic_identical_' + str(i)
file_whittle = 'log_whittle_identical_'+str(i)
file_random = 'log_random_identical_'+str(i)
file_async_qlearning = 'log_async_qlearning_neural_'+str(i)+'_'

finalResult = 'final_result_identical_channel_neural'
f_result = open(finalResult, 'w')

# run_whittleIndex(f_result, p_matrix, file_whittle)
# f_result.write('\n')

run_async_qlearning(f_result, p_matrix, file_async_qlearning)

f_result.write('\n')