from Async_DQN_2 import Async_DQN
from config_2 import *

def run_async_qlearning(f_result, p_matrix = P_DISTINCT_MATRIX, fileName='log_asyn_qlearning_'):

    brain = Async_DQN()

    brain.create_thread(p_matrix, fileName, f_result)
