from test_qtable import *
from test_whittleIndex import *

finalResult = 'final_result_whittle_and_qtable_dinstinct_7_channel'
f_result = open(finalResult, 'w')

for i in range(10):

    # # CASE: same channels

    #
    # temp_prob = [(random.random(), random.random()) for j in range(int(N_CHANNELS))]
    # p_matrix = [[(x, 1 - x), (y, 1 - y)] for x, y in temp_prob]

    # p_matrix = [[(0.2, 0.8), (0.6, 0.4)]] * (N_CHANNELS+i)

    p_01_list = [0.8, 0.6, 0.4, 0.9, 0.8, 0.6, 0.7]
    p_11_list = [0.6, 0.4, 0.2, 0.2, 0.4, 0.1, 0.3]

    p_matrix = []

    for k, l in zip(p_01_list, p_11_list):
        p_matrix.append([(1 - k, k), (1 - l, l)])

    f_result.write('test case ' + str(i) + '\n' + '\n')

    f_result.write('P Matrix is: \n')
    f_result.write(str(p_matrix))
    f_result.write('\n')
    f_result.write('\n')

    file_whittle = 'log_whittle_distinct_7_jul_18' + str(i)
    file_qtable = 'log_q_table_distinct_7_jul_18' + str(i)

    run_whittleIndex(f_result, p_matrix, file_whittle)
    f_result.write('\n')

    run_test(f_result, p_matrix, file_qtable)
    f_result.write('\n')
