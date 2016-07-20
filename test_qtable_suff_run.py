from test_qtable_stat_suff import *

finalResult = 'final_result_qtable_suff_cvg'
f_result = open(finalResult, 'w')

for i in range(1):
    # # CASE: same channels

    #
    # temp_prob = [(random.random(), random.random()) for j in range(int(N_CHANNELS))]
    # p_matrix = [[(x, 1 - x), (y, 1 - y)] for x, y in temp_prob]

    p_matrix = [[(0.8, 0.2), (0.2, 0.8)]] * (N_CHANNELS)

    # p_01_list = [0.8, 0.6, 0.4, 0.9, 0.8, 0.6, 0.7]
    # p_11_list = [0.6, 0.4, 0.2, 0.2, 0.4, 0.1, 0.3]
    #
    # p_matrix = []
    #
    # for k, l in zip(p_01_list, p_11_list):
    #     p_matrix.append([(1-k, k), (1-l, l)])


    f_result.write('test case ' + str(i) + '\n' + '\n')

    f_result.write('P Matrix is: \n')
    f_result.write(str(p_matrix))
    f_result.write('\n')
    f_result.write('\n')

    # file_whittle = 'log_whittle_cvg_2_jul_19'+str(i)
    file_qtable = 'log_q_table_suff_cvg_jul_20' + str(i)

    # run_whittleIndex(f_result, p_matrix, file_whittle)
    # f_result.write('\n')


    run_test(f_result, p_matrix, file_qtable)
    f_result.write('\n')
