from test_async_learner import *
from test_myopic import run_myopic

finalResult = 'final_result_identical_channel_2_inverse_sensing'
f_result = open(finalResult, 'w')

for i in range(1):

    # # CASE: same channels


    p_matrix = [[(0.9, 0.1), (0.1, 0.9)]] * N_CHANNELS

    print p_matrix

    if p_matrix[0][0][1] >= p_matrix[0][1][1]:
        good_channel = False
    else:
        good_channel = True

    print good_channel

    f_result.write('test case ' + str(i) + '\n' + '\n')

    f_result.write('P Matrix is: \n')
    f_result.write(str(p_matrix))
    f_result.write('\n')
    f_result.write('\n')

    lz = list()

    file_myopic = 'log_myopic_identical_jul_22' + str(i)

    run_myopic(f_result, p_matrix, file_myopic, good_channel)
    f_result.write('\n')
