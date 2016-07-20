from test_async_learner import *
from test_whittleIndex import *

finalResult = 'final_result_identical_channel_3_inverse_sensing'
f_result = open(finalResult, 'w')

for i in range(1):
    # # CASE: same channels


    p_matrix = [[(0.6, 0.4), (0.2, 0.8)]] * N_CHANNELS

    print N_CHANNELS

    f_result.write('test case ' + str(i) + '\n' + '\n')

    f_result.write('P Matrix is: \n')
    f_result.write(str(p_matrix))
    f_result.write('\n')
    f_result.write('\n')

    lz = list()

    file_whittle = 'log_whittle_identical_jul_17' + str(i)

    run_whittleIndex(f_result, p_matrix, file_whittle)
    f_result.write('\n')



# f_result.close()
