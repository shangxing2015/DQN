import numpy as np

epsilon = 0.00001

def cal_mix_time(p_matrix):


    a = np.matrix(p_matrix)

    b = a

    s0 = p_matrix[1][0] / (p_matrix[0][1] + p_matrix[1][0])
    s1 = 1 - s0

    stable_dist = np.array([s0, s1])

    # print stable_dist

    for t in range(100):

        for i in range(t + 1):
            b = b * a

        temp = b - stable_dist
        temp_1 = 0.5 * np.sum(np.abs(temp), 1)

        # print t
        #
        # print b
        #
        # print temp_1

        if temp_1[0] < epsilon and temp_1[1] < epsilon:
            #print t
            break

    return t
