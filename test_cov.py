__author__ = 'shangxing'

import numpy as np
import random
from sklearn.covariance import empirical_covariance, GraphLassoCV


def cov(rand_A, rand_B, idx_a, idx_b):
    A = np.array([[0.35738896, 0.14469056, 0.35909998, 0.1388205],
                  [0.25167018, 0.31032806, 0.20215486, 0.23584689],
                  [0.25972398, 0.32713714, 0.31540113, 0.09773775],
                  [0.24322782, 0.23561067, 0.22061496, 0.30054656]])
    B = np.array([[0.28522747, 0.71477253],
                  [0.29006761, 0.70993239]])

    # A = np.random.rand(4,4)
    # sum_A = np.sum(A, axis=1)
    #
    # for i in range(4):
    #
    #     A[i] = A[i]/sum_A[i]
    #
    #
    # B = np.random.rand(2,2)
    # sum_B = np.sum(B, axis=1)
    #
    # for i in range(2):
    #     B[i] = B[i]/sum_B[i]



    S_A = [(0, 0), (0, 1), (1, 0), (1, 1)]
    S_B = [0, 1]

    temp_A = S_A[idx_a]
    temp_B = S_B[idx_b]

    P_A = A[idx_a]
    P_B = B[idx_b]

    if rand_A <= P_A[0]:

        temp_A = S_A[0]
        idx_a = 0

    elif rand_A <= sum(P_A[0:2]) and rand_A > P_A[0]:

        temp_A = S_A[1]
        idx_a = 1

    elif rand_A <= sum(P_A[0:3]) and rand_A > sum(P_A[0:2]):

        temp_A = S_A[2]
        idx_a = 2

    else:
        temp_A = S_A[3]
        idx_a = 3

    if rand_B <= P_B[0]:
        temp_B = S_B[0]
        idx_b = 0
    else:
        temp_B = S_B[1]
        idx_b = 1

    print [temp_A[0], temp_A[1], temp_B]
