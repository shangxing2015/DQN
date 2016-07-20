__author__ = 'Shangxing'

'''
Calculate the LZ complexity of a temporal sequence

Reference: 'Easily calculable measure for the complexity of spatiotemporal patterns' --- Fig. 1
'''

import math


# for real data
def lz_complexity_general(s):
    c = 1
    l = 1
    i = 0
    k = 1
    k_max = 1
    n = len(s)

    while True:
        if s[i + k - 1] == s[l + k - 1]:

            k = k + 1

            if l + k >= n:
                c = c + 1
                break
            else:
                continue
        else:
            if k > k_max:
                k_max = k
            i = i + 1
            if i == l:
                c = c + 1
                l = l + k_max

                if l + 1 > n:
                    break
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
    return c


# for a ergodic Markov process
# reference: 'Complexity of spectrum activity and benefits of reinforcement learning for dynamic channel selection'

def lz_complexity_markov(p):
    s = [(1 - p[1][1]) / ((1 - p[0][0]) + (1 - p[1][1])), (1 - p[0][0]) / ((1 - p[0][0]) + (1 - p[1][1]))]

    h = 0

    for i in range(len(p)):
        for j in range(len(p)):
            h -= s[i] * p[i][j] * math.log(p[i][j], 2)

    return h

#
# lz = lz_complexity_general('0101010101')
# n = 100000000
#
# normalize = n/math.log(n, 2)
#
# print lz
#
# print lz/normalize
#
# print lz_complexity_markov([(0.5, 0.5), (0.5, 0.5)])
