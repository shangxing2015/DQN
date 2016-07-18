import numpy as np
import math

p_matrix = [(0.6, 0.4), (0.2, 0.8)]

a = np.matrix(p_matrix)

b = a

s0 = p_matrix[1][0] / (p_matrix[0][1] + p_matrix[1][0])
s1 = 1-s0

stable_dist = np.array([s0, s1])

print stable_dist

for t in range(100):

    for i in range(t+1):

        b = b*a


    temp = b-stable_dist
    temp_1 = 0.5*np.sum(np.abs(temp),1)

    print t

    print b

    print temp_1

    if temp_1[0] < 0.00001 and temp_1[1] < 0.00001:
        print t
        break

