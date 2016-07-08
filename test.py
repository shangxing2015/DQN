import random
import numpy as np
import Queue
import heapq

import math

import itertools

temp_prob = [(random.random(), random.random()) for i in range(int(3))]
P_DISTINCT_MATRIX = [[(x, 1 - x), (y, 1 - y)] for x, y in temp_prob]

print P_DISTINCT_MATRIX

a = np.array([8,3,2])

print np.max(a)