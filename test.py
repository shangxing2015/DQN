import random
import numpy as np
import Queue


import itertools

a = [(random.random(), random.random()) for i in range (3)]

b = [[(x, 1-x), (y, 1-y)] for x, y in a]

print b