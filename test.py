import random
import numpy as np
import Queue
import heapq

import math

import itertools

a = [6, 1, 3, 5, 2]

b = heapq.nlargest(2, a)

print [a.index(i) for i in b]

print sum(a)