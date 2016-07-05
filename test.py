import random
import numpy as np
import Queue


import itertools

a = [1,2,3,2,3]
b = np.argwhere(a == np.amax(a))



c = b.flatten().tolist()




all_states_list = tuple(itertools.product(range(-1, 2), repeat=2))
all_observations_list = tuple(itertools.product(all_states_list,
                                      repeat=3))

print all_states_list


print all_observations_list

init_state = tuple([tuple([-1 for i in xrange(2)]) for j in
                    xrange(3)])

print init_state

print init_state[1:] + (tuple([1,2,3]),)

e = np.arange(0.1, 1, 0.1)

f = e.tolist()

g = range(1, 5, 1)

h = list(itertools.product(f,g))

for i,j in h:
    print i
    print j
