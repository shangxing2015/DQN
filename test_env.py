from create_pmatrix import  combine_transition

from env import Environment

import itertools

import random

import numpy as np

chan_list_1 = [0,1]

chan_list_2 = [2]

N_CORR = 2
N_IND = 1

n_corr_states = 2**N_CORR
n_ind_states = 2**N_IND

channel_state = [0,1]

S_A = list(itertools.product(channel_state, repeat = N_CORR))
S_B = list(itertools.product(channel_state, repeat = N_IND))




A_1 = np.random.rand(n_corr_states,n_corr_states)
sum_A_1 = np.sum(A_1, axis=1)


for i in range(len(sum_A_1)):

    A_1[i] = A_1[i]/sum_A_1[i]


A_2 = np.random.rand(n_corr_states,n_ind_states)
sum_A_2 = np.sum(A_2, axis=1)


for i in range(len(sum_A_2)):

    A_2[i] = A_2[i]/sum_A_2[i]


p_matrix = combine_transition(chan_list_1,chan_list_2,A_1, A_2)

config = {'n_nodes': 3, 'p_matrix': p_matrix, 'n_channels': 3}

env = Environment(config)

history = []

for i in range(2000):

    history.append(env.get_state())


data = np.array(history)

print data


corr = np.corrcoef(data, rowvar=False)

print corr

# print cov_
# print prec_


threshold = 0.1

low_idx = corr < threshold
high_idx = corr >= threshold

corr[low_idx] = 0
corr[high_idx] = 1

print corr


