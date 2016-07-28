import itertools
import numpy as np


def combine_transition(chan_list_1, chan_list_2, A_1, A_2):


    chan_all_list = chan_list_1+chan_list_2

    chan_all_list.sort()

    chan_all_list_org = chan_list_1+chan_list_2

    chan_idx = [chan_all_list_org.index(i) for i in chan_all_list]

    dim = 2**(len(chan_list_1)+len(chan_list_2))

    all_matrix = np.zeros((dim,dim))


    idx_1 = [i for i in range(2**len(chan_list_1))]
    idx_2 = [i for i in range(2**len(chan_list_2))]

    idx_list_1 = list(itertools.product(idx_1, repeat = 2))
    idx_list_2 = list(itertools.product(idx_2, repeat = 2))

    for (i,j) in idx_list_1:

        p_1 = A_1[i][j]

        curr_1 = [0 for m in range(len(chan_list_1))]
        next_1 = [0 for m in range(len(chan_list_1))]
        bin_1 = [int(x) for x in bin(i)[2:]]
        bin_2 = [int(x) for x in bin(j)[2:]]

        for m in xrange(len(bin_1)):
            curr_1[-1-m] = bin_1[-1-m]

        for m in xrange(len(bin_2)):
            next_1[-1-m] = bin_2[-1-m]


        for (k,l) in idx_list_2:

            p_2 = A_2[k][l]

            curr_2 = [0 for m in range(len(chan_list_2))]
            next_2 = [0 for m in range(len(chan_list_2))]
            bin_1 = [int(x) for x in bin(k)[2:]]
            bin_2 = [int(x) for x in bin(l)[2:]]

            for m in xrange(len(bin_1)):
                curr_2[-1-m] = bin_1[-1-m]

            for m in xrange(len(bin_2)):
                next_2[-1-m] = bin_2[-1-m]

            curr_s = curr_1 + curr_2
            next_s = next_1 + next_2


            curr_s = [curr_s[m] for m in chan_idx]
            next_s = [next_s[m] for m in chan_idx]

            m = int(''.join(map(lambda x: str(x), curr_s)), 2)
            n = int(''.join(map(lambda x: str(x), next_s)), 2)

            all_matrix[m][n] = p_1*p_2

    return all_matrix




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


print(combine_transition(chan_list_1,chan_list_2,A_1, A_2))

