import itertools
import numpy as np


def combine_transition(chan_list_1, chan_list_2, A_1, A_2):


    chan_all_list = chan_list_1+chan_list_2

    chan_all_list.sort()

    chan_all_list_org = chan_list_1+chan_list_2

    chan_idx = [chan_all_list_org.index(i) for i in chan_all_list]

    print chan_idx

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






