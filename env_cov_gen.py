import numpy as np
import random
import math
import itertools

N_CORR = 2
N_IND = 1

n_corr_states = 2**N_CORR
n_ind_states = 2**N_IND

channel_state = [0,1]

S_A = list(itertools.product(channel_state, repeat = N_CORR))
S_B = list(itertools.product(channel_state, repeat = N_IND))




A = np.random.rand(n_corr_states,n_corr_states)
sum_A = np.sum(A, axis=1)


for i in range(len(sum_A)):

    A[i] = A[i]/sum_A[i]


B = np.random.rand(n_ind_states,n_ind_states)

sum_B = np.sum(B, axis=1)

for i in range(len(sum_B)):
    B[i] = B[i]/sum_B[i]


print('A')
print A

print('B')

print B



idx_a = random.randint(0, len(S_A)-1)
idx_b = random.randint(0, len(S_B)-1)

temp_A = S_A[idx_a]
temp_B = S_B[idx_b]

c0 = [temp_A[0]]
c1 = [temp_A[1]]
c2 = [temp_B[0]]

for i in range(100):

    P_A = A[idx_a]
    P_B = B[idx_b]


    P_A_accum = [sum(P_A[0:j]) for j in range(n_corr_states+1)]
    P_B_accum = [sum(P_B[0:j]) for j in range(n_ind_states+1)]




    rand_A = random.random()
    rand_B = random.random()

    for j in range(n_corr_states):
        if rand_A >= P_A_accum[j] and rand_A < P_A_accum[j+1]:
            idx_a = j
            break

    for j in range(n_ind_states):
        if rand_B >= P_B_accum[j] and rand_B < P_B_accum[j+1]:
            idx_b = j
            break


    temp_A = S_A[idx_a]
    temp_B = S_B[idx_b]

    c0.append(temp_A[0])
    c1.append(temp_A[1])
    c2.append(temp_B[0])




data = np.array([c0, c1, c2])

data = data.transpose()



# emp_cov = empirical_covariance(data, assume_centered=False)
#
# print emp_cov


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










