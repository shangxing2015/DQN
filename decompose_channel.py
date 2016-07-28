import math

A1 = [[ 0.28522747,  0.71477253],
 [ 0.29006761,  0.70993239]]
A2 = [[ 0.35738896, 0.14469056,  0.35909998,  0.1388205 ],
 [ 0.25167018,  0.31032806 , 0.20215486,  0.23584689],
 [ 0.25972398,  0.32713714 , 0.31540113 , 0.09773775],
 [ 0.24322782,  0.23561067,  0.22061496, 0.30054656]]


print A1
print A2

C = [[] for i in range(8)]

for i in range(len(C)):

    idx_1 = math.floor(i/2)
    idx_2 = i % 2

    print idx_1

    for entry_1 in A2[int(idx_1)]:



        for entry_2 in A1[int(idx_2)]:

            C[i].append(entry_1*entry_2)



