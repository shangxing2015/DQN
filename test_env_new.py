__author__ = 'shangxing'

from env_new import Environment
import numpy as np

A1 = [[ 0.28522747,  0.71477253],
 [0.70993239, 0.29006761 ]]

A2 = [[ 0.35738896, 0.14469056,  0.35909998,  0.1388205 ],
 [ 0.25167018,  0.31032806 , 0.20215486,  0.23584689],
 [ 0.25972398,  0.32713714 , 0.31540113 , 0.09773775],
 [ 0.24322782,  0.23561067,  0.22061496, 0.30054656]]

config = {}

config['p_matrix'] =[A2, A1]
config['n_nodes'] = 1
config['n_channels'] = 3
config['channel_list'] = [[0,1], [2]]
config['n_subnets'] = 2

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


threshold = 0.03

low_idx = abs(corr) < threshold
high_idx = abs(corr) >= threshold

corr[low_idx] = 0
corr[high_idx] = 1

print corr