from config_2 import *

'''
single user myopic oplicy (reference: ''Indexability of Restless Bandit Problems and Optimality of Whittle Index of Dynamic Multichannel Access'')
at each timeslot:
    observe N_SENSING channels, and choose 1 channel (at the top of the queue) to transmit
'''


class MyopicPolicy():
    def __init__(self):
        self.n_channels = N_CHANNELS
        self.n_nodes = N_NODES
        self.action = [-1 for i in range(N_NODES)]
        self.queue = [i for i in range(self.n_channels)]
        self.n_sensing = N_SENSING

    def getAction_good(self, observation):  # p11 >= p01

        zero_indices = []

        for i in range(self.n_sensing):

            channel_state = observation[self.queue[i]]

            if channel_state == 0:
                zero_indices.append(i)
                self.queue.append(self.queue[i])

        self.queue = [self.queue[i] for i in range(len(self.queue)) if i not in zero_indices]

        return self.queue[0:self.n_sensing]  # which channel to sensing, and the first is the transmit channel

    def getAction_bad(self, observation):  # p11 < p01



        unobs_queue = self.queue[self.n_sensing:]

        unobs_queue.reverse()
        obs_queue = self.queue[0:self.n_sensing]

        one_indices = []

        for i in range(self.n_sensing):

            channel_state = observation[obs_queue[i]]

            if channel_state == 1:
                one_indices.append(i)

                unobs_queue.append(obs_queue[i])

        obs_queue = [obs_queue[i] for i in range(self.n_sensing) if i not in one_indices]

        self.queue = obs_queue + unobs_queue

        return self.queue[0:self.n_sensing]
