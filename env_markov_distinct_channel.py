import random

import numpy as np

from config_4 import *

'''
N_CHANNEL channels with identical 2-state Markov Transition Matrix
observation by default is -1. And good: 1; bad: 0.
'''


class Environment:
    def __init__(self, p_matrix):
        self.n_channels = N_CHANNELS
        self.n_nodes = N_NODES
        self.p_matrix = p_matrix
        self.current_state = [random.randint(0, 1) for i in range(self.n_channels)]
        self.next_state = self.current_state

    def _state_transit(self):

        for i in range(self.n_channels):

            temp = random.random()

            if self.current_state[i] == 0:
                if temp < self.p_matrix[i][0][0]:
                    self.next_state[i] = 0
                else:
                    self.next_state[i] = 1
            else:
                if temp < self.p_matrix[i][1][0]:
                    self.next_state[i] = 0
                else:
                    self.next_state[i] = 1

        self.current_state = self.next_state

    def step(self, action):
        '''
        :param
            action: an array contains the indices of to-be-sensed channels and use the first channel to transmit
        :return:
            observation: an array contains the observed info of each channel
            reward: reward from the chosen transmission channel
            terminal: whether the game is over or not (Always False)
        '''

        obs_state = [self.current_state[i] for i in action]
        reward = sum(obs_state)  # sum of all sensed channels
        # reward = reward
        observation = [-1 for i in range(self.n_channels)]
        terminal = False

        # print 'current state'
        # print self.current_state

        # print 'action'
        # print action

        # print action


        for i in action:

            channel_id = i

            # print 'observe state'
            # print self.current_state[channel_id]

            if self.current_state[channel_id] == 1:

                observation[channel_id] = 1
            else:
                observation[channel_id] = 0

        # print 'observation'
        # print observation

        observation = np.array(observation)

        self._state_transit()

        return observation, reward, terminal
