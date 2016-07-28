__author__ = 'shangxing'


import math
import random



class Environment():
    """

    A state is an integer, in [0...2^{n_channels}-1].
    """

    def __init__(self, config):
        self.n_nodes = config['n_nodes']
        self.p_matrix = config['p_matrix']
        self.n_channels = config['n_channels']
        self.chan_list = config['channel_list']
        self.n_subnets = config['n_subnets']


        for p in self.p_matrix:

            for p_sub in p:
                count = 0
                for i in xrange(len(p_sub)):
                    count += p_sub[i]
                    p_sub[i] = count

        self.chan_idx = self._get_channel_order()



        self.current_state_int = [random.randint(0, 2**len(self.chan_list[i])-1) for i in range(self.n_subnets)]

        bin_temp = []

        for i in range(self.n_subnets):

            state = self.current_state_int[i]

            bin_temp += (self._decode_state(state, len(self.chan_list[i])))



        self.current_state_bin = [bin_temp[i] for i in self.chan_idx]





    def _get_channel_order(self):

        channel_all_list = []

        for i in self.chan_list:
            channel_all_list += i

        channel_all_list_org = channel_all_list
        channel_all_list.sort()

        chan_idx = [channel_all_list_org.index(i) for i in channel_all_list]

        return chan_idx




    def _decode_state(self, integer, size):
        result = [0 for i in xrange(size)]
        binary = [int(x) for x in bin(integer)[2:]]
        for i in xrange(len(binary)):
            result[-1-i] = binary[-1-i]
        return result

    def _encode_state(self, list):
        return int(''.join(map(lambda x: str(x), list)), 2)

    @property
    def observation_size(self):
        return self.n_channels


    def _state_transit(self):

        temp_state = []

        for i in range(self.n_subnets):

            p = self.p_matrix[i][self.current_state_int[i]]

            tmp, flag = random.random(), False

            n_states = 2**len(self.chan_list[i])


            for j in range(n_states-1):
                if tmp > p[j] and tmp <= p[j+1]:
                    flag = True
                    break

            if flag:
                temp_state.append(j+1)
            else:
                temp_state.append(0)



        self.current_state_int = temp_state


        bin_temp = []

        for i in range(self.n_subnets):

            state = self.current_state_int[i]

            bin_temp += (self._decode_state(state, len(self.chan_list[i])))



        self.current_state_bin = [bin_temp[i] for i in self.chan_idx]



    def get_state(self):

        state = self.current_state_bin

        self._state_transit()

        return state


    def step(self, action):
        '''
        :param
            action: an array = [0,...,0, 1, 0,...,0]

        :return:
            obsservation: an array contains the observed info of each channel
            reward: reward from the chosen transmission channel
            terminal: whether the game is over or not (Always False)
        '''
        state_list = self.current_state_bin
        for action_id in xrange(len(action)):
            if action[action_id] == 1:
                break
        observation = [0 if j != action_id
                       else (-1 + 2 * state_list[action_id])
                       for j in xrange(self.n_channels)]
        reward = 1 if state_list[action_id] == 1 else -0.01

        terminal = False
        self._state_transit()

        return observation, reward, terminal