import math
import random


class Environment():
    """

    A state is an integer, in [0...2^{n_channels}].
    """

    def __init__(self, config):
        self.n_nodes = config['n_nodes']
        self.p_matrix = config['p_matrix']
        self.n_channels = config['n_channels']

        for p in self.p_matrix:
            count = 0
            for i in xrange(len(p)):
                count += p[i]
                p[i] = count

        self.current_state = 0
        self.n_states = int(math.pow(2, self.n_channels))

    def _decode_state(self, integer):
        result = [0 for i in xrange(self.n_channels)]
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
        p = self.p_matrix[self.current_state]
        tmp, flag = random.random(), False
        for i in range(self.n_states-1):
            if tmp >= p[i] and tmp <p[i+1]:
                flag = True
                break

        if flag:
            self.current_state = i+1
        else:
            self.current_state = 0


    def get_state(self):

        state = self._decode_state(self.current_state)

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
        state_list = self._decode_state(self.current_state)
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