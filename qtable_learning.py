from random import Random

from config_2 import *
from util import Counter

OBSERVE = 20000  # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon #700000
FINAL_EPSILON = 0.1  # final value of epsilon: for epsilon annealing
INITIAL_EPSILON = 1  # starting value of epsilon
INITIAL_ALPHA = 0.1


class QAgent:
    def __init__(self, transition_func, init_state, all_actions):
        self.Q = Counter()

        self.all_actions = all_actions

        # state and action in last round
        self.state = init_state
        self.action = self.all_actions[0]

        # functions
        self.get_next_state = transition_func

        # q-learning parameters
        self.gamma = 0.99
        self.epsilon = INITIAL_EPSILON

        self.alpha = INITIAL_ALPHA

        self.rand = Random()

    def observe_and_act(self, observation, reward, count):

        next_state = self.get_next_state(self.state, self.action, observation)
        self._update_q(self.state, self.action, next_state, reward)
        self.state = next_state

        if self.rand.uniform(0, 1) < self.epsilon:
            action_id = int(self.rand.uniform(0, len(self.all_actions)))
            self.action = self.all_actions[action_id]
        else:

            # print some info
            log = False
            if count > T_THRESHOLD - 1000:
                if count % 100 == 0:
                    print(count)
                    print(self.epsilon)
                    log = True
            _, self.action = self._get_max_q_value(self.state, log)

        # annealling epsilon
        # change episilon
        if self.epsilon > FINAL_EPSILON and count > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # annealing alpha
        if count > EXPLORE * 4 + 10:
            self.alpha = 1 / float(count - EXPLORE * 4)

        return self.action

    def _get_max_q_value(self, state, log):
        tmp_max, tmp_action = 0, self.all_actions[0]

        # print info
        if log:
            print self.Q

        for action in self.all_actions:
            if self.Q[(state, action)] > tmp_max:
                tmp_max = self.Q[(state, action)]
                tmp_action = action

        action_list = []

        # 'multiple largest q values' case
        for action in self.all_actions:
            if self.Q[(state, action)] == tmp_max:
                action_list.append(action)

        idx = random.randint(0, len(action_list) - 1)
        tmp_action = action_list[idx]

        if len(action_list) > 1:
            print action_list

        return tmp_max, tmp_action

    def _update_q(self, state, action, next_state, reward):
        max_q, _ = self._get_max_q_value(next_state, False)
        self.Q[(state, action)] += self.alpha * (reward + self.gamma * max_q - self.Q[(state, action)])

    def target_observe_and_act(self, observation, reward, count):

        next_state = self.get_next_state(self.state, self.action, observation)

        self.state = next_state

        log = False

        # if count< 10:
        #   log = True

        _, self.action = self._get_max_q_value(self.state, log)

        return self.action
