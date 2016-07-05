import random
import numpy as np
from collections import defaultdict
import itertools

from config import *
CHANNEL_SIZE = N_CHANNELS
CONFLICT_GRAPH = np.zeros((N_NODES, N_NODES))

class _Noise:

  def __init__(self, n_channels, n_states=4):
    self.n_states = int(n_states)
    self.n_channels = n_channels
    self.current_state = 0

    self.state_noise = [[] for i in xrange(self.n_states)]
    self.random = random.Random()
    for i in xrange(self.n_states):
      self.state_noise[i] = [self.random.random()
                             for j in xrange(self.n_channels)] #wy +1 ?
      amount = 0
      for j in xrange(self.n_channels):
        amount += self.state_noise[i][j]
      for j in xrange(self.n_channels):
        if DEBUG_NO_NOISE:
          self.state_noise[i][j] = 0
        else:
          self.state_noise[i][j] /= float(amount) #normalize

  def get_state(self):
    self.current_state = (self.current_state + 1) % self.n_states
    return self.state_noise[self.current_state]


def _get_distance_square(pos1, pos2=(0, 0, 0)):
  assert len(pos1) == 3 and len(pos2) == 3
  return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + \
      (pos1[2] - pos2[2]) ** 2


class Environment:

  def __init__(self, n_nodes, n_channels, n_noises=-1, min_distance=1,
               max_distance=10, const_parameter=True):
    self.n_nodes = n_nodes

    if DEBUG_NO_CONFLICT_GRAPH:
      self.conflict_graph = CONFLICT_GRAPH #conflict graph
    else:
      self.conflict_graph = CONFLICT_GRAPH
      # conflict graph
      self.conflict_graph[0][1] = 1
      self.conflict_graph[0][2] = 1
      self.conflict_graph[0][3] = 1


    self.n_channels = n_channels
    self.n_noises = int(
        self.n_channels * (2 if n_noises == -1 else n_noises))

    self.random = random.Random()
    self.next_int = lambda: self.random.uniform(min_distance,
                                                max_distance)
    self.next_float = self.random.random

    self.noises = [_Noise(self.n_channels, self.next_int())
                   for i in xrange(self.n_noises)]

    if const_parameter:
      self.node_pos = [(1, 1, 1) for i in xrange(self.n_nodes)]
      self.noise_pos = [(1, 1, 1) for i in xrange(self.n_noises)]
      self.node_power = [1 for i in xrange(self.n_nodes)]
      self.noise_power = [1 for i in xrange(self.n_noises)]
    else:
      self.node_pos = [(0.5 + self.next_float(), 0.5 + self.next_float(),
                        0.5 + self.next_float())
                       for i in xrange(self.n_nodes)]
      self.noise_pos = [(1 + self.next_float(), 1 + self.next_float(),
                         1 + self.next_float())
                        for i in xrange(self.n_noises)]
      self.node_power = [self.next_float() for i in xrange(self.n_nodes)]
      self.noise_power = [self.next_float() for i in xrange(self.n_noises)]

  def _get_noise_channel_probability(self):
    return [self.noises[i].get_state()
            for i in xrange(self.n_noises)]

  def _get_channel_noise_mapping(self):
    self.noise_channel_probability = self._get_noise_channel_probability()
    # generate noise affect table
    noise_channel_mapping = [-1 for j in xrange(self.n_noises)]
    for j in xrange(self.n_noises):
      tmp = 1 if DEBUG_NO_NOISE else self.next_float()
      present = 0
      for i in xrange(self.n_channels):
        present += self.noise_channel_probability[j][i]
        if tmp < present:
          noise_channel_mapping[j] = i
          break
      # if noise_channel_mapping[j] == -1:
      #  noise_channel_mapping[j] = self.n_channels - 1

    channel_noise = [[] for j in xrange(self.n_channels)]
    for j in xrange(self.n_noises):
      if noise_channel_mapping[j] != -1:
        channel_noise[noise_channel_mapping[j]].append(j)

    return channel_noise

  def _get_conflict_matrix(self):
    conflict_matrix = [[0 for j in xrange(self.n_nodes)]
                       for i in xrange(self.n_channels)]

    channel_noise = self._get_channel_noise_mapping()
    for i in xrange(self.n_channels):
      noise_list = channel_noise[i]
      noise_power = 0
      for j in noise_list:
        assert type(j) == int
        noise_power += self.noise_power[j] / \
            float(_get_distance_square(self.noise_pos[j]))

      for j in xrange(self.n_nodes):
        node_power = self.node_power[j] / \
            float(_get_distance_square(self.node_pos[j]))
        if node_power > noise_power:
          conflict_matrix[i][j] = 0
        else:
          conflict_matrix[i][j] = 1  # two nodes can use the same channel

    return conflict_matrix

  #make sure conflict users cannot use the same channel
  def _check_concurrent_nodes(self, action, conflict_graph = CONFLICT_GRAPH):

    #In python: mutable variables (e.g. list) is passed by reference

    action_temp = list(action)

    same_channel_users = defaultdict(list)

    for i, item in enumerate(action_temp):
      same_channel_users[item].append(i)

    same_channel_users = {k: v for k, v in same_channel_users.items() if len(v) > 1}

    for key in same_channel_users:

      if key != -1:

        possible_collide_user_pairs = itertools.combinations(same_channel_users[key], 2)

        for user_pair in possible_collide_user_pairs:
          if conflict_graph[user_pair[0]][user_pair[1]] == 1 or conflict_graph[user_pair[1]][user_pair[0]]== 1:
            for index in same_channel_users[key]:
              action_temp[index] = -2


    return action_temp


  def step(self, action):
    assert len(action) == self.n_nodes and type(action[0]) == int

    conflict_matrix = self._get_conflict_matrix()

    observation = [0 for i in xrange(self.n_channels)]
    action_temp = self._check_concurrent_nodes(action, self.conflict_graph)

    node_succ = 0 # num of successful users
    node_sleep = action_temp.count(-1) # num of sleep nodes
    node_collide = action_temp.count(-2) # num of nodes collide due to being in the conflict graph and assinged on the same channel

    for node_id in xrange(self.n_nodes):
        channel_id = action_temp[node_id] #action: -1, or channel_id
        if channel_id != -1 and channel_id != -2 and conflict_matrix[channel_id][node_id] == 0:
            observation[channel_id] = 1
            node_succ += 1

    observation = np.array(observation)

    node_fail = N_NODES-node_succ-node_sleep-node_collide # num of nodes that transmit but failed due to the bad channel quality


    reward = node_succ - (node_fail + node_collide) # node succeeds: +1; node fails/collides: -1
    terminal = False
    return observation, reward, terminal
