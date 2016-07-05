from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
from collections import deque
import math
from config import  *

CHANNEL_SIZE = N_CHANNELS
HISTORY = AGENT_STATE_WINDOWS_SIZE
STATE_SIZE = CHANNEL_SIZE * HISTORY
#ACTION_SIZE = ACTION_SIZE
HIDDEN_UNINITS = 10
GAMMA = 0.99
FRAME_PER_ACTION = 1
OBSERVE = 50  # timesteps to observe before training
EXPLORE = 150000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0  # final value of epsilon: for epsilon annealing
INITIAL_EPSILON = 0.2  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch

"""DQN with online learning"""

class BrainDQN:

    def __init__(self):
        #init replay memory
        self.replayMemory = deque()
        #init Q network
        self.createQNetwork()
        self.BATCH_SIZE = BATCH_SIZE
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON



    def createQNetwork(self):

        self.state_placeholder = tf.placeholder(tf.float32, [None, STATE_SIZE])
        self.action_placeholder = tf.placeholder(tf.float32, [None, ACTION_SIZE])
        self.y_placeholder = tf.placeholder(tf.float32,[None])

        W_1 = tf.Variable(tf.truncated_normal([int(STATE_SIZE), int(HIDDEN_UNINITS)], stddev=1.0 / math.sqrt(float(STATE_SIZE))))
        b_1 = tf.Variable(tf.zeros([HIDDEN_UNINITS]))

        hidden1 = tf.nn.tanh(tf.matmul(self.state_placeholder, W_1) + b_1)

        W_2 = tf.Variable(tf.truncated_normal([int(HIDDEN_UNINITS), int(ACTION_SIZE)], stddev=1.0 / math.sqrt(float(HIDDEN_UNINITS))))
        b_2 = tf.Variable(tf.zeros([ACTION_SIZE]))

        self.q_values = tf.matmul(hidden1, W_2) + b_2

        q_action = tf.reduce_sum(tf.mul(self.q_values, self.action_placeholder), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_placeholder-q_action))
        self.train_op = tf.train.AdadeltaOptimizer(1e-6).minimize(self.cost)

        # saving and loading networks
        #self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        #checkpoint = tf.train.get_checkpoint_state("saved_networks")
        #if checkpoint and checkpoint.model_checkpoint_path:
            #self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            #print("Successfully loaded:", checkpoint.model_checkpoint_path)
        #else:
            #print("Could not find old network weights")

    def trainQNetwork(self):

        #step 1: obtain samples from experience replay
        minibatch = random.sample(self.replayMemory, self.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        #step 2: obtain target values (i.e., y)
        y_batch = []
        q_value_batch = self.q_values.eval(feed_dict = {self.state_placeholder: next_state_batch}) #data structure: [[1, 2, 4], [3,4,5]]
        for i in range(self.BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                temp = np.amax(q_value_batch[i])
                print(temp)
                y_batch.append(reward_batch[i]+GAMMA*temp)

        #step 3: train
        self.train_op.run(feed_dict = {self.state_placeholder: state_batch, self.action_placeholder: action_batch, self.y_placeholder: y_batch})

        # save network every 100000 iteration
        #if self.timeStep % 10000 == 0:
            #self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

    #prepare the experience replay
    def setPerception(self, nextObservation, action, reward, terminal):

        newState = np.concatenate((self.currentState[CHANNEL_SIZE:],nextObservation))
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))

        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        self.currentState = newState
        self.timeStep += 1


    def getAction(self):
        current_state_temp = np.reshape(self.currentState, (-1, STATE_SIZE)) # !!! [[]]
        q_values_temp = self.q_values.eval(feed_dict = {self.state_placeholder: current_state_temp}) #data structure: [[1,2,3]]
        action = np.zeros(int(ACTION_SIZE))
        action_index = 0

        if random.random() <= self.epsilon:
            action_index = random.randrange(ACTION_SIZE)
        else:
            action_index = np.argmax(q_values_temp)

        action[action_index] = 1

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action




    def setInitState(self, observation):
        self.currentState = np.tile(observation, HISTORY)





