from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import time
from collections import deque
import math
from config_4 import  *
from env_markov_distinct_channel import Environment
import threading

CHANNEL_SIZE = N_CHANNELS
HISTORY = AGENT_STATE_WINDOWS_SIZE
STATE_SIZE = CHANNEL_SIZE * HISTORY
#ACTION_SIZE = ACTION_SIZE
HIDDEN_UNINITS_1 = 5
HIDDEN_UNINITS_2 = 5
GAMMA = 1
FRAME_PER_ACTION = 1
OBSERVE = 10000  # timesteps to observe before training
EXPLORE = 700000# frames over which to anneal epsilon
FINAL_EPSILON = 0.1  # final value of epsilon: for epsilon annealing
INITIAL_EPSILON = 0.1 # starting value of epsilon
#REPLAY_MEMORY = 50000  # number of previous transitions to remember
ASYNC_UPDATE_INTERVAL = 32  # size of minibatch
TARGET_UPDATE_INTERVAL = 200000 # target netowrk update period
CONCURRENT_THREADS_NUM = 4 # No. of concurrent learners


"""DQN with separte target estimation network (Atari Nature, Algorithm 1)"""
class Async_DQN:

    def __init__(self):
        #init replay memory
        self.num_learners = CONCURRENT_THREADS_NUM
        #init Q network
        self.state_placeholder, self.W_1, self.b_1, self.W_2, self.b_2, self.W_3, self.b_3, self.q_values = self.createQNetwork()
        self.state_placeholder_T, self.W_1_T, self.b_1_T, self.W_2_T, self.b_2_T, self.W_3_T, self.b_3_T, self.q_values_T = self.createQNetwork()
        self.createTraining()
        self.updateTargetNetwork = [self.W_1_T.assign(self.W_1), self.b_1_T.assign(self.b_1), self.W_2_T.assign(self.W_2), self.b_2_T.assign(self.b_2), self.W_3_T.assign(self.W_3), self.b_3_T.assign(self.b_3)]
        #start session
        # saving and loading networks
        # self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        # checkpoint = tf.train.get_checkpoint_state("saved_networks")
        # if checkpoint and checkpoint.model_checkpoint_path:
        # self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        # print("Successfully loaded:", checkpoint.model_checkpoint_path)
        # else:
        # print("Could not find old network weights")


    def createQNetwork(self):

        state_placeholder = tf.placeholder(tf.float32, [None, STATE_SIZE])



        W_1 = tf.Variable(tf.truncated_normal([int(STATE_SIZE), int(HIDDEN_UNINITS_1)], stddev=1.0 / math.sqrt(float(STATE_SIZE))))
        b_1 = tf.Variable(tf.zeros([HIDDEN_UNINITS_1]))



        hidden1 = tf.nn.tanh(tf.matmul(state_placeholder, W_1) + b_1)

        W_2 = tf.Variable(
            tf.truncated_normal([int(HIDDEN_UNINITS_1), int(HIDDEN_UNINITS_2)], stddev=1.0 / math.sqrt(float(HIDDEN_UNINITS_1))))
        b_2 = tf.Variable(tf.zeros([HIDDEN_UNINITS_2]))

        hidden2 = tf.nn.tanh(tf.matmul(hidden1, W_2) + b_2)

        W_3 = tf.Variable(tf.truncated_normal([int(HIDDEN_UNINITS_2), int(ACTION_SIZE)], stddev=1.0 / math.sqrt(float(HIDDEN_UNINITS_2))))
        b_3 = tf.Variable(tf.zeros([ACTION_SIZE]))

        q_values = tf.matmul(hidden2, W_3) + b_3

        return state_placeholder, W_1, b_1, W_2, b_2, W_3, b_3, q_values

    def createTraining(self):

        self.y_placeholder = tf.placeholder(tf.float32, [None])

        self.action_placeholder = tf.placeholder(tf.float32, [None, ACTION_SIZE])
        q_action = tf.reduce_sum(tf.mul(self.q_values, self.action_placeholder), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_placeholder-q_action))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost) #default: Adadelta

    def q_learner_thread(self, thread_id, fileName, p_matrix, f_result):

        env = Environment(p_matrix)
        action = np.zeros(int(ACTION_SIZE))
        action[0]=1
        action_env = self.process(action)

        observation, reward, terminal = env.step(action_env)

        currentState = self.setInitState(observation)

        state_batch = []
        y_batch = []
        action_batch = []
        nextState_batch = []
        reward_batch = []
        terminal_batch = []
        count = 0
        total = 0

        reward = 0


        total += reward



        epsilon = INITIAL_EPSILON
        final_epsilon = self.sample_final_epsilon()

        print("Start thread %d" % thread_id)
        time.sleep(3*thread_id)

        start_time = time.time()
        f = open(fileName, 'w')

        while count < T_THRESHOLD:
            count += 1
            state_batch.append(currentState)
            action = self.getAction(currentState, epsilon)
            action_batch.append(action)
            action_env = self.process(action)

            observation, reward, terminal = env.step(action_env)

            reward = 50

            total += reward



            reward_batch.append(reward)
            terminal_batch.append(terminal)

            nextState = np.concatenate((currentState[CHANNEL_SIZE:], observation))
            nextState_batch.append(nextState)
            currentState = nextState

            if count % ASYNC_UPDATE_INTERVAL == 0:

                q_values_batch = self.q_values_T.eval(session=self.session, feed_dict={self.state_placeholder_T: nextState_batch})

                for i in range(ASYNC_UPDATE_INTERVAL):

                    if terminal_batch[i]:
                        y_batch.append(reward_batch[i])
                    else:
                        y_batch.append(reward_batch[i] + GAMMA * np.max(q_values_batch[i]))

                self.train_op.run(session = self.session, feed_dict={self.state_placeholder: state_batch, self.action_placeholder: action_batch,
                                             self.y_placeholder: y_batch})
                state_batch = []
                y_batch = []
                action_batch = []
                nextState_batch = []
                reward_batch = []
                terminal_batch = []

            if count % TARGET_UPDATE_INTERVAL == 0:

                self.session.run(self.updateTargetNetwork)

            # change episilon
            if epsilon > final_epsilon and count > OBSERVE:
                epsilon -= (INITIAL_EPSILON - final_epsilon) / EXPLORE



            if (count+1) % PERIOD == 0:

                duration = time.time() - start_time
                accum_reward = total / float(count+1)

                f.write('Index %d: accu_reward is %f, action is: %s and time duration is %f' % (
                    count+1, accum_reward, str(action_env), duration))
                f.write('\n')


        f.close()



        duration = time.time() - start_time
        accum_reward = total / float(count)
        f_result.write('Async Qlearing of thread %d final accu_reward is %f, and time duration is %f\n' % (thread_id, accum_reward, duration))

        print("thread %d end" % thread_id)


    def getAction(self, currentState, epsilon):
        current_state_temp = np.reshape(currentState, (-1, STATE_SIZE)) # !!! [[]]
        q_values_temp = self.q_values.eval(session=self.session, feed_dict = {self.state_placeholder: current_state_temp})[0] #data structure: [[1,2,3]]

        action = np.zeros(int(ACTION_SIZE))

        action_index = 0



        if random.random() <= epsilon:
            action_index = random.randrange(ACTION_SIZE)
        else:
            action_index = np.argwhere(q_values_temp == np.amax(q_values_temp))

            action_index = action_index.flatten().tolist()

            if len(action_index) >= 2:
                print(action_index)

            temp = random.randint(0,len(action_index)-1)
            action_index = action_index[temp]


        action[int(action_index)] = 1


        return action



    def setInitState(self, observation):
        currentState = np.tile(observation, HISTORY)
        return currentState


    def process(self, action):

        action_id = np.nonzero(action)[0]

        action_evn = list(ACTION_SPACE[int(action_id)])

        return action_evn


    def sample_final_epsilon(self):
        """
        Sample a final epsilon value to anneal towards from a distribution.
        These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
        """
        final_epsilons = np.array([.1, .1, .1]) #default: [.1, .01, .5]
        probabilities = np.array([0.4, 0.3, 0.3])
        return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]


    #get action from target network
    def get_target_action(self, currentState, epsilon, count):
        current_state_temp = np.reshape(currentState, (-1, STATE_SIZE))  # !!! [[]]
        q_values_temp = self.q_values_T.eval(session=self.session, feed_dict={self.state_placeholder_T: current_state_temp})[0]  # data structure: [[1,2,3]]

        if count > 1:
            print('q_values')
            print(q_values_temp)

        action = np.zeros(int(ACTION_SIZE))

        action_index = 0


        action_index = np.argwhere(q_values_temp == np.amax(q_values_temp))

        action_index = action_index.flatten().tolist()

        if len(action_index) >= 2:
            print(action_index)

        temp = random.randint(0, len(action_index) - 1)
        action_index = action_index[temp]

        action[int(action_index)] = 1

        return action

    #use target network for interacting with the environment
    def target_network_eval(self, fileName, p_matrix, f_result):
        env = Environment(p_matrix)
        action = np.zeros(int(ACTION_SIZE))
        action[0] = 1
        action_env = self.process(action)

        observation, reward, terminal = env.step(action_env)

        reward = 50

        currentState = self.setInitState(observation)




        count = 0
        total = 0

        total += reward

        epsilon = INITIAL_EPSILON


        print("Start target evaluation" )


        start_time = time.time()
        f = open(fileName, 'w')

        while count < 50:
            count += 1

            if count<= 10:
                action = np.zeros(int(ACTION_SIZE))
                action_index = random.randrange(ACTION_SIZE)
                action[int(action_index)] = 1

            else:

                action = self.get_target_action(currentState, epsilon, count)





            action_env = self.process(action)


            if count > 15:
                print('observation')
                print(observation)

                print('action')
                print(action_env)



            observation, reward, terminal = env.step(action_env)

            reward = 50

            total += reward

            nextState = np.concatenate((currentState[CHANNEL_SIZE:], observation))

            currentState = nextState

            print('currentState')
            print(currentState)



            if (count + 1) % PERIOD == 0:
                duration = time.time() - start_time
                accum_reward = total / float(count + 1)

                f.write('Index %d: accu_reward is %f, action is: %s and time duration is %f' % (
                    count + 1, accum_reward, str(action_env), duration))
                f.write('\n')

        f.close()

        duration = time.time() - start_time
        accum_reward = total / float(count)
        f_result.write('Async Qlearing using target final accu_reward is %f, and time duration is %f\n' % (accum_reward, duration))

        print("target evaluation ends")

    def create_thread(self, p_matrix, fileName, f_result):

        learner_threads = [threading.Thread(target=self.q_learner_thread, args=(
        thread_id, fileName+str(thread_id), p_matrix, f_result)) for thread_id in range(self.num_learners)]
        for t in learner_threads:
            t.start()


        #make sure all threads are over

        for t in learner_threads:
            t.join()


        self.target_network_eval(fileName+'target', p_matrix, f_result)











