#use MNIST data to test
from __future__ import print_function
import tensorflow as tf

import math

from tensorflow.examples.tutorials.mnist import input_data
import model

import time
import numpy as np


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', '0.01', 'optization learning rate')
flags.DEFINE_float('gamma', 0.75, 'future reward discount factor')
flags.DEFINE_float('epsilon', 0.1, 'for epsilon-greedy policy')
flags.DEFINE_float('alpha', 0.01, 'value function learning rate')
flags.DEFINE_integer('experience_add_every', 25, 'number of time steps before we add another experience to')
flags.DEFINE_integer('experience_size', 5000, 'size of experience replay')
flags.DEFINE_integer('learning_steps_per_iteration', 10, 'learning iteration')
flags.DEFINE_float('tderror_clamp', 1.0, 'clamp threshold')
flags.DEFINE_integer('hidden1', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                                        'Must divide evenly into the dataset sizes.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                                         'for unit testing.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags. DEFINE_integer('max_steps', 2000, 'Number of steps to run training' )


STATE_SIZE = model.STATE_SIZE

NUM_ACTIONS = model.NUM_ACTIONS

#fill place holder
def placeholder_iputs(batch_size):

    states_placeholder = tf.placeholder(tf.float32, [batch_size, STATE_SIZE])
    q_values_placeholder = tf.placeholder(tf.float32, [batch_size, NUM_ACTIONS])
    return states_placeholder, q_values_placeholder

#fill dict
def fill_feed_dict(data_set, states_pl, q_values_pl):

    states_feed, labels = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)

    values_feed = np.zeros([FLAGS.batch_size, NUM_ACTIONS])

    for i in xrange(FLAGS.batch_size):
        values_feed[i, labels[i]] = 0

    return {states_pl: states_feed, q_values_pl: values_feed}


#run func

data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

with tf.Graph().as_default():

    states_placeholder, q_values_placeholder = placeholder_iputs(FLAGS.batch_size)
    estimates = model.inference(states_placeholder, FLAGS.hidden1)
    loss = model.loss(estimates, q_values_placeholder)
    train_op = model.training(loss, FLAGS.learning_rate)
    eval = model.evaluation(estimates, q_values_placeholder)

    sess = tf.Session()
    init =  tf.initialize_all_variables()
    sess.run(init)

    start_time = time.time()

    for step in xrange(FLAGS.max_steps):


        duration  = time.time() - start_time

        feed_dict = fill_feed_dict(data_sets.train, states_placeholder, q_values_placeholder)
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict )


        if step % 100 == 0:
            print('Step %d: loss = %.4f %.3f sec()' % (step, loss_value, duration))


    eval_num = data_sets.test.num_examples // FLAGS.batch_size
    mse = 0
    for step in xrange(eval_num):

        feed_dict_test = fill_feed_dict(data_sets.test, states_placeholder, q_values_placeholder )
        mse += sess.run(eval, feed_dict = feed_dict_test)

    print('MSE of the test data %.4f' % (mse/eval_num))

# #main func
# def main(_):
#   run_training()


# if __name__ == '__main__':
#   tf.app.run()
