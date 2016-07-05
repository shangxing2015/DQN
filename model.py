import tensorflow as tf

import math



MNIST_SIZE = 28
STATE_SIZE = MNIST_SIZE*MNIST_SIZE



NUM_ACTIONS = 10

#forward
def inference(states, hidden1_units):

    with tf.name_scope('hidden1'):
        W = tf.Variable(tf.truncated_normal([STATE_SIZE, hidden1_units], stddev=1.0/math.sqrt(float(STATE_SIZE))))
        b = tf.Variable(tf.zeros([hidden1_units]))

        hidden1 = tf.nn.tanh(tf.matmul(states, W) + b)

    with tf.name_scope('softmax_linear'):
        W = tf.Variable(tf.truncated_normal([hidden1_units, NUM_ACTIONS], stddev=1.0/math.sqrt(float(hidden1_units))))
        b = tf.Variable(tf.zeros([NUM_ACTIONS]))

        q_values = tf.matmul(hidden1, W) + b

    return q_values

#MSE
def loss(estimates, targets):
    loss = tf.reduce_mean((estimates-targets)**2)
    return loss

#train
def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op

#evaluate
def evaluation(estimates, targets):
    mse = tf.reduce_mean((estimates-targets)**2)
    return mse