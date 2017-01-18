# coding=utf-8

# reference
# GRU description in http://colah.github.io/posts/2015-08-Understanding-LSTMs/

import tensorflow as tf
import numpy as np
from tensorflow.python import array_ops
from tensorflow.examples.tutorials.mnist import input_data
import cPickle


def weight_variable_normal(shape, stddev=None):
    if stddev is not None:
        std = stddev
    else:
        std = 1.0 / np.sqrt(shape[0])
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=std)
    return tf.Variable(initial)


def weight_variable_uniform(shape, radius):
    initial = tf.random_uniform(shape=shape, minval=-radius, maxval=radius)
    return tf.Variable(initial)


class GRUCell(object):
    def __init__(self, n_input, n_hidden):
        # initialise weights as normal distr
        self.W_z = weight_variable_normal([n_input + n_hidden, n_hidden])
        self.W_r = weight_variable_normal([n_input + n_hidden, n_hidden])
        self.W_c = weight_variable_normal([n_input + n_hidden, n_hidden])

    def __call__(self, h, x):
        hx = array_ops.concat(1, [h, x])
        # z: update gate
        z = tf.sigmoid(tf.matmul(hx, self.W_z))
        # r: reset gate
        r = tf.sigmoid(tf.matmul(hx, self.W_r))
        # h_c: candidate hidden state
        h_candidate = tf.tanh(tf.matmul(array_ops.concat(1, [r * h, x]), self.W_c))
        new_h = (1 - z) * h + z * h_candidate
        return new_h


if __name__ == '__main__':
    tf.reset_default_graph()

    dataset_file = 'dataset/dataset.pkl'
    vocab_file = 'dataset/vocab.pkl'

    dataset = cPickle.load(open(dataset_file, 'rb'))
    vocab_input, vocab_output = cPickle.load(open(vocab_file, 'rb'))

    max_input_len = 0
    max_output_len = 0
    for input, output in dataset:
        max_input_len = max(max_input_len, len(input))
        max_output_len = max(max_output_len, len(output))
    n_steps_input = max_input_len + 1 # plus <end>
    n_steps_output = max_output_len + 1 # plus <end>
        
    input_vocab_size = len(vocab_input) + 2 # plus <start> <end>
    output_vocab_size = len(vocab_output) + 1 # plus <end>
    
    