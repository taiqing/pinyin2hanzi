# coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
import cPickle
import tensorflow as tf


def join_dicts(dict1, dict2):
    """
    Raise exception if two dicts share some keys
    """
    dict_ret = dict1.copy()
    for k, v in dict2.iteritems():
        if k not in dict_ret:
            dict_ret[k] = v
        else:
            raise Exception('Key conflicts in join_dicts')
    return dict_ret


def weight_variable_normal(shape, stddev=None):
    if stddev is None:
        stddev = 1.0 / np.sqrt(shape[0])
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=stddev)
    return tf.Variable(initial)


class GRUCell(object):
    # GRU description in http://colah.github.io/posts/2015-08-Understanding-LSTMs/

    def __init__(self, n_input, n_hidden, stddev=None, variable_values=None, name='GRU'):
        if variable_values is None:
            # update gate
            self.W_z = weight_variable_normal([n_input + n_hidden, n_hidden], stddev)
            self.b_z = tf.Variable(tf.zeros(n_hidden, tf.float32))
            # reset gate
            self.W_r = weight_variable_normal([n_input + n_hidden, n_hidden], stddev)
            self.b_r = tf.Variable(tf.zeros(n_hidden, tf.float32))
            # candidate generation
            self.W_c = weight_variable_normal([n_input + n_hidden, n_hidden], stddev)
            self.b_c = tf.Variable(tf.zeros(n_hidden, tf.float32))
        else:
            self.W_z = tf.Variable(variable_values[':'.join([name, 'W_z'])])
            self.b_z = tf.Variable(variable_values[':'.join([name, 'b_z'])])
            self.W_r = tf.Variable(variable_values[':'.join([name, 'W_r'])])
            self.b_r = tf.Variable(variable_values[':'.join([name, 'b_r'])])
            self.W_c = tf.Variable(variable_values[':'.join([name, 'W_c'])])
            self.b_c = tf.Variable(variable_values[':'.join([name, 'b_c'])])

        self.vars = {':'.join([name, 'W_z']): self.W_z,
                     ':'.join([name, 'b_z']): self.b_z,
                     ':'.join([name, 'W_r']): self.W_r,
                     ':'.join([name, 'b_r']): self.b_r,
                     ':'.join([name, 'W_c']): self.W_c,
                     ':'.join([name, 'b_c']): self.b_c}

    def __call__(self, h, x):
        """
        :param h: must be rank-2
        :param x: must be rank-2
        :return:
        """
        hx = array_ops.concat(1, [h, x])
        # z: update gate
        z = tf.sigmoid(tf.matmul(hx, self.W_z) + self.b_z)
        # r: reset gate
        r = tf.sigmoid(tf.matmul(hx, self.W_r) + self.b_r)
        # h_c: candidate hidden state
        h_candidate = tf.tanh(tf.matmul(array_ops.concat(1, [r * h, x]), self.W_c) + self.b_c)
        new_h = (1 - z) * h + z * h_candidate
        return new_h


def vectorise_list_of_pairs(pairs, vocab_source, vocab_target, n_input, n_output, n_step_input, n_step_output):
    source = np.zeros((len(pairs), n_step_input, n_input), np.float32)
    target = np.zeros((len(pairs), n_step_output, n_output), np.float32)
    for j, pair in zip(range(len(pairs)), pairs):
        s, t = pair
        for k, x in zip(range(len(s)), s):
            source[j, k, vocab_source[x]] = 1
        for k, x in zip(range(len(t)), t):
            target[j, k, vocab_target[x]] = 1
    return source, target


if __name__ == '__main__':
    np.random.seed(1001)

    dataset_file = 'dataset/dataset.pkl'
    vocab_file = 'dataset/vocab.pkl'

    n_input = 28
    n_output = 1103
    n_step_input = 44
    n_step_output = 13
    n_hidden = 128#1024
    weight_stddev = None #0.1
    n_epoch = 10
    batch_size = 100
    validation_steps = 100
    validation_portion = 0.02
    save_param_steps = 100
    learning_rate = 1e-3
    gamma = 1e-3

    encoder_cell = GRUCell(n_input, n_hidden, weight_stddev, name='encoder:0')
    decoder_cell = GRUCell(n_output, n_hidden, weight_stddev, name='decoder:0')
    W_o = weight_variable_normal([n_hidden, n_output], weight_stddev)
    b_o = tf.Variable(np.zeros(n_output, dtype=np.float32))
    variables = join_dicts(join_dicts(encoder_cell.vars, decoder_cell.vars), {'W_o': W_o, 'b_o': b_o})

    x = tf.placeholder(tf.float32, [None, n_step_input, n_input], name='x')
    y = tf.placeholder(tf.float32, [None, n_step_output, n_output], name='y')
    #learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    #gamma = tf.placeholder(tf.float32, name='gamma')

    # encoding
    n_sample = tf.shape(x)[0]
    h0 = tf.zeros((n_sample, n_hidden), tf.float32)
    encoder_states = [h0]
    for i in range(n_step_input - 1, -1, -1):
        h_prev = encoder_states[-1]
        x_t = x[:, i, :] # reads input in the reverse time order
        h_t = encoder_cell(h_prev, x_t)
        encoder_states.append(h_t)

    # decoding
    decoder_states = [encoder_states[-1]]
    initial_input = tf.zeros([n_sample, n_output], tf.float32)
    outputs = list()
    for t in range(0, n_step_output):
        h_prev = decoder_states[-1]
        y_t = initial_input if t == 0 else y[:, t - 1, :]
        h_t = decoder_cell(h_prev, y_t)
        out_t = tf.nn.softmax(tf.matmul(h_t, W_o) + b_o)
        decoder_states.append(h_t)
        outputs.append(out_t)
    outputs = tf.pack(outputs, axis=1)  # outputs: n_samples x n_step x n_output

    # loss
    loss = -tf.reduce_mean(tf.log(outputs) * y)

    # l2-norm of paramters
    regularizer = 0.
    for k, v in variables.iteritems():
        regularizer += tf.reduce_mean(tf.square(v))

    # cost
    cost = loss + gamma * regularizer
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init_vars = tf.global_variables_initializer()

    vocab_source, vocab_target = cPickle.load(open(vocab_file, 'rb'))
    vocab_source_r = dict()
    for k, v in vocab_source.iteritems():
        vocab_source_r[v] = k
    vocab_target_r = dict()
    for k, v in vocab_target.iteritems():
        vocab_target_r[v] = k

    dataset = cPickle.load(open(dataset_file, 'rb'))

    # validation set
    n_sample = len(dataset)
    permutation = np.random.permutation(n_sample)
    selected_idx = permutation[0: int(n_sample * validation_portion)]
    validation_set = [dataset[k] for k in selected_idx]
    selected_idx = permutation[int(n_sample * validation_portion) :]
    train_set = [dataset[k] for k in selected_idx]

    n_sample = len(train_set)
    print '{} training samples'.format(n_sample)
    sess = tf.Session()
    with sess.as_default():
      init_vars.run()
      sample_counter = 0
      for i in range(int(n_epoch * n_sample / batch_size)):
          if i % int(validation_steps) == 0:
              source, target = vectorise_list_of_pairs(validation_set,
                                                       vocab_source, vocab_target,
                                                       n_input, n_output, n_step_input, n_step_output)
              c, l, r = sess.run([cost, loss, regularizer],
                                          feed_dict={x: source,
                                                     y: target})
              print '{i} samples fed in: validation: {n} samples, cost {c:.5f}, loss {l:.5f}, paramter regularizer {r:.5f}'.format(
                  i=sample_counter, n=len(validation_set), c=c, l=l, r=r)

          selected_idx = np.random.permutation(n_sample)[0 : batch_size]
          batch_pairs = [train_set[k] for k in selected_idx]
          source, target = vectorise_list_of_pairs(batch_pairs,
                                                   vocab_source, vocab_target,
                                                   n_input, n_output, n_step_input, n_step_output)
          _, c, l, r = sess.run([train_step, cost, loss, regularizer], feed_dict={x: source, y: target})
          print '{i}-th batch, cost {c:.5f}, loss {l:.5f}, paramter regularizer {r:.5f}'.format(i=i, c=c, l=l, r=r)
          sample_counter += len(batch_pairs)
          if i % int(save_param_steps) == 0:
              parameters = dict()
              for k, v in variables.iteritems():
                  parameters[k] = sess.run(v)
              cPickle.dump(parameters, open('models/parameters_{}.pkl'.format(i), 'wb'))
      parameters = dict()
      for k, v in variables.iteritems():
          parameters[k] = sess.run(v)
      cPickle.dump(parameters, open('models/parameters_final.pkl', 'wb'))
    sess.close()

    # parameters = cPickle.load(open('models/parameters_final.pkl', 'rb'))
    # pair = train_set[1]
    # source, target = vectorise_list_of_pairs([pair], vocab_source, vocab_target, n_input, n_output, n_step_input, n_step_output)
    # feed_dict = dict()
    # for k, v in variables.iteritems():
    #     feed_dict[v] = parameters[k]
    # feed_dict[x] = source
    # feed_dict[y] = target
    # sess = tf.Session()
    # out = sess.run(outputs, feed_dict=feed_dict)
    # pred = [vocab_target_r[d] for d in out[0].argmax(axis=1)]
    # print "source: " + pair[0]
    # print "target: " + pair[1]
    # print "prediction: " + "".join(pred)

