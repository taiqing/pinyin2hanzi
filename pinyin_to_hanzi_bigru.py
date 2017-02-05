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


def vectorise(string, vocab):
    coding = np.zeros((len(string), len(vocab)), np.float32)
    for k, x in zip(range(len(string)), string):
        coding[k, vocab[x]] = 1
    return coding


def vectorise_list_of_pairs(pairs, vocab_source, vocab_target):
    """
    The strings in pairs must be aligned, that is, their lengths are padded to the same
    """
    source = []
    target = []
    for s, t in pairs:
        source.append(vectorise(s, vocab_source))
        target.append(vectorise(t, vocab_target))
    source = np.stack(source, axis=0)
    target = np.stack(target, axis=0)
    return source, target


if __name__ == '__main__':
    np.random.seed(1001)

    dataset_file = 'dataset/dataset.pkl'
    vocab_file = 'dataset/vocab.pkl'

    n_input = 28
    n_output = 1104
    n_step_input = 44
    n_hidden = 256
    weight_stddev = 0.1
    n_epoch = 5
    batch_size = 100
    validation_steps = 100
    validation_portion = 0.025
    test_portion = 0.025
    save_param_steps = 100
    learning_rate = 1e-2
    gamma = 1e-1
    verbose = True
    
    # -- build the graph -- 
    
    encoder_cell = GRUCell(n_input, n_hidden, weight_stddev, name='encoder:0')
    encoder_cell_r = GRUCell(n_input, n_hidden, weight_stddev, name='encoder:1')
    W_o = weight_variable_normal([2 * n_hidden, n_output], weight_stddev)
    b_o = tf.Variable(np.zeros(n_output, dtype=np.float32))
    variables = join_dicts(join_dicts(encoder_cell.vars, encoder_cell_r.vars), {'W_o': W_o, 'b_o': b_o})

    x = tf.placeholder(tf.float32, [None, n_step_input, n_input], name='x')
    y = tf.placeholder(tf.float32, [None, n_step_input, n_output], name='y')
    #learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    #gamma = tf.placeholder(tf.float32, name='gamma')

    # encoding
    n_sample = tf.shape(x)[0]
    h_init = tf.zeros((n_sample, n_hidden), tf.float32)
    encoder_states = []
    for i in range(n_step_input):
        h_prev = h_init if i == 0 else encoder_states[-1]
        x_t = x[:, i, :]
        h_t = encoder_cell(h_prev, x_t)
        encoder_states.append(h_t)

    encoder_states_r = []
    for i in range(n_step_input):
        h_prev = h_init if i == 0 else encoder_states_r[-1]
        x_t = x[:, n_step_input - i - 1, :] # read the input in reverse order
        h_t = encoder_cell_r(h_prev, x_t)
        encoder_states_r.append(h_t)
    encoder_states_r = encoder_states_r[::-1]

    # decoding
    outputs = list()
    for i in range(n_step_input):
        h_t = tf.concat(1, [encoder_states[i], encoder_states_r[i]])
        out_t = tf.nn.softmax(tf.matmul(h_t, W_o) + b_o)
        outputs.append(out_t)
    outputs = tf.pack(outputs, axis=1)  # outputs: n_samples x n_step x n_output

    # loss
    loss = -tf.reduce_sum(tf.log(outputs) * y) / (tf.cast(n_sample, tf.float32) * n_step_input)

    # l2-norm of paramters
    regularizer = 0.
    for k, v in variables.iteritems():
        regularizer += tf.reduce_mean(tf.square(v))

    # cost
    cost = loss + gamma * regularizer
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init_vars = tf.global_variables_initializer()
    
    # -- run the graph --

    vocab_source, vocab_target = cPickle.load(open(vocab_file, 'rb'))
    vocab_source_r = dict()
    for k, v in vocab_source.iteritems():
        vocab_source_r[v] = k
    vocab_target_r = dict()
    for k, v in vocab_target.iteritems():
        vocab_target_r[v] = k

    dataset = cPickle.load(open(dataset_file, 'rb'))

    n_sample = len(dataset)
    permutation = np.random.permutation(n_sample)
    selected_idx = permutation[0: int(n_sample * validation_portion)]
    validation_set = [dataset[k] for k in selected_idx]
    selected_idx = permutation[int(n_sample * validation_portion) : int(n_sample * validation_portion) + int(n_sample * test_portion)]
    test_set = [dataset[k] for k in selected_idx]
    selected_idx = permutation[int(n_sample * validation_portion) + int(n_sample * test_portion) : ]
    train_set = [dataset[k] for k in selected_idx]
    print '{tr} training samples, {v} validation samples, {te} test samples'.format(tr=len(train_set), v=len(validation_set), te=len(test_set))

    n_sample = len(train_set)
    print '{} training samples'.format(n_sample)
    sess = tf.Session()
    with sess.as_default():
        init_vars.run()
        sample_counter = 0
        for i in range(int(n_epoch * n_sample / batch_size)):
            if i % int(validation_steps) == 0:
                source, target = vectorise_list_of_pairs(validation_set, vocab_source, vocab_target)
                c, l, r = sess.run([cost, loss, regularizer],
                                   feed_dict={x: source,
                                              y: target})
                print '{i} samples fed in: validation: {n} samples, cost {c:.5f}, loss {l:.5f}, paramter regularizer {r:.5f}'.format(
                    i=sample_counter, n=len(validation_set), c=c, l=l, r=r)

            if i % int(save_param_steps) == 0:
                parameters = dict()
                for k, v in variables.iteritems():
                    parameters[k] = sess.run(v)
                cPickle.dump(parameters, open('models/parameters_{}.pkl'.format(i), 'wb'))

            selected_idx = np.random.permutation(n_sample)[0 : batch_size]
            batch_pairs = [train_set[k] for k in selected_idx]
            source, target = vectorise_list_of_pairs(batch_pairs, vocab_source, vocab_target)
            _, c, l, r = sess.run([train_step, cost, loss, regularizer], feed_dict={x: source, y: target})
            if verbose:
                print '{i}-th batch, cost {c:.5f}, loss {l:.5f}, paramter regularizer {r:.5f}'.format(i=i, c=c, l=l, r=r)
            sample_counter += len(batch_pairs)

        parameters = dict()
        for k, v in variables.iteritems():
            parameters[k] = sess.run(v)
        cPickle.dump(parameters, open('models/parameters_final.pkl', 'wb'))

        # evaluate on test set
        source, target = vectorise_list_of_pairs(test_set, vocab_source, vocab_target)
        l = sess.run(loss, feed_dict={x: source, y: target})
        print 'test set: {n} samples, loss {l:.8f}'.format(n=len(test_set), l=l)
    sess.close()

    parameters = cPickle.load(open('models/parameters_final.pkl', 'rb'))
    # evaluate on test set
    source, target = vectorise_list_of_pairs(test_set, vocab_source, vocab_target)
    sess = tf.Session()
    feed_dict = dict()
    for k, v in variables.iteritems():
        feed_dict[v] = parameters[k]
    feed_dict[x] = source
    feed_dict[y] = target
    l = sess.run(loss, feed_dict=feed_dict)
    print 'test set: {n} samples, loss {l:.8f}'.format(n=len(test_set), l=l)
    
    pair = test_set[1]
    source, target = vectorise_list_of_pairs([pair], vocab_source, vocab_target)
    feed_dict = dict()
    for k, v in variables.iteritems():
        feed_dict[v] = parameters[k]
    feed_dict[x] = source
    feed_dict[y] = target
    out = sess.run(outputs, feed_dict=feed_dict)
    pred = [vocab_target_r[d] for d in out[0].argmax(axis=1)]
    print "source:     " + pair[0]
    print "target:     " + pair[1]
    print "prediction: " + "".join(pred)