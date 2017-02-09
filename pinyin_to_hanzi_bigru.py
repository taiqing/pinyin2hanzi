# coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
import cPickle
import tensorflow as tf

from config import filling_symbol, aligned_input_len


def join_dicts(dict_list):
    """
    Raise exception if two dicts share some keys
    """
    dict_ret = dict()
    for d in dict_list:
        for k, v in d.iteritems():
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

        self.variables = {':'.join([name, 'W_z']): self.W_z,
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


def edit_distance(input_x, input_y):
    xlen = len(input_x) + 1
    ylen = len(input_y) + 1

    dp = np.zeros(shape=(xlen, ylen), dtype=int)
    for i in range(0, xlen):
        dp[i][0] = i
    for j in range(0, ylen):
        dp[0][j] = j

    for i in range(1, xlen):
        for j in range(1, ylen):
            if input_x[i - 1] == input_y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[xlen - 1][ylen - 1]
    

if __name__ == '__main__':
    np.random.seed(1001)

    dataset_file = 'dataset/dataset.pkl'
    vocab_file = 'dataset/vocab.pkl'
    
    n_input = 28
    n_output = 1104
    n_step_input = 44
    n_layer = 2
    n_hidden = [256, 128]
    weight_stddev = 0.1
    n_epoch = 20
    batch_size = 100
    validation_steps = 100
    validation_portion = 0.025
    test_portion = 0.025
    save_param_steps = 100
    learning_rate = 1e-2
    gamma = 1e-1
    verbose = True
    
    # -- build the graph --
    x = tf.placeholder(tf.float32, [None, n_step_input, n_input], name='x')
    y = tf.placeholder(tf.float32, [None, n_step_input, n_output], name='y')

    encoder_cells = []
    encoder_r_cells = []
    variables = dict()
    for l in range(n_layer):
        input_size = n_input if l == 0 else n_hidden[l - 1]
        layer_size = n_hidden[l]
        encoder = GRUCell(input_size, layer_size, weight_stddev, name='encoder:{}'.format(l))
        encoder_r = GRUCell(input_size, layer_size, weight_stddev, name='encoder_r:{}'.format(l))
        variables = join_dicts([variables, encoder.variables, encoder_r.variables])
        encoder_cells.append(encoder)
        encoder_r_cells.append(encoder_r)
    W_o = weight_variable_normal([2 * n_hidden[-1], n_output], weight_stddev)
    b_o = tf.Variable(np.zeros(n_output, dtype=np.float32))
    variables = join_dicts([variables, {'W_o': W_o, 'b_o': b_o}])

    # encoding
    n_sample = tf.shape(x)[0]

    def build_encoder_layers(reverse_input=False):
        inputs = tf.unstack(x, axis=1)
        if reverse_input: inputs = inputs[::-1]
        states_layers = []
        for l in range(n_layer):
            states_prev = inputs if l == 0 else states_layers[l - 1]
            states = []
            h_init = tf.zeros((n_sample, n_hidden[l]), tf.float32)
            for t in range(n_step_input):
                h_prev = h_init if t == 0 else states[-1]
                input_t = states_prev[t]
                h_t = encoder_cells[l](h_prev, input_t)
                states.append(h_t)
            states_layers.append(states)
        return states_layers

    states_layers = build_encoder_layers()
    states_r_layers = build_encoder_layers(reverse_input=True)

    # decoding
    outputs = list()
    for t in range(n_step_input):
        h_t = tf.concat(1, [states_layers[-1][t], states_r_layers[-1][-t-1]])
        out_t = tf.nn.softmax(tf.matmul(h_t, W_o) + b_o)
        outputs.append(out_t)
    outputs = tf.pack(outputs, axis=1)  # outputs: n_samples x n_step x n_output
    predictions = tf.argmax(outputs, axis=2, name='predictions') # predictions: n_samples x n_step

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
    l, pred = sess.run([loss, predictions], feed_dict=feed_dict)
    print 'test set: {n} samples, loss {l:.8f}'.format(n=len(test_set), l=l)
    edit_num = 0
    char_num = 0
    targets = [a[1] for a in test_set]
    for t, p in zip(targets, pred):
        t = t.replace('#', '').strip('.')
        p = u"".join([vocab_target_r[d] for d in p])
        p = p.replace('#', '').strip('.')
        edit_num += edit_distance(t, p)
        char_num += len(t)
    print '{c} hanzi, {e} edits, error rate is {er:.2f}%'.format(c=char_num, e=edit_num, er=100. * edit_num / char_num)
    print ''

    pair = test_set[1000]
    source, target = vectorise_list_of_pairs([pair], vocab_source, vocab_target)
    feed_dict = dict()
    for k, v in variables.iteritems():
        feed_dict[v] = parameters[k]
    feed_dict[x] = source
    feed_dict[y] = target
    pred = sess.run(predictions, feed_dict=feed_dict)
    pred = "".join([vocab_target_r[d] for d in pred[0]])
    pred = pred.replace('#', '').strip('.')
    source = pair[0]
    target = pair[1].replace('#', '').strip('.')
    print "source:     " + source
    print "target:     " + target
    print "prediction: " + pred
    print 'edit dist is {}'.format(edit_distance(target, pred))
    print ''


    # interactive testing
    inputs = ['womenyouxinxinnengyingdezhechangbisai', 'youyujingyanbuzu', 'tebieshizuijin_nianlai']
    for input in inputs:
        aligned_input = input + filling_symbol * (aligned_input_len - len(input))
        source = vectorise(aligned_input, vocab_source)
        source = source.reshape((1, ) + source.shape)
        feed_dict = dict()
        for k, v in variables.iteritems():
            feed_dict[v] = parameters[k]
        feed_dict[x] = source
        pred = sess.run(predictions, feed_dict=feed_dict)
        pred = "".join([vocab_target_r[d] for d in pred[0]])
        pred = pred.replace('#', '')
        pred = pred.strip('.')
        print "source:     " + input
        print "prediction: " + "".join(pred)
        print ""
