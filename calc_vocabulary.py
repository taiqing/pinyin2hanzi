# coding=utf-8

import cPickle

def make_vocabulary(vocab):
    vocab2 = dict()
    i = 0
    for k in vocab:
        vocab2[k] = i
        i += 1
    return vocab2
    

if __name__ == '__main__':
    labeled_corpus_path = 'dataset/labeled_corpus_no_rare.txt'
    vocab_file = 'dataset/vocab.pkl'

    vocab_source = dict()
    vocab_target = dict()
    with open(labeled_corpus_path, 'r') as f:
        for line in f:
            line = line.decode('utf-8')
            if line.startswith('P:'):
                for uchar in line[2:]:
                    if uchar != u'\n':
                        if uchar not in vocab_source:
                            vocab_source[uchar] = 1
                        else:
                            vocab_source[uchar] += 1
            if line.startswith('Z:'):
                for uchar in line[2:]:
                    if uchar != u'\n':
                        if uchar not in vocab_target:
                            vocab_target[uchar] = 1
                        else:
                            vocab_target[uchar] += 1
    print 'source vocabulary size is {}'.format(len(vocab_source))
    print 'target vocabulary size is {}'.format(len(vocab_target))
    
    vocab_source = make_vocabulary(vocab_source)
    vocab_target = make_vocabulary(vocab_target)
    cPickle.dump((vocab_source, vocab_target), open(vocab_file, 'wb'))