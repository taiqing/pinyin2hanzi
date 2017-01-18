# coding=utf-8


if __name__ == '__main__':
    labeled_corpus_path = 'dataset/labeled_corpus_no_rare.txt'

    vocab_source = dict()
    vocab_target = dict()
    with open(labeled_corpus_path, 'r') as f:
        for line in f:
            line = line.decode('utf-8')
            if line.startswith('P:'):
                for uchar in line:
                    if uchar != u'\n':
                        if uchar not in vocab_source:
                            vocab_source[uchar] = 1
                        else:
                            vocab_source[uchar] += 1
            if line.startswith('Z:'):
                for uchar in line:
                    if uchar != u'\n':
                        if uchar not in vocab_target:
                            vocab_target[uchar] = 1
                        else:
                            vocab_target[uchar] += 1
    print 'source vocabulary size is {}'.format(len(vocab_source))
    print 'target vocabulary size is {}'.format(len(vocab_target))