# coding=utf-8

# format dataset as the form [(source, target), (source, target), ...]

import cPickle
import numpy as np


if __name__ == '__main__':
    labeled_corpus_path = 'dataset/labeled_corpus.txt'
    dataset_file = 'dataset/dataset.pkl'

    dataset = list()
    input_len = []
    with open(labeled_corpus_path, 'r') as f:
        for line in f:
            line = line.decode('utf-8')[:-1] # remove \n
            if line.startswith('Z:'):
                target = line[2:]
            if line.startswith('P:'):
                source = line[2:]
                input_len.append(len(source))
                dataset.append((source, target))
    cPickle.dump(dataset, open(dataset_file, 'wb'))
    print 'input length: {min} ~ {max}, average {a:.2f}'.format(min=np.min(input_len), max=np.max(input_len), a=np.mean(input_len))
    