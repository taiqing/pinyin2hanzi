# coding=utf-8

# format dataset as the form [(source, target), (source, target), ...]

import cPickle
import numpy as np


if __name__ == '__main__':
    aligned_corpus_path = 'dataset/aligned_labeled_corpus.txt'
    dataset_file = 'dataset/dataset.pkl'

    dataset = list()
    src_len = []
    target_len = []
    with open(aligned_corpus_path, 'r') as f:
        for line in f:
            line = line.decode('utf-8')[:-1] # remove \n
            if line.startswith('Z:'):
                target = line[2:]
            if line.startswith('P:'):
                source = line[2:]
                src_len.append(len(source))
                target_len.append(len(target))
                dataset.append((source, target))
    cPickle.dump(dataset, open(dataset_file, 'wb'))
    print 'source length: {min} ~ {max}, average {a:.2f}'.format(min=np.min(src_len), max=np.max(src_len), a=np.mean(src_len))
    print 'target length: {min} ~ {max}, average {a:.2f}'.format(min=np.min(target_len), max=np.max(target_len), a=np.mean(target_len))
    