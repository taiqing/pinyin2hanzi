# coding=utf-8

import cPickle
import numpy as np


if __name__ == '__main__':
    dataset_file = 'dataset/dataset.pkl'
    vocab_file = 'dataset/vocab.pkl'
    vectorised_dataset_file = 'dataset/vectorised_dataset.pkl'
    
    dataset = cPickle.load(open(dataset_file, 'rb'))
    vocab_source, vocab_target = cPickle.load(open(vocab_file, 'rb'))
    
    vocab_source_size = len(vocab_source)
    vocab_target_size = len(vocab_target)
    
    dataset_vectorised = []
    i = 0
    for source, target in dataset:
        source_vectorised = [vocab_source[x] for x in source]
        target_vectorised = [vocab_target[x] for x in target]
        dataset_vectorised.append((source_vectorised, target_vectorised))
        i += 1
        if i % 1000 == 0:
            print '{} pairs processed'.format(i)
    cPickle.dump(dataset_vectorised, open(vectorised_dataset_file, 'wb'))