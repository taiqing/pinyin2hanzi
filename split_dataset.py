# coding=utf-8

import cPickle
import numpy as np


if __name__ == '__main__':
    dataset_file = 'dataset/dataset.pkl'
    dataset_split_file = 'dataset/dataset_split.pkl'
    validation_portion = 0.025
    test_portion = 0.025

    dataset = cPickle.load(open(dataset_file, 'rb'))

    n_sample = len(dataset)
    permutation = np.random.permutation(n_sample)
    selected_idx = permutation[0: int(n_sample * validation_portion)]
    validation_set = [dataset[k] for k in selected_idx]
    selected_idx = permutation[
                   int(n_sample * validation_portion): int(n_sample * validation_portion) + int(n_sample * test_portion)]
    test_set = [dataset[k] for k in selected_idx]
    selected_idx = permutation[int(n_sample * validation_portion) + int(n_sample * test_portion):]
    train_set = [dataset[k] for k in selected_idx]
    print '{tr} training samples, {v} validation samples, {te} test samples'.format(tr=len(train_set),
                                                                                    v=len(validation_set), te=len(test_set))

    cPickle.dump((train_set, validation_set, test_set), open(dataset_split_file, 'wb'))
