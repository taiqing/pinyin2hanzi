# coding=utf-8

import pandas as pd

from utils import *
from config import *


if __name__ == '__main__':
    src_corpus_fpath = 'dataset/sentence_corpus.txt'
    dst_corpus_fpath = 'dataset/sentence_corpus_no_rare.txt'

    word_freq = dict()
    with open(src_corpus_fpath, 'r') as f:
        for line in f:
            line = line.decode('utf-8')
            for word in line:
                if is_hanzi(word):
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
    word_freq_series = pd.Series(word_freq)
    word_freq_series.sort_values(axis=0, ascending=False, inplace=True)
    frequent_words = word_freq_series[word_freq_series >= min_word_freq].to_dict()
    print '{f} frequent words out of {t}'.format(f=len(frequent_words), t=word_freq_series.shape[0])
    
    with open(dst_corpus_fpath, 'w') as dst_f:
        with open(src_corpus_fpath, 'r') as f:
            for line in f:
                line = line.decode('utf-8')
                line2 = list()
                for word in line:
                    if is_hanzi(word) and word not in frequent_words:
                        line2.append(rare_word_symbol)
                    else:
                        line2.append(word)
                line2 = u''.join(line2)
                line2 = line2.encode('utf-8')
                dst_f.write(line2)