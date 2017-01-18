# coding=utf-8

import pypinyin
from pypinyin import pinyin, lazy_pinyin

from utils import *


def count_hanzi(ustring):
    i = 0
    for uchar in ustring:
        if is_hanzi(uchar):
            i += 1
    return i


def count_punctuations(ustring):
    i = 0
    for uchar in ustring:
        if is_other(uchar):
            i += 1
    return i


if __name__ == '__main__':
    corpus_fpath = 'dataset/sentence_corpus.txt'
    labeled_corpus_fpath = 'dataset/labeled_corpus.txt'
    
    i = 0
    with open(labeled_corpus_fpath, 'w') as labeled_corpus_file:
        with open(corpus_fpath, 'r') as corpus_file:
            for line in corpus_file:
                line = line.decode('utf-8')
                hanzi_cnt = count_hanzi(line)
                punc_cnt = count_punctuations(line)
                if hanzi_cnt >= 3 and punc_cnt <= 3:
                    hanzi = ':'.join(['Z', line])
                    labeled_corpus_file.write(hanzi.encode('utf-8'))
                    pinyin = ':'.join(['P', ''.join(lazy_pinyin(line))])
                    #pinyin = ':'.join(['P', '-'.join(lazy_pinyin(line))])
                    labeled_corpus_file.write(pinyin.encode('utf-8'))
                    
                    i += 1
                    if i % 1000 == 0:
                        print '{} lines processed'.format(i)
