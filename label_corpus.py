# coding=utf-8

# Annotate sentences with their pinyin
# Replace non-hanzi characters with a single special symbol

import pypinyin
from pypinyin import pinyin, lazy_pinyin

from utils import *
from config import *


def count_punctuations(ustring):
    i = 0
    for uchar in ustring:
        if is_other(uchar):
            i += 1
    return i


def replace_nonhanzi(ustring, uchar):
    # 将非汉字替换为uchar
    ustring2 = list()
    for c in ustring:
        if is_hanzi(c) is False:
            ustring2.append(uchar)
        else:
            ustring2.append(c)

    # 将连续出现的uchar替换为一个uchar
    ustring3 = list(ustring2[0])
    for i in range(1, len(ustring2)):
        if ustring2[i] == uchar and ustring2[i-1] == uchar:
            pass
        else:
            ustring3.append(ustring2[i])
    return u''.join(ustring3)


if __name__ == '__main__':
    corpus_fpath = 'dataset/sentence_corpus_no_rare.txt'
    labeled_corpus_fpath = 'dataset/labeled_corpus.txt'
    
    i = 0
    with open(labeled_corpus_fpath, 'w') as labeled_corpus_file:
        with open(corpus_fpath, 'r') as corpus_file:
            for line in corpus_file:
                try:
                    line = line.decode('utf-8')[:-1] # remove \n
                    n_hanzi = count_hanzi(line)
                    if n_hanzi >= min_hanzi_len and n_hanzi <= max_hanzi_len:
                        line = replace_nonhanzi(line, nonhanzi_symbol)
                        hanzi = ':'.join(['Z', line])
                        labeled_corpus_file.write(hanzi.encode('utf-8'))
                        labeled_corpus_file.write('\n')
                        pinyin = ':'.join(['P', ''.join(lazy_pinyin(line))])
                        #pinyin = ':'.join(['P', '-'.join(lazy_pinyin(line))])
                        labeled_corpus_file.write(pinyin.encode('utf-8'))
                        labeled_corpus_file.write('\n')
                except:
                    pass
                i += 1
                if i % 1000 == 0:
                    print '{} lines processed'.format(i)
