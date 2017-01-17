# coding=utf-8


import pypinyin
from pypinyin import pinyin, lazy_pinyin

if __name__ == '__main__':
    corpus_fpath = 'dataset/sentence_corpus.txt'
    labeled_corpus_fpath = 'dataset/labeled_corpus.txt'
    
    i = 0
    with open(labeled_corpus_fpath, 'w') as labeled_corpus_file:
        with open(corpus_fpath, 'r') as corpus_file:
            for line in corpus_file:
                line = line.decode('utf-8')
                if len(line) >= 3:
                    hanzi = ':'.join(['Z', line])
                    labeled_corpus_file.write(hanzi.encode('utf-8'))
                    pinyin = ':'.join(['P', '-'.join(lazy_pinyin(line))])
                    labeled_corpus_file.write(pinyin.encode('utf-8'))
                    
                    i += 1
                    if i % 1000 == 0:
                        print '{} lines processed'.format(i)
