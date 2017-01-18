# coding=utf-8

import re
import os

from utils import *


if __name__ == '__main__':
    data_dir = 'dataset/corpus'
    dst_fpath = 'dataset/sentence_corpus.txt'
    
    file_cnt = 0
    with open(dst_fpath, 'w') as f_write:
        for root, dirs, files in os.walk(data_dir):
            for file_path in files:
                if file_path.endswith('TXT') or file_path.endswith('txt'):
                    file_cnt += 1
                    with open(os.path.join(root, file_path), 'r') as f:
                        text = f.read()
                    text = text.decode('GBK', 'ignore')
                    sents = segment_text_into_sentences(text)
                    for s in sents:
                        s = stringQ2B(s)
                        f_write.write(s.encode('utf-8'))
                        f_write.write('\n')
    print '{} files processed'.format(file_cnt)