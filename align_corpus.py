# coding=utf-8

import numpy as np

from config import aligned_input_len, filling_symbol


if __name__ == '__main__':
    labeled_corpus_path = 'dataset/labeled_corpus.txt'
    aligned_corpus_path = 'dataset/aligned_labeled_corpus.txt'

    with open(aligned_corpus_path, 'w') as f_wrt:
        with open(labeled_corpus_path, 'r') as f:
            for line in f:
                line = line.decode('utf-8')[:-1]# remove \n
                if line.startswith('P:'):
                    source = line[2:]
                if line.startswith('Z:'):
                    target = line[2:]
                    target2 = []
                    for x, y in zip(source.split('-'), list(target)):
                        z = [u'#'] * (len(x) - len(y)) + [y]
                        target2 += z
                    target2 = ''.join(target2)
                    target2 += filling_symbol * (aligned_input_len - len(target2))
                    source2 = source.replace('-', '')
                    source2 += filling_symbol * (aligned_input_len - len(source2))
                    assert len(source2) == len(target2)
                    f_wrt.write(('P:' + source2 + '\n').encode('utf-8'))
                    f_wrt.write(('Z:' + target2 + '\n').encode('utf-8'))
    