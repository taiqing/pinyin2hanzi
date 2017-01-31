# coding=utf-8

import numpy as np

from config import aligned_input_len, aligned_output_len, filling_symbol


if __name__ == '__main__':
    labeled_corpus_path = 'dataset/labeled_corpus.txt'
    aligned_corpus_path = 'dataset/aligned_labeled_corpus.txt'
    
    src_len = []
    target_len = []
    with open(labeled_corpus_path, 'r') as f:
        for line in f:
            line = line.decode('utf-8')[:-1] # remove \n
            if line.startswith('Z:'):
                target = line[2:]
            if line.startswith('P:'):
                source = line[2:]
                src_len.append(len(source))
                target_len.append(len(target))
    print 'source length: {min} ~ {max}, average {a:.2f}'.format(min=np.min(src_len), max=np.max(src_len), a=np.mean(src_len))
    print 'target length: {min} ~ {max}, average {a:.2f}'.format(min=np.min(target_len), max=np.max(target_len), a=np.mean(target_len))

    with open(aligned_corpus_path, 'w') as f_wrt:
        with open(labeled_corpus_path, 'r') as f:
            for line in f:
                line = line.decode('utf-8')[:-1]# remove \n
                if line.startswith('Z:'):
                    target = line[2:] + filling_symbol * (aligned_output_len - len(line[2:]))
                    f_wrt.write(('Z:' + target + '\n').encode('utf-8'))
                if line.startswith('P:'):
                    source = line[2:] + filling_symbol * (aligned_input_len - len(line[2:]))
                    f_wrt.write(('P:' + source + '\n').encode('utf-8'))
    