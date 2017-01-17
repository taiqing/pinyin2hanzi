# coding=utf-8

import re


def segment_text_into_sentences(text):
    text = text.strip('\r\n ')
    text = text.replace('\r\n', '')
    
    # 分割成子句
    sents = re.split(u'[，。？：；！,.?:;!]', text)
    sents = [s.replace(' ', '').replace(u'\u3000', '') for s in sents]
    return sents

if __name__ == '__main__':
    src_fpath = 'dataset/samples/5189.txt'
    dst_fpath = 'dataset/sentence_corpus.txt'
    
    with open(dst_fpath, 'w') as f_write:
        with open(src_fpath, 'r') as f:
            text = f.read()
        text = text.decode('GBK', 'ignore')
        sents = segment_text_into_sentences(text)
        for s in sents:
            f_write.write(s.encode('utf-8'))
            f_write.write('\n')