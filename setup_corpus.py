# coding=utf-8


import re
import os


def Q2B(uchar):
    """全角转半角"""
    if uchar == u'\u3010': # 【 -> [
        return u'['
    if uchar == u'\u3011': # 】-> ]
        return u']'
        
    inside_code=ord(uchar)
    if inside_code==0x3000:
        inside_code=0x0020
    else:
        inside_code-=0xfee0
    if inside_code<0x0020 or inside_code>0x7e:
        return uchar
    return unichr(inside_code)

def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])
        
def uniform(ustring):
    """格式化字符串，完成全角转半角，大写转小写的工作"""
    return stringQ2B(ustring).lower()

def segment_text_into_sentences(text):
    text = text.strip('\r\n ')
    text = text.replace('\r\n', '')
    
    # 分割成子句
    sents = re.split(u'[，。？：；！,.?:;!…]', text)
    sents = [s.replace(' ', '').replace(u'\u3000', '') for s in sents]
    return sents

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