## A Gated Recurrent Neural Network for End-to-End Translation of Chinese Pinyin

setup_corpus.py
-->
handle_rare_words.py
-->
label_corpus.py
-->
align_corpus.py
-->
calc_vocab.py & format_dataset.py
-->
pinyin_to_hanzi_bigru.py

网络结构采用双向GRU，训练数据使用67K个中文短句。

输入是一串英文字母（"womenyouxinxinyingdezhecibisai"），输出是汉字（"我们有信心赢得这次比赛"）。

### Examples
input:     womenyouxinxinyingdezhecibisai

output: 我们有信心赢得这次比赛

input:     youyujingyanbuzu

output: 由于经验不足

input:     tebieshizuijin_nianlai

output: 特别是最近_年来
