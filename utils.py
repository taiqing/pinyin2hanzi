# coding=utf-8

def is_hanzi(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_hanzi(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


def Q2B(uchar):
    """全角转半角"""
    if uchar == u'\u3010':  # 【 -> [
        return u'['
    if uchar == u'\u3011':  # 】-> ]
        return u']'

    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:
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