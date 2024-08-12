import re
import unicodedata
from transformers import AutoTokenizer

from ug_dictionary import english_dictionary ,etc_dictionary
import symbols
import MeCab

# 维吾尔语的标点符号
punctuation = [
    "!", "?", "…", ",", ".", "'", "-",
    "؛", "«", "»", "ـ", "،", "؟", "٪", "ـ"
]

pre_trained_model = "xlm-roberta-base"

def normalize(text:str):
    text = text.strip()
    text = unicodedata.normalize('NFKC', text)
    # 删除除维吾尔语、阿拉伯语和常用标点符号外的所有字符
    # 维吾尔语使用阿拉伯字母，且维吾尔语文本可能包含拉丁字母和阿拉伯数字
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\u0041-\u005A\u0061-\u007A0-9 \t\n\r\f\v' + ''.join(map(re.escape, punctuation)) + ']', '', text)
    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    return text

def normalize_with_dictionary(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile("|".join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    return text
def normalize_english(text):
    def fn(m):
        word = m.group().upper()
        print(word)
        if word in english_dictionary:
            return english_dictionary.get(word)
        return word

    text = re.sub("([A-Za-z]+)", fn, text)
    return text


if __name__ == '__main__':
    text = "بۇ بىر سىناش ماتېرىيال! 这是测试文本。"
    normalized_text = normalize(text)
    print(normalized_text)

