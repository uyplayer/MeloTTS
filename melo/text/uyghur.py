import re
import unicodedata
from num2words import num2words
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
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
def normalize(text:str):
    text = text.strip()
    text = unicodedata.normalize('NFKC', text)
    # 删除除维吾尔语、阿拉伯语和常用标点符号外的所有字符
    # 维吾尔语使用阿拉伯字母，且维吾尔语文本可能包含拉丁字母和阿拉伯数字
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\u0041-\u005A\u0061-\u007A0-9 \t\n\r\f\v' + ''.join(map(re.escape, punctuation)) + ']', '', text)
    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    text = text.lower()
    return text

def normalize_with_dictionary(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile("|".join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    return text
def normalize_english(text:str):
    def fn(m):
        word = m.group().upper()
        if word in english_dictionary:
            return english_dictionary.get(word)
        return word

    text = re.sub("([A-Za-z]+)", fn, text)
    return text

def text_normalize(text):
    text = uyghur_convert_numbers_to_words(text)
    text = uyghur_convert_alpha_symbols_to_words(text)
    text = normalize(text)
    return text

def uyghur_text_to_phonemes(text:str):
    return text

def uyghur_convert_numbers_to_words(text:str):
    return text
def uyghur_convert_alpha_symbols_to_words(text:str):
    text = re.sub(r'[A-Za-z]', lambda x: english_dictionary.get(x.group().upper(), x.group()), text)
    return text


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def g2p_uyghur(norm_text):
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))

    word2ph = []
    for group in ph_groups:
        text = ""
        for ch in group:
            text += ch
        if text == '[UNK]':
            phs += ['_']
            word2ph += [1]
            continue
        elif text in punctuation:
            phs += [text]
            word2ph += [1]
            continue
        phonemes = uyghur_text_to_phonemes(text)
        phone_len = len(phonemes)
        word_len = len(group)

        aaa = distribute_phone(phone_len, word_len)
        assert len(aaa) == word_len
        word2ph += aaa

        phs += phonemes

    phones = ["_"] + phs + ["_"]
    tones = [0 for _ in phones]
    word2ph = [1] + word2ph + [1]
    assert len(word2ph) == len(tokenized) + 2

    return phones, tones, word2ph


if __name__ == '__main__':
    text = "بۇ بىر سىناش ماتېرىيال! 这是测试文本。"
    normalized_text = normalize(text)
    print(normalized_text)
    model_id = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
    res = tokenizer.tokenize(normalized_text)
    for item in res:
        print(item)

