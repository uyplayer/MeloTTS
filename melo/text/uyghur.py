import re
import symbols
from melo.text.es_phonemizer import cleaner as es_cleaner
from melo.text.es_phonemizer import es_to_ipa
from transformers import AutoTokenizer
from melo.text.ug_utils.text_processing.text_cleaner import TextCleaner
from melo.text.ug_utils.tokenizer.FairseqXLMRTokenizer import FairseqXLMRTokenizer
from melo.text.ug_utils.tokenizer.XLMRobertaTokenizer import XLMRobertaTokenizer
import epitran, torch

import warnings
warnings.filterwarnings("ignore")

def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


cleaner = TextCleaner()


def text_normalize(text):
    cleaned_text = cleaner.clean_text(text)
    return cleaned_text


def post_replace_ph(ph):
    rep_map = {
        "،": ",",
        "؛": ",",
        "?": "؟",
        "！": "!",
        "。": ".",
        "٫": ",",
        "：": ":",
        "\n": ".",
        "...": "…",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph


def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone


def refine_syllables(syllables):
    tones = []
    phonemes = []
    for phn_list in syllables:
        for i in range(len(phn_list)):
            phn = phn_list[i]
            phn, tone = refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


# tokenizer = FairseqXLMRTokenizer()
tokenizer = XLMRobertaTokenizer()
ipa = epitran.Epitran('uig-Arab')

def g2p(text, pad_start_end=True, tokenized=None):
    if tokenized is None:
        tokenized = tokenizer.tokenize(text)
    print(tokenized)
    print(f"tokenized len : {len(tokenized)}")
    phs = []
    ph_groups = []
    i = 0
    remove_index = []
    while i < len(tokenized):
        if tokenized[i] == "▁":
            if i + 1 < len(tokenized):
                tokenized[i + 1] = tokenized[i] + tokenized[i + 1]
            tokenized.pop(i)
            remove_index.append(i)
        else:
            i += 1
    for idx in reversed(remove_index):
        tokenized.insert(idx, "▁")
    print(tokenized)
    print(f"tokenized len : {len(tokenized)}")
    for token in tokenized:
        if token.startswith("▁") or len(token) == 1:
            ph_groups.append([token.replace("▁", "")])
        else:
            if token == "UNK":
                ph_groups.append(["UNK"])
            else:
                ph_groups[-1].append(token)
    phones = []
    tones = []
    word2ph = []
    # print(ph_groups)
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w == 'UNK':
            phone_list = ['UNK']
        else:
            phone_list = list(filter(lambda p: p != " ", ipa.trans_list(w)))

        for ph in phone_list:
            phones.append(ph)
            tones.append(0)
            phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph
def get_bert_feature(text, word2ph, device=None):
    from melo.text import uyghur_bert
    return uyghur_bert.get_bert_feature(text, word2ph, device=device)


if __name__ == "__main__":
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ.111"
    print(text)
    cleaner = TextCleaner()
    text = cleaner.clean_text(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)
    print(phones)
    print(len(phones), tones, sum(word2ph), bert.shape)
    print(bert)

