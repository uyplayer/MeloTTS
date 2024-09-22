import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import logging as transformers_logging
import sys
import warnings
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()


model_id = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = None

def get_bert_feature(text, word2ph, device=None):
    global model
    if (
            sys.platform == "darwin"
            and torch.backends.mps.is_available()
            and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if model is None:
        model = AutoModelForMaskedLM.from_pretrained(model_id,ignore_mismatched_sizes=True ).to( device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    print(inputs["input_ids"].shape[-1] , len(word2ph))
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T


if __name__ == '__main__':
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    print(text)
    res = get_bert_feature(text)