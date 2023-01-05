import torch
from torch.nn import functional as F


def get_inputs(tokenizer, text):
    encoded_input = tokenizer.encode_plus(text, return_tensors="pt")
    mask_index = torch.where(encoded_input["input_ids"][0] == tokenizer.mask_token_id)
    # print(encoded_input.keys)
    return encoded_input["input_ids"], encoded_input["token_type_ids"], encoded_input["attention_mask"], mask_index


def validate(logits, mask_index, tokenizer, text):
    print("------------------------- validation")
    logits = torch.from_numpy(logits)
    softmax = F.softmax(logits, dim = -1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    print("model test topk10 output:")
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)