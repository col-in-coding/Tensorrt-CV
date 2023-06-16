import time
import torch
import onnxruntime as rt
from transformers import BertTokenizer
from utils import get_inputs, validate

BERT_PATH = 'bert-base-uncased'


if __name__ == "__main__":
    so = rt.SessionOptions()
    so.log_severity_level = 3
    # model = "onnx/model.onnx"
    model = "onnx/model_sim.onnx"

    sess = rt.InferenceSession(model, so, providers=['CUDAExecutionProvider'])

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."

    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    # print(output_names)

    input_ids, token_type_ids, attention_mask, mask_index = get_inputs(tokenizer, text)
    print(input_ids.shape, input_ids.dtype, input_ids)
    print(token_type_ids.shape, token_type_ids.dtype, token_type_ids)
    print(attention_mask.shape, attention_mask.dtype, attention_mask)
    # exit(0)

    outputs = sess.run(output_names, {
        "input_ids": input_ids.numpy(),
        "token_type_ids": token_type_ids.numpy(),
        "attention_mask": attention_mask.numpy()
    })
    logits = outputs[0]
    print("===> ", logits, logits.dtype)
    validate(logits, mask_index, tokenizer, text)

    print("-------------------------------- timing")
    # warmup
    for _ in range(100):
        outputs = sess.run(output_names, {
            "input_ids": input_ids.numpy(),
            "token_type_ids": token_type_ids.numpy(),
            "attention_mask": attention_mask.numpy()
        })
    runs = 100
    start_time = time.time()
    for _ in range(runs):
        outputs = sess.run(output_names, {
            "input_ids": input_ids.numpy(),
            "token_type_ids": token_type_ids.numpy(),
            "attention_mask": attention_mask.numpy()
        })
    print("time avg: ", (time.time() - start_time) / runs)