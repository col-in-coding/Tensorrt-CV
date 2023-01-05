import onnx
from onnxsim import simplify
from model import BertForMaskedLMTRT
from transformers import BertTokenizer
from utils import get_inputs, validate

BERT_PATH = 'bert-base-uncased'


if __name__ == "__main__":
    filename = "onnx/model.onnx"
    sim_filename = "onnx/model_sim.onnx"
    plan_file_path = "engine/model.plan"
    
    print('------------------------ onnx simplifying')
    # load your predefined ONNX model
    model = onnx.load(filename)
    # # convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, sim_filename)

    # print("----------------------- building tensorrt engine")
    onnx_file_path = sim_filename
    optimization_profiles = [{
        0: ((1, 1), (1, 16), (1, 512)),
        1: ((1, 1), (1, 16), (1, 512)),
        2: ((1, 1), (1, 16), (1, 512))
    }]
    BertForMaskedLMTRT.build_engine(onnx_file_path, plan_file_path, use_fp16=False, optimization_profiles=optimization_profiles, workspace=20)

    print("--------------------- tensorrt inference")
    trt_model = BertForMaskedLMTRT(plan_file_path)

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
    # text = "The capital of China, " + tokenizer.mask_token + ", contains the Tian'an Men."
    input_ids, token_type_ids, attention_mask, mask_index = get_inputs(tokenizer, text)
    # print(input_ids.shape, input_ids.dtype, input_ids)
    # print(token_type_ids.shape, token_type_ids.dtype, token_type_ids)
    # print(attention_mask.shape, attention_mask.dtype, attention_mask)
    # exit(0)

    trt_outputs = trt_model(input_ids.numpy(), attention_mask.numpy(), token_type_ids.numpy())
    logits = trt_outputs[0]
    validate(logits, mask_index, tokenizer, text)