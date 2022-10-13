import argparse
import time
import numpy as np

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import onnxruntime

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

def infer(args):
    # load pretrained model
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.load_state_dict(torch.load("pytorch_model.bin"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # load test data
    context = "Beyonce Giselle Knowles-Carter (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles 'Crazy in Love' and 'Baby Boy'."
    questions = ["When was Beyonce born?", "What areas did Beyonce compete in when she was growing up?", "When did Beyonce leave Destiny's Child and become a solo singer?", "What was the name of Beyonce's debut album?"]
    
    # preprocess test data
    inputs = []
    for question in questions:
        inputs.append((question, context))

    # tokenize test data
    encoding = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    # if using onnnxruntime, convert to onnx format
    if args.ort:
        torch.onnx.export(model, (input_ids, attention_mask), "model.onnx", input_names=['input_ids', 'attention_mask'], output_names=['start_logits', "end_logits"])                       
        sess = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])
        ort_input = {
            'input_ids': np.ascontiguousarray(input_ids.numpy()),
            'attention_mask' : np.ascontiguousarray(attention_mask.numpy()),
        }

    # run inference
    start = time.time()
    if args.ort:
        output = sess.run(None, ort_input)
    else:
        output = model(input_ids, attention_mask=attention_mask)
    end = time.time()

    # postprocess test data
    print("Context: ", context)
    for i in range(len(questions)):
        if args.ort:
            max_start_logits = output[0][i].argmax()
            max_end_logits = output[1][i].argmax()
        else:
            max_start_logits = output.start_logits[i].argmax()
            max_end_logits = output.end_logits[i].argmax()
        ans_tokens = input_ids[i][max_start_logits: max_end_logits + 1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
        answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
        print("Question: ", questions[i])
        print("Answer: ", answer_tokens_to_string)

    # brag about how fast we are
    print("Inference time: ", end - start, "seconds")

def main(raw_args=None):
    parser = argparse.ArgumentParser(description="DistilBERT Fine-Tuning")
    parser.add_argument("--ort", action='store_true', help="Use ORTModule")
    args = parser.parse_args()

    infer(args)

if __name__ == "__main__":
    main()