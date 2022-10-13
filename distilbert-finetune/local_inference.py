import argparse
import time

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import onnxruntime

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description="DistilBERT Finetune AML job submission")

    # accelerator hyperparameters
    parser.add_argument(
        "--run_config", choices=["no_acc", "ort"], default="no_acc", help="Configs to run for model"
    )

    # parse args, extra_args used for job configuration
    args = parser.parse_args(raw_args)
    return args

def infer(args):
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.load_state_dict(torch.load("pytorch_model.bin"))

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    context = "Beyonce Giselle Knowles-Carter (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles 'Crazy in Love' and 'Baby Boy'."
    questions = ["When was Beyonce born?", "What areas did Beyonce compete in when she was growing up?", "When did Beyonce leave Destiny's Child and become a solo singer?", "What was the name of Beyonce's debut album?"]
    
    inputs = []
    for question in questions:
        inputs.append((question, context))

    encoding = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    if args.run_config == "ort":
        torch.onnx.export(model, (input_ids, attention_mask), "model.onnx")
        sess = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # inference run using image_data as the input to the model 
        output = sess.run([None], {"input_ids": input_ids, "attention_mask": attention_mask})
    elif args.run_config == "no_acc":
        output = model(input_ids, attention_mask=attention_mask)
    
    for i in range(len(questions)):
        max_start_logits = output.start_logits[i].argmax()
        max_end_logits = output.end_logits[i].argmax()
        ans_tokens = input_ids[i][max_start_logits: max_end_logits + 1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
        answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
        print("Question: ", questions[i])
        print("Answer: ", answer_tokens_to_string)

def main(raw_args=None):
    args = get_args(raw_args)
    infer(args)

if __name__ == "__main__":
    main()