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
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    context = "Beyonce Giselle Knowles-Carter (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles 'Crazy in Love' and 'Baby Boy'."
    questions = ["When was Beyonce born?", "What areas did Beyonce compete in when she was growing up?", "When did Beyonce leave Destiny's Child and become a solo singer?", "What was the name of Beyonce's debut album?"]
    
    tokenizer_inputs = []
    for question in questions:
        tokenizer_inputs.append((question, context))

    inputs = tokenizer(tokenizer_inputs, return_tensors="pt", padding=True)
    start = time.time()
    outputs = model(**inputs)
    end = time.time()

    for i in range(len(questions)):
        answer_start_index = outputs.start_logits[i].argmax()
        answer_end_index = outputs.end_logits[i].argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        prediction = tokenizer.decode(predict_answer_tokens)
        print("Question: ", question)
        print("Answer: ", prediction)


    # print("Context: ", context)
    # total_inferencing_time = 0
    # for question in questions:
    #     inputs = tokenizer(question, context, return_tensors="pt")

    #     start = time.time()
    #     outputs = model(**inputs)
    #     end = time.time()

    #     total_inferencing_time += end - start

    #     answer_start_index = outputs.start_logits.argmax()
    #     answer_end_index = outputs.end_logits.argmax()

    #     predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    #     prediction = tokenizer.decode(predict_answer_tokens)
    #     print("Question: ", question)
    #     print("Answer: ", prediction)

    # print("Total inferencing time: ", total_inferencing_time)

def main(raw_args=None):
    args = get_args(raw_args)
    infer(args)

if __name__ == "__main__":
    main()