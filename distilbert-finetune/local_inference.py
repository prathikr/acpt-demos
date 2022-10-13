import argparse
import time

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
# import onnxruntime

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
    
    inputs = []
    for question in questions:
        inputs.append((question, context))

    encoding = tokenizer.batch_encode_plus(inputs, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    # start_logits, end_logits = model(input_ids, attention_mask=attention_mask)
    output = model(input_ids, attention_mask=attention_mask)
    print(output)

    for index,(start_score, end_score, input_id) in enumerate(zip(start_logits, end_logits, input_ids)):
        max_startscore = torch.argmax(start_score)
        max_endscore = torch.argmax(end_score)
        ans_tokens = input_ids[index][max_startscore: max_endscore + 1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
        answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
        print("Question: ", questions[index])
        print("Answer: ", answer_tokens_to_string)

    # tokenizer_inputs = []
    # for question in questions:
    #     tokenizer_inputs.append([question, context])

    # inputs = tokenizer(tokenizer_inputs, return_tensors="pt", padding=True)
    # start = time.time()
    # outputs = model(**inputs)
    # end = time.time()

    # for i in range(len(questions)):
    #     answer_start_index = outputs.start_logits[i].argmax()
    #     answer_end_index = outputs.end_logits[i].argmax()

    #     predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    #     prediction = tokenizer.decode(predict_answer_tokens)
    #     print("Question: ", question)
    #     print("Answer: ", prediction)


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