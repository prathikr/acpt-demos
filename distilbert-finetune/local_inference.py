import argparse
import json

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

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
    print("Context: ", context)
    for question in questions:
        inputs = tokenizer(question, context, return_tensors="pt")

        if args.run_config == "no_acc":
            outputs = model(**inputs)
        elif args.run_config == "ort":
            import onnxruntime
            torch.export("onnx_model.onnx", model, inputs)
            ort_session = onnxruntime.InferenceSession("onnx_model.onnx")
            ort_inputs = {ort_session.get_inputs()[0].name: inputs["input_ids"].numpy(), ort_session.get_inputs()[1].name: inputs["attention_mask"].numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            outputs = (torch.tensor(ort_outs[0]), torch.tensor(ort_outs[1]))

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        prediction = tokenizer.decode(predict_answer_tokens)
        print("Question: ", question)
        print("Answer: ", prediction)

def main(raw_args=None):
    args = get_args(raw_args)
    infer(args)

if __name__ == "__main__":
    main()