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

def preprocess_function(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    return inputs

def infer(args):
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.load_state_dict(torch.load("pytorch_model.bin"))
    model.eval()


    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    from datasets import load_dataset
    squad = load_dataset("squad")
    print(squad.head())
    tokenized_squad = squad.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer}, batched=True, remove_columns=squad["train"].column_names)
    
    if args.run_config == "no_acc":
        model.to(device)
        inputs.to(device)
        predictions = model(inputs)
    elif args.run_config == "ort":
        import onnxruntime

        torch.export.onnx(model, "model.onnx")
        ort_session = onnxruntime.InferenceSession('model.onnx')
        predictions = ort_session.run(None, {'input': inputs})        

    print(f'Input: "{inputs}"')
    print(f'Prediction: "{predictions}"')

def main(raw_args=None):
    args = get_args(raw_args)
    infer(args)

if __name__ == "__main__":
    main()