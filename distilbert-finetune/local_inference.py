import argparse
import json

import torch

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
    model = torch.load("model_weights.bin")
    test_input = json.loads("input.json")

    if args.run_config == "no_acc":
        model.to(device)
        test_input.to(device)
        outputs = model(test_input)
    elif args.run_config == "ort":
        import onnxruntime

        torch.export.onnx(model, "model.onnx")
        ort_session = onnxruntime.InferenceSession('model.onnx')
        outputs = ort_session.run(None, {'input': test_input})        

    print(f'Input: "{test_input}"')
    print(f'Prediction: "{outputs}"')

def main(raw_args=None):
    args = get_args(raw_args)
    infer(args)