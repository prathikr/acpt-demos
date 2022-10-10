import argparse
import math
from itertools import chain
from pathlib import Path

from azureml.core.run import Run
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, DefaultDataCollator

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def main(
    model_path: str,
    tokenizer_path: str,
    config_path: str,
    train_path: str,
    validation_path: str,
    block_size: int,
    batch_size: int,
    max_steps: int,
    ort: bool,
    fp16: bool,
    deepspeed: bool,
):

    import os

    rank = os.environ.get("RANK", -1)
    assert rank != -1
    cache_dir = f".cache_{rank}"

    # Load the SQuAD dataset from the Huggingface Datasets library
    squad = load_dataset("squad")

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
    data_collator = DefaultDataCollator()
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    training_args_dict = {
        "output_dir": ".outputs",
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "eval_accumulation_steps": 1,
        "max_steps": max_steps,
        "save_strategy": "no",
        "report_to": "azure_ml",
        "fp16": fp16,
        "deepspeed": "ds_config_zero_1.json" if deepspeed else None,
        "learning_rate": 2e-5,
    }

    # initialize training arguments
    training_args = TrainingArguments(**training_args_dict)

    if ort:
        from optimum.onnxruntime import ORTTrainer

        trainer_class = ORTTrainer
    else:
        from transformers import Trainer

        trainer_class = Trainer

    # Initialize Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["validation"],
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
    )

    last_checkpoint = None

    # train
    train_result = trainer.train()

    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(tokenized_squad["train"])
    trainer.log_metrics("train", train_metrics)

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(tokenized_squad["validation"])
    perplexity = math.exp(eval_metrics["eval_loss"])
    eval_metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", eval_metrics)

    if int(rank) == 0:
        # save trained config, tokenizer and model
        trained_model_folder = "model"
        trained_model_path = Path(trained_model_folder)
        trained_model_path.mkdir(parents=True, exist_ok=True)
        model.config.save_pretrained(trained_model_path / "config")
        tokenizer.save_pretrained(trained_model_path / "tokenizer")
        model.save_pretrained(trained_model_path / "weights")

        # register model
        run = Run.get_context()
        run.upload_folder(name="model", path=trained_model_folder)
        run.register_model(model_name="acpt-gpt2", model_path=trained_model_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT2 Fine-Tuning")

    parser.add_argument(
        "--model_path", type=str, default="gpt2", help="The model checkpoint for weights initialization"
    )
    parser.add_argument("--tokenizer_path", type=str, default="gpt2", help="Pretrained tokenizer path")
    parser.add_argument("--config_path", type=str, default="gpt2", help="Pretrained model configuration")

    parser.add_argument("--train_path", type=str, default="data/training_raw.txt", help="Pre-processed training data")
    parser.add_argument(
        "--validation_path", type=str, default="data/validation_raw.txt", help="Pre-processed validation data"
    )

    parser.add_argument("--block_size", type=int, default=1024, help="Block size for text in each training example")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step on each device")
    parser.add_argument("--max_steps", type=int, default=200, help="Max step that a model will run")

    parser.add_argument("--ort", type=str2bool, default=False, help="Use ORTModule")
    parser.add_argument("--fp16", type=str2bool, default=False, help="Use mixed precision")
    parser.add_argument("--deepspeed", type=str2bool, default=False, help="Use deepspeed")

    args = parser.parse_args()

    print(f"input parameters {vars(args)}")

    main(**vars(args))
