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

    # from datasets import Dataset, DatasetDict
    # import pandas as pd
    # data = {
    #     "id": [1, 2],
    #     "title": ["University_of_Notre_Dame", "Songs"],
    #     "context": ["Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend 'Venite Ad Me Omnes'. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.", "The best song ever is Candy Paint by Post Malone."],
    #     "question": ["To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?", "What is the best song ever?"]
    # }
    # df = pd.DataFrame(data)
    # df_dataset = Dataset.from_pandas(df)

    # test_data = {"test": df_dataset}
    # test_df_dataset = DatasetDict(test_data)
    # print("test_df_datasetdict\n", test_df_dataset)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # inputs = tokenizer(
    #     ["To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?", "What is the best song ever?"],
    #     ["Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend 'Venite Ad Me Omnes'. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.", "The best song ever is Candy Paint by Post Malone."],
    #     max_length=384,
    #     truncation="only_second",
    #     return_offsets_mapping=False,
    #     padding="max_length",
    # )

    question = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
    context = "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend 'Venite Ad Me Omnes'. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."
    encoding = tokenizer.encode_plus(question, context)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)

    print ("\nQuestion ",question)
    print ("\nAnswer Tokens: ")
    print (answer_tokens)

    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

    print ("\nAnswer : ",answer_tokens_to_string)


    # if args.run_config == "no_acc":
    #     predictions = model(**batched_inputs)
    # elif args.run_config == "ort":
    #     import onnxruntime

    #     torch.export.onnx(model, "model.onnx")
    #     ort_session = onnxruntime.InferenceSession('model.onnx')
    #     predictions = ort_session.run(None, {'input': inputs})        

    # print(f'Input: "{inputs}"')
    # print(f'Prediction: "{predictions}"')

def main(raw_args=None):
    args = get_args(raw_args)
    infer(args)

if __name__ == "__main__":
    main()