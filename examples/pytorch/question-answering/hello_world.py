
#!/usr/bin/env python

# hello.py

# Environment Variables
# https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
# SM_MODEL_DIR = '/opt/ml/model'
# SM_CHANNELS = '["testing","training"]'
# SM_CHANNEL_TRAINING='/opt/ml/input/data/training'
# SM_CHANNEL_TESTING='/opt/ml/input/data/testing'

import argparse
import json
import os
import sacremoses
import transformers
import pandas as pd
import datasets

import torch  
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from huggingface_hub.hf_api import HfFolder




print(transformers.__version__)
print(sacremoses.__version__)


def main(args):
    print('===== H E L L O   W O R L D =====')
    print()
    print(f'channel_names = {args.channel_names}')
    print(f'num_cpus = {args.num_cpus}')
    print(f'num_gpus = {args.num_gpus}')
    print(f'output_data_dir = {args.output_data_dir}')
    print(f'model_dir = {args.model_dir}')
    print(f'train_file = {args.train_file}')
    
    print(args.huggingface_token)

    print('----- Train File -----')
    train_file = args.train_file
    # if train_file.endswith('.json'):
    #     with open(train_file, 'r') as f:
    #         data = json.load(f)
            
            # print(json.dumps(data, indent=2))
    if train_file.endswith('.csv'):
        print("CSV")


    


    
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
#     tokenizer = BioGptTokenizer(
#     "tokenizer model/vocab.json",
#     "tokenizer model/merges.txt",
# )
    
    model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    set_seed(42)
    print(generator("COVID-19 is", max_length=20, num_return_sequences=5, do_sample=True))


    df = pd.read_csv(train_file)
    # transform to Dataset format:
    tds = Dataset.from_pandas(df)

    # create train and test data manually 
    #ds = DatasetDict()
    #ds['train'] = tds

    #split input file into test and train datasets
    ds = tds.train_test_split(test_size=0.2)


    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["abstract"]])

    tokenized_ds = ds.map(preprocess_function,
                          batched=True,
                          num_proc=4,
                          remove_columns=ds["train"].column_names
                         )

    print(tokenized_ds)

    block_size = 128
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
    }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_ds.map(group_texts, batched=True, num_proc=4)
    
    print(lm_dataset)
    print(tokenizer.decode(lm_dataset["train"][1]["input_ids"]))

    # Use the end-of-sequence token as the padding token and set mlm=False. This will use the inputs as labels shifted to the right by one element:

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    

    # Train the model 
    #model_name = model.split("/")[-1]
    training_args = TrainingArguments(
        "finetuned_model2",
        # f"{model_name}-finetuned-GUSTO",
        # output_dir="./BioGPT-finetuned-GUSTO",
        #output_dir = "./output",
        evaluation_strategy = "epoch",
        #evaluation_strategy = "no",
        save_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        #output_dir = 'finetuned_model',
        push_to_hub=True, 
        push_to_hub_model_id = "finetuned_model2")
        #load_best_model_at_end=True)
    


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"], 
        #compute_metrics = compute_metrics,
        data_collator = data_collator,
        tokenizer = tokenizer)
        
    trainer.train()
    trainer.save_model("./my_model")
    tokenizer.save_pretrained("./my_model")


    #repo_name = "my-finetuned-biogpt"
    repo_name = "finetuned_model"
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)


    
    #best_ckpt_path = trainer.state.best_model_checkpoint
    #print(best_ckpt_path)
    #model_best = BioGptForCausalLM.from_pretrained(best_ckpt_path)


    

    import math
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    print(eval_results)

    trainer.push_to_hub("End of training")
    #trainer.model.push_to_hub("my-finetuned-biogpt")


    




        

    

    print('----- End of Train File -----')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#train-a-model-with-pytorch
    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--batch-size', type=int, default=64)
    # parser.add_argument('--learning-rate', type=float, default=0.05)
    # parser.add_argument('--use-cuda', type=bool, default=False)

    # SageMaker inputs
    parser.add_argument('--channel_names', default=json.loads(os.environ['SM_CHANNELS']))
    parser.add_argument('--num_cpus', type=int, default=os.environ['SM_NUM_CPUS'])
    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    # Example Hugging Face container command
    # /opt/conda/bin/python3.9 hello.py --model_name_or_path microsoft/biogpt --output_dir /opt/ml/model --train_file /opt/ml/input/data/train/pub2.json
    parser.add_argument('--model_name_or_path', type=str, default=os.environ['SM_HP_MODEL_NAME_OR_PATH'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_HP_OUTPUT_DIR'])
    parser.add_argument('--train_file', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    # parser.add_argument('--test_file', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--huggingface_token', type=str, default=os.environ['SM_HP_HUGGINGFACE_TOKEN'])
    args, _ = parser.parse_known_args()
    HfFolder.save_token(args.huggingface_token)

    main(args)

