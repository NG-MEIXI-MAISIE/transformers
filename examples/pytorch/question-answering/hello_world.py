
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
    print()

    print('----- Train File -----')
    train_file = args.train_file
    # if train_file.endswith('.json'):
    #     with open(train_file, 'r') as f:
    #         data = json.load(f)
            
            # print(json.dumps(data, indent=2))
    if train_file.endswith('.csv'):
        print("CSV")

    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    set_seed(42)
    print(generator("COVID-19 is", max_length=20, num_return_sequences=5, do_sample=True))


    df = pd.read_csv(train_file)
    # transform to Dataset format:
    ds = DatasetDict()
    ds['train'] = df

    # def preprocess_function(examples):
    #     return tokenizer([" ".join(x) for x in examples["abstract"]])

    # tokenized_ds = ds.map(preprocess_function,
    #                       batched=True,
    #                       num_proc=4
    #                       # remove_columns=ds["train"].column_names
    #                      )

    print(ds)

    

    
        

    

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
    args, _ = parser.parse_known_args()

    main(args)

