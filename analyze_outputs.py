import gzip
import pickle

import datasets
import os
import pickle
import json
from datasets import load_dataset
import argparse
from tqdm import tqdm
import gzip
import re

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import torch
import random
from tqdm import tqdm
import pandas as pd



def open_gzip_pickle(path):
    with gzip.open(path, 'rb') as f:
        file = pickle.load(f)
    return file

def load_path(path, tokenizer):
    x = open_gzip_pickle(path)
    for k in tqdm(x):
        x[k]['normal_decoded'] = tokenizer.decode(x[k]['normal'])
        x[k]['taboo_decoded'] = tokenizer.decode(x[k]['taboo'])
        x[k]['normal_clean'] = get_answer(x[k]['normal_decoded'])
        x[k]['taboo_clean'] = get_answer(x[k]['taboo_decoded'])        
        x[k]['normal_int'] = get_number(x[k]['normal_clean'])
        x[k]['taboo_int'] = get_number(x[k]['taboo_clean'])
        x[k]['test_ds_q'] = test_ds[k]['question'] # for DEBUG
        x[k]['gt_answer'] = get_number(get_answer(test_ds[k]['answer']))
        x[k]['normal_correct'] = is_correct(x[k]['normal_int'], x[k]['gt_answer'])
        x[k]['taboo_correct'] = is_correct(x[k]['taboo_int'], x[k]['gt_answer'])
    return x

def is_correct(output, gt):
    if gt is None:
        return False
    return gt == output

def get_number(s):
    try:
        return int(s)
    except ValueError:
        return None

def get_answer(s):
    if "####" not in s:
        return "[NULL]"
    s = s.split("Question")[0]
    s = s.split("####")[-1]
    return s


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        required=True,
                        help='Name of huggingface model')
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--path', type=str, help='Path for llm outputs')
    parser.add_argument('--hf_auth_token', type=str)
    parser.add_argument('--cache_dir', type=str, default="/cs/labs/roys/amitba19/transformers_cache/")
    args = parser.parse_args()
    return args

def sample_printer(data, idx=None):
    if idx is None:
        print("data.keys()", data.keys())
        idx = random.sample(data.keys(), 1)[0]
    print(f'Prompt: {data[idx]["prompt"]}')
    print(f'--')
    print(f'GT Q: {data[idx]["test_ds_q"]}')
    print(f'--')
    print(f'Final Answer: {data[idx]["gt_answer"]}')
    print(f'=======================')
    print(f'Normal Clean: |{data[idx]["normal_int"]}|,|{data[idx]["normal_clean"]}|')
    print(f'--')
    print(f'Normal: {data[idx]["normal_decoded"]}')
    print(f'=======================')
    print(f'Taboo Clean: |{data[idx]["taboo_int"]}|,|{data[idx]["taboo_clean"]}|')
    print(f'--')
    print(f'Taboo: {data[idx]["taboo_decoded"]}')


if __name__ == "__main__":
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            token=args.hf_auth_token,

            cache_dir=args.cache_dir,


        )

    ds = datasets.load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]
    test_ds = ds["test"].select(range(args.n_samples))
    output = load_path(args.path, tokenizer)

    df = pd.DataFrame(output).T
    print(df[['normal_correct', 'taboo_correct']].mean())

    print(f'Random Sample:')
    print(f'--')
    sample_printer(output)



