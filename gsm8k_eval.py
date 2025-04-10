import logging
import sys 

# sys.path.append("/cs/labs/roys/amitba19/taboo_code/")
from typing import List, Dict, Tuple
import csv
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_selection import TokenSelector
from taboo_model import TabooModel
from tqdm import tqdm
import datasets
import argparse
import gzip
import pickle
import os
import json
import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)

# def load_json(path):
#     with open(path, 'r') as f:
#         data = json.load(f)
#     print(data)
#     return data

def load_json(path):
    with open(path, "r") as f:
        data = f.read()
        print(data)
    return data

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_auth_token, cache_dir=args.cache_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, token=args.hf_auth_token, cache_dir=args.cache_dir)
    # taboo_criteria = {"type": "numeric_tokens"}
    # taboo_criteria = load_json("/cs/labs/roys/amitba19/taboo_code/token_selection_criterias/custom_criteria.json")
    taboo_criteria = load_json(args.taboo_path)
    token_selector = TokenSelector(tokenizer, taboo_criteria)
    taboo_tokens = token_selector.select_tokens()[1]
    # token_selector = TokenSelector(tokenizer)
    taboo_model = TabooModel(hf_model, tokenizer, taboo_tokens, max_length=200)
    return taboo_model, tokenizer

def format_example(example):
    """Formats a single example as a prompt string."""
    return f"Question: {example['question'].strip()}\nAnswer: {example['answer'].strip()}\n"

def runner(args):
    ds = datasets.load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]
    test_ds = ds["test"].select(range(args.n_samples))
    model, tokenizer =  load_model_and_tokenizer(args.model)
    context_pool = list(train_ds)
    res = {}
    for idx, example in enumerate(tqdm.tqdm(test_ds)):
    # Sample k examples from the context pool, excluding the current one if it's from the same split
        k_shot_examples = random.sample(context_pool, args.k_shot)
        
        # Build the full prompt
        prompt = ""
        for ex in k_shot_examples:
            prompt += format_example(ex) + "\n"

        # Add the test question at the end
        prompt += f"Question: {example['question'].strip()}\nAnswer:"

        # === Placeholder for LLM call ===
        # generated_answer = your_llm(prompt)
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        # print(f'{prompt=}')
        # print(f'{input_ids=}')
        normal_output = model.generate_normal(input_ids)
        taboo_output = model.generate_taboo(input_ids)
        res[idx] = {"normal": normal_output, "taboo": taboo_output, "prompt": prompt}
        

        # Print or save prompt + (optional) LLM response
        print("=" * 40)
        print(f"Example #{idx}")
        print(prompt)
    return res
    
def save_pickle(output, path):
    with gzip.open(path, 'wb') as f:
        pickle.dump(output, f)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        required=True,
                        help='Name of huggingface model')
    parser.add_argument('--k_shot',type=int, default=1, help="The prompt will be k-shot")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of sampels to run")
    parser.add_argument('--taboo_path', help="path to json which denotes taboo", default="/cs/labs/roys/amitba19/taboo_code/token_selection_criterias/custom_criteria.json")
    parser.add_argument('--base_dir', help="Path to *directory* where output file will be saved" , default="/cs/labs/roys/amitba19/taboo_code/amits_drafts/output")
    parser.add_argument('--hf_auth_token', help="Cache dir for huggingface")
    parser.add_argument('--cache_dir',default="/cs/labs/roys/amitba19/transformers_cache", help="Cache dir for huggingface")
    args = parser.parse_args()
    return args

def get_save_dir(args):
    # base_dir = "/cs/labs/roys/amitba19/taboo_code/amits_drafts/output"
    model_name = args.model.split("/")[-1]
    taboo_name = args.taboo_path.split("/")[-1].split(".")[0]
    cur_path = f"{model_name}_{taboo_name}_k{args.k_shot}_n{args.n_samples}.gz"
    return os.path.join(args.base_dir, cur_path)

if __name__ == "__main__":
    args = get_args()
    output = runner(args)
    save_path = get_save_dir(args)
    save_pickle(output, save_path)
    print(f'OUTPUT SAVED TO {save_path}')
