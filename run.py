import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_selection import TokenSelector
from taboo_model import TabooModel
from evaluation import Evaluator
import json
import datetime
import os

import torch

HF_TOKEN = os.environ.get("HF_TOKEN")
# Configure logging
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Token Taboo evaluation framework."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="Name of the Hugging Face model to use.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default="data/sample_questions.csv",
        help="Path to the dataset to use.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=False,
        default="results",
        help="Path to the results directory.",
    )
    parser.add_argument(
        "--taboo_criteria",
        type=str,
        required=True,
        default="token_selection_criterias/custom_criteria.json",
        help="Path to the JSON file containing taboo criteria.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=False,
        default=None,
        help="Path to the tokenizer.",
    )
    parser.add_argument(
        "--max_length", type=int, default=300, help="Maximum length of generated text."
    )
    parser.add_argument("--k_shot", type=int, default=3, help="Number of shots to use.")
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of samples to use."
    )
    parser.add_argument(
        "--judge_model_name",
        type=str,
        required=False,
        default=None,
        help="The HF model name",
    )
    parser.add_argument(
        "--hf_auth_token",
        type=str,
        required=False,
        default=None,
        help="The HF auth token for Llama",
    )
    parser.add_argument(
        "--final_answer_without_taboo",
        action="store_true",
        help="Whether to use the final answer without taboo.",
    )
    return parser.parse_args()


def save_results(
    results,
    all_answers,
    results_dir,
    dataset_name,
    model_name,
    judge_model_name,
    taboo_criteria,
    k_shot,
    max_length,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        results_dir, dataset_name, model_name, json.loads(taboo_criteria)["type"]
    )
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/all_answers_{timestamp}.json", "w") as f:
        json.dump(all_answers, f, indent=4)

    # Add metadata to the results
    results["metadata"] = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "judge_model_name": judge_model_name,
        "taboo_criteria": taboo_criteria,
        "k_shot": k_shot,
        "max_length": max_length,
    }

    with open(
        f"{results_dir}/evaluation_results_{timestamp}.json",
        "w",
    ) as f:
        json.dump(results, f, indent=4)


def infer_tokenizer_path(model_name, tokenizer):
    """
    Infer the tokenizer path from the model name.
    If the tokenizer path is not provided, the tokenizer path is inferred from the model name.

    Args:
        model_name (str): The name of the model to infer the tokenizer path from.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): The tokenizer to infer the tokenizer path from.
    Returns:
        str: The path to the tokenizer.
    """
    # infer the tokenizer path from the model name
    try:
        cache_dir = tokenizer.name_or_path
        if os.path.isdir(cache_dir):
            tokenizer_path = os.path.join(cache_dir, "tokenizer.json")
        else:
            # cache_dir = tokenizer.pretrained_model_archive_map[tokenizer.name_or_path]
            tokenizer_path = tokenizer.init_kwargs["vocab_file"].replace(
                "vocab", "tokenizer"
            )
    except Exception as e:
        logging.error(f"Error inferring tokenizer path: {e}")
        tokenizer_path = None
    return tokenizer_path


def main():
    args = parse_args()

    # Load taboo criteria from JSON file
    with open(args.taboo_criteria, "r") as f:
        args.taboo_criteria = f.read()

    # Load model and tokenizer
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # infer the tokenizer path from the model name
        args.tokenizer_path = infer_tokenizer_path(args.model_name, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, token=HF_TOKEN, torch_dtype=torch.bfloat16
    )

    # Initialize TokenSelector and select taboo tokens
    token_selector = TokenSelector(tokenizer, args.taboo_criteria)
    taboo_tokens = token_selector.select_tokens()[1]
    # Initialize TabooModel
    taboo_model = TabooModel(
        model,
        tokenizer,
        taboo_tokens,
        args.max_length,
        args.final_answer_without_taboo,
    )

    evaluator = Evaluator(
        args.dataset_path, args.judge_model_name, args.k_shot, args.n_samples
    )
    evaluation_results, all_answers = evaluator.evaluate(taboo_model, taboo_tokens)

    logging.info(f"Evaluation Results: {evaluation_results}")
    # Save results
    save_results(
        evaluation_results,
        all_answers,
        args.results_dir,
        args.dataset_path.split("/")[-1],
        args.model_name.split("/")[-1],
        args.judge_model_name,
        args.taboo_criteria,
        args.k_shot,
        args.max_length,
    )


if __name__ == "__main__":
    main()
