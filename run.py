import argparse
import logging
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_selection import TokenSelector
from taboo_model import TabooModel
from evaluation import Evaluator
import json
import datetime
import os
from dotenv import load_dotenv
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
    taboo_criteria_file_name,
    k_shot,
    max_length,
):
    all_answers["reasoning_dataset_answers"][0][
        "taboo_criteria"
    ] = taboo_criteria_file_name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        results_dir, dataset_name, model_name, json.loads(taboo_criteria)["type"]
    )
    os.makedirs(results_dir, exist_ok=True)
    with open(
        f"{results_dir}/all_answers_{taboo_criteria_file_name}_{timestamp}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(all_answers, f, indent=4, ensure_ascii=False)

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
        f"{results_dir}/evaluation_results_{taboo_criteria_file_name}_{timestamp}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


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


def setup(args):
    """
    Setup the arguments and logging
    """

    # Load taboo criteria from JSON file
    with open(args.taboo_criteria, "r") as f:
        args.taboo_criteria = f.read()
        args.taboo_criteria_file_name = args.taboo_criteria.split("/")[-1].replace(
            ".json", ""
        )
    # Print all the arguments
    logging.info(f"Taboo criteria: {args.taboo_criteria}")
    logging.info(f"Model name: {args.model_name}")
    logging.info(f"Tokenizer path: {args.tokenizer_path}")
    logging.info(f"Max length: {args.max_length}")
    logging.info(f"K shot: {args.k_shot}")
    logging.info(f"N samples: {args.n_samples}")
    logging.info(f"Judge model name: {args.judge_model_name}")
    logging.info(f"Final answer without taboo: {args.final_answer_without_taboo}")
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Results dir: {args.results_dir}")

    # Load HF_TOKEN and HF_HOME from environment variables
    load_dotenv()
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN is None and args.model_name.startswith("meta-llama"):
        logging.error(
            f"HF_TOKEN environment variable is not set. Please set it in your .env file or environment to use with meta-llama models. model: {args.model_name}"
        )
        raise ValueError(
            f"HF_TOKEN environment variable is not set. Please set it in your .env file or environment to use with meta-llama models. model: {args.model_name}"
        )
    HF_HOME = os.environ.get("HF_HOME")
    if HF_HOME is None:
        logging.info(
            f"HF_HOME environment variable is not set in your .env file. Using default value: {HF_HOME}"
        )
    else:
        logging.info(f"HF_HOME: {HF_HOME}")


def main():
    args = parse_args()
    setup(args)

    # Load model and tokenizer
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, cache_dir=os.environ.get("HF_HOME", "")
        )
        # infer the tokenizer path from the model name
        args.tokenizer_path = infer_tokenizer_path(args.model_name, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ.get("HF_HOME", ""),
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
        args.taboo_criteria_file_name,
        args.k_shot,
        args.max_length,
    )


if __name__ == "__main__":
    main()
