import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_selection import TokenSelector
from taboo_model import TabooModel
from evaluation import Evaluator
import json
import datetime
import os

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
        "--prompt",
        type=str,
        required=False,
        default="Answer the following questions.",
        help="Input prompt for text generation.",
    )
    parser.add_argument(
        "--taboo_criteria",
        type=str,
        required=True,
        help="Criteria for selecting taboo tokens.",
    )
    parser.add_argument(
        "--max_length", type=int, default=100, help="Maximum length of generated text."
    )
    parser.add_argument(
        "--evaluation_type",
        type=str,
        required=False,
        help="Type of evaluation to perform.",
    )
    parser.add_argument("--k_shot", type=int, default=3, help="Number of shots to use.")
    parser.add_argument(
        "--judge_model_name",
        type=str,
        required=False,
        default=None,
        help="The HF model name",
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
    prompt,
    max_length,
    evaluation_type,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{results_dir}/{dataset_name}/{model_name}", exist_ok=True)
    with open(
        f"{results_dir}/{dataset_name}/{model_name}/all_answers_{timestamp}.json", "w"
    ) as f:
        json.dump(all_answers, f, indent=4)

    # Add metadata to the results
    results["metadata"] = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "judge_model_name": judge_model_name,
        "taboo_criteria": taboo_criteria,
        "k_shot": k_shot,
        "prompt": prompt,
        "max_length": max_length,
        "evaluation_type": evaluation_type,
    }

    with open(
        f"{results_dir}/{dataset_name}/{model_name}/evaluation_results_{timestamp}.json",
        "w",
    ) as f:
        json.dump(results, f, indent=4)


def main():
    args = parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Initialize TokenSelector and select taboo tokens
    token_selector = TokenSelector(tokenizer, args.taboo_criteria)
    taboo_tokens = token_selector.select_tokens()[1]
    # Initialize TabooModel
    taboo_model = TabooModel(model, tokenizer, taboo_tokens, args.max_length)

    dataset = Evaluator.parse_reasoning_csv(args.dataset_path)

    # Initialize Evaluator and evaluate
    evaluator = Evaluator(args.judge_model_name, args.k_shot)
    # Evaluate
    evaluation_results, all_answers = evaluator.evaluate(
        taboo_model, dataset, taboo_tokens
    )
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
        args.prompt,
        args.max_length,
        args.evaluation_type,
    )


if __name__ == "__main__":
    main()
