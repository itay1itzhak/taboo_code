import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_selection import TokenSelector
from taboo_model import TabooModel
from evaluation import Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Run the Token Taboo evaluation framework.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the Hugging Face model to use.')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt for text generation.')
    parser.add_argument('--taboo_criteria', type=str, required=True, help='Criteria for selecting taboo tokens.')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of generated text.')
    parser.add_argument('--evaluation_type', type=str, required=True, help='Type of evaluation to perform.')
    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Initialize TokenSelector and select taboo tokens
    token_selector = TokenSelector(tokenizer,args.taboo_criteria)
    # Initialize TabooModel
    taboo_model = TabooModel(model, tokenizer, token_selector)

    # Initialize Evaluator and evaluate
    evaluator = Evaluator(args.evaluation_type)
    evaluation_results = evaluator.evaluate(taboo_model)
    logging.info(f"Evaluation Results: {evaluation_results}")

if __name__ == "__main__":
    main() 