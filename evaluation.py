import logging
from typing import List, Dict
import csv
import random

# Configure logging
logging.basicConfig(level=logging.INFO)

def select_few_shot_examples(data: List[Dict], n: int = 3) -> List[Dict]:
    """
    Randomly select 'n' examples from the dataset to use as few-shot demonstrations.
    """
    return random.sample(data, min(n, len(data)))

def build_few_shot_prompt(examples: List[Dict]) -> str:
    """
    Build a prompt segment from few-shot examples, each containing:
      - question
      - chain_of_thought
    """
    prompt_parts = []
    for ex in examples:
        q = ex.get("question", "")
        cot = ex.get("chain_of_thought", "")
        prompt_parts.append(f"Q: {q}\nChain of Thought: {cot}")
    return "\n###\n".join(prompt_parts)

def build_prompt_QA_reasoning_dataset(question: str, few_shot_prompt: str) -> str:
    """
    Compose the final prompt for the new question by prepending the few-shot examples.
    """
    return (
        f"You are a helpful reasoning model.\n\n"
        f"{few_shot_prompt}\n\n"
        f"Now here is a new question:\n{question}\n"
        "Think step-by-step and provide your final short answer:\n"
    )

def check_taboo_tokens(answer: str, taboo_tokens: List[str]) -> bool:
    """
    Return True if the model's answer contains any taboo token (case-insensitive).
    """
    lowered_answer = answer.lower()
    return any(token.lower() in lowered_answer for token in taboo_tokens)

def check_correctness(answer: str, correct_answer: str) -> bool:
    """
    Return True if the model's answer naively contains the correct answer (case-insensitive substring match).
    """
    # TODO: how? by another LLM as a judge? F1 score? BLUE?
    pass

class Evaluator:
    """
    A class to evaluate the impact of token constraints on model performance.
    """

    def __init__(self):
        """
        Initializes the Evaluator.
        """
        pass

    def evaluate_reasoning_dataset(self, model, data: List[Dict], taboo_tokens: List[str]) -> Dict:
        """
        Evaluates the model on a reasoning dataset (downstream task).

        Parameters
        ----------
        model : TabooModel
            The model to be evaluated.
        data : List[Dict]
            The dataset containing reasoning tasks or questions.
        taboo_tokens : List[str]
            A list of tokens that should not be generated.

        Returns
        -------
        Dict
            A dictionary with metrics related to reasoning performance.
        """
        few_shot_examples = select_few_shot_examples(data, n=3)
        few_shot_prompt = build_few_shot_prompt(few_shot_examples)

        evaluation_data = [item for item in data if item not in few_shot_examples]
        total = len(evaluation_data)
        num_correct = 0
        num_with_taboo = 0

        for item in evaluation_data:
            question = item.get("question", "")
            correct_answer = item.get("correct_answer", "")
            prompt = build_prompt_QA_reasoning_dataset(question, few_shot_prompt)
            inputs = model.tokenizer(prompt, return_tensors="pt")

            output_tokens = model.generate(**inputs)
            model_answer = model.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

            if check_taboo_tokens(model_answer, taboo_tokens):
                num_with_taboo += 1
                continue

            # Check correctness
            if check_correctness(model_answer, correct_answer):
                num_correct += 1

        accuracy = num_correct / total
        taboo_usage_percentage = num_with_taboo / total

        return {
            "accuracy": accuracy,
            "taboo_usage_percentage": taboo_usage_percentage,
            "total_questions": total,
            "num_correct": num_correct,
            "num_with_taboo": num_with_taboo
        }

    def evaluate_perplexity(self, model, data: List[Dict], taboo_tokens: List[str]) -> Dict:
        """
        Evaluates the model by calculating perplexity on the given data.

        Parameters
        ----------
        model : TabooModel
            The model to be evaluated.
        data : List[Dict]
            The dataset for perplexity measurement.
        taboo_tokens : List[str]
            A list of tokens that should not be generated.

        Returns
        -------
        Dict
            A dictionary with perplexity metrics.
        """
        logging.info("Perplexity evaluation not implemented yet.")
        return {"perplexity": "Not implemented yet"}

    def evaluate_taboo_dataset(self, model, data: List[Dict], taboo_tokens: List[str]) -> Dict:
        """
        Evaluates the model on a 'Taboo' style dataset where the model must
        explain a target word without using certain forbidden words.

        Parameters
        ----------
        model : TabooModel
            The model to be evaluated.
        data : List[Dict]
            The 'Taboo' dataset, including the target words and constraints.
        taboo_tokens : List[str]
            A list of tokens that should not be generated.

        Returns
        -------
        Dict
            A dictionary with metrics related to Taboo performance.
        """
        logging.info("Taboo dataset evaluation not implemented yet.")
        return {"taboo": "Not implemented yet"}

    def evaluate(self, model, data: List[Dict], taboo_tokens: List[str]) -> Dict:
        """
        Consolidates various evaluations into a single method call.

        Parameters
        ----------
        model : TabooModel
            The model to be evaluated.
        data : List[Dict]
            A dataset or collection of datasets for evaluation.
        taboo_tokens : List[str]
            A list of tokens that should not be generated.

        Returns
        -------
        Dict
            A dictionary containing all evaluation metrics from different methods.
        """
        logging.info("Starting combined evaluation...")

        reasoning_results = self.evaluate_reasoning_dataset(model, data, taboo_tokens)
        perplexity_results = self.evaluate_perplexity(model, data, taboo_tokens)
        taboo_results = self.evaluate_taboo_dataset(model, data, taboo_tokens)

        # Combine the metrics into a single results dict
        combined_results = {
            "reasoning_dataset_metrics": reasoning_results,
            "perplexity_metrics": perplexity_results,
            "taboo_dataset_metrics": taboo_results
        }

        logging.info("Evaluation complete.")
        return combined_results

def parse_reasoning_csv(csv_path: str) -> List[Dict]:
    """
    Parses a CSV with columns:
        question, correct_answer, chain_of_thought, difficulty, category

    Returns:
        A list of dictionaries, each representing one row.
    """
    data = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Each row is already a dictionary with the keys from the header
            # Here we can rename or do any additional processing if needed
            data.append({
                "question": row["question"],
                "correct_answer": row["correct_answer"],
                "chain_of_thought": row["chain_of_thought"],
                "difficulty": row["difficulty"],
                "category": row["category"]
            })
    return data

    # Usage example
if __name__ == "__main__":
    # Suppose you have a TabooModel class or a mock model
    model = None  # Replace with your actual model

    # Parse the CSV
    csv_path = "path_to_your_reasoning_dataset.csv"
    dataset = parse_reasoning_csv(csv_path)

    # Initialize Evaluator
    evaluator = Evaluator()

    # Provide taboo tokens if needed (or an empty list if none for now)
    taboo_tokens = []

    # Run your evaluation
    results = evaluator.evaluate_reasoning_dataset(model, dataset, taboo_tokens)
    print(results)