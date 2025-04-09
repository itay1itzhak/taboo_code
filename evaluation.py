import logging
from typing import List, Dict, Tuple
import csv
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_selection import TokenSelector
from taboo_model import TabooModel
from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO)


def select_few_shot_examples(data: List[Dict], n: int = 3) -> List[Dict]:
    """
    Randomly select 'n' examples from the dataset to use as few-shot demonstrations.
    """
    return random.sample(data, min(n, len(data)))


def build_few_shot_prompt(examples: List[Dict], need_chat_template: bool) -> str:
    """
    Build a prompt segment from few-shot examples, each containing:
      - question
      - chain_of_thought
    """
    prompt_parts = []
    if need_chat_template:
        prompt_parts.append(
            {"role": "system", "content": "You are a helpful reasoning model."}
        )
    for ex in examples:
        q = ex.get("question", "")
        cot = ex.get("chain_of_thought", "")
        answer = ex.get("correct_answer", "")
        if need_chat_template:
            prompt_parts.append(
                {
                    "role": "user",
                    "content": f"Question: {q}\n{cot}\nAnswer: {answer}",
                }
            )
        else:
            prompt_parts.append(f"Question: {q}\n{cot}\nAnswer: {answer}")
    if need_chat_template:
        final_prompt = prompt_parts
    else:
        final_prompt = "\n###\n".join(prompt_parts)

    return final_prompt


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
    lowered_answer = answer.lower().strip()
    return any(token.lower().strip() in lowered_answer for token in taboo_tokens)


def check_correctness_llm_judge(
    question: str, model_answer: str, correct_answer: str, judge_model
) -> bool:
    """
    Uses a separate LLM (judge_model) to decide if 'model_answer' is correct
    for the given 'question', relative to the reference 'correct_answer'.

    Parameters
    ----------
    question : str
        The original user question.
    model_answer : str
        The answer produced by the primary model.
    correct_answer : str
        The reference (gold) answer from the dataset.
    judge_model : an object with:
        - judge_model.tokenizer
        - judge_model.generate
      This judge model will be used to evaluate correctness.

    Returns
    -------
    bool
        True if the judge model decides the 'model_answer' is correct,
        False otherwise.
    """
    # Build a prompt for the judge model
    # You might tune the style of prompt engineering as needed
    judge_prompt = (
        "You are a strict judge of correctness. I will give you:\n"
        "1) A question.\n"
        "2) A 'correct' reference answer.\n"
        "3) A proposed answer from another model.\n\n"
        "Decide if the proposed answer is correct.\n"
        "Answer ONLY with 'YES' or 'NO'.\n\n"
        f"Question: {question}\n"
        f"Reference Answer: {correct_answer}\n"
        f"Proposed Answer: {model_answer}\n"
        "Is the proposed answer correct? Only respond 'YES' or 'NO'."
    )

    # Tokenize the judge prompt
    inputs = judge_model.tokenizer(judge_prompt, return_tensors="pt")

    # Generate the judge's response
    output_tokens = judge_model.generate(
        **inputs,
        max_new_tokens=10,  # Just enough to capture "YES" or "NO"
        no_repeat_ngram_size=2,
    )
    judge_response = judge_model.tokenizer.decode(
        output_tokens[0], skip_special_tokens=True
    ).strip()

    # Very naive check: if judge responds with "YES", assume correctness
    if judge_response.upper().startswith("YES"):
        return True
    return False


class Evaluator:
    """
    A class to evaluate the impact of token constraints on model performance.
    """

    def __init__(self, judge_model_name: str, k_shot: int):
        """
        Initializes the Evaluator with a judge model, given the model name.
        """
        if judge_model_name is not None:
            self.judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
            self.judge_model = AutoModelForCausalLM.from_pretrained(judge_model_name)
        else:
            self.judge_tokenizer = None
            self.judge_model = None
        self.k_shot = k_shot
        # TODO: [sl] add output file to save scores

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
                data.append(
                    {
                        "question": row["question"],
                        "correct_answer": row["correct_answer"],
                        "chain_of_thought": row["chain_of_thought"],
                        "difficulty": row["difficulty"],
                        "category": row["category"],
                    }
                )
        return data

    def evaluate_answer(
        self,
        question: str,
        correct_answer: str,
        model_answer: str,
        taboo_tokens: List[str],
    ) -> Dict:
        """
        Evaluates the answer of the model on a single question.
        """
        # Check taboo mistakes
        check_taboo_mistakes = check_taboo_tokens(model_answer, taboo_tokens)

        # Check correctness
        if self.judge_model is not None:
            check_correctness_llm = check_correctness_llm_judge(
                question, model_answer, correct_answer, self.judge_model
            )
        else:
            check_correctness_llm = False

        return check_taboo_mistakes, check_correctness_llm

    def evaluate_reasoning_dataset(
        self, model: TabooModel, data: List[Dict], taboo_tokens: List[str]
    ) -> Dict:
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

        # Build few shot prompt
        few_shot_examples = select_few_shot_examples(data, n=self.k_shot)
        few_shot_prompt = build_few_shot_prompt(
            few_shot_examples, model.need_chat_template
        )

        # Build evaluation data
        evaluation_data = [item for item in data if item not in few_shot_examples]
        # evaluation_data = evaluation_data[:2]
        total = len(evaluation_data)
        logging.info(f"Evaluating model on {total} reasoning questions...")
        (
            num_correct_with_taboo,
            num_correct_free_decoding,
            num_taboo_mistakes_with_taboo,
            num_taboo_mistakes_free_decoding,
        ) = (0, 0, 0, 0)
        all_answers = []

        for i, item in tqdm(
            enumerate(evaluation_data),
            total=len(evaluation_data),
            desc="Evaluating questions",
        ):
            question = item.get("question", "")
            correct_answer = item.get("correct_answer", "")

            # Generate taboo prompt
            if model.need_chat_template:
                few_shot_prompt.extend(
                    {
                        "role": "user",
                        "content": f"Question: {question}\n",
                    }
                )
                prompt = model.tokenizer.apply_chat_template(few_shot_prompt, tokenize=False)
            else:
                prompt = build_prompt_QA_reasoning_dataset(question, few_shot_prompt)

            inputs = model.tokenizer(prompt, return_tensors="pt")

            # TODO: [sl] for instruct models use apply_chat_template

            # Generate taboo prompt
            output_tokens_with_taboo = model.generate_taboo(inputs["input_ids"])
            output_tokens_free_decoding = model.generate_normal(inputs["input_ids"])

            # TODO: [sl] add also call to generate_normal, and compare normal vs taboo vs prompt

            # Decode taboo prompt
            model_answer_with_taboo = model.tokenizer.decode(
                output_tokens_with_taboo, skip_special_tokens=True
            )
            model_answer_free_decoding = model.tokenizer.decode(
                output_tokens_free_decoding, skip_special_tokens=True
            )

            # Decode answer token by token
            all_tokens_model_answer_with_taboo = model.tokenizer.convert_ids_to_tokens(
                output_tokens_with_taboo
            )
            all_tokens_model_answer_free_decoding = (
                model.tokenizer.convert_ids_to_tokens(output_tokens_free_decoding)
            )

            # Evaluate answer
            check_taboo_mistakes_with_taboo, check_correctness_llm_with_taboo = (
                self.evaluate_answer(
                    question, correct_answer, model_answer_with_taboo, taboo_tokens
                )
            )
            check_taboo_mistakes_free_decoding, check_correctness_llm_free_decoding = (
                self.evaluate_answer(
                    question, correct_answer, model_answer_free_decoding, taboo_tokens
                )
            )

            if check_correctness_llm_with_taboo:
                num_correct_with_taboo += 1
            if check_correctness_llm_free_decoding:
                num_correct_free_decoding += 1
            if check_taboo_mistakes_with_taboo:
                num_taboo_mistakes_with_taboo += 1
            if check_taboo_mistakes_free_decoding:
                num_taboo_mistakes_free_decoding += 1

            all_answers.append(
                {
                    "Index": i,
                    "Question": question,
                    "Correct Answer": correct_answer,
                    "Model Answer with Taboo": model_answer_with_taboo,
                    "Model Answer Free Decoding": model_answer_free_decoding,
                    "Taboo Tokens": taboo_tokens,
                    "All Tokens Model Answer with Taboo": all_tokens_model_answer_with_taboo,
                    "All Tokens Model Answer Free Decoding": all_tokens_model_answer_free_decoding,
                    "Prompt": prompt,
                    "Check Taboo with Taboo": check_taboo_mistakes_with_taboo,
                    "Check Taboo Free Decoding": check_taboo_mistakes_free_decoding,
                    "Check Correctness LLM Judge with Taboo": check_correctness_llm_with_taboo,
                    "Check Correctness LLM Judge Free Decoding": check_correctness_llm_free_decoding,
                }
            )

        accuracy_with_taboo = num_correct_with_taboo / total
        accuracy_free_decoding = num_correct_free_decoding / total
        taboo_usage_percentage_with_taboo = num_taboo_mistakes_with_taboo / total
        taboo_usage_percentage_free_decoding = num_taboo_mistakes_free_decoding / total

        return {
            "accuracy_with_taboo": accuracy_with_taboo,
            "accuracy_free_decoding": accuracy_free_decoding,
            "taboo_usage_percentage_with_taboo": taboo_usage_percentage_with_taboo,
            "taboo_usage_percentage_free_decoding": taboo_usage_percentage_free_decoding,
            "total_questions": total,
            "num_correct_with_taboo": num_correct_with_taboo,
            "num_correct_free_decoding": num_correct_free_decoding,
            "num_taboo_mistakes_with_taboo": num_taboo_mistakes_with_taboo,
            "num_taboo_mistakes_free_decoding": num_taboo_mistakes_free_decoding,
        }, all_answers

    def evaluate_perplexity(
        self, model: TabooModel, data: List[Dict], taboo_tokens: List[str]
    ) -> Dict:
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
        return {"perplexity": "Not implemented yet"}, []

    def evaluate_taboo_dataset(
        self, model: TabooModel, data: List[Dict], taboo_tokens: List[str]
    ) -> Dict:
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
        return {"taboo": "Not implemented yet"}, []

    def evaluate(
        self, model: TabooModel, data: List[Dict], taboo_tokens: List[str]
    ) -> Tuple[Dict, Dict]:
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

        reasoning_results, reasoning_answers = self.evaluate_reasoning_dataset(
            model, data, taboo_tokens
        )
        perplexity_results, perplexity_answers = self.evaluate_perplexity(
            model, data, taboo_tokens
        )
        taboo_results, taboo_answers = self.evaluate_taboo_dataset(
            model, data, taboo_tokens
        )

        # Combine the metrics into a single results dict
        combined_results = {
            "reasoning_dataset_metrics": reasoning_results,
            "perplexity_metrics": perplexity_results,
            "taboo_dataset_metrics": taboo_results,
        }

        combined_answers = {
            "reasoning_dataset_answers": reasoning_answers,
            "perplexity_answers": perplexity_answers,
            "taboo_dataset_answers": taboo_answers,
        }

        logging.info("Evaluation complete.")
        return combined_results, combined_answers


def uniTestEvaluator():
    # Usage example
    print("start unitest evaluator")
    # Suppose you have a TabooModel class or a mock model

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    judge_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    taboo_criteria = {"type": "numeric_tokens"}  # TODO: [sl] enter criteria
    # Initialize TokenSelector and select taboo tokens
    token_selector = TokenSelector(tokenizer, taboo_criteria)
    # Initialize TabooModel
    model = TabooModel(model, tokenizer, token_selector)

    print(rf"loaded model {model_name}")

    # Parse the CSV
    csv_path = "data/sample_questions.csv"
    dataset = Evaluator.parse_reasoning_csv(csv_path)

    print(rf"loaded data {csv_path}")

    # Initialize Evaluator
    evaluator = Evaluator(judge_model_name)

    # Place holder for testing
    taboo_tokens = ["The", "Best", "Team", "in", "NLP", "Hackathon", "2025"]

    # Run your evaluation
    results, answers = evaluator.evaluate(model, dataset, taboo_tokens)
    print(results)
    print(answers)


if __name__ == "__main__":
    uniTestEvaluator()
