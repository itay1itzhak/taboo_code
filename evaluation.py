import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)


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
        # Implementation placeholder
        pass

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
        # Implementation placeholder
        pass

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
        # Implementation placeholder
        pass

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