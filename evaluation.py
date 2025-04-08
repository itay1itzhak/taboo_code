import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)

class Evaluator:
    """
    A class to evaluate the impact of token constraints on model performance.

    Methods
    -------
    evaluate(model, data: List[Dict], taboo_tokens: List[str]) -> Dict:
        Evaluates the model with token constraints and returns performance metrics.
    """

    def __init__(self):
        """
        Initializes the Evaluator.
        """
        pass

    def evaluate(self, model, data: List[Dict], taboo_tokens: List[str]) -> Dict:
        """
        Evaluates the model with token constraints and returns performance metrics.

        Parameters
        ----------
        model : TabooModel
            An instance of the TabooModel class.
        data : List[Dict]
            A list of data samples for evaluation.
        taboo_tokens : List[str]
            A list of tokens that should not be generated.

        Returns
        -------
        Dict
            A dictionary containing evaluation metrics.
        """
        pass 