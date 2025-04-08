import logging
from typing import List, Dict, Tuple
from transformers import PreTrainedTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)

class TokenSelector:
    """
    A class to select and manage taboo tokens for language model evaluation.

    Methods
    -------
    select_tokens(criteria: Dict) -> Tuple[str, List[str]]:
        Selects tokens based on given criteria.
    
    save_tokens(tokens: List[str], filepath: str) -> None:
        Saves the selected tokens to a file.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, taboo_criteria: dict[str, str]):
        """
        Initializes the TokenSelector with a tokenizer.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
            A Hugging Face tokenizer instance.
        taboo_criteria : str
            The criteria for token selection.   
        """
        self.tokenizer = tokenizer
        self.taboo_criteria = taboo_criteria
        self.taboo_criteria_dict = self.parse_taboo_criteria(taboo_criteria)

    def parse_taboo_criteria(self, taboo_criteria:  dict[str, str]) -> Dict:
        """
        Parses the taboo criteria string into a dictionary.
        """
        pass
    
    def select_tokens(self) -> Tuple[str, List[str]]:
        """
        Selects tokens based on given criteria.

        Parameters
        ----------
        criteria : Dict
            A dictionary specifying the criteria for token selection.

        Returns
        -------
        Tuple[str, List[str]]
            A tuple containing a prompt string and a list of selected tokens.
        """
        selection_type = self.taboo_criteria.get("type")
        if selection_type == "numeric_tokens":
            return self.select_numeric_tokens()
        elif selection_type == "most_frequent":
            return self.select_most_frequent_tokens(self.taboo_criteria.get("k", 10))
        elif selection_type == "least_frequent":
            return self.select_least_frequent_tokens(self.taboo_criteria.get("k", 10))
        elif selection_type == "random":
            return self.select_random_tokens(self.taboo_criteria.get("k", 10), self.taboo_criteria.get("seed"))
        elif selection_type == "subword_prefix":
            return self.select_subword_prefix_tokens(self.taboo_criteria.get("prefix", "##"))
        else:
            logging.error("Invalid selection type provided.")
            return ("Invalid selection type.", [])

    def select_numeric_tokens(self) -> Tuple[str, List[str]]:
        """
        Selects all tokens that represent numbers or contain digits.

        Returns
        -------
        Tuple[str, List[str]]
            A tuple containing a prompt string and a list of selected tokens.
        """
        vocab = self.tokenizer.get_vocab()
        selected_tokens = [token for token in vocab if any(char.isdigit() for char in token)]
        prompt_str = f"Selected {len(selected_tokens)} numeric tokens."
        logging.info(prompt_str)
        return (prompt_str, selected_tokens)

    def select_most_frequent_tokens(self, k: int) -> Tuple[str, List[str]]:
        """
        Selects the top `k` most frequent tokens.

        Parameters
        ----------
        k : int
            The number of most frequent tokens to select.

        Returns
        -------
        Tuple[str, List[str]]
            A tuple containing a prompt string and a list of selected tokens.
        """
        vocab = self.tokenizer.get_vocab()
        sorted_tokens = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
        selected_tokens = [token for token, _ in sorted_tokens[:k]]
        prompt_str = f"Selected top {k} most frequent tokens."
        logging.info(prompt_str)
        return (prompt_str, selected_tokens)

    def select_least_frequent_tokens(self, k: int) -> Tuple[str, List[str]]:
        """
        Selects the bottom `k` least frequent tokens.

        Parameters
        ----------
        k : int
            The number of least frequent tokens to select.

        Returns
        -------
        Tuple[str, List[str]]
            A tuple containing a prompt string and a list of selected tokens.
        """
        vocab = self.tokenizer.get_vocab()
        sorted_tokens = sorted(vocab.items(), key=lambda item: item[1])
        selected_tokens = [token for token, _ in sorted_tokens[:k]]
        prompt_str = f"Selected bottom {k} least frequent tokens."
        logging.info(prompt_str)
        return (prompt_str, selected_tokens)

    def select_random_tokens(self, k: int, seed: int = None) -> Tuple[str, List[str]]:
        """
        Selects `k` random tokens.

        Parameters
        ----------
        k : int
            The number of random tokens to select.
        seed : int, optional
            A seed for random number generation, by default None.

        Returns
        -------
        Tuple[str, List[str]]
            A tuple containing a prompt string and a list of selected tokens.
        """
        import random
        if seed is not None:
            random.seed(seed)
        vocab = list(self.tokenizer.get_vocab().keys())
        selected_tokens = random.sample(vocab, k)
        prompt_str = f"Selected {k} random tokens."
        logging.info(prompt_str)
        return (prompt_str, selected_tokens)

    def select_subword_prefix_tokens(self, prefix: str) -> Tuple[str, List[str]]:
        """
        Selects tokens starting with a given prefix.

        Parameters
        ----------
        prefix : str
            The prefix to filter tokens by.

        Returns
        -------
        Tuple[str, List[str]]
            A tuple containing a prompt string and a list of selected tokens.
        """
        vocab = self.tokenizer.get_vocab()
        selected_tokens = [token for token in vocab if token.startswith(prefix)]
        prompt_str = f"Selected {len(selected_tokens)} tokens with prefix '{prefix}'."
        logging.info(prompt_str)
        return (prompt_str, selected_tokens)

    def save_tokens(self, tokens: List[str], filepath: str) -> None:
        """
        Saves the selected tokens to a file.

        Parameters
        ----------
        tokens : List[str]
            A list of tokens to save.
        filepath : str
            The path to the file where tokens will be saved.
        """
        pass 