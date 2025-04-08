import logging
from token_selection import TokenSelector
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)


class TabooModel:
    """
    A class to wrap a Hugging Face model and enforce token generation constraints.

    Methods
    -------
    generate(input_ids: List[int], taboo_tokens: List[str]) -> List[int]:
        Generates text while restricting the use of taboo tokens.
    generate_from_prompt(prompt: str, taboo_tokens: List[str], max_length: int = 50) -> str:
        Generates text from a given prompt while restricting the use of taboo tokens.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        taboo_tokens: List[str],
        max_length: int = 50,
    ):
        """
        Initializes the TabooModel with a model and tokenizer.

        Parameters
        ----------
        model : PreTrainedModel
            A Hugging Face pre-trained model instance.
        tokenizer : PreTrainedTokenizer
            A Hugging Face tokenizer instance.
        taboo_tokens : List[str]
            A list of taboo tokens.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.taboo_tokens = taboo_tokens
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_normal(self, input_ids: List[int]) -> List[int]:
        """
        Generates text without any restrictions.

        Parameters
        ----------
        input_ids : List[int]
            A list of input token IDs.

        Returns
        -------
        List[int]
            A list of generated token IDs.
        """
        # Generate text normally without any taboo constraints
        generated_ids = self.model.generate(
            input_ids.to(self.device), max_new_tokens=self.max_length, do_sample=False
        )
        # Remove the input tokens from the generated IDs according to the length of the input
        return generated_ids[0].tolist()[input_ids.shape[1] :]

    def generate_taboo(self, input_ids: List[int]) -> List[int]:
        """
        Generates text while restricting the use of taboo tokens.

        Parameters
        ----------
        input_ids : List[int]
            A list of input token IDs.

        Returns
        -------
        List[int]
            A list of generated token IDs.
        """
        # Convert taboo tokens to IDs
        taboo_token_ids = [
            self.tokenizer.convert_tokens_to_ids(token)
            for token in self.taboo_tokens[1]
        ]

        # Implement logic to restrict taboo tokens during generation
        # This is a placeholder for the actual implementation
        # You might need to use a custom generation loop to enforce taboo constraints
        generated_ids = self.model.generate(
            input_ids.to(self.device),
            max_new_tokens=self.max_length,
            do_sample=False,
            bad_words_ids=[[token_id] for token_id in taboo_token_ids],
        )
        # Remove the input tokens from the generated IDs according to the length of the input
        return generated_ids[0].tolist()[input_ids.shape[1] :]

    # Delete this function, should be implemented in the evaluation class
    # def generate_with_taboo_prompt(self, taboo_prompt: str, input_ids: List[int], max_length: int = 50) -> str:
    #     """
    #     Generates text from a given prompt while restricting the use of taboo tokens.

    #     Parameters
    #     ----------
    #     taboo_prompt : str
    #         The input text prompt to explain which tokens are taboo.
    #     input_ids : List[int]
    #         A list of input token IDs.
    #     max_length : int, optional
    #         The maximum length of the generated text, by default 50.

    #     Returns
    #     -------
    #     str
    #         The generated text.
    #     """
    #     # Convert prompt to input_ids
    #     taboo_prompt_input_ids = self.tokenizer.encode(taboo_prompt, return_tensors='pt')
    #     input_ids = torch.cat([taboo_prompt_input_ids, input_ids], dim=1)

    #     # Implement logic to restrict taboo tokens during generation
    #     # This is a placeholder for the actual implementation
    #     generated_ids = self.model.generate(input_ids, max_length=max_length)

    #     # Decode the generated token IDs to text
    #     generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    #     return generated_text
