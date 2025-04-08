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

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, token_selector: TokenSelector):
        """
        Initializes the TabooModel with a model and tokenizer.

        Parameters
        ----------
        model : PreTrainedModel
            A Hugging Face pre-trained model instance.
        tokenizer : PreTrainedTokenizer
            A Hugging Face tokenizer instance.
        token_selector : TokenSelector
            A TokenSelector instance.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.token_selector = token_selector
        self.taboo_tokens = self.token_selector.select_tokens()
        # self.taboo_prompt = self.token_selector.get_taboo_prompt()
        
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
        generated_ids = self.model.generate(input_ids, max_length=50)
        return generated_ids[0].tolist()

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
        taboo_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in self.taboo_tokens[1]]

        # Implement logic to restrict taboo tokens during generation
        # This is a placeholder for the actual implementation
        # You might need to use a custom generation loop to enforce taboo constraints
        generated_ids = self.model.generate(
            input_ids,
            max_length=50,
            bad_words_ids=[[token_id] for token_id in taboo_token_ids]
        )
        return generated_ids[0].tolist()

    def generate_from_prompt(self, prompt: str, input_ids: List[int], max_length: int = 50) -> str:
        """
        Generates text from a given prompt while restricting the use of taboo tokens.

        Parameters
        ----------
        prompt : str
            The input text prompt to generate text from.
        input_ids : List[int]
            A list of input token IDs.
        max_length : int, optional
            The maximum length of the generated text, by default 50.

        Returns
        -------
        str
            The generated text.
        """
        # Convert prompt to input_ids
        prompt_input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        input_ids = torch.cat([prompt_input_ids, input_ids], dim=1)
        
        # Implement logic to restrict taboo tokens during generation
        # This is a placeholder for the actual implementation
        generated_ids = self.model.generate(input_ids, max_length=max_length)
        
        # Decode the generated token IDs to text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text 