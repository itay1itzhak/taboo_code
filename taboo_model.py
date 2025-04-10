import logging
from token_selection import TokenSelector
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import List
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer, stops=[], encounters=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stops = [stop.to(self.device) for stop in stops]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False


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
        final_answer_without_taboo: bool = False,
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
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.taboo_tokens = taboo_tokens
        self.taboo_token_ids = self.get_taboo_token_ids(taboo_tokens)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # If model is instrcuted model, set the chat template
        self.need_chat_template = (
            "instruct" in self.model.name_or_path.lower()
            or "chat" in self.model.name_or_path.lower()
        )
        self.final_answer_without_taboo = final_answer_without_taboo

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

    def get_taboo_token_ids(self, taboo_tokens: List[str]) -> List[int]:
        """
        Returns the token IDs of the taboo tokens.
        """
        taboo_token_ids = []
        logging.info(f"Loading taboo tokens ids:\n{taboo_tokens}\n...")
        for taboo_token in taboo_tokens:
            try:
                taboo_token_ids.append(self.tokenizer.get_vocab()[taboo_token])
                # taboo_token_ids.append(
                #     self.tokenizer.encode(taboo_token, add_prefix_space=True)
                # )
            except Exception as e:
                logging.warning(
                    f"Token {taboo_token} not found in vocabulary, getting last token"
                )
                tokens = self.tokenizer.tokenize(taboo_token)
                # ids = self.tokenizer.convert_tokens_to_ids(tokens)
                # Get all tokens ids
                taboo_token_ids.extend(tokens)
        logging.info(f"Finished loading taboo tokens ids.")
        return taboo_token_ids

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
        # First tokenize the taboo words properly, then convert to IDs
        # taboo_token_ids = []
        # for word in self.taboo_tokens:
        #     tokens = self.tokenizer.tokenize(word)
        #     ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #     taboo_token_ids.extend(ids)

        if not self.final_answer_without_taboo:
            # Restrict taboo tokens during generation
            generated_ids = self.model.generate(
                input_ids.to(self.device),
                max_new_tokens=self.max_length,
                do_sample=False,
                bad_words_ids=[[token_id] for token_id in self.taboo_token_ids],
            )
        else:
            # Restrict taboo tokens only until final answer or last '####'
            stop_words = ["##", "####", "Answer:"]
            stop_words_ids = [
                self.tokenizer(
                    stop_word, return_tensors="pt", add_special_tokens=False
                )["input_ids"].squeeze()
                for stop_word in stop_words
            ]
            stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(tokenizer=self.tokenizer, stops=stop_words_ids)]
            )
            intermidated_generated_ids = self.model.generate(
                input_ids.to(self.device),
                max_new_tokens=self.max_length,
                do_sample=False,
                bad_words_ids=[[token_id] for token_id in self.taboo_token_ids],
                stopping_criteria=stopping_criteria,
            )
            # continute free generation after the stopping criteria is met
            generated_ids = self.model.generate(
                intermidated_generated_ids,
                max_new_tokens=self.max_length,
                do_sample=False,
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
