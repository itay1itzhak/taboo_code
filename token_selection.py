import os
import json
import logging
import random
from typing import List, Dict, Tuple, Set

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from huggingface_hub import hf_hub_download

# Optional: load spaCy for synthetic POS tagging
try:
    import spacy

    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)


class TokenSelector:
    """
    A class to select and manage taboo tokens for language model evaluation.

    Supports several selection types:
      - "frequency": select tokens by frequency (using the tokenizer vocabulary),
         optionally restricting to leaf tokens determined via BPE merges.
      - "synthetic": select tokens whose part-of-speech (POS) tags (via spaCy) match a given list.
      - "custom": use a custom provided list.
      - "combined": select tokens that satisfy all of several sub-criteria.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, taboo_criteria: str):
        """
        Initializes the TokenSelector with a tokenizer and a taboo criteria JSON string.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
            A Hugging Face tokenizer instance.
        taboo_criteria : str
            A JSON string specifying the selection criteria.
            For a combined criterion, the JSON should be similar to:
            {
                "type": "combined",
                "criteria": [
                    {"criterion": "frequency", "leaf": true},
                    {"criterion": "synthetic", "pos": ["VERB"]}
                ],
                "k": 10
            }
        """
        self.tokenizer = tokenizer
        self.taboo_criteria = taboo_criteria
        self.taboo_criteria_dict = self.parse_taboo_criteria(taboo_criteria)
        # For frequency selection with leaf filtering, load BPE ranks from merges file.
        if (self.taboo_criteria_dict.get("type") == "frequency" and
                self.taboo_criteria_dict.get("leaf", False)):
            self.bpe_ranks = self.load_bpe_ranks()
        else:
            self.bpe_ranks = None
        # For synthetic selection, load spaCy if available.
        self.nlp = spacy.load("en_core_web_sm")
        # if self.taboo_criteria_dict.get("type") in ["synthetic", "combined"]:
        #     # If any sub-criterion is synthetic, then we need spaCy.
        #     if any(
        #             crit.get("criterion") == "synthetic"
        #             for crit in self.taboo_criteria_dict.get("type", [])
        #     ):
        #         if NLP_AVAILABLE:
        #             self.nlp = spacy.load("en_core_web_sm")
        #         else:
        #             logging.error("spaCy is not available. Install spaCy and en_core_web_sm model.")
        #             self.nlp = None

    def parse_taboo_criteria(self, taboo_criteria: str) -> Dict:
        """
        Parses the taboo criteria string (JSON format) into a dictionary.
        """
        try:
            criteria = json.loads(taboo_criteria)
            logging.info("Taboo criteria parsed successfully.")
            return criteria
        except Exception as e:
            logging.error("Failed to parse taboo criteria: " + str(e))
            return {}

    def load_bpe_ranks(self) -> Dict[str, int]:
        """
        Loads the BPE merge rules from the merges file associated with the tokenizer,
        and returns a dictionary mapping the merge rule (as a string, e.g. "Ġ t") to its rank.
        """
        vocab_files = self.tokenizer.vocab_files_names
        merges_filename = vocab_files.get("merges_file")
        if not merges_filename:
            raise ValueError("merges_file not found in tokenizer.vocab_files_names.")
        # Download the merges file from HF Hub:
        merges_file_path = hf_hub_download(repo_id=self.tokenizer.name_or_path, filename=merges_filename)
        logging.info(f"Downloaded merges file: {merges_file_path}")

        merge_rules = []
        with open(merges_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip header/comments (lines starting with '#' or empty)
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                # Concatenate the two parts with a space to match our key format
                merge_rule = f"{parts[0]} {parts[1]}"
                merge_rules.append(merge_rule)
        logging.info(f"Loaded {len(merge_rules)} merge rules.")
        return {rule: rank for rank, rule in enumerate(merge_rules)}

    def decompose_token(self, token: str) -> Tuple[List[str], int]:
        """
        Decomposes a token by simulating the reverse of BPE merge operations.
        Splits the token into characters and then repeatedly checks for adjacent pairs that
        appear in the bpe_ranks. Counts the number of merges (i.e. the merge depth).

        Returns:
            components: final list of sub-components after reverse merging.
            merge_count: the number of merge operations applied.
        """
        if not self.bpe_ranks:
            return [token], 0

        components = list(token)
        merge_count = 0

        while True:
            candidate_index = None
            candidate_rank = None
            # Examine each adjacent pair of current components.
            for i in range(len(components) - 1):
                pair_str = f"{components[i]} {components[i + 1]}"
                if pair_str in self.bpe_ranks:
                    rank = self.bpe_ranks[pair_str]
                    if candidate_rank is None or rank < candidate_rank:
                        candidate_rank = rank
                        candidate_index = i
            if candidate_index is None:
                break
            merge_count += 1
            new_component = components[candidate_index] + components[candidate_index + 1]
            components = components[:candidate_index] + [new_component] + components[candidate_index + 2:]
        return components, merge_count

    # ------------------------------------
    # Existing selection methods:
    # ------------------------------------
    def select_frequency_tokens(self, criteria: Dict) -> Tuple[str, List[str]]:
        """
        Selects tokens from the tokenizer vocabulary based on frequency.
        If criteria includes "leaf": true then only tokens with merge depth 0 are used.
        Tokens are sorted by their token IDs in descending order.

        Expected keys in criteria:
          - source: should be "tokenizer" (currently the only supported option)
          - leaf: (bool) if true, restrict to tokens with merge depth 0.
          - k: integer number of tokens to return.
        """
        vocab = self.tokenizer.get_vocab()  # token -> token_id
        tokens = list(vocab.keys())
        if criteria.get("leaf", False):
            leaf_tokens = []
            for token in tokens:
                _, merge_depth = self.decompose_token(token)
                if merge_depth == 0:
                    leaf_tokens.append(token)
            tokens = leaf_tokens

        # Sort tokens by token_id in descending order.
        sorted_tokens = sorted(tokens, key=lambda token: vocab[token], reverse=True)
        k = criteria.get("k", None)  # for combined selection, we may not use k here.
        if k is not None:
            sorted_tokens = sorted_tokens[:k]
        prompt_str = f"Frequency criterion: selected {len(sorted_tokens)} tokens (leaf filter: {criteria.get('leaf', False)})."
        logging.info(prompt_str)
        return (prompt_str, sorted_tokens)

    def select_synthetic_tokens(self, criteria: Dict) -> Tuple[str, List[str]]:
        """
        Selects tokens based on synthetic criteria. Uses spaCy to determine the POS tag of each token
        and selects only those tokens whose POS tag is in the provided list.

        Expected keys in criteria:
          - pos: a list of POS tags (e.g., ["NOUN", "VERB"]).
          - k: the maximum number of tokens to return.
        """
        if not self.nlp:
            return ("spaCy not available.", [])
        desired_pos = criteria.get("pos", [])
        vocab = list(self.tokenizer.get_vocab().keys())
        selected = []
        for token in tqdm(vocab):
            clean_token = token.lstrip("Ġ")
            doc = self.nlp(clean_token)
            if not doc:
                continue
            token_pos = doc[0].pos_
            if token_pos in desired_pos:
                selected.append(token)
        k = criteria.get("k", None)
        if k is not None:
            selected = selected[:k]
        prompt_str = f"Synthetic criterion: selected {len(selected)} tokens matching POS tags: {desired_pos}."
        logging.info(prompt_str)
        return (prompt_str, selected)

    def select_custom_tokens(self, criteria: Dict) -> Tuple[str, List[str]]:
        """
        Returns the custom list of tokens specified in criteria.

        Expected key in criteria:
          - tokens: a list of tokens.
        """
        custom_tokens = criteria.get("tokens", [])
        prompt_str = f"Custom criterion: selected {len(custom_tokens)} tokens."
        logging.info(prompt_str)
        return (prompt_str, custom_tokens)

    # ------------------------------------
    # New combined selection methods:
    # ------------------------------------
    def get_all_frequency_tokens(self, criteria: Dict) -> Set[str]:
        """
        Returns the full set of tokens satisfying the frequency criteria.
        (If 'leaf' is true, only tokens with merge depth 0 are returned; otherwise, all tokens.)
        """
        vocab = self.tokenizer.get_vocab()
        tokens = set(vocab.keys())
        if criteria.get("leaf", False):
            tokens = {token for token in tokens if self.decompose_token(token)[1] == 0}
        return tokens

    def get_all_synthetic_tokens(self, criteria: Dict) -> Set[str]:
        """
        Returns the full set of tokens satisfying the synthetic criterion (POS filtering).
        """
        if not self.nlp:
            return set()
        desired_pos = criteria.get("pos", [])
        vocab = list(self.tokenizer.get_vocab().keys())
        selected = set()
        for token in vocab:
            clean_token = token.lstrip("Ġ")
            doc = self.nlp(clean_token)
            if not doc:
                continue
            token_pos = doc[0].pos_
            if token_pos in desired_pos:
                selected.add(token)
        return selected

    def get_all_custom_tokens(self, criteria: Dict) -> Set[str]:
        """
        Returns the set of custom tokens from the provided criteria.
        """
        custom_tokens = criteria.get("tokens", [])
        return set(custom_tokens)

    def select_combined_tokens(self, criteria: Dict) -> Tuple[str, List[str]]:
        """
        Combines multiple criteria (frequency, synthetic, custom) by taking the intersection
        of the tokens satisfying each sub-criterion.

        Expected structure for combined criteria:
            {
                "type": "combined",
                "criteria": [
                    {"criterion": "frequency", "leaf": true},
                    {"criterion": "synthetic", "pos": ["VERB"]}
                ],
                "k": 10
            }
        """
        sub_criteria = criteria.get("criteria", [])
        if not sub_criteria:
            logging.error("No sub-criteria provided for combined selection.")
            return ("No sub-criteria provided.", [])

        # For each sub-criterion, obtain the set of tokens satisfying that criterion.
        sets = []
        for sub in sub_criteria:
            crit_type = sub.get("criterion")
            if crit_type == "frequency":
                s = self.get_all_frequency_tokens(sub)
            elif crit_type == "synthetic":
                s = self.get_all_synthetic_tokens(sub)
            elif crit_type == "custom":
                s = self.get_all_custom_tokens(sub)
            else:
                logging.error(f"Unknown sub-criterion type: {crit_type}")
                s = set()
            sets.append(s)

        # Compute intersection over all sets.
        if sets:
            common_tokens = set.intersection(*sets)
        else:
            common_tokens = set()

        # Sort the tokens by token ID in descending order (frequency ranking)
        vocab = self.tokenizer.get_vocab()
        sorted_tokens = sorted(common_tokens, key=lambda token: vocab[token], reverse=True)
        k = criteria.get("k", len(sorted_tokens))
        prompt_str = f"Combined criterion: {len(sorted_tokens)} tokens satisfy all sub-criteria."
        logging.info(prompt_str)
        return (prompt_str, sorted_tokens[:k])

    def select_tokens(self) -> Tuple[str, List[str]]:
        """
        Selects tokens based on the provided taboo criteria.
        """
        selection_type = self.taboo_criteria_dict.get("type")
        if selection_type == "frequency":
            return self.select_frequency_tokens(self.taboo_criteria_dict)
        elif selection_type == "synthetic":
            return self.select_synthetic_tokens(self.taboo_criteria_dict)
        elif selection_type == "custom":
            return self.select_custom_tokens(self.taboo_criteria_dict)
        elif selection_type == "combined":
            return self.select_combined_tokens(self.taboo_criteria_dict)
        else:
            logging.error("Invalid selection type provided.")
            return ("Invalid selection type.", [])

    def save_tokens(self, tokens: List[str], filepath: str) -> None:
        """
        Saves the selected tokens to a file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            for token in tokens:
                f.write(token + "\n")
        logging.info(f"Saved {len(tokens)} tokens to {filepath}.")


if __name__ == "__main__":
    # Load a tokenizer from HF Hub.
    model_name = "allenai/OLMo-7B-0724-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Example 1: Frequency criterion using tokenizer source, leaf tokens only.
    freq_criteria = json.dumps({
        "type": "frequency",
        "source": "tokenizer",
        "leaf": False,
        "k": 10
    })
    ts_freq = TokenSelector(tokenizer, freq_criteria)
    prompt, tokens = ts_freq.select_tokens()
    print(prompt)
    for token in tokens:
        print(f"Token: {token:20s} | Token ID: {tokenizer.get_vocab()[token]}")

    print("\n" + "=" * 40 + "\n")
    # Example 2: Synthetic criterion, select tokens that are NOUNs or VERBs.
    synthetic_criteria = json.dumps({
        "type": "synthetic",
        "pos": ["NOUN", "VERB"],
        "k": 10
    })
    ts_syn = TokenSelector(tokenizer, synthetic_criteria)
    prompt, tokens = ts_syn.select_tokens()
    print(prompt)
    for token in tokens:
        print(f"Token: {token:20s}")

    print("\n" + "=" * 40 + "\n")
    # Example 3: Custom criterion, user provides a custom list.
    custom_criteria = json.dumps({
        "type": "custom",
        "tokens": ["hello", "world", "test", "example"]
    })
    ts_custom = TokenSelector(tokenizer, custom_criteria)
    prompt, tokens = ts_custom.select_tokens()
    print(prompt)
    for token in tokens:
        print(f"Token: {token:20s}")

    print("\n" + "=" * 40 + "\n")
    # Example 4: Combined criterion - both frequency (leaf only) and synthetic (only VERBs)
    combined_criteria = json.dumps({
        "type": "combined",
        "criteria": [
            {"criterion": "frequency", "leaf": True},
            {"criterion": "synthetic", "pos": ["VERB"]}
        ],
        "k": 10
    })
    ts_combined = TokenSelector(tokenizer, combined_criteria)
    prompt, tokens = ts_combined.select_tokens()
    print(prompt)
    for token in tokens:
        print(f"Token: {token:20s} | Token ID: {tokenizer.get_vocab()[token]}")
