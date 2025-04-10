import os
import re
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
      - "regex": select tokens that match a given regular expression.
      - "combined": select tokens that satisfy all of several sub-criteria.
    """

    def __init__(
            self, tokenizer: PreTrainedTokenizer, taboo_criteria: str, json_config=None
    ):
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
        self.exclude_regexes = self.taboo_criteria_dict.get(
            "exclude_regexes",
            [r'^<\|.*?\|>$', r'^\|\|\|.*?\|\|\|$']
        )
        # For frequency selection with leaf filtering, load BPE ranks from merges file.
        if self.taboo_criteria_dict.get("type") in ["frequency", "combined"]:
            self.bpe_ranks = self.load_bpe_ranks(json_config)
        else:
            self.bpe_ranks = None
        # For synthetic selection, load spaCy if available.
        self.nlp = spacy.load("en_core_web_sm")

    def _filter_excluded_tokens(self, tokens: List[str]) -> List[str]:
        filtered = []
        for token in tokens:
            # Skip this token if it matches any exclusion regex
            if any(re.match(pattern, token) for pattern in self.exclude_regexes):
                continue
            filtered.append(token)
        return filtered

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
            raise ValueError("Failed to parse taboo criteria.")

    def load_bpe_ranks(self, path_file=None) -> Dict[str, int]:
        """
        Loads the BPE merge rules from the merges file associated with the tokenizer,
        and returns a dictionary mapping the merge rule (as a string, e.g. "Ġ t") to its rank.
        """
        if path_file:
            try:
                with open(path_file, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    print("Parsed JSON data:", data)
            except FileNotFoundError:
                print(f"File not found at path: {path_file}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            merge_rules = data["model"]["merges"]
            print(f"Loaded {len(merge_rules)} merge rules.")

            # Create a dictionary mapping each merge pair to its rank (order of appearance).
            bpe_ranks = {pair: rank for rank, pair in enumerate(merge_rules)}
            return bpe_ranks
        vocab_files = self.tokenizer.vocab_files_names
        merges_filename = vocab_files.get("merges_file")
        if not merges_filename:
            raise ValueError("merges_file not found in tokenizer.vocab_files_names.")
        # Download the merges file from HF Hub:
        merges_file_path = hf_hub_download(
            repo_id=self.tokenizer.name_or_path, filename=merges_filename
        )
        logging.info(f"Downloaded merges file: {merges_file_path}")

        merge_rules = []
        with open(merges_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
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
            new_component = (
                    components[candidate_index] + components[candidate_index + 1]
            )
            components = (
                    components[:candidate_index]
                    + [new_component]
                    + components[candidate_index + 2:]
            )
        return components, merge_count

    # ------------------------------------
    # Existing selection methods:
    # ------------------------------------
    def select_frequency_tokens(self, criteria: Dict) -> Tuple[str, List[str]]:
        vocab = self.tokenizer.get_vocab()  # token -> token_id
        tokens = list(vocab.keys())

        if criteria.get("leaf", False):
            merge_components = set()
            for rule in self.bpe_ranks:
                a, b = rule.split()
                merge_components.add(a)
                merge_components.add(b)
            tokens = [token for token in tokens if token not in merge_components]

        order = criteria.get("order", "least_frequent")
        reverse_order = True if order.lower() == "least_frequent" else False
        sorted_tokens = sorted(tokens, key=lambda token: vocab[token], reverse=reverse_order)
        k = criteria.get("k", None)
        if k is not None:
            sorted_tokens = sorted_tokens[:k]
        sorted_tokens = self._filter_excluded_tokens(sorted_tokens)

        prompt_str = (
            f"Frequency criterion: selected {len(sorted_tokens)} tokens "
            f"(leaf filter: {criteria.get('leaf', False)}; order: {order})."
        )
        logging.info(prompt_str)
        return (prompt_str, sorted_tokens)

    def select_synthetic_tokens(self, criteria: Dict) -> Tuple[str, List[str]]:
        if not self.nlp:
            return ("spaCy not available.", [])
        desired_pos = criteria.get("pos", [])
        vocab = list(self.tokenizer.get_vocab().keys())
        selected = []
        logging.info(f"Selecting synthetic tokens for {len(vocab)} tokens.")
        for token in tqdm(vocab):
            clean_token = token.lstrip("Ġ")
            doc = self.nlp(clean_token)
            if not doc:
                continue
            token_pos = doc[0].pos_
            if token_pos in desired_pos:
                selected.append(token)
        selected = self._filter_excluded_tokens(selected)

        prompt_str = f"Synthetic criterion: selected {len(selected)} tokens matching POS tags: {desired_pos}."
        logging.info(prompt_str)
        return (prompt_str, selected)

    def select_custom_tokens(self, criteria: Dict) -> Tuple[str, List[str]]:
        custom_tokens = criteria.get("tokens", [])
        prompt_str = f"Custom criterion: selected {len(custom_tokens)} tokens."
        logging.info(prompt_str)
        return (prompt_str, custom_tokens)

    # ------------------------------------
    # New regex selection method:
    # ------------------------------------
    def select_regex_tokens(self, criteria: Dict) -> Tuple[str, List[str]]:
        """
        Selects tokens based on a regex pattern provided in the criteria.

        Expected keys in criteria:
          - pattern: a regex pattern (e.g., ".*\\d.*" to match tokens containing a digit).
          - k: (optional) maximum number of tokens to return.

        The method always applies the exclusion filter (filter_excluded_tokens)
        to the regex-selected tokens.
        """
        pattern = criteria.get("pattern")
        if not pattern:
            logging.error("No regex pattern provided in criteria.")
            return ("No regex pattern provided.", [])

        vocab = list(self.tokenizer.get_vocab().keys())
        # Use re.search to find tokens that contain a match to the regex pattern.
        matched_tokens = [token for token in vocab if re.search(pattern, token)]
        # Always apply the exclusion filter.
        filtered_tokens = self._filter_excluded_tokens(matched_tokens)

        k = criteria.get("k")
        if k is not None:
            filtered_tokens = filtered_tokens[:k]

        prompt_str = f"Regex criterion: selected {len(filtered_tokens)} tokens matching pattern '{pattern}'."
        logging.info(prompt_str)
        return (prompt_str, filtered_tokens)

    # ------------------------------------
    # New combined selection methods:
    # ------------------------------------
    def get_all_frequency_tokens(self, criteria: Dict) -> Set[str] or List[str]:
        vocab = self.tokenizer.get_vocab()
        tokens = set(vocab.keys())

        if criteria.get("leaf", False):
            merge_components = set()
            for rule in self.bpe_ranks:
                a, b = rule.split()
                merge_components.add(a)
                merge_components.add(b)
            tokens = {token for token in tokens if token not in merge_components}

        order = criteria.get("order", None)
        if order:
            reverse_order = True if order.lower() == "least_frequent" else False
            sorted_tokens = sorted(tokens, key=lambda token: vocab[token], reverse=reverse_order)
            return self._filter_excluded_tokens(sorted_tokens)

        return tokens

    def get_all_synthetic_tokens(self, criteria: Dict) -> Set[str]:
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
        custom_tokens = criteria.get("tokens", [])
        return set(custom_tokens)

    def select_combined_tokens(self, criteria: Dict) -> Tuple[str, List[str]]:
        sub_criteria = criteria.get("criteria", [])
        if not sub_criteria:
            logging.error("No sub-criteria provided for combined selection.")
            return ("No sub-criteria provided.", [])

        token_sets = []
        for sub in sub_criteria:
            crit_type = sub.get("criterion")
            if crit_type == "frequency":
                tokens_result = self.get_all_frequency_tokens(sub)
            elif crit_type == "synthetic":
                tokens_result = self.get_all_synthetic_tokens(sub)
            elif crit_type == "custom":
                tokens_result = self.get_all_custom_tokens(sub)
            elif crit_type == "regex":
                # For combined criteria, if one of the sub-criteria is regex-based:
                prompt, tokens_result = self.select_regex_tokens(sub)
                tokens_result = set(tokens_result)
            else:
                logging.error(f"Unknown sub-criterion type: {crit_type}")
                tokens_result = set()

            if not isinstance(tokens_result, set):
                tokens_result = set(tokens_result)
            token_sets.append(tokens_result)

        common_tokens = set.intersection(*token_sets) if token_sets else set()

        vocab = self.tokenizer.get_vocab()
        sorted_tokens = sorted(common_tokens, key=lambda token: vocab[token], reverse=True)
        k = criteria.get("k", len(sorted_tokens))
        prompt_str = f"Combined criterion: {len(sorted_tokens)} tokens satisfy all sub-criteria."
        logging.info(prompt_str)
        return (prompt_str, sorted_tokens[:k])

    def select_tokens(self) -> Tuple[str, List[str]]:
        selection_type = self.taboo_criteria_dict.get("type")
        if selection_type == "frequency":
            return self.select_frequency_tokens(self.taboo_criteria_dict)
        elif selection_type == "synthetic":
            return self.select_synthetic_tokens(self.taboo_criteria_dict)
        elif selection_type == "custom":
            return self.select_custom_tokens(self.taboo_criteria_dict)
        elif selection_type == "combined":
            return self.select_combined_tokens(self.taboo_criteria_dict)
        elif selection_type == "regex":
            return self.select_regex_tokens(self.taboo_criteria_dict)
        else:
            logging.error("Invalid selection type provided.")
            raise ValueError("Invalid selection type.")

    def save_tokens(self, tokens: List[str], filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            for token in tokens:
                f.write(token + "\n")
        logging.info(f"Saved {len(tokens)} tokens to {filepath}.")


if __name__ == "__main__":
    # Load a tokenizer from HF Hub.
    model_name = "allenai/OLMo-7B-0724-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_json_path = (
        "/Users/guykaplan/Dev/OLMo/test_fixtures/test-olmo-model/tokenizer.json"
    )
    #
    # # Example 1: Frequency criterion using tokenizer source, leaf tokens only.
    # freq_criteria = json.dumps(
    #     {"type": "frequency", "source": "tokenizer", "leaf": True, "k": 100, "order": "least_frequent"}
    # )
    # ts_freq = TokenSelector(tokenizer, freq_criteria, tokenizer_json_path)
    # prompt, tokens = ts_freq.select_tokens()
    # print(prompt)
    # for token in tokens:
    #     print(f"Token: {token:20s} | Token ID: {tokenizer.get_vocab()[token]}")
    #
    # print("\n" + "=" * 40 + "\n")
    # # # Example 2: Synthetic criterion, select tokens that are NOUNs or VERBs.
    # synthetic_criteria = json.dumps(
    #     {"type": "synthetic", "pos": ["NOUN", "VERB"], "k": 10}
    # )
    # ts_syn = TokenSelector(tokenizer, synthetic_criteria)
    # prompt, tokens = ts_syn.select_tokens()
    # print(prompt)
    # for token in tokens:
    #     print(f"Token: {token:20s}")
    #
    # print("\n" + "=" * 40 + "\n")
    # # Example 3: Custom criterion, user provides a custom list.
    # custom_criteria = json.dumps(
    #     {"type": "custom", "tokens": ["hello", "world", "test", "example"]}
    # )
    # ts_custom = TokenSelector(tokenizer, custom_criteria)
    # prompt, tokens = ts_custom.select_tokens()
    # print(prompt)
    # for token in tokens:
    #     print(f"Token: {token:20s}")
    #
    # print("\n" + "=" * 40 + "\n")
    # # Example 4: Combined criterion - both frequency (leaf only) and synthetic (only VERBs)
    # combined_criteria = json.dumps(
    #     {
    #         "type": "combined",
    #         "criteria": [
    #             {"criterion": "frequency", "leaf": True, "order": "most_frequent"},
    #             {"criterion": "synthetic", "pos": ["VERB"]},
    #         ],
    #         "k": 100,
    #     }
    # )
    # ts_combined = TokenSelector(tokenizer, combined_criteria, tokenizer_json_path)
    # prompt, tokens = ts_combined.select_tokens()
    # print(prompt)
    # for token in tokens:
    #     print(f"Token: {token:20s} | Token ID: {tokenizer.get_vocab()[token]}")

    # freq_criteria_desc = json.dumps(
    #     {"type": "frequency", "source": "tokenizer", "leaf": True, "k": 100, "order": "most_frequent"}
    # )
    # ts_freq_desc = TokenSelector(tokenizer, freq_criteria_desc, tokenizer_json_path)
    # prompt, tokens = ts_freq_desc.select_tokens()
    # print(prompt)
    # for token in tokens:
    #     print(f"Token: {token:20s} | Token ID: {tokenizer.get_vocab()[token]}")

    # Example: Regex criterion to select tokens that contain a digit.
    # The regex pattern ".*\d.*" will match any token with at least one digit.
    regex_criteria = json.dumps(
        {"type": "regex", "pattern": r".*\d.*", "k": 50}
    )
    ts_regex = TokenSelector(tokenizer, regex_criteria, tokenizer_json_path)
    prompt, tokens = ts_regex.select_tokens()
    print(prompt)
    for token in tokens:
        print(f"Token: {token:20s} | Token ID: {tokenizer.get_vocab()[token]}")

    print("\n" + "=" * 40 + "\n")

    # Example: Combined criterion that includes regex as one of the sub-criteria.
    # This example selects tokens that are both among the most frequent (with leaf filtering)
    # and match a regex (contain a digit).
    combined_criteria = json.dumps(
        {
            "type": "combined",
            "criteria": [
                {"criterion": "frequency", "leaf": True, "order": "most_frequent"},
                {"criterion": "regex", "pattern": r".*\d.*"}
            ],
            "k": 100,
        }
    )
    ts_combined = TokenSelector(tokenizer, combined_criteria, tokenizer_json_path)
    prompt, tokens = ts_combined.select_tokens()
    print(prompt)
    for token in tokens:
        print(f"Token: {token:20s} | Token ID: {tokenizer.get_vocab()[token]}")