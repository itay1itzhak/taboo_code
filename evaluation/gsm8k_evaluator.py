from typing import Dict, List
from taboo_model import TabooModel
from transformers import PreTrainedTokenizer
from evaluation.gsm8k_loader import GSM8KLoader
import random


class GSM8KEvaluator:
    def __init__(self, model: TabooModel, tokenizer: PreTrainedTokenizer, k_shot: int):
        self.model = model
        self.tokenizer = tokenizer
        self.k_shot = k_shot

    def evaluate(self, n_samples: int) -> Dict:
        loader = GSM8KLoader(n_samples)
        data = loader.load_dataset()
        train_data = data["train"]
        test_data = data["test"]

        results = {}
        for idx, example in enumerate(test_data):
            k_shot_examples = random.sample(train_data, self.k_shot)
            prompt = "".join([loader.format_example(ex) for ex in k_shot_examples])
            prompt += f"Question: {example['question'].strip()}\nAnswer:"

            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            normal_output = self.model.generate_normal(input_ids)
            taboo_output = self.model.generate_taboo(input_ids)

            results[idx] = {
                "normal": normal_output,
                "taboo": taboo_output,
                "prompt": prompt,
            }
        return results
