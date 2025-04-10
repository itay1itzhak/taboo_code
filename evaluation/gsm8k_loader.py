import datasets
from typing import List, Dict


class GSM8KLoader:
    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def load_dataset(self) -> Dict[str, List[Dict]]:
        ds = datasets.load_dataset("openai/gsm8k", "main")
        train_ds = ds["train"]
        test_ds = ds["test"].select(range(self.n_samples))
        return {"train": list(train_ds), "test": list(test_ds)}

    def format_example(self, example: Dict) -> str:
        """Formats a single example as a prompt string."""
        return f"Question: {example['question'].strip()}\nAnswer: {example['answer'].strip()}\n"
