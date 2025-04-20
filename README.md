# Token Taboo Evaluation Framework

## Overview

The Token Taboo Evaluation Framework is designed to evaluate language models by imposing constraints on token generation. This framework allows users to define "taboo" tokens that the model should avoid generating, and then assess the model's performance under these constraints. The framework supports various criteria for selecting taboo tokens, including frequency, part-of-speech (POS), custom lists, regular expressions (regex), and combined criteria.

The framework evaluates models on reasoning tasks (like GSM8K or custom datasets), measures the adherence to taboo constraints, and compares performance with and without these constraints. It can optionally use another language model as a "judge" to evaluate the correctness of the generated answers.

## Project Structure

- **`run.py`**: The main script to configure and run the evaluation framework. It handles loading models, tokenizers, datasets, and taboo criteria, orchestrates the evaluation process, and saves the results.
- **`token_selection.py`**: Contains the `TokenSelector` class, responsible for selecting taboo tokens based on various criteria (frequency, POS, custom, regex, combined). It also caches selected tokens.
- **`evaluation.py`**: Contains the `Evaluator` class, which performs the actual evaluation on datasets (e.g., GSM8K, custom CSV). It calculates metrics like accuracy and taboo token usage, optionally using an LLM judge.
- **`taboo_model.py`**: Defines the `TabooModel` class, a wrapper around a Hugging Face `PreTrainedModel` that enforces the taboo token constraints during text generation.
- **`token_selection_criterias/`**: Directory containing JSON files that define different taboo token selection criteria (e.g., `frequency_criteria.json`, `pos_criteria.json`, `custom_criteria.json`, `regex_criteria.json`).
- **`data/`**: Directory for datasets and cached data.
  - **`sample_questions.csv`**: A sample dataset used for evaluating the model's reasoning capabilities.
  - **`cache_selected_tokens/`**: Automatically created directory to cache selected taboo tokens for specific models and criteria, speeding up subsequent runs.
- **`results/`**: Directory where evaluation results (metrics and detailed answers) are saved in JSON format, organized by dataset, model, and criteria type.
- **`requirements.txt`**: Lists the Python dependencies required to run the project.
- **`.env`**: Configuration file for environment variables like Hugging Face Hub token and cache directory (optional but recommended).
- **`playground.ipynb`**: A Jupyter notebook for experimenting with the framework's components and testing different configurations interactively.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/token-taboo-evaluation.git # Replace with your repo URL
    cd token-taboo-evaluation
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy model (required for POS criteria):**
    The code attempts to load `en_core_web_lg` but falls back to `en_core_web_sm`. Ensure at least the small model is downloaded:

    ```bash
    python -m spacy download en_core_web_sm
    # Optionally, download the larger model for potentially better POS tagging:
    # python -m spacy download en_core_web_lg
    ```

5.  **Configure Environment Variables (Optional but Recommended):**
    Create a `.env` file in the project root directory to store your Hugging Face Hub token and desired cache directory:

    ```dotenv:.env
    # .env
    HF_HOME="/path/to/your/huggingface/cache" # Optional: Default is ~/.cache/huggingface
    HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN"     # Recommended: Especially for gated models like Llama
    ```

    The script will load these variables automatically.

## Running the Framework

The main script is `run.py`. You can configure the evaluation using command-line arguments.

**Key Arguments:**

- `--model_name`: Name of the Hugging Face model (e.g., `allenai/OLMo-2-1124-7B-Instruct`).
- `--taboo_criteria`: Path to the JSON file defining taboo criteria (e.g., `token_selection_criterias/frequency_criteria.json`).
- `--dataset_path`: Path to the evaluation dataset (e.g., `data/sample_questions.csv` or `openai/gsm8k`).
- `--results_dir`: Directory to save evaluation results (default: `results`).
- `--k_shot`: Number of few-shot examples to include in the prompt (default: 3).
- `--n_samples`: Number of samples to evaluate from the dataset (default: 100).
- `--max_length`: Maximum number of new tokens to generate (default: 300).
- `--judge_model_name`: (Optional) HF model name to use as a judge for correctness (e.g., `allenai/OLMo-2-1124-7B-Instruct`).
- `--final_answer_without_taboo`: (Experimental) Apply taboo constraints only up to the final answer marker.

### Example Commands:

1.  **Frequency Criterion (Top 100 most frequent leaf tokens) on GSM8K:**

    ```bash
    python run.py \
        --model_name allenai/OLMo-2-1124-7B-Instruct \
        --taboo_criteria token_selection_criterias/frequency_criteria_top_k_100.json \
        --dataset_path openai/gsm8k \
        --results_dir results \
        --n_samples 50 \
        --k_shot 5 \
        --max_length 512 \
        --judge_model_name allenai/OLMo-2-1124-7B-Instruct
    ```

2.  **POS Criterion (Nouns) on Sample Data:**

    ```bash
    python run.py \
        --model_name allenai/OLMo-2-1124-7B-Instruct \
        --taboo_criteria token_selection_criterias/noun_criteria.json \
        --dataset_path data/sample_questions.csv \
        --results_dir results \
        --n_samples 10 \
        --k_shot 3
    ```

3.  **Custom Criterion on GSM8K:**

    ```bash
    python run.py \
        --model_name allenai/OLMo-2-1124-7B-Instruct \
        --taboo_criteria token_selection_criterias/custom_criteria.json \
        --dataset_path openai/gsm8k \
        --results_dir results \
        --n_samples 20
    ```

4.  **Regex Criterion (Tokens containing digits) on Sample Data:**

    ```bash
    python run.py \
        --model_name allenai/OLMo-2-1124-7B-Instruct \
        --taboo_criteria token_selection_criterias/regex_digits_criteria.json \
        --dataset_path data/sample_questions.csv \
        --results_dir results \
        --n_samples 10
    ```

    _(Ensure `regex_digits_criteria.json` exists with `{"type": "regex", "pattern": "._\\d._"}`)_

## Customizing Taboo Criteria

Define custom criteria by creating JSON files in the `token_selection_criterias/` directory.

**Supported Types:**

- **`frequency`**: Selects tokens based on frequency in the tokenizer's vocabulary.
  - `k`: Number of tokens to select.
  - `order`: `"most_frequent"` or `"least_frequent"`.
  - `leaf`: `true` to select only "leaf" tokens (not composed of other tokens via BPE merges), `false` otherwise. Requires tokenizer merges file access.
  - `exclude_regexes` (Optional): List of regex patterns to exclude tokens (e.g., special tokens). Defaults provided in `token_selection.py`.
  ```json
  {
    "type": "frequency",
    "k": 100,
    "order": "most_frequent",
    "leaf": true
  }
  ```
- **`synthetic`**: Selects tokens based on Part-of-Speech (POS) tags using spaCy.
  - `pos`: List of spaCy POS tags (e.g., `["NOUN", "VERB"]`).
  - `k` (Optional): Maximum number of tokens.
  - `exclude_regexes` (Optional): List of regex patterns to exclude tokens.
  ```json
  {
    "type": "synthetic",
    "pos": ["NOUN"],
    "k": 500
  }
  ```
- **`custom`**: Uses a predefined list of taboo words. The selector finds corresponding tokens in the vocabulary.
  - `tokens`: List of taboo words (strings).
  - `exclude_regexes` (Optional): List of regex patterns to exclude tokens.
  ```json
  {
    "type": "custom",
    "tokens": ["the", "a", "is", "are", "example"]
  }
  ```
- **`regex`**: Selects tokens matching a Python regular expression.
  - `pattern`: The regex pattern string (e.g., `".*\\d.*"` for tokens with digits).
  - `k` (Optional): Maximum number of tokens.
  - `exclude_regexes` (Optional): List of regex patterns to exclude tokens.
  ```json
  {
    "type": "regex",
    "pattern": "^[A-Z].*", // Tokens starting with an uppercase letter
    "k": 200
  }
  ```
- **`combined`**: Selects tokens that satisfy _all_ specified sub-criteria.
  - `criteria`: A list of criteria dictionaries (each like the examples above, but using `"criterion"` instead of `"type"`).
  - `k` (Optional): Maximum number of tokens from the intersection.
  - `exclude_regexes` (Optional): List of regex patterns to exclude tokens (applied _after_ intersection).
  ```json
  {
    "type": "combined",
    "criteria": [
      { "criterion": "frequency", "leaf": true, "order": "most_frequent" },
      { "criterion": "synthetic", "pos": ["VERB"] }
    ],
    "k": 50
  }
  ```

## Evaluation Metrics

The framework outputs evaluation results to the specified `--results_dir`. For each run, two JSON files are created:

1.  **`evaluation_results_*.json`**: Contains summary metrics:
    - `accuracy_with_taboo`: Accuracy on the dataset when taboo constraints are applied.
    - `accuracy_free_decoding`: Accuracy when no constraints are applied.
    - `agreement_percentage`: How often the taboo-constrained model was correct when the free model was correct.
    - `taboo_usage_percentage_with_taboo`: Percentage of answers generated _with_ constraints that still contained taboo tokens (ideally 0).
    - `taboo_usage_percentage_free_decoding`: Percentage of answers generated _without_ constraints that contained taboo tokens.
    - Counts for correct answers and taboo mistakes.
    - Metadata about the run (model, dataset, criteria, etc.).
2.  **`all_answers_*.json`**: Contains detailed information for each sample evaluated:
    - Question, correct answer.
    - Model-generated answer with taboo constraints.
    - Model-generated answer without constraints (free decoding).
    - Whether each answer was judged correct (direct comparison and optionally LLM judge).
    - Whether each answer contained taboo tokens.
    - The prompt used for generation.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find bugs, have suggestions, or want to add new features.

## License

All rights reserved to Itay Itzhak, Guy Kaplan, Shahar Levy, and Amit Ben-Artzi.
