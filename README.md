# Token Taboo Evaluation Framework

## Overview

The Token Taboo Evaluation Framework is designed to evaluate language models by imposing constraints on token generation. This framework allows users to define "taboo" tokens that the model should avoid generating, and then assess the model's performance under these constraints. The framework supports various criteria for selecting taboo tokens, including frequency, part-of-speech (POS), custom lists, and combined criteria.

## Project Structure

- **token_selection.py**: Contains the `TokenSelector` class, which is responsible for selecting taboo tokens based on different criteria.
- **run.py**: The main script to run the evaluation framework. It loads models, tokenizers, and taboo criteria, and performs evaluations.
- **evaluation.py**: Contains the `Evaluator` class, which evaluates the model's performance on reasoning datasets and other tasks.
- **taboo_model.py**: Defines the `TabooModel` class, which wraps a language model to enforce token generation constraints.
- **token_selection_criterias/**: Directory containing JSON files that define different taboo token selection criteria.
- **data/sample_questions.csv**: A sample dataset used for evaluating the model's reasoning capabilities.
- **requirements.txt**: Lists the Python dependencies required to run the project.
- **playground.ipynb**: A Jupyter notebook for experimenting with the framework and testing different components.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/token-taboo-evaluation.git
   cd token-taboo-evaluation
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the spaCy model (if using POS criteria):**

   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the Framework

### Example 1: Frequency Criterion

To run the framework using a frequency criterion, execute the following command:

```bash
python run.py --model_name allenai/OLMo-2-1124-7B-Instruct --taboo_criteria token_selection_criterias/frequency_criteria.json --dataset_path data/sample_questions.csv --results_dir results
```

### Example 2: Synthetic Criterion

To run the framework using a synthetic criterion (POS-based), use:

```bash
python run.py --model_name allenai/OLMo-2-1124-7B-Instruct --taboo_criteria token_selection_criterias/pos_criteria.json --dataset_path data/sample_questions.csv --results_dir results
```

### Example 3: Combined Criterion

To run the framework using a combined criterion, execute:

```bash
python run.py --model_name allenai/OLMo-2-1124-7B-Instruct --taboo_criteria token_selection_criterias/combined_criteria.json --dataset_path data/sample_questions.csv --results_dir results
```

## Customizing Taboo Criteria

You can define your own taboo criteria by creating a JSON file in the `token_selection_criterias` directory. The JSON structure should specify the type of criterion and any additional parameters required for that type. For example:

```json
{
  "type": "custom",
  "tokens": ["example", "token", "list"]
}
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
