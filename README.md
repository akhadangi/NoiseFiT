# NoiseFiT

This repository provides an implementation of our preprint **[NoiseFiT](https://arxiv.org/abs/2504.03302)**, which introduces a novel method for adaptive noise injection in transformer models. By dynamically injecting noise based on the signal-to-noise ratio (SNR) into specific transformer layers, NoiseFiT seeks to reduce hallucinated responses in LLMs.

Key features include:
- **Adaptive Noise Injection:** Dynamically injects noise into transformer layers using token-specific scaling.
- **Custom Trainer:** Extends the standard SFTTrainer to compute a hybrid loss that combines cross-entropy, soft target, and consistency losses.
- **Flexible CLI:** Train your model with customizable training parameters directly from the command line.

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- [PEFT](https://github.com/huggingface/peft)
- [Accelerate](https://github.com/huggingface/accelerate)
- [Weights & Biases (wandb)](https://wandb.ai/site)
- [TRL](https://github.com/lvwerra/trl)

Install the dependencies using pip:

```bash
pip install torch transformers datasets peft accelerate wandb trl
```

## Usage

The main script is `NoiseFiT.py`, which you can run from the command line. It accepts several arguments with default values; you can override them as needed.

### Command-Line Arguments

- `--model`: Pre-trained model identifier (e.g., `gpt2`). **(Required)**
- `--train_data`: Path to a CSV file containing training data with `prompt` and `response` columns. **(Required)**
- `--output_model`: Directory for saving output checkpoints (default: `output_model`).
- `--batch_size`: Per-device training batch size (default: 4).
- `--epochs`: Number of training epochs (default: 5).
- `--max_steps`: Maximum training steps (default: 1000).
- `--learning_rate`: Learning rate (default: 5e-5).
- `--noise_std`: Base noise standard deviation (default: 0.1).
- `--lambda_consistency`: Consistency loss weight (default: 0.1).
- `--snr_format`: SNR selection method for layer noise injection (`Largest` or `Lowest`, default: `Largest`).
- `--num_noise_layers`: Number of transformer layers to inject noise into (default: 3).
- `--hybrid_loss_alpha`: Weight for hybrid loss component (default: 0.5).
- `--hf_token`: Hugging Face token for login (optional, if model not already in your cache or not download locally).

## Example Command

```bash
python NoiseFiT.py \
   --model gpt2 \
   --train_data "dataset/train.csv" \
   --output_model my_output \
   --batch_size 4 \
   --epochs 5 \
   --hf_token YOUR_HF_TOKEN
```

## Inference

In addition to training, we provide an inference script (`Inference.py`) to generate responses from the fine-tuned NoiseFiT models using a PEFT adapter. This script leverages Hugging Face Accelerate for efficient distributed processing and supports customizable generation parameters.

### Inference Script Overview

The inference script performs the following steps:

1. **Model and Tokenizer Loading:**  
   - Loads a base model (default: `mistralai/Mistral-7B-v0.1`) and its corresponding tokenizer.
   - Merges a PEFT adapter checkpoint with the base model.

2. **Dataset Partitioning:**  
   - Reads prompts from an input CSV file (expects a column named `prompt`).
   - Splits the dataset evenly across available GPUs for parallel processing using Accelerate.

3. **Response Generation:**  
   - Formats prompts using special tokens to separate user input and the assistant's response.
   - Generates responses with configurable parameters like maximum new tokens, temperature, top-p, top-k, and repetition penalty.
   - Cleans and extracts the generated assistant responses.

4. **Result Gathering:**  
   - Gathers results from all processes and outputs a consolidated CSV file with the generated responses.

### Command-Line Arguments for Inference

- `--model_id`: Hugging Face model identifier for the base model (default: `mistralai/Mistral-7B-v0.1`).
- `--model_path`: Path to the PEFT adapter checkpoint.
- `--input_csv`: CSV file containing the prompts (expects a `prompt` column). Default: `test_ground_truth.csv`.
- `--output_csv`: Output CSV file name. If not provided, a default name based on the model path is used.
- `--rounds`: Number of rounds to generate responses per prompt (default: 5).
- `--max_new_tokens`: Maximum number of new tokens to generate per response (default: 50).
- `--temperature`: Temperature for sampling (default: 0.5).
- `--top_p`: Top-p nucleus sampling probability (default: 0.9).
- `--top_k`: Top-k sampling parameter (default: 40).
- `--repetition_penalty`: Repetition penalty for generation (default: 1.2).
- `--hf_token`: Hugging Face authentication token (optional).

### Example Command for Inference

```bash
python Inference.py \
    --model_id "base_model_id" \
    --model_path "path/to/your/PEFT_checkpoint" \
    --input_csv "dataset/test_ground_truth.csv" \
    --output_csv "final_output.csv" \
    --rounds 5 \
    --hf_token "YOUR_HF_TOKEN"

