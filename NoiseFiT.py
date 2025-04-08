#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 3 00:00:00 2025

@author: Afshin Khadangi

A command-line tool for training a transformer model with adaptive noise injection.

Example usage:
    python NoiseFiT.py --model meta-llama/Llama-3.2-1B --train_data train.csv --output_model my_output --batch_size 4 --epochs 5 --hf_token YOUR_HF_TOKEN

"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import argparse
import os
import warnings

# =============================================================================
# Third-party Imports
# =============================================================================
from transformers import logging, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
import torch.nn.functional as F
import pandas as pd
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from accelerate import Accelerator
import wandb

# =============================================================================
# Warnings and Logging Configuration
# =============================================================================
warnings.filterwarnings("ignore")
logging.set_verbosity_error()  # Suppress transformer warnings

# =============================================================================
# Global Constants
# =============================================================================
EPSILON = 1e-6         # Small constant for numerical stability
DEFAULT_BETA = 0.1     # Default beta for adaptive noise injection

# =============================================================================
# Accelerator and Device Setup
# =============================================================================
accelerator = Accelerator()
torch.cuda.set_device(accelerator.local_process_index)

# =============================================================================
# Data Preparation
# =============================================================================
def formatted_train(input_text: str, response: str) -> str:
    """
    Formats a single training example with user and assistant tokens.
    
    Args:
        input_text (str): The prompt text.
        response (str): The corresponding response text.
    
    Returns:
        str: The formatted training string.
    """
    return (
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>\n"
    )

def prepare_train_data(data: pd.DataFrame) -> Dataset:
    """
    Validates and prepares the training dataset by combining prompt and response
    columns into a single text field suitable for causal language modeling.
    
    Args:
        data (pd.DataFrame): The training dataframe expected to have 'prompt' and 'response' columns.
    
    Returns:
        Dataset: A Hugging Face Dataset object created from the processed dataframe.
    
    Raises:
        ValueError: If the required columns are missing.
    """
    if not {"prompt", "response"}.issubset(data.columns):
        raise ValueError("Input data must contain 'prompt' and 'response' columns")
    
    # Apply formatting to each row
    data["text"] = data[["prompt", "response"]].apply(
        lambda x: (
            f"<|im_start|>user\n{x['prompt']} <|im_end|>\n"
            f"<|im_start|>assistant\n{x['response']}<|im_end|>\n"
        ),
        axis=1
    )
    return Dataset.from_pandas(data)


# =============================================================================
# Model and Tokenizer Loading
# =============================================================================
def get_model_and_tokenizer(model_id: str):
    """
    Loads the tokenizer and the quantized model based on the provided model_id.
    
    Args:
        model_id (str): Identifier or path for the model.
    
    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Set padding token for causal language models
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure 4-bit quantization settings
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )
    
    # Load the model with the quantization configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


# =============================================================================
# Adaptive Noise Injection Utility
# =============================================================================
def adaptive_noise_injection(
    target_embeds: torch.Tensor,
    base_std: float,
    beta: float = DEFAULT_BETA,
    noise_scale: float = 1.0,
    noise_gate: float = 1.0,
    logits: torch.Tensor = None
) -> torch.Tensor:
    """
    Injects adaptive zero-mean Gaussian noise into target embeddings with token-specific scaling.
    
    Args:
        target_embeds (Tensor): Tensor of shape [B, L, H] representing target embeddings.
        base_std (float): Base standard deviation for noise.
        beta (float): Controls exponential weighting sensitivity.
        noise_scale (float): Additional scaling factor.
        noise_gate (float): Direct scaling factor for noise magnitude.
        logits (Tensor, optional): Tensor of shape [B, L, vocab_size] for entropy-based modulation.
    
    Returns:
        Tensor: The target embeddings after noise injection.
    """
    # Calculate token-specific noise factor using logits if available
    if logits is not None:
        p = F.softmax(logits, dim=-1)  # [B, L, vocab_size]
        entropy = -(p * torch.log(p + EPSILON)).sum(dim=-1, keepdim=True)
        noise_factor = torch.exp(entropy)
    else:
        embed_var = target_embeds.var(dim=-1, keepdim=True)
        noise_factor = torch.exp(embed_var / (embed_var.mean() + EPSILON))
    
    # Clamp noise factor to prevent extreme values
    noise_factor = torch.clamp(noise_factor, max=5.0)
    
    # Robust statistics for the embeddings
    median = target_embeds.median(dim=-1, keepdim=True).values
    mad = (target_embeds - median).abs().median(dim=-1, keepdim=True).values
    token_norm = target_embeds.norm(dim=-1, keepdim=True)
    mad = mad.clamp(min=EPSILON * token_norm)
    
    # Exponential weighting based on deviation from median
    weight = torch.exp(-beta * torch.abs(target_embeds - median) / (mad + EPSILON))
    
    # Direct gating for controlling noise magnitude
    gate = torch.tensor(noise_gate, device=target_embeds.device).clamp(0, 1)
    
    # Calculate effective noise scale and add noise
    effective_scale = base_std * noise_scale * gate * mad * weight * noise_factor
    noise = torch.randn_like(target_embeds) * effective_scale
    return target_embeds + noise

# =============================================================================
# Transformer Layers Extraction
# =============================================================================
def get_transformer_layers(model):
    """
    Extracts transformer blocks from a given model. Handles various model wrappers and architectures.
    
    Args:
        model: The model object.
    
    Returns:
        list: A list of transformer layer modules.
    
    Raises:
        ValueError: If transformer layers cannot be found.
    """
    # Unwrap model if wrapped by PEFT
    if hasattr(model, "base_model"):
        model = model.base_model
    
    # Check for nested model attributes
    if hasattr(model, "model"):
        sub_model = model.model
        if hasattr(sub_model, "layers"):
            return sub_model.layers
        if hasattr(sub_model, "model"):
            deeper = sub_model.model
            if hasattr(deeper, "layers"):
                return deeper.layers
        if hasattr(sub_model, "h"):
            return sub_model.h
    
    # Direct layer access
    if hasattr(model, "layers"):
        return model.layers
    
    # Fallback for GPT-2 style models
    if hasattr(model, "transformer"):
        sub_model = model.transformer
        if hasattr(sub_model, "h"):
            return sub_model.h
    
    raise ValueError("Cannot find transformer layers in the model.")

# =============================================================================
# Custom Trainer Class Definition: NoiseFitTrainer
# =============================================================================
class NoiseFitTrainer(SFTTrainer):
    """
    Custom trainer that injects adaptive noise into transformer layers during training.
    Inherits from SFTTrainer and extends functionality with noise injection hooks
    and a hybrid loss combining cross-entropy, soft targets, and consistency loss.
    """
    def __init__(
        self, *args, 
        noise_std: float = 0.1,
        temperature: float = 1.0,
        hybrid_loss_alpha: float = 0.5,
        lambda_consistency: float = 0.1,
        snr_format: str = "Largest",
        num_noise_layers: int = 3,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.noise_std = noise_std
        self.temperature = temperature
        self.hybrid_loss_alpha = hybrid_loss_alpha
        self.lambda_consistency = lambda_consistency
        self.inject_noise = False  # Controls noise injection during forward passes
        self.selected_layer_indices = None  # Stores indices of layers for noise injection
        self.snr_format = snr_format
        self.num_noise_layers = num_noise_layers
        self.hooks = []  # List to store registered forward hooks

    def select_layers(self):
        """
        Identifies transformer layers for noise injection based on Signal-to-Noise Ratio (SNR).
        Selects layers with either the highest or lowest SNR values as specified.
        """
        model = self.model
        model.eval()  # Set model to evaluation mode for SNR calculation
        
        # Get a batch from the training dataloader and move to model's device
        dataloader = self.get_train_dataloader()
        batch = next(iter(dataloader))
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        with torch.no_grad():
            # Clean forward pass for baseline hidden states
            outputs_clean = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True
            )
            # Exclude the input embeddings (index 0) and use all transformer layer hidden states
            hidden_clean = outputs_clean.hidden_states[1:]
            num_layers = len(hidden_clean)
            signal_per_layer = [h.abs().mean() for h in hidden_clean]
        
            num_noisy_passes = 10  # Number of noisy passes for SNR estimation
            # Prepare a tensor to accumulate noise differences over multiple passes per layer
            noise_diff = torch.zeros((num_noisy_passes, num_layers), device=model.device)
    
            # For each noisy pass, inject noise directly into each hidden state
            for pass_idx in range(num_noisy_passes):
                for i, h_clean in enumerate(hidden_clean):
                    # Compute a noise standard deviation for this hidden state
                    noise_std_hidden = self.noise_std * h_clean.std().item()
                    # Inject adaptive noise directly to the hidden state
                    h_noisy = adaptive_noise_injection(
                        h_clean,
                        base_std=noise_std_hidden,
                        logits=None
                    )
                    # Compute and store the mean absolute difference between noisy and clean hidden states
                    noise_diff[pass_idx, i] = (h_noisy - h_clean).abs().mean()
    
            # Average the noise difference for each layer across passes
            avg_noise_diff = noise_diff.mean(dim=0)
            # Compute the SNR for each layer (adding a small constant for stability)
            snr = torch.tensor(
                [signal_per_layer[i] / (avg_noise_diff[i] + 1e-6) for i in range(num_layers)],
                device=model.device
            )
    
        # Select top k layers based on SNR according to the specified format ("Largest" or not)
        k = self.num_noise_layers
        largest = True if self.snr_format == "Largest" else False
        _, indices = torch.topk(snr, k, largest=largest)
        self.selected_layer_indices = indices.tolist()
    
        print(f"Selected layers for noise injection (SNR based): {self.selected_layer_indices}")
        print(f"SNR values of selected layers: {snr[self.selected_layer_indices]}")

    def register_noise_hooks(self):
        """
        Registers forward hooks on the selected transformer layers to inject noise.
        """
        self.hooks = []
        layers = list(get_transformer_layers(self.model))
        for idx in self.selected_layer_indices:
            hook = layers[idx].register_forward_hook(self.noise_hook)
            self.hooks.append(hook)

    def noise_hook(self, module, input, output):
        """
        Forward hook that injects adaptive noise into the module's output if enabled.
        """
        if self.inject_noise:
            hidden_states = output[0]
            noise_std_hidden = self.noise_std * hidden_states.std().item()
            noisy_hidden_states = adaptive_noise_injection(
                hidden_states,
                base_std=noise_std_hidden * self.noise_std,
                logits=None
            )
            # Return tuple with noised hidden states replacing the original ones
            return (noisy_hidden_states,) + output[1:]
        return output  # Return unchanged output if noise injection is disabled

    def train(self, *args, **kwargs):
        """
        Overrides the training process by selecting layers and registering noise hooks before standard training.
        """
        self.select_layers()        # Identify layers based on SNR
        self.register_noise_hooks()   # Attach noise injection hooks
        super().train(*args, **kwargs)  # Proceed with the standard training loop

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the total loss by combining cross-entropy loss with a hybrid loss
        that includes soft target loss and consistency loss between noisy forward passes.
        """
        # Clean forward pass
        self.inject_noise = False
        outputs = model(**inputs)
        ce_loss = outputs.loss  # Cross-entropy loss
        logits = outputs.logits
        labels = inputs["labels"]
        target_mask = labels != -100  # Mask to ignore padding or irrelevant tokens

        # Generate noisy outputs with injected noise
        self.inject_noise = True
        noisy_outputs1 = model(**inputs)
        logits_noise1 = noisy_outputs1.logits
        noisy_outputs2 = model(**inputs)  # Second noisy pass for consistency check
        logits_noise2 = noisy_outputs2.logits

        # Compute soft cross-entropy loss using clean logits as targets
        soft_targets = F.softmax(logits / self.temperature, dim=-1)
        soft_ce_loss1 = F.kl_div(
            F.log_softmax(logits_noise1, dim=-1), soft_targets, reduction='none'
        ).sum(dim=-1)
        soft_ce_loss2 = F.kl_div(
            F.log_softmax(logits_noise2, dim=-1), soft_targets, reduction='none'
        ).sum(dim=-1)
        soft_ce_loss = 0.5 * (soft_ce_loss1 + soft_ce_loss2)
        soft_ce_loss = (soft_ce_loss * target_mask.float()).sum() / target_mask.sum()

        # Consistency loss between two noisy outputs
        consistency_loss = F.kl_div(
            F.log_softmax(logits_noise1, dim=-1),
            F.softmax(logits_noise2, dim=-1),
            reduction='none'
        ).sum(dim=-1)
        consistency_loss = (consistency_loss * target_mask.float()).sum() / target_mask.sum()

        # Combine losses: hybrid loss blends CE loss and soft target loss, plus consistency regularization
        hybrid_loss = self.hybrid_loss_alpha * ce_loss + (1 - self.hybrid_loss_alpha) * soft_ce_loss
        total_loss = hybrid_loss + self.lambda_consistency * consistency_loss

        return (total_loss, outputs) if return_outputs else total_loss

# =============================================================================
# Main Training Routine with Argument Parsing
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train a transformer model with adaptive noise injection (NoiseFiT)."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Pre-trained model identifier (e.g. 'meta-llama/Llama-3.2-1B').")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to CSV file containing training data.")
    parser.add_argument("--output_model", type=str, default="output_model",
                        help="Directory for saving output checkpoints (default: output_model).")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device training batch size (default: 4).")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5).")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum training steps (default: 1000).")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5).")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Base noise standard deviation (default: 0.1).")
    parser.add_argument("--lambda_consistency", type=float, default=0.1,
                        help="Consistency loss weight (default: 0.1).")
    parser.add_argument("--snr_format", type=str, choices=["Largest", "Lowest"],
                        default="Largest",
                        help="SNR selection method for layer noise injection (default: Largest).")
    parser.add_argument("--num_noise_layers", type=int, default=3,
                        help="Number of transformer layers to inject noise into (default: 3).")
    parser.add_argument("--hybrid_loss_alpha", type=float, default=0.5,
                        help="Weight for hybrid loss component (default: 0.5).")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token for login (default: None).")
    
    args = parser.parse_args()
    
    if args.hf_token is not None:
        from huggingface_hub import login
        login(token=args.hf_token)
        print("Hugging Face login successful.")

    # Load training data
    train_df = pd.read_csv(args.train_data)
    dataset = prepare_train_data(train_df)
    
    # Load model and tokenizer using the provided model id
    model, tokenizer = get_model_and_tokenizer(args.model)
    
    # Set up WandB and TrainingArguments
    wandb.login()
    os.environ["WANDB_PROJECT"] = args.output_model
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    
    training_arguments = TrainingArguments(
        output_dir=args.output_model,
        run_name=args.output_model,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=5,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        fp16=True,
        max_grad_norm=1.0,
        report_to="wandb"
    )
    
    # Configure PEFT for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Instantiate the custom trainer with command-line parameters
    trainer = NoiseFitTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_arguments,
        tokenizer=tokenizer,
        hybrid_loss_alpha=args.hybrid_loss_alpha,
        noise_std=args.noise_std,
        lambda_consistency=args.lambda_consistency,
        snr_format=args.snr_format,
        num_noise_layers=args.num_noise_layers
    )
    
    # Start the training process
    trainer.train()

# =============================================================================
# Module Execution Entry Point
# =============================================================================
if __name__ == "__main__":
    main()
