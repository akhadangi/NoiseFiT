#!/usr/bin/env python
"""
Created on Thu Apr 3 00:00:00 2025

@author: Afshin Khadangi

Inference Script for NoiseFiT with PEFT adapter using Accelerate.

This script loads a base model (default: mistralai/Mistral-7B-v0.1) and a PEFT adapter 
(PEFT_CHECKPOINT), processes an input CSV of prompts, generates responses in multiple rounds, 
and aggregates the results into a final CSV output.

It supports passing Hugging Face token for gated repositories (if any) and allows to override various generation parameters.
"""

import re
import argparse
import warnings
import pandas as pd
import torch
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import PeftModel

# Enable Hugging Face authentication (if token provided)
# (This can also be done via the huggingface-cli or environment variable HUGGINGFACE_TOKEN)
from huggingface_hub import login

# Suppress warnings and transformer logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def formatted_prompt(user_input: str) -> str:
    """
    Format the prompt with special tokens for user and assistant.
    """
    return f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant"


def generate_response(user_input: str, model, tokenizer, gen_kwargs: dict) -> str:
    """
    Generate a response from the model for a given user prompt.
    
    The function formats the prompt, tokenizes it, performs generation with the provided
    generation parameters, and cleans up the output.
    """
    # Format the prompt with the required special tokens.
    prompt = formatted_prompt(user_input)
    # Tokenize input and move it to the model's device.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate output with specified decoding parameters.
    outputs = model.generate(
        **inputs,
        max_new_tokens=gen_kwargs.get("max_new_tokens", 50),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=gen_kwargs.get("temperature", 0.5),
        top_p=gen_kwargs.get("top_p", 0.9),
        top_k=gen_kwargs.get("top_k", 40),
        repetition_penalty=gen_kwargs.get("repetition_penalty", 1.2)
    )
    
    # Decode the generated tokens.
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Clean the output: remove any text before the first '?' (if exists) and flatten newlines.
    if "?" in full_response:
        full_response = " ".join(full_response.split("?")[1:])
    full_response = " ".join(full_response.split("\n"))
    return full_response


def extract_from_response(generated_response: str) -> str:
    """
    Extract the assistant's response from the generated text.
    
    Supports different token patterns in the generated text.
    """
    assistant_response = ""
    # Check for the special token format.
    if "<|im_start|>user" in generated_response:
        if "<|im_start|>assistant\n" in generated_response:
            start_assistant = generated_response.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        else:
            start_assistant = generated_response.find("<|im_start|>assistant") + len("<|im_start|>assistant")
        end_assistant = generated_response.find("<|im_end|>", start_assistant)
        if end_assistant == -1:
            end_assistant = len(generated_response)
        assistant_response = generated_response[start_assistant:end_assistant].strip()
    
    # Fallback to a different format if special tokens are not used.
    elif "user:" in generated_response and "assistant:" in generated_response:
        start_assistant = generated_response.find("assistant:") + len("assistant:")
        assistant_response = generated_response[start_assistant:].strip()
    
    # Default extraction if no known format is found.
    else:
        start_assistant = generated_response.find("<|im_start|>assistant") + len("<|im_start|>assistant")
        end_assistant = generated_response.find("<|im_end|>", start_assistant)
        if end_assistant == -1:
            end_assistant = len(generated_response)
        assistant_response = generated_response[start_assistant:end_assistant].strip()
    
    return assistant_response


def process_chunk(local_df: pd.DataFrame, model, tokenizer, rounds: int, gen_kwargs: dict) -> pd.DataFrame:
    """
    Process a chunk of the dataset for a specified number of rounds.
    
    For each round, generate a response for each prompt and record the results.
    """
    dfs = []
    for rnd in tqdm(range(1, rounds + 1), desc="Processing rounds"):
        df_temp = local_df.copy()
        df_temp['noisy fine tuning'] = df_temp.prompt.apply(
            lambda x: extract_from_response(generate_response(x, model, tokenizer, gen_kwargs))
        )
        df_temp['round'] = rnd
        dfs.append(df_temp)
    return pd.concat(dfs, ignore_index=True)


def main(args):
    """
    Main function to run the inference.
    
    Loads model and tokenizer (with optional authentication), processes the dataset in parallel across GPUs,
    and gathers the results into a final CSV output.
    """
    # Log in to Hugging Face if a token is provided.
    if args.hf_token:
        login(args.hf_token)
    
    # Initialize accelerator for distributed processing.
    accelerator = Accelerator()
    device = accelerator.device

    # Load the base model on the designated device with half precision.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
        use_auth_token=args.hf_token if args.hf_token else None
    )

    # Load and merge the PEFT adapter into the base model.
    peft_model = PeftModel.from_pretrained(
        model, 
        args.model_path, 
        device_map={"": device},
        use_auth_token=args.hf_token if args.hf_token else None
    )
    model = peft_model.merge_and_unload()

    # Load the tokenizer and set the pad token.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_auth_token=args.hf_token if args.hf_token else None
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset containing prompts.
    test_df = pd.read_csv(args.input_csv)

    # Determine the data chunk for this process.
    local_rank = accelerator.local_process_index
    num_processes = accelerator.num_processes
    chunk_size = len(test_df) // num_processes
    start_idx = local_rank * chunk_size
    end_idx = start_idx + chunk_size if local_rank < num_processes - 1 else len(test_df)
    local_df = test_df.iloc[start_idx:end_idx]

    # Prepare generation parameters.
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty
    }

    # Process the local chunk of data and generate responses over multiple rounds.
    local_final_df = process_chunk(local_df, model, tokenizer, args.rounds, generation_kwargs)

    # Convert the local DataFrame to a list of dictionaries for gathering across processes.
    local_results = local_final_df.to_dict('records')
    all_results = accelerator.gather(local_results)

    # Only the main process writes the final combined CSV output.
    if accelerator.is_main_process:
        all_results_flat = [item for sublist in all_results for item in sublist]
        final_df = pd.DataFrame(all_results_flat)
        output_filename = args.output_csv if args.output_csv else args.model_path.split('/')[0] + "_final_output.csv"
        final_df.to_csv(output_filename, index=False)
        print(f"Final output saved to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="General Inference Script for Causal LM with PEFT and Accelerate"
    )
    # Model and checkpoint parameters.
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-v0.1",
                        help="HuggingFace model ID for the base model.")
    parser.add_argument("--model_path", type=str, default="PEFT_CHECKPOINT",
                        help="Path to the PEFT adapter checkpoint (default: PEFT_CHECKPOINT).")
    
    # Data I/O parameters.
    parser.add_argument("--input_csv", type=str, default="test_ground_truth.csv",
                        help="CSV file containing the prompts (expects a 'prompt' column).")
    parser.add_argument("--output_csv", type=str, default="",
                        help="Output CSV filename. If not provided, a default name will be used.")
    
    # Generation parameters.
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of rounds to generate responses per prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum number of new tokens to generate per response.")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p nucleus sampling probability.")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling value.")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty for generation.")
    
    # Hugging Face authentication token.
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face authentication token for private models.")

    args = parser.parse_args()
    main(args)
