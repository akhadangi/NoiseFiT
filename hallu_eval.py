#!/usr/bin/env python3
"""
hallu_eval.py

Evaluate NoiseFiT variants and BaseFiT on the HaluEval QA split and save the results to CSV.

Usage:
    python hallu_eval.py \
        --model_name your-model-name \
        --split test \
        --max_samples 5000 \
        --output_path results.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def evaluate_hallucination(
    model_name: str,
    split: str,
    max_samples: int,
) -> pd.DataFrame:
    """
    Load the HaluEval dataset split, run the model to generate answers, and collect results.

    Args:
        model_name: Hugging Face model identifier.
        split: Which split of HaluEval to load (e.g. "test").
        max_samples: Maximum number of examples to process.

    Returns:
        A pandas DataFrame with columns ["response", "ground_truth"].
    """
    # 1. Load dataset
    ds = load_dataset("flowaicom/HaluEval", split=split)

    # 2. Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=36,
        do_sample=False,
        temperature=0.0,
    )

    # 3. Generate and collect
    responses = []
    ground_truths = []

    for idx, sample in enumerate(tqdm(ds, desc="Evaluating")):
        if idx >= max_samples:
            break

        # Build prompt from passage and question
        passage = sample.get("passage") or sample.get("knowledge") or ""
        question = sample["question"]
        answer = sample.get("answer") or sample.get("right_answer") or ""

        prompt = f"{passage}\n\n{question}\n\n"

        out = gen(
            prompt,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=True,
        )[0]["generated_text"]
        # Strip off the prompt from the model output
        generated = out[len(prompt) :].strip()

        responses.append(generated)
        ground_truths.append(answer)

    # Create DataFrame
    df = pd.DataFrame({
        "response": responses,
        "ground_truth": ground_truths,
    })
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on HaluEval QA split.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model identifier (e.g. 'your-model-name').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum number of samples to process (default: 5000).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("hallu_eval_results.csv"),
        help="Output CSV file path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"Loading model '{args.model_name}' and dataset split '{args.split}'")

    df = evaluate_hallucination(
        model_name=args.model_name,
        split=args.split,
        max_samples=args.max_samples,
    )

    # Show first few rows
    logging.info("Sample outputs:\n%s", df.head().to_string(index=False))

    # Save to CSV
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)
    logging.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
