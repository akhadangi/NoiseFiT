#!/usr/bin/env python3
"""
Evaluate zero‐shot accuracy of NoiseFiT variants and BaseFiT on the
TruthfulQA multiple‐choice (MC) dataset.
"""

import re
import argparse
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero‐shot evaluation on TruthfulQA‐MC"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="your-model-name",
        help="Pretrained model identifier",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="EleutherAI/truthful_qa_mc",
        help="Hugging Face dataset (default: EleutherAI/truthful_qa_mc)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="multiple_choice",
        help="Dataset subset (default: multiple_choice)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Which split to use (train/validation/test)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3,
        help="How many new tokens to generate per prompt",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model placement (default: auto)",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    return tokenizer, model


def make_generator(model, tokenizer, max_new_tokens: int):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy decoding
    )


def evaluate_zero_shot(gen, tokenizer, dataset) -> float:
    correct = 0
    total = len(dataset)

    for example in tqdm(dataset, desc="Evaluating"):
        # Build prompt
        question = example["question"]
        choices = example["choices"]
        label = example["label"]

        prompt = question + "\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(ord('A')+i)}. {choice}\n"
        prompt += "Answer:"

        # Generate and parse
        out = gen(
            prompt,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=True,
        )[0]["generated_text"]
        # Strip away prompt from the generated text
        generated = out[len(prompt) :].strip()

        # Look for A-E in the first few characters
        m = re.search(r"\b([A-E])\b", generated)
        if m:
            pred = ord(m.group(1)) - ord("A")
            if pred == label:
                correct += 1

    return correct / total if total > 0 else 0.0


def main():
    args = parse_args()

    print(f"Loading dataset {args.dataset} ({args.subset}) split={args.split}...")
    ds = load_dataset(args.dataset, args.subset, split=args.split)

    print(f"Loading model & tokenizer: '{args.model_name}'...")
    tokenizer, model = load_model_and_tokenizer(args.model_name, args.device_map)

    print("Preparing generation pipeline...")
    gen = make_generator(model, tokenizer, args.max_new_tokens)

    print("Running evaluation...")
    acc = evaluate_zero_shot(gen, tokenizer, ds)
    print(f"Zero‐shot accuracy on TruthfulQA‐MC: {acc:.2%}")


if __name__ == "__main__":
    main()
