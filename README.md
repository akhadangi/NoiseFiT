# NoiseFiT

This repository provides the implementation of **[NoiseFiT](https://arxiv.org/abs/2504.03302)** for hallucination-aware supervised fine-tuning of causal language models. NoiseFiT injects adaptive hidden-state noise into transformer layers and optimizes the model with clean supervision, clean-to-noisy alignment, and noisy-branch consistency.

## Version 2 update

The training entry point, `NoiseFiT.py`, now preserves the original manual SNR-layer NoiseFiT trainer while adding the second-version training stack. Use `--trainer_version v1` for the first-version trainer or the default `--trainer_version v2` for the updated auto/baseline stack. The v2 script supports:

- **BaseFiT**: clean supervised fine-tuning baseline.
- **NEFTune**: embedding-noise baseline with configurable `--neftune_alpha`.
- **R-Drop**: two-pass dropout consistency baseline with configurable KL weight and dropout.
- **Auto NoiseFiT**: automatically calibrates noise layers and per-layer noise scales from the training data.
- **Prefix Auto NoiseFiT**: optional prefix sharing for the full NoiseFiT objective to avoid recomputing deterministic early layers before the first noisy layer.
- **NoiseFiT loss ablations**: `ce_noisy`, `ce_kl`, `kl_only`, and `consistency_only`.
- **Diagnostics and logging**: writes `auto_noise_policy.json` to the output directory and logs loss components, selected-layer statistics, runtime noise statistics, memory, and speed metrics when W&B is enabled.

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- [PEFT](https://github.com/huggingface/peft)
- [Accelerate](https://github.com/huggingface/accelerate)
- [TRL](https://github.com/huggingface/trl)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [Weights & Biases](https://wandb.ai/site) for optional experiment logging

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data format

Training data should be a CSV with these columns:

| column | description |
| --- | --- |
| `prompt` | Instruction or user query. |
| `response` | Target assistant response. |

`NoiseFiT.py` converts each row into an instruction-response text field internally.


## Version selection

This repository now includes both training paths:

| version | command | purpose |
| --- | --- | --- |
| v1 manual SNR NoiseFiT | `python NoiseFiT.py --trainer_version v1 ...` | Original first-version trainer with `--snr_format`, `--num_noise_layers`, fixed `--noise_std`, and the original hybrid CE/KL/consistency objective. |
| v2 updated stack | `python NoiseFiT.py --trainer_version v2 ...` or omit the flag | Second-version trainer with baselines, automatic noise calibration, prefix sharing, and loss ablations. |

The original first-version script is also saved as `NoiseFiT_v1_manual_snr.py` for reproducibility.

## Training

The main script is `NoiseFiT.py`.


### First-version manual SNR NoiseFiT

```bash
python NoiseFiT.py \
  --trainer_version v1 \
  --model "meta-llama/Llama-2-7b-hf" \
  --train_data "dataset/train.csv" \
  --output_model "outputs/noisefit-v1-manual-snr" \
  --snr_format Largest \
  --num_noise_layers 3 \
  --noise_std 0.1 \
  --lambda_consistency 0.1 \
  --hybrid_loss_alpha 0.5
```

This keeps the first-version behavior: select the requested number of layers by SNR, inject adaptive hidden-state noise into those layers, and optimize the original hybrid objective.

### Full Auto NoiseFiT

```bash
python NoiseFiT.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --train_data "dataset/train.csv" \
  --output_model "outputs/noisefit-full" \
  --loss_mode full \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --epochs 3 \
  --learning_rate 5e-5 \
  --hf_token "YOUR_HF_TOKEN"
```

`loss_mode=full` performs one clean branch and two noisy branches. It optimizes clean CE, clean-to-noisy KL, and noisy-noisy consistency.

### Prefix Auto NoiseFiT

```bash
python NoiseFiT.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --train_data "dataset/train.csv" \
  --output_model "outputs/noisefit-prefix" \
  --loss_mode full \
  --use_prefix_sharing \
  --prefix_share_min_saved_layers 4 \
  --prefix_share_min_layer_frac 0.25 \
  --prefix_share_exclude_early_layers \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --epochs 3
```

Prefix sharing is most useful when auto calibration selects noisy layers after a reusable deterministic prefix. For distributed training, enable `--allow_prefix_sharing_ddp` only after a smoke test on your setup.

### Baselines

Base supervised fine-tuning:

```bash
python NoiseFiT.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --train_data "dataset/train.csv" \
  --output_model "outputs/basefit" \
  --loss_mode basefit
```

NEFTune:

```bash
python NoiseFiT.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --train_data "dataset/train.csv" \
  --output_model "outputs/neftune" \
  --loss_mode neftune \
  --neftune_alpha 5.0
```

R-Drop:

```bash
python NoiseFiT.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --train_data "dataset/train.csv" \
  --output_model "outputs/rdrop" \
  --loss_mode rdrop \
  --rdrop_alpha 1.0 \
  --rdrop_dropout 0.05 \
  --rdrop_kl_chunk_size 64
```

### Loss ablations

Use `--loss_mode` to isolate different parts of the NoiseFiT objective:

| mode | objective |
| --- | --- |
| `ce_noisy` | Cross-entropy on a noisy branch only. |
| `ce_kl` | Clean CE plus clean-to-noisy KL. |
| `kl_only` | Clean-to-noisy KL only. |
| `consistency_only` | Consistency between two noisy branches only. |
| `full` | Clean CE, clean-to-noisy KL, and noisy-noisy consistency. |

Example:

```bash
python NoiseFiT.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --train_data "dataset/train.csv" \
  --output_model "outputs/ablation-ce-kl" \
  --loss_mode ce_kl
```

## Important command-line arguments

### Core arguments

| argument | default | description |
| --- | --- | --- |
| `--trainer_version` | `v2` | Choose `v1` for the original manual SNR trainer or `v2` for the updated stack. |
| `--model` | required | Base model ID or local path. |
| `--train_data` | required | CSV file with `prompt` and `response`. |
| `--output_model` | `output_model` | Checkpoint/output directory. |
| `--batch_size` | `4` | Per-device training batch size. |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps. |
| `--epochs` | `3` | Training epochs. |
| `--max_steps` | `-1` | Maximum optimizer steps; `-1` uses epochs. |
| `--learning_rate` | `5e-5` | Learning rate. |
| `--logging_steps` | `20` | Logging interval. |
| `--hf_token` | `None` | Optional Hugging Face token. |


### First-version manual SNR arguments

These arguments are active when `--trainer_version v1` is used:

| argument | default | description |
| --- | --- | --- |
| `--snr_format` | `Largest` | Original layer-selection rule: choose layers with largest or lowest SNR. |
| `--num_noise_layers` | `3` | Number of transformer layers to perturb. |
| `--noise_std` | `0.1` | Base hidden-state noise standard deviation. |
| `--lambda_consistency` | `0.1` | Weight on noisy-branch consistency in the original objective. |
| `--hybrid_loss_alpha` | `0.5` | Weighting between clean CE and soft/noisy alignment in the original hybrid loss. |

### Objective arguments

| argument | default | description |
| --- | --- | --- |
| `--loss_mode` | `full` | Choose `basefit`, `neftune`, `rdrop`, `ce_noisy`, `ce_kl`, `consistency_only`, `kl_only`, or `full`. |
| `--temperature` | `1.0` | Temperature for KL-based losses. |
| `--hybrid_loss_alpha` | `0.5` | Weight on clean CE in hybrid NoiseFiT modes. |
| `--lambda_consistency` | `0.05` | Weight on noisy-noisy consistency in full NoiseFiT. |
| `--neftune_alpha` | `5.0` | NEFTune embedding-noise coefficient. |
| `--rdrop_alpha` | `1.0` | R-Drop KL coefficient. |
| `--rdrop_dropout` | `0.05` | Dropout probability enabled for R-Drop. |

### Auto NoiseFiT calibration

| argument | default | description |
| --- | --- | --- |
| `--auto_calib_batches` | `2` | Number of train batches used for calibration. |
| `--auto_scale_factors` | `0.25,0.5,1.0,1.5,2.0` | Scale grid around `1 / sqrt(hidden_dim)`. |
| `--auto_scale_score_tolerance` | `0.90` | Chooses the smallest scale within this fraction of the best score. |
| `--auto_depth_bands` | `3` | Number of depth bands for diverse layer selection. |
| `--auto_use_depth_diversity` | enabled | Keeps selected layers from collapsing into a single depth region. |
| `--auto_use_target_kl_scale` | enabled | Selects per-layer scales using a target KL rule. |
| `--auto_target_kl_fraction` | `0.75` | Fraction of selected-layer median KL used as the target. |
| `--auto_max_ce_delta_ratio` | `0.02` | Rejects scales that increase CE too much during calibration. |
| `--auto_scale_l2_penalty` | `0.05` | Soft penalty against unnecessarily large scales. |

### Prefix sharing

| argument | default | description |
| --- | --- | --- |
| `--use_prefix_sharing` | disabled | Enables prefix-shared full NoiseFiT. |
| `--prefix_share_min_saved_layers` | `4` | Minimum number of deterministic prefix layers to share. |
| `--prefix_share_min_layer_frac` | `0.25` | Minimum prefix as a fraction of total layers. |
| `--prefix_share_exclude_early_layers` | enabled | Encourages selected noisy layers to start after the prefix boundary. |
| `--allow_prefix_sharing_ddp` | disabled | Allows prefix sharing under DDP after manual validation. |

### Logging

Enable W&B logging with:

```bash
python NoiseFiT.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --train_data "dataset/train.csv" \
  --output_model "outputs/noisefit-wandb" \
  --loss_mode full \
  --report_to wandb \
  --wandb_project "NoiseFiT-revision"
```

You can pass `--report_api_key` or rely on an existing `wandb login` session.

## Outputs

The training script saves checkpoints under `--output_model`. Auto NoiseFiT modes additionally save:

```text
auto_noise_policy.json
```

This file contains selected layers, calibrated scales, scale-grid scores, KL/CE diagnostics, and layer-level statistics.

## Inference

The repository also includes `Inference.py` for generating responses from a fine-tuned PEFT adapter.

```bash
python Inference.py \
  --model_id "meta-llama/Llama-3.2-1B" \
  --model_path "path/to/your/PEFT_checkpoint" \
  --input_csv "dataset/test_ground_truth.csv" \
  --output_csv "final_output.csv" \
  --rounds 5 \
  --hf_token "YOUR_HF_TOKEN"
```

## Evaluation utilities

The repository includes:

- `evaluate_truthfulqa.py`
- `hallu_eval.py`

These scripts are kept for downstream evaluation workflows and can be adapted to compare BaseFiT, NEFTune, R-Drop, Auto NoiseFiT, Prefix Auto NoiseFiT, and ablation runs.
