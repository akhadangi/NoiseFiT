#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NoiseFiT v2 training entry point.

This CLI keeps the original repository interface while adding the second-version
training modes: BaseFiT, NEFTune, R-Drop, automated NoiseFiT, prefix-shared
NoiseFiT, and loss ablations.
"""

import argparse
import json
import math
import os
import runpy
import sys
import time
import warnings
import inspect

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset
import pandas as pd
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import SFTTrainer

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

accelerator = Accelerator()
if torch.cuda.is_available():
    torch.cuda.set_device(accelerator.local_process_index)

def formatted_train(input_text, response) -> str:
    return f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"

def prepare_train_datav2(data):
    data_df = data.copy()

    # Validate dataset structure
    if not {"prompt", "response"}.issubset(data_df.columns):
        raise ValueError("train_df must contain 'prompt' and 'response' columns")

    # Avoid NaN / float concatenation errors
    data_df["prompt"] = data_df["prompt"].fillna("").astype(str)
    data_df["response"] = data_df["response"].fillna("").astype(str)

    # Remove empty rows
    data_df = data_df[
        (data_df["prompt"].str.strip() != "") &
        (data_df["response"].str.strip() != "")
    ].reset_index(drop=True)

    # Alpaca-style instruction format
    data_df["text"] = data_df[["prompt", "response"]].apply(
        lambda x:
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        + x["prompt"].strip()
        + "\n\n### Response:\n"
        + x["response"].strip(),
        axis=1,
    )

    return Dataset.from_pandas(data_df)

# --------------------------------------------
# Model and Tokenizer Loading Function
# --------------------------------------------
def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": accelerator.local_process_index},
    )

    model.config.use_cache = False

    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    return model, tokenizer


# --------------------------------------------
# Adaptive Noise Injection Function
# --------------------------------------------
EPSILON = 1e-6  # Increased for numerical stability
DEFAULT_BETA = 0.1

def adaptive_noise_injection(target_embeds, base_std, beta=DEFAULT_BETA, noise_scale=1.0, noise_gate=1.0, logits=None):
    """
    Injects adaptive zero-mean Gaussian noise into target embeddings with token-specific scaling.
    
    Args:
        target_embeds (Tensor): [B, L, H] target embeddings.
        base_std (float): Base noise standard deviation (scaled dynamically).
        beta (float): Controls exponential weighting sensitivity.
        noise_scale (float): Additional scaling factor.
        noise_gate (float): Direct scaling factor for noise magnitude.
        logits (Tensor, optional): [B, L, vocab_size] model logits for entropy-based noise factor.
    
    Returns:
        Tensor: Noised target embeddings.
    """
    # Token-specific noise factor
    if logits is not None:
        p = F.softmax(logits, dim=-1)  # [B, L, vocab_size]
        entropy = -(p * torch.log(p + EPSILON)).sum(dim=-1, keepdim=True)  # [B, L, 1]
        noise_factor = torch.exp(entropy)
    else:
        embed_var = target_embeds.var(dim=-1, keepdim=True)  # [B, L, 1]
        noise_factor = torch.exp(embed_var / (embed_var.mean() + EPSILON))
    
    # Cap noise factor to prevent extreme values
    noise_factor = torch.clamp(noise_factor, max=5.0) #reduced from 10 to 5

    # Robust statistics
    median = target_embeds.median(dim=-1, keepdim=True).values  # [B, L, 1]
    mad = (target_embeds - median).abs().median(dim=-1, keepdim=True).values
    token_norm = target_embeds.norm(dim=-1, keepdim=True)
    mad = mad.clamp(min=EPSILON * token_norm)

    # Exponential weighting
    weight = torch.exp(-beta * torch.abs(target_embeds - median) / (mad + EPSILON))  # [B, L, H]

    # Direct gating
    gate = torch.tensor(noise_gate, device=target_embeds.device).clamp(0, 1)

    # Effective noise scale
    effective_scale = base_std * noise_scale * gate * mad * weight * noise_factor  # [B, L, H]
    noise = torch.randn_like(target_embeds) * effective_scale
    return target_embeds + noise





def get_transformer_layers(model):
    """
    Returns the list of transformer blocks from the model.
    For many Llama models (including PEFT-wrapped ones), the layers might be found under
    model.base_model.model.layers, model.model.layers, or similar.
    """
    # Unwrap if model is wrapped (e.g., by PEFT).
    if hasattr(model, "base_model"):
        model = model.base_model
    # Try checking model.model first.
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
    # Fallback: check if the model itself has layers.
    if hasattr(model, "layers"):
        return model.layers
    # Fallback for GPT-2 style.
    if hasattr(model, "transformer"):
        sub_model = model.transformer
        if hasattr(sub_model, "h"):
            return sub_model.h
    raise ValueError("Cannot find transformer layers in the model.")



# --------------------------------------------
# MAIN
# --------------------------------------------
class Noise2NoiseTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        noise_std=0.1,
        temperature=1.0,
        hybrid_loss_alpha=0.5,
        lambda_consistency=0.1,
        rdrop_alpha=1.0,
        rdrop_kl_chunk_size=64,
        rdrop_dropout=0.05,
        loss_mode="full",
        neftune_alpha=5.0,
    
        # Auto-noise calibration
        auto_calib_batches=2,
        auto_scale_factors=(0.25, 0.5, 1.0, 1.5, 2.0),
        auto_scale_score_tolerance=0.90,
        auto_depth_bands=3,
        auto_use_depth_diversity=True,

        use_prefix_sharing=False,
        prefix_share_min_saved_layers=4,
        prefix_share_min_layer_frac=0.25,
        prefix_share_exclude_early_layers=True,
        allow_prefix_sharing_ddp=False,

        auto_use_target_kl_scale=True,
        auto_target_kl_fraction=0.75,
        
        
        # Scale/grid regularization
        auto_max_ce_delta_ratio=0.02,
        auto_scale_l2_penalty=0.05,
        
    
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    
        self.noise_std = noise_std
        self.temperature = temperature
        self.hybrid_loss_alpha = hybrid_loss_alpha
        self.lambda_consistency = lambda_consistency
    
        self.rdrop_alpha = rdrop_alpha
        self.rdrop_kl_chunk_size = rdrop_kl_chunk_size
        self.rdrop_dropout = rdrop_dropout
    
        self.loss_mode = loss_mode
        self.neftune_alpha = neftune_alpha
    
        self.inject_noise = False
        self.selected_layer_indices = None
        self.hooks = []
    
        # Auto-noise state
        self.auto_calib_batches = int(auto_calib_batches)
        self.auto_scale_factors = tuple(float(x) for x in auto_scale_factors)
        self.auto_scale_score_tolerance = float(auto_scale_score_tolerance)
        self.auto_depth_bands = int(auto_depth_bands)
        self.auto_use_depth_diversity = bool(auto_use_depth_diversity)

        self.auto_max_ce_delta_ratio = float(auto_max_ce_delta_ratio)
        self.auto_scale_l2_penalty = float(auto_scale_l2_penalty)
    
        self.layer_noise_scales = {}
        self.layer_selection_report = {}
        self._noise_runtime_stats = {}

        self.use_prefix_sharing = bool(use_prefix_sharing)
        self.prefix_share_min_saved_layers = int(prefix_share_min_saved_layers)
        self.allow_prefix_sharing_ddp = bool(allow_prefix_sharing_ddp)
        self.prefix_share_min_layer_frac = float(prefix_share_min_layer_frac)
        self.prefix_share_exclude_early_layers = bool(prefix_share_exclude_early_layers)
        self._prefix_share_fallback_warned = False
        self._prefix_share_enabled_logged = False
        
        self._prefix_shared_first_layer = -1
        self._prefix_share_active_last = False
        self._prefix_share_approx_forward_equiv_last = -1.0
        self._prefix_share_theoretical_speedup_last = -1.0

        self.auto_use_target_kl_scale = bool(auto_use_target_kl_scale)
        self.auto_target_kl_fraction = float(auto_target_kl_fraction)
    
        # Logging / timing state
        self._last_aux_log_step = -1
        self._train_start_time = None
        self._last_perf_log_time = None
        self._last_perf_log_step = 0
        self._rdrop_debug_logged = False

        self._prefix_share_packed_tail_last = False

    def _accumulate_noise_runtime_stats(self, layer_idx, hidden_states, noise_delta, scale):
        """
        Accumulate runtime noise statistics between logging intervals.
    
        Logs the actual applied noise, not only the calibrated scale.
        """
        with torch.no_grad():
            hidden_std = hidden_states.detach().float().std()
            noise_delta_std = noise_delta.detach().float().std()
    
            ratio = noise_delta_std / hidden_std.clamp_min(1e-12)
    
            if layer_idx not in self._noise_runtime_stats:
                self._noise_runtime_stats[layer_idx] = {
                    "count": 0,
                    "scale_sum": 0.0,
                    "hidden_std_sum": 0.0,
                    "noise_delta_std_sum": 0.0,
                    "noise_to_hidden_std_sum": 0.0,
                }
    
            stats = self._noise_runtime_stats[layer_idx]
            stats["count"] += 1
            stats["scale_sum"] += float(scale)
            stats["hidden_std_sum"] += float(hidden_std.detach().cpu())
            stats["noise_delta_std_sum"] += float(noise_delta_std.detach().cpu())
            stats["noise_to_hidden_std_sum"] += float(ratio.detach().cpu())

    def _depth_aware_select_layers(self, layer_scores, candidate_indices, max_layers):
        """
        Select high-scoring layers while avoiding unnecessary late-layer collapse.
    
        This does not force weak early/middle layers into the policy. It only adds
        depth diversity among layers that already passed the robust threshold.
        """
        device = layer_scores.device
        num_layers = int(layer_scores.numel())
    
        if candidate_indices.numel() == 0:
            return torch.topk(layer_scores, k=1, largest=True).indices
    
        if (not self.auto_use_depth_diversity) or max_layers <= 1:
            candidate_scores = layer_scores[candidate_indices]
            k = min(max_layers, candidate_indices.numel())
            top_pos = torch.topk(candidate_scores, k=k, largest=True).indices
            return candidate_indices[top_pos]
    
        bands = max(1, min(int(self.auto_depth_bands), num_layers))
        selected = []
    
        # For 32 layers and 3 bands:
        # early: 0-10
        # middle: 11-21
        # late: 22-31
        per_band_quota = max(1, int(math.ceil(max_layers / bands)))
    
        for band_idx in range(bands):
            start = int(math.floor(band_idx * num_layers / bands))
            end = int(math.floor((band_idx + 1) * num_layers / bands))
    
            band_mask = (candidate_indices >= start) & (candidate_indices < end)
            band_candidates = candidate_indices[band_mask]
    
            if band_candidates.numel() == 0:
                continue
    
            band_scores = layer_scores[band_candidates]
            k_band = min(per_band_quota, band_candidates.numel())
            top_band_pos = torch.topk(band_scores, k=k_band, largest=True).indices
    
            selected.extend(
                [int(x) for x in band_candidates[top_band_pos].detach().cpu().tolist()]
            )
    
        selected_set = set(selected)
    
        # Fill remaining slots globally from the strongest remaining candidates.
        if len(selected) < max_layers:
            global_scores = layer_scores[candidate_indices]
            global_order = torch.argsort(global_scores, descending=True)
    
            for pos in global_order.detach().cpu().tolist():
                idx = int(candidate_indices[pos].detach().cpu())
    
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
    
                if len(selected) >= max_layers:
                    break
    
        selected = selected[:max_layers]
        selected = sorted(selected)
    
        return torch.tensor(selected, device=device, dtype=torch.long)

    def select_layers(self):
        """
        Fully automatic internal calibration.
    
        What is automated:
        1. Candidate noise scales are centered around 1 / sqrt(hidden_dim).
        2. Each layer gets its own best scale from an internal scale grid.
        3. Layers are selected by useful perturbation response:
           high KL(clean || noisy), penalized by CE degradation.
        4. Number of selected layers is automatic:
           layers above median + MAD, capped by ceil(sqrt(num_layers)).
        5. Optional depth-aware selection reduces late-layer collapse.
    
        No downstream validation benchmark is used.
        No manual high-SNR/low-SNR choice is used.
        No manually fixed k is used.
        No global noise_std=0.1 is used for auto hidden-state noise.
        """
        model = self.model
        model.eval()
    
        device = next(model.parameters()).device
        layers = list(get_transformer_layers(model))
        num_layers = len(layers)
        dataloader = self.get_train_dataloader()
    
        hidden_dim = None
        base_scale = None
        candidate_scales = None
        num_scales = len(self.auto_scale_factors)
    
        kl_sum = torch.zeros((num_layers, num_scales), device=device)
        ce_delta_sum = torch.zeros((num_layers, num_scales), device=device)
        grid_count = torch.zeros((num_layers, num_scales), device=device)
    
        snr_sum = torch.zeros(num_layers, device=device)
        snr_count = torch.zeros(num_layers, device=device)
    
        clean_ce_sum = torch.zeros((), device=device)
        clean_count = torch.zeros((), device=device)
    
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.auto_calib_batches:
                break
    
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            shift_mask = labels[:, 1:].ne(-100)
    
            with torch.no_grad():
                clean_outputs = model(
                    **batch,
                    output_hidden_states=True,
                )
    
                clean_loss = clean_outputs.loss.detach()
                clean_logits = clean_outputs.logits.detach()
                clean_hidden = clean_outputs.hidden_states[1:num_layers + 1]
    
                if hidden_dim is None:
                    hidden_dim = int(clean_hidden[0].shape[-1])
                    base_scale = 1.0 / math.sqrt(float(hidden_dim))
                    candidate_scales = torch.tensor(
                        [base_scale * f for f in self.auto_scale_factors],
                        device=device,
                        dtype=torch.float32,
                    )
    
                clean_ce_sum += clean_loss
                clean_count += 1.0
    
                clean_shift_logits = clean_logits[:, :-1, :]
    
                for layer_idx, layer in enumerate(layers):
                    h = clean_hidden[layer_idx].detach()
    
                    # SNR is diagnostic only. It is not the selection rule.
                    signal = h.abs().mean()
                    expected_noise_amp = base_scale * h.std().clamp_min(1e-8)
                    snr = signal / expected_noise_amp
                    snr_sum[layer_idx] += snr
                    snr_count[layer_idx] += 1.0
    
                    for scale_idx, scale in enumerate(candidate_scales):
                        hook = layer.register_forward_hook(
                            self._make_probe_noise_hook(float(scale.detach().cpu()))
                        )
    
                        try:
                            noisy_outputs = model(**batch)
                        finally:
                            hook.remove()
    
                        noisy_loss = noisy_outputs.loss.detach()
                        noisy_logits = noisy_outputs.logits.detach()
                        noisy_shift_logits = noisy_logits[:, :-1, :]
    
                        kl = self.symmetric_kl_from_logits(
                            logits_a=clean_shift_logits,
                            logits_b=noisy_shift_logits,
                            mask=shift_mask,
                        ).detach()
    
                        ce_delta = (noisy_loss - clean_loss).clamp_min(0.0)
    
                        kl_sum[layer_idx, scale_idx] += kl
                        ce_delta_sum[layer_idx, scale_idx] += ce_delta
                        grid_count[layer_idx, scale_idx] += 1.0
    
                        del noisy_outputs, noisy_logits, noisy_shift_logits
    
                del clean_outputs, clean_logits, clean_hidden, clean_shift_logits
    
        if hidden_dim is None:
            raise RuntimeError(
                "Auto-noise calibration failed: no calibration batches were available."
            )
    
        # Aggregate raw sums across DDP ranks before averaging.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for tensor in [
                kl_sum,
                ce_delta_sum,
                grid_count,
                snr_sum,
                snr_count,
                clean_ce_sum,
                clean_count,
            ]:
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    
        grid_count = grid_count.clamp_min(1.0)
        snr_count = snr_count.clamp_min(1.0)
        clean_count = clean_count.clamp_min(1.0)
    
        kl_mean_grid = kl_sum / grid_count
        ce_delta_mean_grid = ce_delta_sum / grid_count
    
        snr_mean = snr_sum / snr_count
        clean_ce_mean = clean_ce_sum / clean_count
    
        # ------------------------------------------------------------
        # Grid score:
        # high KL response is useful
        # large CE degradation is penalized
        # ------------------------------------------------------------
        ce_ref = clean_ce_mean.abs().clamp_min(1e-8)
    
        # ------------------------------------------------------------
        # Grid score:
        #   high KL response is useful
        #   large CE degradation is penalized
        #   very large noise scales are softly discouraged
        # ------------------------------------------------------------
        ce_delta_ratio_grid = ce_delta_mean_grid / ce_ref
        
        raw_score_grid = kl_mean_grid / (
            1.0 + ce_delta_ratio_grid
        )
        
        scale_factor_tensor = torch.tensor(
            list(self.auto_scale_factors),
            device=device,
            dtype=torch.float32,
        )
        
        # Soft scale prior. This prevents the grid from always selecting
        # the largest scale when KL grows monotonically with noise magnitude.
        scale_penalty = 1.0 + float(self.auto_scale_l2_penalty) * (
            torch.log2(scale_factor_tensor).abs() ** 2
        )
        
        score_grid = raw_score_grid / scale_penalty.unsqueeze(0)
        
        # Hard internal safety filter: reject scales that damage clean CE too much.
        # This is not benchmark tuning. It is an internal stability constraint.
        valid_scale_mask = ce_delta_ratio_grid <= float(self.auto_max_ce_delta_ratio)
        
        score_grid = torch.where(
            valid_scale_mask,
            score_grid,
            torch.full_like(score_grid, -float("inf")),
        )
        
        # If a layer has no valid scale, fall back to the raw penalized score
        # rather than crashing.
        invalid_layers = torch.isinf(score_grid).all(dim=1)
        
        if bool(invalid_layers.any()):
            score_grid[invalid_layers] = raw_score_grid[invalid_layers] / scale_penalty.unsqueeze(0)
    
        # Best score per layer across candidate scales.
        best_score_per_layer, best_scale_pos_per_layer = torch.max(score_grid, dim=1)
    
        # Choose the smallest scale within tolerance of the best scale.
        chosen_scale_pos = torch.zeros(num_layers, device=device, dtype=torch.long)
        chosen_scale_per_layer = torch.zeros(num_layers, device=device)
        chosen_score_per_layer = torch.zeros(num_layers, device=device)
        chosen_kl_per_layer = torch.zeros(num_layers, device=device)
        chosen_ce_delta_per_layer = torch.zeros(num_layers, device=device)
    
        tol = float(self.auto_scale_score_tolerance)
    
        for layer_idx in range(num_layers):
            layer_scores = score_grid[layer_idx]
            best_score = best_score_per_layer[layer_idx]
    
            if best_score <= 0:
                # Fallback to the central scale, usually factor 1.0.
                central_idx = min(
                    range(num_scales),
                    key=lambda i: abs(self.auto_scale_factors[i] - 1.0),
                )
                pos = int(central_idx)
            else:
                acceptable = torch.nonzero(
                    layer_scores >= tol * best_score,
                    as_tuple=False,
                ).flatten()
    
                if acceptable.numel() == 0:
                    pos = int(best_scale_pos_per_layer[layer_idx].detach().cpu())
                else:
                    # Candidate scales are sorted from small to large.
                    # Pick the smallest scale that is near-best.
                    pos = int(acceptable[0].detach().cpu())
    
            chosen_scale_pos[layer_idx] = pos
            chosen_scale_per_layer[layer_idx] = candidate_scales[pos]
            chosen_score_per_layer[layer_idx] = score_grid[layer_idx, pos]
            chosen_kl_per_layer[layer_idx] = kl_mean_grid[layer_idx, pos]
            chosen_ce_delta_per_layer[layer_idx] = ce_delta_mean_grid[layer_idx, pos]
    
        # ------------------------------------------------------------
        # Layer selection:
        # robust threshold on chosen/best internal score
        # ------------------------------------------------------------
        layer_score = best_score_per_layer
    
        score_median = layer_score.median()
        score_mad = (layer_score - score_median).abs().median().clamp_min(1e-12)
        threshold = score_median + score_mad
    
        candidate_indices = torch.nonzero(
            layer_score >= threshold,
            as_tuple=False,
        ).flatten()
        
        # Avoid selecting layers with exactly zero useful response.
        candidate_indices = candidate_indices[layer_score[candidate_indices] > 0]
        
        # ------------------------------------------------------------
        # Prefix-sharing-aware filter
        #
        # If prefix sharing is enabled, selecting layer 0 destroys the speedup.
        # So for the prefix-sharing variant, restrict selected noisy layers to
        # begin after an architecture-derived prefix boundary.
        #
        # For LLaMA-2-7B with 32 layers and prefix_share_min_layer_frac=0.25:
        #   min_prefix_layer = 8
        # ------------------------------------------------------------
        if self.use_prefix_sharing and self.prefix_share_exclude_early_layers:
            min_prefix_layer = max(
                int(self.prefix_share_min_saved_layers),
                int(math.ceil(float(self.prefix_share_min_layer_frac) * float(num_layers))),
            )
        
            prefix_candidate_indices = candidate_indices[candidate_indices >= min_prefix_layer]
        
            if prefix_candidate_indices.numel() > 0:
                candidate_indices = prefix_candidate_indices
            else:
                # Fallback: choose the best available layers after the prefix boundary.
                late_indices = torch.arange(
                    min_prefix_layer,
                    num_layers,
                    device=device,
                    dtype=torch.long,
                )
        
                late_indices = late_indices[layer_score[late_indices] > 0]
        
                if late_indices.numel() > 0:
                    candidate_indices = late_indices
                else:
                    # Last fallback: keep original candidates.
                    # This avoids failure if all useful response is early.
                    pass
        
        max_layers = max(1, int(math.ceil(math.sqrt(num_layers))))
        
        selected_tensor = self._depth_aware_select_layers(
            layer_scores=layer_score,
            candidate_indices=candidate_indices,
            max_layers=max_layers,
        )
    
        selected = sorted([int(x) for x in selected_tensor.detach().cpu().tolist()])
    
        # ------------------------------------------------------------
        # Final per-layer scale assignment
        #
        # Default grid behavior chooses the smallest scale within tolerance
        # of each layer's best score.
        #
        # Target-KL behavior makes the printed calibrated scale more genuinely
        # layer-specific:
        #   - compute a target internal KL from selected layers
        #   - choose the smallest valid scale that reaches that target
        #   - sensitive layers get smaller scales
        #   - insensitive layers get larger scales
        # ------------------------------------------------------------
        scale_selection_target_kl = None
        
        if self.auto_use_target_kl_scale and len(selected) > 0:
            selected_tensor_for_scale = torch.tensor(
                selected,
                device=device,
                dtype=torch.long,
            )
        
            finite_score_grid = torch.where(
                torch.isfinite(score_grid),
                score_grid,
                torch.full_like(score_grid, -float("inf")),
            )
        
            best_valid_pos_per_layer = torch.argmax(finite_score_grid, dim=1)
        
            layer_arange = torch.arange(
                num_layers,
                device=device,
                dtype=torch.long,
            )
        
            best_valid_kl_per_layer = kl_mean_grid[
                layer_arange,
                best_valid_pos_per_layer,
            ]
        
            scale_selection_target_kl = (
                best_valid_kl_per_layer[selected_tensor_for_scale].median()
                * float(self.auto_target_kl_fraction)
            ).clamp_min(1e-12)
        
            for idx in selected:
                idx = int(idx)
        
                valid_pos = torch.nonzero(
                    valid_scale_mask[idx]
                    & torch.isfinite(score_grid[idx])
                    & (kl_mean_grid[idx] >= scale_selection_target_kl),
                    as_tuple=False,
                ).flatten()
        
                if valid_pos.numel() > 0:
                    # Candidate scales are sorted small -> large.
                    # Pick the smallest scale that reaches target KL.
                    pos = int(valid_pos[0].detach().cpu())
        
                else:
                    valid_pos = torch.nonzero(
                        valid_scale_mask[idx]
                        & torch.isfinite(score_grid[idx]),
                        as_tuple=False,
                    ).flatten()
        
                    if valid_pos.numel() > 0:
                        # If no scale reaches target KL, choose the valid scale whose KL
                        # is closest to the target on a log scale.
                        kl_vals = kl_mean_grid[idx, valid_pos].clamp_min(1e-12)
        
                        kl_distance = (
                            torch.log(kl_vals)
                            - torch.log(scale_selection_target_kl)
                        ).abs()
        
                        best_local_pos = torch.argmin(kl_distance)
                        pos = int(valid_pos[best_local_pos].detach().cpu())
        
                    else:
                        # Last fallback.
                        pos = int(best_scale_pos_per_layer[idx].detach().cpu())
        
                chosen_scale_pos[idx] = pos
                chosen_scale_per_layer[idx] = candidate_scales[pos]
                chosen_score_per_layer[idx] = score_grid[idx, pos]
                chosen_kl_per_layer[idx] = kl_mean_grid[idx, pos]
                chosen_ce_delta_per_layer[idx] = ce_delta_mean_grid[idx, pos]
        
        layer_noise_scales = {
            int(idx): float(chosen_scale_per_layer[idx].detach().cpu())
            for idx in selected
        }
    
        self.selected_layer_indices = selected
        self.layer_noise_scales = layer_noise_scales
    
        candidate_scale_values = [
            float(x.detach().cpu()) for x in candidate_scales
        ]
    
        report = {
            "num_layers": int(num_layers),
            "hidden_dim": int(hidden_dim),
            "base_scale": float(base_scale),
            "auto_scale_factors": list(self.auto_scale_factors),
            "candidate_scales": candidate_scale_values,
            "auto_scale_score_tolerance": float(self.auto_scale_score_tolerance),
            "auto_depth_bands": int(self.auto_depth_bands),
            "auto_use_depth_diversity": bool(self.auto_use_depth_diversity),
            "max_layers": int(max_layers),
            "clean_ce_mean": float(clean_ce_mean.detach().cpu()),
            "score_threshold_median_plus_mad": float(threshold.detach().cpu()),
            "score_median": float(score_median.detach().cpu()),
            "score_mad": float(score_mad.detach().cpu()),
            "selected_layers": selected,
            "layer_noise_scales": layer_noise_scales,
            "per_layer": [
                {
                    "layer": int(i),
                    "snr_mean": float(snr_mean[i].detach().cpu()),
                    "best_score": float(best_score_per_layer[i].detach().cpu()),
                    "best_scale_pos": int(best_scale_pos_per_layer[i].detach().cpu()),
                    "best_scale": float(
                        candidate_scales[best_scale_pos_per_layer[i]].detach().cpu()
                    ),
                    "chosen_score": float(chosen_score_per_layer[i].detach().cpu()),
                    "chosen_scale_pos": int(chosen_scale_pos[i].detach().cpu()),
                    "chosen_scale": float(chosen_scale_per_layer[i].detach().cpu()),
                    "chosen_kl_mean": float(chosen_kl_per_layer[i].detach().cpu()),
                    "chosen_ce_delta_mean": float(
                        chosen_ce_delta_per_layer[i].detach().cpu()
                    ),
                    "selected": int(i) in selected,
                    "scale_grid": [
                        {
                            "scale_factor": float(self.auto_scale_factors[j]),
                            "scale": float(candidate_scales[j].detach().cpu()),
                            "kl_mean": float(kl_mean_grid[i, j].detach().cpu()),
                            "ce_delta_mean": float(
                                ce_delta_mean_grid[i, j].detach().cpu()
                            ),
                            "score": float(score_grid[i, j].detach().cpu()),
                        }
                        for j in range(num_scales)
                    ],
                }
                for i in range(num_layers)
            ],
        }
    
        self.layer_selection_report = report
    
        if self.is_world_process_zero():
            print("Automatic scale-grid noise calibration complete.", flush=True)
            print(f"Base scale: {base_scale:.8f}", flush=True)
            print(f"Candidate scales: {candidate_scale_values}", flush=True)
            print(f"Selected layers: {selected}", flush=True)
            print(f"Layer noise scales: {layer_noise_scales}", flush=True)
            print(f"Clean CE mean: {report['clean_ce_mean']:.6f}", flush=True)
            print(
                f"Score threshold: {report['score_threshold_median_plus_mad']:.8f}",
                flush=True,
            )
    
            os.makedirs(self.args.output_dir, exist_ok=True)
            report_path = os.path.join(self.args.output_dir, "auto_noise_policy.json")
    
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
    
            print(f"Saved auto noise policy to: {report_path}", flush=True)
    
        model.train()

    def _make_train_noise_hook(self, layer_idx):
        def hook(module, input, output):
            if not self.inject_noise:
                return output
    
            hidden_states = output[0]
    
            scale = self.layer_noise_scales.get(layer_idx, None)
    
            if scale is None:
                hidden_dim = hidden_states.shape[-1]
                scale = 1.0 / math.sqrt(float(hidden_dim))
    
            noisy_hidden_states = adaptive_noise_injection(
                hidden_states,
                base_std=float(scale),
                logits=None,
            )
    
            noise_delta = (noisy_hidden_states - hidden_states).detach()
    
            self._accumulate_noise_runtime_stats(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                noise_delta=noise_delta,
                scale=float(scale),
            )
    
            noisy_hidden_states = hidden_states + noise_delta
    
            return (noisy_hidden_states,) + output[1:]
    
        return hook

    def _repeat_for_three_branches(self, x, batch_size):
        """
        Repeat batch-shaped tensors as:
            [clean batch, noisy1 batch, noisy2 batch]
    
        Leaves non-batch tensors unchanged, such as cache_position [L].
        Handles tensors, tuples, and lists.
        """
        if x is None:
            return None
    
        if torch.is_tensor(x):
            if x.dim() > 0 and x.size(0) == batch_size:
                return torch.cat([x, x, x], dim=0)
            return x
    
        if isinstance(x, tuple):
            return tuple(self._repeat_for_three_branches(y, batch_size) for y in x)
    
        if isinstance(x, list):
            return [self._repeat_for_three_branches(y, batch_size) for y in x]
    
        return x
    
    
    def _apply_noise_to_packed_noisy_branches(self, layer_idx, hidden_states, batch_size):
        """
        hidden_states layout:
            [0:B]     clean branch
            [B:2B]    noisy branch 1
            [2B:3B]   noisy branch 2
    
        Only noisy branches receive NoiseFiT perturbation.
        Clean branch remains untouched.
        """
        if layer_idx not in self.layer_noise_scales:
            return hidden_states
    
        clean_hidden = hidden_states[:batch_size]
        noisy_hidden1 = hidden_states[batch_size:2 * batch_size]
        noisy_hidden2 = hidden_states[2 * batch_size:3 * batch_size]
    
        scale = self.layer_noise_scales.get(layer_idx, None)
    
        if scale is None:
            hidden_dim = hidden_states.shape[-1]
            scale = 1.0 / math.sqrt(float(hidden_dim))
    
        scale = float(scale)
    
        noised_hidden1 = adaptive_noise_injection(
            noisy_hidden1,
            base_std=scale,
            logits=None,
        )
    
        noise_delta1 = (noised_hidden1 - noisy_hidden1).detach()
        noisy_hidden1 = noisy_hidden1 + noise_delta1
    
        self._accumulate_noise_runtime_stats(
            layer_idx=layer_idx,
            hidden_states=noisy_hidden1,
            noise_delta=noise_delta1,
            scale=scale,
        )
    
        noised_hidden2 = adaptive_noise_injection(
            noisy_hidden2,
            base_std=scale,
            logits=None,
        )
    
        noise_delta2 = (noised_hidden2 - noisy_hidden2).detach()
        noisy_hidden2 = noisy_hidden2 + noise_delta2
    
        self._accumulate_noise_runtime_stats(
            layer_idx=layer_idx,
            hidden_states=noisy_hidden2,
            noise_delta=noise_delta2,
            scale=scale,
        )
    
        return torch.cat(
            [clean_hidden, noisy_hidden1, noisy_hidden2],
            dim=0,
        )
    
    
    def _run_prefix_share_layers_packed_tail(
        self,
        layers,
        hidden_states,
        start_idx,
        end_idx,
        causal_mask,
        position_ids,
        cache_position,
        position_embeddings,
        batch_size,
    ):
        """
        Runs the tail once on a packed batch:
            clean + noisy1 + noisy2
    
        Noise is injected manually after selected layers, only into the noisy
        branches. Existing forward hooks are bypassed because self.inject_noise
        remains False during this packed pass.
        """
        selected_set = set(int(x) for x in (self.selected_layer_indices or []))
    
        for layer_idx in range(start_idx, end_idx):
            hidden_states = self._call_prefix_share_decoder_layer(
                layer=layers[layer_idx],
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
    
            if layer_idx in selected_set:
                hidden_states = self._apply_noise_to_packed_noisy_branches(
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                    batch_size=batch_size,
                )
    
        return hidden_states

    def _add_noise_runtime_stats_to_log_dict(self, log_dict):
        """
        Add runtime noise statistics to W&B log_dict.
    
        This is called from rank 0 inside log_loss_components.
        """
        if not self._noise_runtime_stats:
            return log_dict
    
        selected_noise_stds = []
        selected_ratios = []
    
        for layer_idx, stats in sorted(self._noise_runtime_stats.items()):
            count = max(1, int(stats["count"]))
    
            mean_scale = stats["scale_sum"] / count
            mean_hidden_std = stats["hidden_std_sum"] / count
            mean_noise_delta_std = stats["noise_delta_std_sum"] / count
            mean_ratio = stats["noise_to_hidden_std_sum"] / count
    
            log_dict[f"auto_noise/layer_{layer_idx}_calibrated_scale"] = mean_scale
            log_dict[f"auto_noise/layer_{layer_idx}_hidden_std"] = mean_hidden_std
            log_dict[f"auto_noise/layer_{layer_idx}_noise_delta_std"] = mean_noise_delta_std
            log_dict[f"auto_noise/layer_{layer_idx}_noise_to_hidden_std"] = mean_ratio
    
            selected_noise_stds.append(mean_noise_delta_std)
            selected_ratios.append(mean_ratio)
    
        if selected_noise_stds:
            log_dict["auto_noise/mean_noise_delta_std_selected_layers"] = (
                sum(selected_noise_stds) / len(selected_noise_stds)
            )
            log_dict["auto_noise/max_noise_delta_std_selected_layers"] = max(selected_noise_stds)
            log_dict["auto_noise/min_noise_delta_std_selected_layers"] = min(selected_noise_stds)
    
        if selected_ratios:
            log_dict["auto_noise/mean_noise_to_hidden_std_selected_layers"] = (
                sum(selected_ratios) / len(selected_ratios)
            )
            log_dict["auto_noise/max_noise_to_hidden_std_selected_layers"] = max(selected_ratios)
            log_dict["auto_noise/min_noise_to_hidden_std_selected_layers"] = min(selected_ratios)
    
        return log_dict
    
    
    def register_noise_hooks(self):
        self.hooks = []
    
        layers = list(get_transformer_layers(self.model))
    
        for idx in self.selected_layer_indices:
            layer = layers[idx]
            hook = layer.register_forward_hook(self._make_train_noise_hook(idx))
            self.hooks.append(hook)
    
        if self.is_world_process_zero():
            print(
                f"Registered calibrated noise hooks on layers: {self.selected_layer_indices}",
                flush=True,
            )


    def _make_probe_noise_hook(self, probe_scale):
        def hook(module, input, output):
            hidden_states = output[0]
    
            noisy_hidden_states = adaptive_noise_injection(
                hidden_states,
                base_std=float(probe_scale),
                logits=None,
            )
    
            noise_delta = (noisy_hidden_states - hidden_states).detach()
            noisy_hidden_states = hidden_states + noise_delta
    
            return (noisy_hidden_states,) + output[1:]
    
        return hook

    def _neftune_embedding_hook(self, module, inputs, output):
        """
        NEFTune embedding noise.

        During training only:
            embedding = embedding + U(-1, 1) * alpha / sqrt(seq_len * hidden_dim)
        """
        if self.loss_mode != "neftune":
            return output

        if not module.training:
            return output

        if self.neftune_alpha is None or self.neftune_alpha <= 0:
            return output

        if not torch.is_tensor(output):
            return output

        if output.dim() != 3:
            return output

        _, seq_len, hidden_dim = output.shape
        scale = float(self.neftune_alpha) / math.sqrt(seq_len * hidden_dim)

        noise = torch.empty_like(output).uniform_(-1.0, 1.0) * scale
        return output + noise

    def register_neftune_hook(self):
        embedding_layer = self.model.get_input_embeddings()
        hook = embedding_layer.register_forward_hook(self._neftune_embedding_hook)
        self.hooks.append(hook)

        if self.is_world_process_zero():
            print(
                f"Registered NEFTune embedding hook with alpha={self.neftune_alpha}",
                flush=True,
            )

    def needs_noise_hooks(self):
        return self.loss_mode in {
            "ce_noisy",
            "ce_kl",
            "consistency_only",
            "kl_only",
            "full",
        }

    def log_loss_components(
        self,
        ce_loss,
        soft_ce_loss,
        consistency_loss,
        total_loss,
        noisy_ce_loss=None,
        num_forwards=None,
    ):
        step = int(self.state.global_step)
        logging_steps = max(1, int(self.args.logging_steps))

        if step <= 0:
            return
        if step == self._last_aux_log_step:
            return
        if step % logging_steps != 0:
            return

        self._last_aux_log_step = step

        # CUDA kernels are asynchronous. Synchronize before timing,
        # otherwise wall-time speed can look unchanged or noisy.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        now = time.time()

        if self._train_start_time is None:
            self._train_start_time = now
        if self._last_perf_log_time is None:
            self._last_perf_log_time = now
            self._last_perf_log_step = step

        total_wall_time_sec = now - self._train_start_time
        interval_time_sec = now - self._last_perf_log_time
        interval_steps = max(1, step - self._last_perf_log_step)

        sec_per_optimizer_step = interval_time_sec / interval_steps

        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()

        effective_batch_size = (
            int(self.args.per_device_train_batch_size)
            * int(self.args.gradient_accumulation_steps)
            * int(world_size)
        )

        examples_per_sec = effective_batch_size / sec_per_optimizer_step

        if torch.cuda.is_available():
            device = torch.cuda.current_device()

            mem_stats = torch.tensor(
                [
                    float(torch.cuda.memory_allocated(device) / 1024**3),
                    float(torch.cuda.memory_reserved(device) / 1024**3),
                    float(torch.cuda.max_memory_allocated(device) / 1024**3),
                    float(torch.cuda.max_memory_reserved(device) / 1024**3),
                ],
                device=device,
                dtype=torch.float32,
            )
        else:
            mem_stats = torch.zeros(4, dtype=torch.float32)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered_mem = [torch.zeros_like(mem_stats) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_mem, mem_stats)
        else:
            gathered_mem = [mem_stats]

        if self.is_world_process_zero():
            ce = float(ce_loss.detach().float().cpu())
            soft = float(soft_ce_loss.detach().float().cpu())
            cons = float(consistency_loss.detach().float().cpu())
            total = float(total_loss.detach().float().cpu())

            log_dict = {
                "noisefit/ce_loss": ce,
                "noisefit/soft_ce_loss": soft,
                "noisefit/consistency_loss": cons,
                "noisefit/total_loss": total,
                "noisefit/noise_std": float(self.noise_std),
                "noisefit/lambda_consistency": float(self.lambda_consistency),
                "noisefit/hybrid_loss_alpha": float(self.hybrid_loss_alpha),
                "neftune/alpha": float(self.neftune_alpha),
                "noisefit/auto_calib_batches": float(self.auto_calib_batches),
                "noisefit/num_selected_layers": float(len(self.selected_layer_indices or [])),

                "prefix_share/requested": float(self.use_prefix_sharing),
                "prefix_share/active": float(self._prefix_share_active_last),
                "prefix_share/packed_tail": float(self._prefix_share_packed_tail_last),
                "prefix_share/first_noisy_layer": float(self._prefix_shared_first_layer),
                "prefix_share/approx_forward_equiv": float(self._prefix_share_approx_forward_equiv_last),
                "prefix_share/theoretical_speedup": float(self._prefix_share_theoretical_speedup_last),

                "speed/forward_passes_per_microbatch": float(num_forwards) if num_forwards is not None else -1.0,
                "speed/wall_time_sec": float(total_wall_time_sec),
                "speed/interval_time_sec": float(interval_time_sec),
                "speed/sec_per_optimizer_step": float(sec_per_optimizer_step),
                "speed/effective_batch_size": float(effective_batch_size),
                "speed/examples_per_sec": float(examples_per_sec),
            }

            if noisy_ce_loss is not None:
                log_dict["noisefit/noisy_ce_loss"] = float(noisy_ce_loss.detach().float().cpu())

            loss_mode_id = {
                "basefit": 0,
                "neftune": 1,
                "rdrop": 2,
                "ce_noisy": 3,
                "ce_kl": 4,
                "consistency_only": 5,
                "kl_only": 6,
                "full": 7,
            }.get(self.loss_mode, -1)

            log_dict["noisefit/loss_mode_id"] = loss_mode_id

            allocated_values = []
            reserved_values = []
            max_allocated_values = []
            max_reserved_values = []

            for rank_idx, rank_mem in enumerate(gathered_mem):
                rank_mem_cpu = rank_mem.detach().float().cpu().tolist()

                allocated_gb = rank_mem_cpu[0]
                reserved_gb = rank_mem_cpu[1]
                max_allocated_gb = rank_mem_cpu[2]
                max_reserved_gb = rank_mem_cpu[3]

                allocated_values.append(allocated_gb)
                reserved_values.append(reserved_gb)
                max_allocated_values.append(max_allocated_gb)
                max_reserved_values.append(max_reserved_gb)

                log_dict[f"memory/rank_{rank_idx}_allocated_gb"] = allocated_gb
                log_dict[f"memory/rank_{rank_idx}_reserved_gb"] = reserved_gb
                log_dict[f"memory/rank_{rank_idx}_max_allocated_gb"] = max_allocated_gb
                log_dict[f"memory/rank_{rank_idx}_max_reserved_gb"] = max_reserved_gb

            log_dict["memory/max_allocated_gb_all_ranks"] = max(allocated_values)
            log_dict["memory/max_reserved_gb_all_ranks"] = max(reserved_values)
            log_dict["memory/peak_allocated_gb_all_ranks"] = max(max_allocated_values)
            log_dict["memory/peak_reserved_gb_all_ranks"] = max(max_reserved_values)
            log_dict["memory/mean_allocated_gb_all_ranks"] = sum(allocated_values) / len(allocated_values)
            log_dict["memory/mean_reserved_gb_all_ranks"] = sum(reserved_values) / len(reserved_values)

            # Add automated NoiseFiT runtime STD logs
            log_dict = self._add_noise_runtime_stats_to_log_dict(log_dict)

            try:
                import wandb
            
                if wandb.run is not None:
                    wandb.log(log_dict, step=step)
                else:
                    # Fallback if W&B is not initialized
                    self.log(log_dict)
            
            except Exception as e:
                print(f"[rank0] W&B direct logging failed: {e}", flush=True)
                self.log(log_dict)

        self._last_perf_log_time = now
        self._last_perf_log_step = step
        self._noise_runtime_stats = {}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())


    def symmetric_kl_from_logits(self, logits_a, logits_b, mask):
        """
        R-Drop symmetric KL over causal-LM target positions only.
    
        logits_a/logits_b: [B, L-1, V]
        mask:              [B, L-1], True for valid target positions
        """
        denom = mask.sum().clamp_min(1)
        total = None
        chunk_size = max(1, int(self.rdrop_kl_chunk_size))
    
        for start in range(0, logits_a.size(1), chunk_size):
            end = min(start + chunk_size, logits_a.size(1))
            chunk_mask = mask[:, start:end]
    
            if not bool(chunk_mask.any()):
                continue
    
            a = logits_a[:, start:end, :]
            b = logits_b[:, start:end, :]
    
            if self.temperature != 1.0:
                a = a / self.temperature
                b = b / self.temperature
    
            logp = F.log_softmax(a, dim=-1)
            logq = F.log_softmax(b, dim=-1)
            p = logp.exp()
            q = logq.exp()
    
            # F.kl_div(logq, p) = KL(p || q)
            kl_p_to_q = F.kl_div(logq, p, reduction="none").sum(dim=-1)
    
            # F.kl_div(logp, q) = KL(q || p)
            kl_q_to_p = F.kl_div(logp, q, reduction="none").sum(dim=-1)
    
            token_kl = 0.5 * (kl_p_to_q + kl_q_to_p)
            chunk_total = token_kl.masked_select(chunk_mask).sum()
    
            total = chunk_total if total is None else total + chunk_total
    
        if total is None:
            return logits_a.new_zeros(())
    
        loss = total / denom
    
        if self.temperature != 1.0:
            loss = loss * (self.temperature ** 2)
    
        return loss

    def log_rdrop_components(self, ce_loss, kl_loss, total_loss):
        step = int(self.state.global_step)
        logging_steps = max(1, int(self.args.logging_steps))
    
        if step <= 0:
            return
        if step == self._last_aux_log_step:
            return
        if step % logging_steps != 0:
            return
    
        self._last_aux_log_step = step
    
        if not self.is_world_process_zero():
            return
    
        log_dict = {
            "rdrop/ce_loss": float(ce_loss.detach().float().cpu()),
            "rdrop/kl_loss": float(kl_loss.detach().float().cpu()),
            "rdrop/total_loss": float(total_loss.detach().float().cpu()),
            "rdrop/alpha": float(self.rdrop_alpha),
            "speed/forward_passes_per_microbatch": 2.0,
            "rdrop/loss_mode_id": 2,
        }
    
        try:
            import wandb
    
            if wandb.run is not None:
                # Direct W&B logging avoids Hugging Face adding "train/" prefix.
                wandb.log(log_dict, step=step)
            else:
                # Fallback. This will likely appear as train/rdrop/... in W&B.
                self.log(log_dict)
    
        except Exception as e:
            print(f"[rank0] W&B direct logging failed: {e}", flush=True)
            self.log(log_dict)


    def enable_rdrop_dropout(self):
        """
        Enable stochastic dropout for true R-Drop.
    
        LLaMA-2 usually has attention_dropout=0.0, so the two R-Drop
        forwards can otherwise be almost identical except for LoRA dropout.
        This method enables attention dropout for R-Drop only.
        """
        p = float(self.rdrop_dropout)
    
        if p <= 0:
            if self.is_world_process_zero():
                print(
                    "R-Drop dropout is <= 0. R-Drop will rely only on existing stochasticity, "
                    "such as LoRA dropout.",
                    flush=True,
                )
            return
    
        # Update config objects where present.
        possible_models = [self.model]
    
        for attr in ["base_model", "model"]:
            if hasattr(self.model, attr):
                possible_models.append(getattr(self.model, attr))
    
        for m in possible_models:
            if hasattr(m, "config"):
                if hasattr(m.config, "attention_dropout"):
                    m.config.attention_dropout = p
                if hasattr(m.config, "hidden_dropout_prob"):
                    m.config.hidden_dropout_prob = p
                if hasattr(m.config, "dropout"):
                    m.config.dropout = p
    
        # Update already-created modules.
        changed_attention_modules = 0
        changed_dropout_modules = 0
    
        for module in self.model.modules():
            # LLaMA attention modules commonly store this as a float attribute.
            if hasattr(module, "attention_dropout"):
                try:
                    module.attention_dropout = p
                    changed_attention_modules += 1
                except Exception:
                    pass
    
            # If any real nn.Dropout modules exist, update them too.
            if isinstance(module, torch.nn.Dropout):
                module.p = p
                changed_dropout_modules += 1
    
        if self.is_world_process_zero():
            print(
                f"Enabled R-Drop dropout: p={p}. "
                f"Updated attention_dropout on {changed_attention_modules} modules; "
                f"updated nn.Dropout.p on {changed_dropout_modules} modules.",
                flush=True,
            )
            

    def train(self, *args, **kwargs):
        if self.loss_mode == "neftune":
            self.register_neftune_hook()
    
        elif self.loss_mode == "rdrop":
            self.enable_rdrop_dropout()
    
            if self.is_world_process_zero():
                print(
                    f"Running R-Drop with rdrop_alpha={self.rdrop_alpha}, "
                    f"rdrop_dropout={self.rdrop_dropout}, "
                    f"rdrop_kl_chunk_size={self.rdrop_kl_chunk_size}",
                    flush=True,
                )
    
        elif self.needs_noise_hooks():
            self.select_layers()
            self.register_noise_hooks()
    
        else:
            if self.is_world_process_zero():
                print(
                    f"Skipping NoiseFiT layer selection and hooks for loss_mode={self.loss_mode}",
                    flush=True,
                )
    
        # Reset speed timer after calibration, not before.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
        self._train_start_time = time.time()
        self._last_perf_log_time = self._train_start_time
        self._last_perf_log_step = int(self.state.global_step)
    
        return super().train(*args, **kwargs)

    def _prefix_share_fallback(self, reason):
        self._prefix_share_active_last = False
        self._prefix_shared_first_layer = -1
        self._prefix_share_approx_forward_equiv_last = 3.0
        self._prefix_share_theoretical_speedup_last = 1.0
        self._prefix_share_packed_tail_last = False
    
        if self.is_world_process_zero() and not self._prefix_share_fallback_warned:
            print(
                f"[prefix-sharing] Falling back to normal full forward: {reason}",
                flush=True,
            )
            self._prefix_share_fallback_warned = True
    
    
    def _get_prefix_share_lm_parts(self, model):
        """
        Locate LLaMA-style CausalLM parts while preserving PEFT/LoRA modules.
        """
        try:
            if hasattr(self, "accelerator"):
                model = self.accelerator.unwrap_model(model)
        except Exception:
            pass
    
        if hasattr(model, "module"):
            model = model.module
    
        candidates = []
    
        def add_candidate(x):
            if x is not None and all(id(x) != id(y) for y in candidates):
                candidates.append(x)
    
        add_candidate(model)
    
        if hasattr(model, "base_model"):
            add_candidate(model.base_model)
    
        if hasattr(model, "model"):
            add_candidate(model.model)
    
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            add_candidate(model.base_model.model)
    
        if hasattr(model, "model") and hasattr(model.model, "model"):
            add_candidate(model.model.model)
    
        for cand in candidates:
            decoder = getattr(cand, "model", None)
            lm_head = getattr(cand, "lm_head", None)
    
            if (
                decoder is not None
                and lm_head is not None
                and hasattr(decoder, "layers")
                and hasattr(decoder, "embed_tokens")
                and hasattr(decoder, "norm")
            ):
                return cand, decoder, decoder.layers, decoder.embed_tokens, decoder.norm, lm_head
    
        return None
    
    
    def _make_fallback_4d_causal_mask(self, attention_mask, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        min_dtype = torch.finfo(dtype).min
    
        causal_mask = torch.full(
            (seq_len, seq_len),
            fill_value=min_dtype,
            device=device,
            dtype=dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
    
        causal_mask = causal_mask[None, None, :, :].expand(
            batch_size,
            1,
            seq_len,
            seq_len,
        )
    
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].eq(0)
            causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)
    
        return causal_mask
    
    
    def _prepare_prefix_share_position_and_mask(
        self,
        decoder_model,
        input_ids,
        attention_mask,
        inputs_embeds,
        position_ids=None,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
    
        cache_position = torch.arange(
            0,
            seq_len,
            device=device,
            dtype=torch.long,
        )
    
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
    
        causal_mask = None
    
        if hasattr(decoder_model, "_update_causal_mask"):
            try:
                causal_mask = decoder_model._update_causal_mask(
                    attention_mask,
                    inputs_embeds,
                    cache_position,
                    None,
                    False,
                )
            except Exception:
                causal_mask = None
    
        if causal_mask is None:
            causal_mask = self._make_fallback_4d_causal_mask(
                attention_mask=attention_mask,
                hidden_states=inputs_embeds,
            )
    
        position_embeddings = None
    
        if hasattr(decoder_model, "rotary_emb"):
            try:
                position_embeddings = decoder_model.rotary_emb(
                    inputs_embeds,
                    position_ids,
                )
            except Exception:
                position_embeddings = None
    
        return position_ids, cache_position, causal_mask, position_embeddings
    
    
    def _call_prefix_share_decoder_layer(
        self,
        layer,
        hidden_states,
        causal_mask,
        position_ids,
        cache_position,
        position_embeddings,
    ):
        kwargs = {}
    
        try:
            params = inspect.signature(layer.forward).parameters
        except Exception:
            params = {}
    
        if "attention_mask" in params:
            kwargs["attention_mask"] = causal_mask
    
        if "position_ids" in params:
            kwargs["position_ids"] = position_ids
    
        if "past_key_value" in params:
            kwargs["past_key_value"] = None
    
        if "output_attentions" in params:
            kwargs["output_attentions"] = False
    
        if "use_cache" in params:
            kwargs["use_cache"] = False
    
        if "cache_position" in params:
            kwargs["cache_position"] = cache_position
    
        if "position_embeddings" in params and position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
    
        layer_outputs = layer(
            hidden_states,
            **kwargs,
        )
    
        if isinstance(layer_outputs, tuple):
            return layer_outputs[0]
    
        return layer_outputs
    
    
    def _run_prefix_share_layers(
        self,
        layers,
        hidden_states,
        start_idx,
        end_idx,
        causal_mask,
        position_ids,
        cache_position,
        position_embeddings,
    ):
        for layer_idx in range(start_idx, end_idx):
            hidden_states = self._call_prefix_share_decoder_layer(
                layer=layers[layer_idx],
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
    
        return hidden_states
    
    
    def _causal_lm_ce_from_logits(self, logits, labels):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
    
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    
    
    def _forward_full_standard(self, model, inputs, inputs_no_labels):
        self._prefix_share_active_last = False
        self._prefix_shared_first_layer = -1
        self._prefix_share_approx_forward_equiv_last = 3.0
        self._prefix_share_theoretical_speedup_last = 1.0
        self._prefix_share_packed_tail_last = False
    
        self.inject_noise = False
        outputs = model(**inputs)
        ce_loss = outputs.loss
        logits = outputs.logits
    
        self.inject_noise = True
        noisy_outputs1 = model(**inputs_no_labels)
        logits_noise1 = noisy_outputs1.logits
    
        noisy_outputs2 = model(**inputs_no_labels)
        logits_noise2 = noisy_outputs2.logits
    
        self.inject_noise = False
    
        return outputs, ce_loss, logits, logits_noise1, logits_noise2, 3.0
    
    
    def _forward_full_prefix_shared(self, model, inputs):
        """
        Prefix-shared + branch-packed full NoiseFiT forward.
    
        It computes:
    
            shared prefix once
    
        then packs:
    
            clean tail
            noisy tail 1
            noisy tail 2
    
        into one larger batch and runs the tail once.
    
        This is faster than three separate tail passes only if the GPU benefits
        from the larger packed batch. It keeps the same clean/noisy objective.
        """
        if "input_ids" not in inputs:
            self._prefix_share_fallback("inputs do not contain input_ids")
            return None
    
        if "labels" not in inputs:
            self._prefix_share_fallback("inputs do not contain labels")
            return None
    
        if not self.selected_layer_indices:
            self._prefix_share_fallback("no selected noise layers available")
            return None
    
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and not self.allow_prefix_sharing_ddp
        ):
            self._prefix_share_fallback(
                "DDP is active. Set allow_prefix_sharing_ddp=True only after a smoke test."
            )
            return None
    
        parts = self._get_prefix_share_lm_parts(model)
    
        if parts is None:
            self._prefix_share_fallback("could not locate LLaMA-style decoder/lm_head parts")
            return None
    
        causal_lm, decoder_model, layers, embed_tokens, final_norm, lm_head = parts
    
        num_layers = len(layers)
        first_noisy_layer = int(min(self.selected_layer_indices))
    
        if first_noisy_layer <= 0:
            self._prefix_share_fallback("first selected noise layer is 0, no prefix can be shared")
            return None
    
        if first_noisy_layer < self.prefix_share_min_saved_layers:
            self._prefix_share_fallback(
                f"only {first_noisy_layer} prefix layers can be shared, "
                f"less than prefix_share_min_saved_layers={self.prefix_share_min_saved_layers}"
            )
            return None
    
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)
        position_ids = inputs.get("position_ids", None)
    
        batch_size = int(input_ids.shape[0])
    
        inputs_embeds = embed_tokens(input_ids)
    
        (
            position_ids,
            cache_position,
            causal_mask,
            position_embeddings,
        ) = self._prepare_prefix_share_position_and_mask(
            decoder_model=decoder_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
    
        # ------------------------------------------------------------
        # Shared deterministic prefix.
        # Keep self.inject_noise=False so registered hooks do not perturb
        # the prefix.
        # ------------------------------------------------------------
        self.inject_noise = False
    
        shared_hidden = self._run_prefix_share_layers(
            layers=layers,
            hidden_states=inputs_embeds,
            start_idx=0,
            end_idx=first_noisy_layer,
            causal_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    
        # ------------------------------------------------------------
        # Pack clean/noisy1/noisy2 into batch dimension.
        #
        # Layout:
        #   [0:B]     clean
        #   [B:2B]    noisy1
        #   [2B:3B]   noisy2
        # ------------------------------------------------------------
        packed_hidden = torch.cat(
            [shared_hidden, shared_hidden, shared_hidden],
            dim=0,
        )
    
        packed_causal_mask = self._repeat_for_three_branches(
            causal_mask,
            batch_size=batch_size,
        )
    
        packed_position_ids = self._repeat_for_three_branches(
            position_ids,
            batch_size=batch_size,
        )
    
        packed_position_embeddings = self._repeat_for_three_branches(
            position_embeddings,
            batch_size=batch_size,
        )
    
        # ------------------------------------------------------------
        # Tail once, packed.
        # Noise is applied manually only to noisy branches.
        # Existing forward hooks remain inactive because self.inject_noise=False.
        # ------------------------------------------------------------
        self.inject_noise = False
    
        packed_hidden = self._run_prefix_share_layers_packed_tail(
            layers=layers,
            hidden_states=packed_hidden,
            start_idx=first_noisy_layer,
            end_idx=num_layers,
            causal_mask=packed_causal_mask,
            position_ids=packed_position_ids,
            cache_position=cache_position,
            position_embeddings=packed_position_embeddings,
            batch_size=batch_size,
        )
    
        packed_hidden = final_norm(packed_hidden)
        packed_logits = lm_head(packed_hidden)
    
        logits = packed_logits[:batch_size]
        logits_noise1 = packed_logits[batch_size:2 * batch_size]
        logits_noise2 = packed_logits[2 * batch_size:3 * batch_size]
    
        ce_loss = self._causal_lm_ce_from_logits(
            logits=logits,
            labels=labels,
        )
    
        self.inject_noise = False
    
        outputs = CausalLMOutputWithPast(
            loss=ce_loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
        prefix_fraction = float(first_noisy_layer) / float(num_layers)
        tail_fraction = float(num_layers - first_noisy_layer) / float(num_layers)
    
        # FLOP-equivalent is still prefix + 3 tails.
        # But the three tails are now one packed call, which may improve utilization.
        approx_forward_equiv = prefix_fraction + 3.0 * tail_fraction
    
        self._prefix_share_active_last = True
        self._prefix_share_packed_tail_last = True
        self._prefix_shared_first_layer = first_noisy_layer
        self._prefix_share_approx_forward_equiv_last = float(approx_forward_equiv)
        self._prefix_share_theoretical_speedup_last = 3.0 / float(approx_forward_equiv)
    
        if self.is_world_process_zero() and not self._prefix_share_enabled_logged:
            print(
                "[prefix-sharing] Enabled with packed tail. "
                f"first_noisy_layer={first_noisy_layer}, "
                f"num_layers={num_layers}, "
                f"approx_forward_equiv={approx_forward_equiv:.3f} instead of 3.000, "
                f"theoretical_layer_speedup={3.0 / float(approx_forward_equiv):.3f}x",
                flush=True,
            )
            self._prefix_share_enabled_logged = True
    
        return (
            outputs,
            ce_loss,
            logits,
            logits_noise1,
            logits_noise2,
            approx_forward_equiv,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_mode = self.loss_mode

        # ------------------------------------------------------------
        # BaseFiT / NEFTune:
        #
        # BaseFiT:
        #   clean CE only
        #
        # NEFTune:
        #   clean CE only, with embedding-noise hook active during training
        #
        # No NoiseFiT hidden-state hooks.
        # No noisy hidden-state passes.
        # No KL.
        # No consistency.
        # ------------------------------------------------------------
        if loss_mode in {"basefit", "neftune"}:
            self.inject_noise = False

            outputs = model(**inputs)
            ce_loss = outputs.loss

            zero = ce_loss.new_zeros(())

            soft_ce_loss = zero
            consistency_loss = zero
            noisy_ce_loss = zero
            total_loss = ce_loss

            self.log_loss_components(
                ce_loss=ce_loss,
                soft_ce_loss=soft_ce_loss,
                consistency_loss=consistency_loss,
                total_loss=total_loss,
                noisy_ce_loss=noisy_ce_loss,
                num_forwards=1,
            )

            return (total_loss, outputs) if return_outputs else total_loss

        if loss_mode == "rdrop":
            self.inject_noise = False
        
            outputs1 = model(**inputs)
            outputs2 = model(**inputs)
        
            ce_loss = 0.5 * (outputs1.loss + outputs2.loss)
        
            labels = inputs["labels"]
        
            # Causal LM: logits[:, t, :] predict labels[:, t+1]
            shift_mask = labels[:, 1:].ne(-100)
        
            logits1 = outputs1.logits[:, :-1, :]
            logits2 = outputs2.logits[:, :-1, :]

            if self.is_world_process_zero() and not self._rdrop_debug_logged:
                with torch.no_grad():
                    # Small slice to avoid creating a huge temporary tensor.
                    probe_diff = (
                        logits1[:, :4, :2048].detach().float()
                        - logits2[:, :4, :2048].detach().float()
                    ).abs().mean()
            
                print(
                    f"R-Drop stochasticity check, mean |logits1-logits2| slice = "
                    f"{float(probe_diff.cpu()):.8f}",
                    flush=True,
                )
            
                self._rdrop_debug_logged = True
        
            kl_loss = self.symmetric_kl_from_logits(
                logits_a=logits1,
                logits_b=logits2,
                mask=shift_mask,
            )
        
            total_loss = ce_loss + self.rdrop_alpha * kl_loss
        
            self.log_rdrop_components(
                ce_loss=ce_loss,
                kl_loss=kl_loss,
                total_loss=total_loss,
            )
        
            return (total_loss, outputs1) if return_outputs else total_loss

        labels = inputs["labels"]
        target_mask = labels != -100

        inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}

        def masked_mean(token_loss):
            mask = target_mask.to(dtype=token_loss.dtype)
            denom = mask.sum().clamp_min(1.0)
            return (token_loss * mask).sum() / denom

        # ------------------------------------------------------------
        # CE(noisy): one noisy forward only
        # Objective: CE(noisy)
        # Forwards: 1 noisy with labels
        # ------------------------------------------------------------
        if loss_mode == "ce_noisy":
            self.inject_noise = True
            noisy_outputs1 = model(**inputs)
            self.inject_noise = False

            noisy_ce_loss = noisy_outputs1.loss
            zero = noisy_ce_loss.new_zeros(())

            ce_loss = zero
            soft_ce_loss = zero
            consistency_loss = zero
            total_loss = noisy_ce_loss

            self.log_loss_components(
                ce_loss=ce_loss,
                soft_ce_loss=soft_ce_loss,
                consistency_loss=consistency_loss,
                total_loss=total_loss,
                noisy_ce_loss=noisy_ce_loss,
                num_forwards=1,
            )

            return (total_loss, noisy_outputs1) if return_outputs else total_loss

        # ------------------------------------------------------------
        # CE + KL: one clean forward + one noisy forward
        # Objective: alpha * CE(clean) + (1 - alpha) * KL(clean -> noisy)
        # No consistency, no second noisy pass
        # ------------------------------------------------------------
        if loss_mode == "ce_kl":
            self.inject_noise = False
            outputs = model(**inputs)
            ce_loss = outputs.loss
            logits = outputs.logits
        
            self.inject_noise = True
            noisy_outputs1 = model(**inputs_no_labels)
            logits_noise1 = noisy_outputs1.logits
        
            self.inject_noise = False
        
            soft_targets = F.softmax(
                (logits / self.temperature).detach(),
                dim=-1,
            )
        
            soft_ce_token_loss = F.kl_div(
                F.log_softmax(logits_noise1 / self.temperature, dim=-1),
                soft_targets,
                reduction="none",
            ).sum(dim=-1)
        
            soft_ce_loss = masked_mean(soft_ce_token_loss)
        
            if self.temperature != 1.0:
                soft_ce_loss = soft_ce_loss * (self.temperature ** 2)
        
            consistency_loss = ce_loss.new_zeros(())
            noisy_ce_loss = ce_loss.new_zeros(())
        
            total_loss = (
                self.hybrid_loss_alpha * ce_loss
                + (1.0 - self.hybrid_loss_alpha) * soft_ce_loss
            )
        
            self.log_loss_components(
                ce_loss=ce_loss,
                soft_ce_loss=soft_ce_loss,
                consistency_loss=consistency_loss,
                total_loss=total_loss,
                noisy_ce_loss=noisy_ce_loss,
                num_forwards=2,
            )
        
            return (total_loss, outputs) if return_outputs else total_loss

        # ------------------------------------------------------------
        # KL only: one clean forward + one noisy forward
        # Objective: KL(clean -> noisy)
        # No CE, no consistency, no second noisy pass
        # ------------------------------------------------------------
        if loss_mode == "kl_only":
            self.inject_noise = False
            outputs = model(**inputs_no_labels)
            logits = outputs.logits

            self.inject_noise = True
            noisy_outputs1 = model(**inputs_no_labels)
            logits_noise1 = noisy_outputs1.logits
            self.inject_noise = False

            zero = logits.new_zeros(())

            soft_targets = F.softmax(logits / self.temperature, dim=-1)

            soft_ce_token_loss = F.kl_div(
                F.log_softmax(logits_noise1, dim=-1),
                soft_targets,
                reduction="none",
            ).sum(dim=-1)

            soft_ce_loss = masked_mean(soft_ce_token_loss)

            ce_loss = zero
            consistency_loss = zero
            noisy_ce_loss = zero
            total_loss = soft_ce_loss

            self.log_loss_components(
                ce_loss=ce_loss,
                soft_ce_loss=soft_ce_loss,
                consistency_loss=consistency_loss,
                total_loss=total_loss,
                noisy_ce_loss=noisy_ce_loss,
                num_forwards=2,
            )

            return (total_loss, outputs) if return_outputs else total_loss

        # ------------------------------------------------------------
        # Consistency only: two noisy forwards
        # Objective: KL(noisy_2 -> noisy_1)
        # No clean forward, no CE, no clean-teacher KL
        # ------------------------------------------------------------
        if loss_mode == "consistency_only":
            self.inject_noise = True

            noisy_outputs1 = model(**inputs_no_labels)
            logits_noise1 = noisy_outputs1.logits

            noisy_outputs2 = model(**inputs_no_labels)
            logits_noise2 = noisy_outputs2.logits

            self.inject_noise = False

            consistency_token_loss = F.kl_div(
                F.log_softmax(logits_noise1, dim=-1),
                F.softmax(logits_noise2, dim=-1),
                reduction="none",
            ).sum(dim=-1)

            consistency_loss = masked_mean(consistency_token_loss)

            zero = consistency_loss.new_zeros(())
            ce_loss = zero
            soft_ce_loss = zero
            noisy_ce_loss = zero
            total_loss = consistency_loss

            self.log_loss_components(
                ce_loss=ce_loss,
                soft_ce_loss=soft_ce_loss,
                consistency_loss=consistency_loss,
                total_loss=total_loss,
                noisy_ce_loss=noisy_ce_loss,
                num_forwards=2,
            )

            return (total_loss, noisy_outputs1) if return_outputs else total_loss

        # ------------------------------------------------------------
        # Full NoiseFiT: one clean forward + two noisy forwards
        # Objective:
        #   alpha * CE(clean)
        #   + (1 - alpha) * averaged KL(clean -> noisy)
        #   + lambda * consistency
        # ------------------------------------------------------------
        if loss_mode == "full":
            prefix_result = None
        
            if self.use_prefix_sharing:
                prefix_result = self._forward_full_prefix_shared(
                    model=model,
                    inputs=inputs,
                )
        
            if prefix_result is None:
                (
                    outputs,
                    ce_loss,
                    logits,
                    logits_noise1,
                    logits_noise2,
                    approx_forward_equiv,
                ) = self._forward_full_standard(
                    model=model,
                    inputs=inputs,
                    inputs_no_labels=inputs_no_labels,
                )
            else:
                (
                    outputs,
                    ce_loss,
                    logits,
                    logits_noise1,
                    logits_noise2,
                    approx_forward_equiv,
                ) = prefix_result
        
            zero = ce_loss.new_zeros(())
            noisy_ce_loss = zero
        
            # Clean CE-anchored teacher.
            soft_targets = F.softmax(
                (logits / self.temperature).detach(),
                dim=-1,
            )
        
            soft_ce_loss1 = F.kl_div(
                F.log_softmax(logits_noise1 / self.temperature, dim=-1),
                soft_targets,
                reduction="none",
            ).sum(dim=-1)
        
            soft_ce_loss2 = F.kl_div(
                F.log_softmax(logits_noise2 / self.temperature, dim=-1),
                soft_targets,
                reduction="none",
            ).sum(dim=-1)
        
            soft_ce_loss = 0.5 * (soft_ce_loss1 + soft_ce_loss2)
            soft_ce_loss = masked_mean(soft_ce_loss)
        
            if self.temperature != 1.0:
                soft_ce_loss = soft_ce_loss * (self.temperature ** 2)
        
            # Noisy-noisy consistency.
            consistency_token_loss = F.kl_div(
                F.log_softmax(logits_noise1 / self.temperature, dim=-1),
                F.softmax((logits_noise2 / self.temperature).detach(), dim=-1),
                reduction="none",
            ).sum(dim=-1)
        
            consistency_loss = masked_mean(consistency_token_loss)
        
            if self.temperature != 1.0:
                consistency_loss = consistency_loss * (self.temperature ** 2)
        
            hybrid_loss = (
                self.hybrid_loss_alpha * ce_loss
                + (1.0 - self.hybrid_loss_alpha) * soft_ce_loss
            )
        
            total_loss = hybrid_loss + self.lambda_consistency * consistency_loss
        
            self.log_loss_components(
                ce_loss=ce_loss,
                soft_ce_loss=soft_ce_loss,
                consistency_loss=consistency_loss,
                total_loss=total_loss,
                noisy_ce_loss=noisy_ce_loss,
                num_forwards=approx_forward_equiv,
            )
        
            return (total_loss, outputs) if return_outputs else total_loss

        raise ValueError(f"Unknown loss_mode: {loss_mode}")


def parse_float_tuple(value: str):
    """Parse comma-separated floats, e.g. '0.25,0.5,1.0,1.5,2.0'."""
    if isinstance(value, (tuple, list)):
        return tuple(float(x) for x in value)
    return tuple(float(x.strip()) for x in value.split(",") if x.strip())


def print_dropout_config(model):
    if int(os.environ.get("RANK", "0")) != 0:
        return
    print("Model dropout-related config:", flush=True)
    for key in [
        "attention_dropout",
        "hidden_dropout",
        "hidden_dropout_prob",
        "dropout",
        "resid_pdrop",
        "embd_pdrop",
        "attn_pdrop",
    ]:
        if hasattr(model.config, key):
            print(f"{key}: {getattr(model.config, key)}", flush=True)



def dispatch_legacy_v1_if_requested():
    """
    Preserve the first-version manual SNR NoiseFiT trainer behind the main
    entry point. When --trainer_version v1 is passed, this function delegates
    execution to NoiseFiT_v1_manual_snr.py after removing the version flag.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--trainer_version", choices=["v1", "v2"], default="v2")
    known, remaining = pre_parser.parse_known_args()

    if known.trainer_version != "v1":
        return False

    legacy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NoiseFiT_v1_manual_snr.py")
    if not os.path.exists(legacy_path):
        raise FileNotFoundError(
            "Requested --trainer_version v1, but NoiseFiT_v1_manual_snr.py was not found."
        )

    sys.argv = [legacy_path] + remaining
    runpy.run_path(legacy_path, run_name="__main__")
    return True

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train a causal language model with NoiseFiT v2, NEFTune, R-Drop, "
            "or loss-ablation objectives."
        )
    )

    parser.add_argument("--trainer_version", type=str, default="v2", choices=["v1", "v2"],
                        help="Use v2 for the updated auto/baseline stack, or v1 for the original manual SNR trainer.")

    # Core I/O
    parser.add_argument("--model", type=str, required=True,
                        help="Base Hugging Face model identifier or local model path.")
    parser.add_argument("--train_data", type=str, required=True,
                        help="CSV file with 'prompt' and 'response' columns.")
    parser.add_argument("--output_model", type=str, default="output_model",
                        help="Output directory for checkpoints and reports.")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Optional Hugging Face token for gated/private models.")

    # Training schedule
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device train batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=float, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum optimizer steps. Use -1 to train for --epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ddp_find_unused_parameters", action=argparse.BooleanOptionalAction,
                        default=False)

    # Reporting
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"],
                        help="Set to 'wandb' to log to Weights & Biases.")
    parser.add_argument("--report_api_key", type=str, default=None,
                        help="Optional W&B API key. If omitted, wandb uses existing login.")
    parser.add_argument("--wandb_project", type=str, default="NoiseFiT-revision")
    parser.add_argument("--run_name", type=str, default=None)

    # PEFT / LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj",
                        help="Comma-separated LoRA target modules, e.g. q_proj,v_proj.")

    # Objective selection
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="full",
        choices=[
            "basefit",
            "neftune",
            "rdrop",
            "ce_noisy",
            "ce_kl",
            "consistency_only",
            "kl_only",
            "full",
        ],
        help=(
            "Training objective. 'full' is full auto NoiseFiT; 'basefit', 'neftune', "
            "and 'rdrop' are baselines; the remaining modes are NoiseFiT loss ablations."
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hybrid_loss_alpha", type=float, default=0.5)
    parser.add_argument("--lambda_consistency", type=float, default=0.05)
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Legacy/base noise std used for logging and fallback compatibility.")

    # NEFTune and R-Drop
    parser.add_argument("--neftune_alpha", type=float, default=5.0)
    parser.add_argument("--rdrop_alpha", type=float, default=1.0)
    parser.add_argument("--rdrop_kl_chunk_size", type=int, default=64)
    parser.add_argument("--rdrop_dropout", type=float, default=0.05)

    # Auto NoiseFiT calibration
    parser.add_argument("--auto_calib_batches", type=int, default=2)
    parser.add_argument("--auto_scale_factors", type=str, default="0.25,0.5,1.0,1.5,2.0")
    parser.add_argument("--auto_scale_score_tolerance", type=float, default=0.90)
    parser.add_argument("--auto_depth_bands", type=int, default=3)
    parser.add_argument("--auto_use_depth_diversity", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--auto_use_target_kl_scale", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--auto_target_kl_fraction", type=float, default=0.75)
    parser.add_argument("--auto_max_ce_delta_ratio", type=float, default=0.02)
    parser.add_argument("--auto_scale_l2_penalty", type=float, default=0.05)

    # Prefix-shared Auto NoiseFiT variant
    parser.add_argument("--use_prefix_sharing", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--prefix_share_min_saved_layers", type=int, default=4)
    parser.add_argument("--prefix_share_min_layer_frac", type=float, default=0.25)
    parser.add_argument("--prefix_share_exclude_early_layers", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--allow_prefix_sharing_ddp", action=argparse.BooleanOptionalAction,
                        default=False)

    # Backward-compatible v1 arguments. They are accepted but no longer used by
    # the auto-calibrated v2 NoiseFiT policy.
    parser.add_argument("--snr_format", type=str, default=None,
                        choices=[None, "Largest", "Lowest"],
                        help=argparse.SUPPRESS)
    parser.add_argument("--num_noise_layers", type=int, default=None,
                        help=argparse.SUPPRESS)

    return parser


def main():
    if dispatch_legacy_v1_if_requested():
        return

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.hf_token is not None:
        from huggingface_hub import login
        login(token=args.hf_token)
        print("Hugging Face login successful.", flush=True)

    if args.report_to == "wandb":
        import wandb
        if args.report_api_key:
            wandb.login(key=args.report_api_key)
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        report_to = "wandb"
    else:
        report_to = []

    train_df = pd.read_csv(args.train_data)
    dataset = prepare_train_datav2(train_df)

    model, tokenizer = get_model_and_tokenizer(args.model)
    print_dropout_config(model)

    run_name = args.run_name or args.output_model
    training_arguments = TrainingArguments(
        output_dir=args.output_model,
        run_name=run_name,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        report_to=report_to,
    )

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = Noise2NoiseTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_arguments,
        tokenizer=tokenizer,
        noise_std=args.noise_std,
        temperature=args.temperature,
        hybrid_loss_alpha=args.hybrid_loss_alpha,
        lambda_consistency=args.lambda_consistency,
        loss_mode=args.loss_mode,
        neftune_alpha=args.neftune_alpha,
        rdrop_alpha=args.rdrop_alpha,
        rdrop_kl_chunk_size=args.rdrop_kl_chunk_size,
        rdrop_dropout=args.rdrop_dropout,
        auto_calib_batches=args.auto_calib_batches,
        auto_scale_factors=parse_float_tuple(args.auto_scale_factors),
        auto_scale_score_tolerance=args.auto_scale_score_tolerance,
        auto_depth_bands=args.auto_depth_bands,
        auto_use_depth_diversity=args.auto_use_depth_diversity,
        use_prefix_sharing=args.use_prefix_sharing,
        prefix_share_min_saved_layers=args.prefix_share_min_saved_layers,
        prefix_share_min_layer_frac=args.prefix_share_min_layer_frac,
        prefix_share_exclude_early_layers=args.prefix_share_exclude_early_layers,
        allow_prefix_sharing_ddp=args.allow_prefix_sharing_ddp,
        auto_use_target_kl_scale=args.auto_use_target_kl_scale,
        auto_target_kl_fraction=args.auto_target_kl_fraction,
        auto_max_ce_delta_ratio=args.auto_max_ce_delta_ratio,
        auto_scale_l2_penalty=args.auto_scale_l2_penalty,
    )

    trainer.train()


if __name__ == "__main__":
    main()
