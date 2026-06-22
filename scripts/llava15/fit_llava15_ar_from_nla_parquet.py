#!/usr/bin/env python3
"""Fit AR-lite probe directly from a LLaVA-NLA `ar_sft.parquet`.

Input schema:
  prompt: critic-style text prompt
  activation_vector: target LLaVA activation

This verifies the NLA-style AR parquet is sufficient for text->activation
training without relying on the original synthetic generation columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


def normalize_to_sqrt_d(v: torch.Tensor) -> torch.Tensor:
    scale = v.shape[-1] ** 0.5
    return v / v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) * scale


def mse_nrm(pred: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    return ((normalize_to_sqrt_d(pred) - normalize_to_sqrt_d(gold)) ** 2).mean(dim=-1)


def cosine(pred: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(pred.float(), gold.float(), dim=-1)


def ridge_predict(x_train: torch.Tensor, y_train: torch.Tensor, x_eval: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    ones_train = torch.ones(x_train.shape[0], 1, dtype=x_train.dtype)
    ones_eval = torch.ones(x_eval.shape[0], 1, dtype=x_eval.dtype)
    x_train = torch.cat([x_train, ones_train], dim=1)
    x_eval = torch.cat([x_eval, ones_eval], dim=1)
    gram = x_train @ x_train.T + ridge_lambda * torch.eye(x_train.shape[0], dtype=x_train.dtype)
    alpha = torch.linalg.solve(gram, y_train)
    return x_eval @ x_train.T @ alpha


def load_ar_parquet(parquet_path: str) -> tuple[list[str], torch.Tensor, list[str]]:
    table = pq.read_table(parquet_path, columns=["prompt", "activation_vector", "source_sample_id"])
    prompts = table.column("prompt").to_pylist()
    sample_ids = table.column("source_sample_id").to_pylist()
    activation_col = table.column("activation_vector").combine_chunks()
    flat = activation_col.values.to_numpy(zero_copy_only=False).astype(np.float32)
    activations = torch.from_numpy(flat.reshape(len(prompts), -1))
    return prompts, activations, sample_ids


def compute_prompt_features(
    model: LlavaForConditionalGeneration,
    tokenizer,
    prompts: list[str],
    layer_index: int,
    batch_size: int,
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def layer_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["layer_hidden"] = hidden.detach().float().cpu()

    handle = model.model.language_model.layers[layer_index].register_forward_hook(layer_hook)
    feats: list[torch.Tensor] = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            tok = tokenizer(batch, padding=True, return_tensors="pt", add_special_tokens=True)
            tok = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in tok.items()}
            captured.clear()
            with torch.no_grad():
                model.model.language_model(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
            hidden = captured["layer_hidden"]
            last_idx = tok["attention_mask"].detach().cpu().sum(dim=1) - 1
            feats.append(hidden[torch.arange(hidden.shape[0]), last_idx])
    finally:
        handle.remove()
    return torch.cat(feats, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ar-parquet", required=True)
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--layer-index", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", required=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    prompts, activations, sample_ids = load_ar_parquet(args.ar_parquet)
    n = len(prompts)
    rng = np.random.default_rng(args.seed)
    perm = torch.from_numpy(rng.permutation(n))
    n_train = max(4, int(n * args.train_frac))
    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]

    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    processor.tokenizer.padding_side = "right"
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    features = compute_prompt_features(model, processor.tokenizer, prompts, args.layer_index, args.batch_size)
    x_train = features[train_idx].float()
    y_train = activations[train_idx].float()
    x_eval = features[eval_idx].float()
    y_eval = activations[eval_idx].float()

    pred_train = ridge_predict(x_train, y_train, x_train, args.ridge_lambda)
    pred_eval = ridge_predict(x_train, y_train, x_eval, args.ridge_lambda)
    mean_pred = y_train.mean(dim=0, keepdim=True).expand_as(y_eval)
    shuffle = train_idx[torch.from_numpy(rng.permutation(len(train_idx)))]
    pred_shuffle = ridge_predict(x_train, activations[shuffle].float(), x_eval, args.ridge_lambda)

    eval_mse = mse_nrm(pred_eval, y_eval)
    train_mse = mse_nrm(pred_train, y_train)
    mean_mse = mse_nrm(mean_pred, y_eval)
    shuffle_mse = mse_nrm(pred_shuffle, y_eval)

    summary = {
        "ar_parquet": args.ar_parquet,
        "model_id": args.model_id,
        "layer_index": args.layer_index,
        "n_total": n,
        "n_train": int(len(train_idx)),
        "n_eval": int(len(eval_idx)),
        "ridge_lambda": args.ridge_lambda,
        "feature_shape": list(features.shape),
        "activation_shape": list(activations.shape),
        "train_mse_nrm_mean": float(train_mse.mean().item()),
        "train_cos_mean": float(cosine(pred_train, y_train).mean().item()),
        "eval_mse_nrm_mean": float(eval_mse.mean().item()),
        "eval_mse_nrm_std": float(eval_mse.std(unbiased=False).item()),
        "eval_cos_mean": float(cosine(pred_eval, y_eval).mean().item()),
        "mean_baseline_mse_nrm_mean": float(mean_mse.mean().item()),
        "mean_baseline_cos_mean": float(cosine(mean_pred, y_eval).mean().item()),
        "shuffle_control_mse_nrm_mean": float(shuffle_mse.mean().item()),
        "shuffle_control_cos_mean": float(cosine(pred_shuffle, y_eval).mean().item()),
        "fve_vs_mean_baseline": float(1.0 - eval_mse.mean().item() / mean_mse.mean().item()),
        "fve_vs_shuffle_control": float(1.0 - eval_mse.mean().item() / shuffle_mse.mean().item()),
        "eval_sample_ids": [sample_ids[i] for i in eval_idx.tolist()],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

