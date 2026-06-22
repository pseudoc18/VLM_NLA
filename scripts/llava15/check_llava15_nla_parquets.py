#!/usr/bin/env python3
"""Sanity-check LLaVA-NLA AV/AR parquets and optional injection forward path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoProcessor, LlavaForConditionalGeneration


def sidecar_path(path: str) -> Path:
    return Path(path + ".nla_meta.yaml")


def read_activation(row: dict) -> torch.Tensor:
    return torch.tensor(row["activation_vector"], dtype=torch.float32).view(1, -1)


def normalize(v: torch.Tensor, scale: float | None) -> torch.Tensor:
    if scale is None:
        return v
    return v / v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) * scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av-parquet", required=True)
    parser.add_argument("--ar-parquet", required=True)
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--forward-check", action="store_true")
    parser.add_argument("--injection-scale", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    av_table = pq.read_table(args.av_parquet)
    ar_table = pq.read_table(args.ar_parquet)
    if av_table.num_rows != ar_table.num_rows:
        raise RuntimeError(f"row mismatch av={av_table.num_rows} ar={ar_table.num_rows}")

    meta = yaml.safe_load(sidecar_path(args.av_parquet).read_text())
    ar_meta = yaml.safe_load(sidecar_path(args.ar_parquet).read_text())
    if meta != ar_meta:
        raise RuntimeError("AV and AR sidecars differ")

    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    tokenizer = processor.tokenizer

    first_av = av_table.slice(0, 1).to_pylist()[0]
    first_ar = ar_table.slice(0, 1).to_pylist()[0]
    token_meta = meta["tokens"]
    inj_id = int(token_meta["injection_token_id"])
    ids = tokenizer.encode(first_av["prompt"], add_special_tokens=True)
    matches = [i for i, tid in enumerate(ids) if tid == inj_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one injection token in first AV prompt, got {len(matches)}")
    p = matches[0]
    if ids[p - 1] != token_meta["injection_left_neighbor_id"] or ids[p + 1] != token_meta["injection_right_neighbor_id"]:
        raise RuntimeError("injection neighbor ids do not match sidecar")

    ar_ids = tokenizer.encode(first_ar["prompt"], add_special_tokens=True)
    summary = {
        "av_parquet": args.av_parquet,
        "ar_parquet": args.ar_parquet,
        "row_count": av_table.num_rows,
        "d_model": meta["d_model"],
        "activation_layer": meta["activation_layer"],
        "target_token": meta["target_token"],
        "injection_token_id": inj_id,
        "injection_position": p,
        "av_prompt_token_count": len(ids),
        "ar_prompt_token_count_first": len(ar_ids),
        "first_response": first_av["response"],
        "first_ar_prompt": first_ar["prompt"],
    }

    if args.forward_check:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            local_files_only=args.local_files_only,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            embeds = model.get_input_embeddings()(ids_tensor).float()
        raw_embed = embeds[0, p].detach().cpu()
        activation = normalize(read_activation(first_av), args.injection_scale).to(embeds.device, embeds.dtype)
        embeds_injected = embeds.clone()
        embeds_injected[0, p] = activation[0]

        with torch.no_grad():
            out = model.model.language_model(inputs_embeds=embeds_injected.to(model.dtype), use_cache=False, return_dict=True)
            logits = model.lm_head(out.last_hidden_state[:, -1:, :])
        summary.update(
            {
                "forward_check": True,
                "inputs_embeds_shape": list(embeds_injected.shape),
                "last_hidden_state_shape": list(out.last_hidden_state.shape),
                "last_token_logits_shape": list(logits.shape),
                "raw_marker_embed_norm": float(raw_embed.norm().item()),
                "injected_activation_norm": float(activation[0].float().norm().item()),
                "marker_delta_norm": float((embeds_injected[0, p].detach().cpu() - raw_embed).norm().item()),
                "finite_logits": bool(torch.isfinite(logits).all().item()),
            }
        )
    else:
        summary["forward_check"] = False

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

