#!/usr/bin/env python3
"""Activation-sensitivity check for Qwen3-VL NLA AV models.

Candidate ranking can be dominated by response language priors. This script
holds each target response fixed and compares its teacher-forced NLL under the
matched activation versus a shifted/shuffled activation. If the AV model uses
the injected layer activation, matched activations should give lower NLL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def sidecar_path(path: str) -> Path:
    return Path(path + ".nla_meta.yaml")


def unwrap_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def normalize(v: torch.Tensor, scale: float | None) -> torch.Tensor:
    if scale is None:
        return v
    return v / v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) * scale


def map_activation(
    activation_adapter: torch.nn.Module | None,
    activation: torch.Tensor,
    injection_scale: float | None,
    num_injection_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    activation = normalize(activation.view(1, -1), injection_scale).to(device)
    if activation_adapter is not None:
        mapped = activation_adapter(activation.float()).view(activation.shape[0], num_injection_tokens, -1)
    else:
        mapped = activation.view(activation.shape[0], 1, -1).expand(-1, num_injection_tokens, -1)
        if num_injection_tokens > 1:
            mapped = mapped / (num_injection_tokens**0.5)
    return mapped.to(device, dtype)


def load_activation_adapter(
    path: str | None,
    d_model: int,
    num_injection_tokens: int,
    device: torch.device,
) -> torch.nn.Module | None:
    if path is None:
        return None
    adapter_path = Path(path)
    if adapter_path.is_dir():
        adapter_path = adapter_path / "activation_adapter.pt"
    ckpt = torch.load(adapter_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    adapter = torch.nn.Linear(
        d_model,
        d_model * num_injection_tokens,
        bias=True,
        device=device,
        dtype=torch.float32,
    )
    adapter.load_state_dict(state_dict)
    adapter.eval()
    return adapter


def load_rows(path: str, max_rows: int | None) -> list[dict]:
    table = pq.read_table(path)
    if max_rows is not None:
        table = table.slice(0, max_rows)
    return table.to_pylist()


def expand_injection_prompt(prompt: str, injection_token: str, num_injection_tokens: int) -> str:
    if num_injection_tokens == 1:
        return prompt
    if prompt.count(injection_token) != 1:
        raise RuntimeError(f"expected one textual injection token {injection_token!r}")
    return prompt.replace(injection_token, injection_token * num_injection_tokens, 1)


def encode_prompt(
    tokenizer,
    prompt: str,
    inj_id: int,
    left_id: int,
    right_id: int,
    injection_token: str,
    num_injection_tokens: int,
) -> tuple[list[int], list[int]]:
    prompt = expand_injection_prompt(prompt, injection_token, num_injection_tokens)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    matches = [i for i, tid in enumerate(ids) if tid == inj_id]
    if len(matches) != num_injection_tokens:
        raise RuntimeError(f"expected {num_injection_tokens} injection markers, got {len(matches)}")
    start = matches[0]
    if matches != list(range(start, start + num_injection_tokens)):
        raise RuntimeError(f"injection tokens are not contiguous: {matches}")
    if ids[start - 1] != left_id or ids[start + num_injection_tokens] != right_id:
        raise RuntimeError("injection neighbors do not match sidecar")
    return ids, matches


def encode_response(tokenizer, response: str) -> list[int]:
    ids = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return ids


@torch.no_grad()
def score_responses(
    model,
    activation_adapter: torch.nn.Module | None,
    prompt_ids: list[int],
    inj_positions: list[int],
    activations: list[torch.Tensor],
    response_ids: list[list[int]],
    pad_id: int,
    injection_scale: float | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    base = unwrap_model(model)
    prompt_len = len(prompt_ids)
    scores: list[torch.Tensor] = []

    for start in range(0, len(response_ids), batch_size):
        batch_resp = response_ids[start : start + batch_size]
        batch_act = activations[start : start + batch_size]
        seqs = [prompt_ids + resp for resp in batch_resp]
        max_len = max(len(x) for x in seqs)
        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
        labels = torch.full((len(seqs), max_len), -100, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)

        for i, (seq, resp) in enumerate(zip(seqs, batch_resp, strict=True)):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            labels[i, prompt_len : len(seq)] = torch.tensor(resp, dtype=torch.long, device=device)
            attention_mask[i, : len(seq)] = 1

        embeds = base.get_input_embeddings()(input_ids)
        embeds = embeds.clone()
        mapped = [
            map_activation(
                activation_adapter,
                activation,
                injection_scale,
                len(inj_positions),
                device,
                embeds.dtype,
            )[0]
            for activation in batch_act
        ]
        embeds[:, inj_positions, :] = torch.stack(mapped, dim=0)
        out = base.model(
            input_ids=None,
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = base.lm_head(out.last_hidden_state).float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.shape)
        counts = (shift_labels != -100).sum(dim=1).clamp_min(1)
        scores.append(losses.sum(dim=1) / counts)
    return torch.cat(scores, dim=0).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av-parquet", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--activation-adapter", default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--injection-scale", type=float, default=57.75)
    parser.add_argument("--num-injection-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = load_rows(args.av_parquet, args.max_rows)
    meta = yaml.safe_load(sidecar_path(args.av_parquet).read_text())
    token_meta = meta["tokens"]

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    left_id = int(token_meta.get("injection_left_neighbor_id", token_meta.get("vision_start_token_id")))
    right_id = int(token_meta.get("injection_right_neighbor_id", token_meta.get("vision_end_token_id")))
    prompt_ids, inj_positions = encode_prompt(
        tokenizer,
        rows[0]["prompt"],
        int(token_meta["injection_token_id"]),
        left_id,
        right_id,
        str(token_meta.get("injection_token", "<|image_pad|>")),
        args.num_injection_tokens,
    )
    activations = [torch.tensor(row["activation_vector"], dtype=torch.float32) for row in rows]
    response_ids = [encode_response(tokenizer, row["response"]) for row in rows]

    shifted_activations = activations[1:] + activations[:1]
    perm = rng.permutation(len(activations))
    if len(perm) > 1:
        for i in range(len(perm)):
            if perm[i] == i:
                perm[i], perm[(i + 1) % len(perm)] = perm[(i + 1) % len(perm)], perm[i]
    shuffled_activations = [activations[int(i)] for i in perm.tolist()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).to(device)
    model.eval()
    activation_adapter = load_activation_adapter(
        args.activation_adapter,
        int(len(rows[0]["activation_vector"])),
        args.num_injection_tokens,
        device,
    )

    matched_scores = score_responses(
        model,
        activation_adapter,
        prompt_ids,
        inj_positions,
        activations,
        response_ids,
        pad_id,
        args.injection_scale,
        args.batch_size,
        device,
    )
    shifted_scores = score_responses(
        model,
        activation_adapter,
        prompt_ids,
        inj_positions,
        shifted_activations,
        response_ids,
        pad_id,
        args.injection_scale,
        args.batch_size,
        device,
    )
    shuffled_scores = score_responses(
        model,
        activation_adapter,
        prompt_ids,
        inj_positions,
        shuffled_activations,
        response_ids,
        pad_id,
        args.injection_scale,
        args.batch_size,
        device,
    )

    shifted_delta = shifted_scores.numpy() - matched_scores.numpy()
    shuffled_delta = shuffled_scores.numpy() - matched_scores.numpy()
    examples = []
    for i, row in enumerate(rows[:8]):
        examples.append(
            {
                "index": i,
                "sample_id": row["source_sample_id"],
                "source_description": row.get("source_description", ""),
                "matched_nll": float(matched_scores[i].item()),
                "shifted_nll": float(shifted_scores[i].item()),
                "shuffled_nll": float(shuffled_scores[i].item()),
                "shifted_minus_matched": float(shifted_delta[i]),
                "shuffled_minus_matched": float(shuffled_delta[i]),
            }
        )

    summary = {
        "av_parquet": args.av_parquet,
        "adapter": args.adapter,
        "activation_adapter": args.activation_adapter,
        "model_id": args.model_id,
        "num_rows": len(rows),
        "injection_scale": args.injection_scale,
        "num_injection_tokens": args.num_injection_tokens,
        "prompt_token_count": len(prompt_ids),
        "injection_positions": [int(x) for x in inj_positions],
        "matched_mean_nll": float(matched_scores.mean().item()),
        "shifted_mean_nll": float(shifted_scores.mean().item()),
        "shuffled_mean_nll": float(shuffled_scores.mean().item()),
        "shifted_minus_matched_mean": float(shifted_delta.mean()),
        "shuffled_minus_matched_mean": float(shuffled_delta.mean()),
        "shifted_match_better_fraction": float((shifted_delta > 0).mean()),
        "shuffled_match_better_fraction": float((shuffled_delta > 0).mean()),
        "examples": examples,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
