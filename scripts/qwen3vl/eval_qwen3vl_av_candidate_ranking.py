#!/usr/bin/env python3
"""Evaluate AV candidate ranking for Qwen3-VL NLA.

For each injected activation, score a pool of candidate explanations by
teacher-forced NLL:

  activation -> replace <|image_pad|> marker embeddings -> language model
  score(prompt + candidate_response)

If the trained AV uses the activation, the correct response should rank above
other candidates. This catches signal even when greedy decoding still
mode-collapses.
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
    weight = state_dict["weight"]
    in_features = int(weight.shape[1])
    out_features = int(weight.shape[0])
    if out_features % num_injection_tokens != 0:
        raise RuntimeError(
            f"activation adapter out_features={out_features} is not divisible by {num_injection_tokens} tokens"
        )
    if in_features != d_model:
        raise RuntimeError(f"activation dim mismatch: parquet has {d_model}, adapter expects {in_features}")
    adapter = torch.nn.Linear(
        in_features,
        out_features,
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
def score_candidates(
    model,
    activation_adapter: torch.nn.Module | None,
    prompt_ids: list[int],
    inj_positions: list[int],
    activation: torch.Tensor,
    candidate_response_ids: list[list[int]],
    pad_id: int,
    injection_scale: float | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    base = unwrap_model(model)
    scores: list[torch.Tensor] = []
    prompt_len = len(prompt_ids)

    for start in range(0, len(candidate_response_ids), batch_size):
        batch_resp = candidate_response_ids[start : start + batch_size]
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
        mapped_activation = map_activation(
            activation_adapter,
            activation,
            injection_scale,
            len(inj_positions),
            device,
            embeds.dtype,
        )
        embeds[:, inj_positions, :] = mapped_activation.expand(len(seqs), -1, -1)
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
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--candidate-mode", choices=["prefix", "all"], default="prefix")
    parser.add_argument("--score-mode", choices=["nll", "activation_gain"], default="nll")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--injection-scale", type=float, default=57.75)
    parser.add_argument("--num-injection-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows = load_rows(args.av_parquet, args.max_rows)
    if args.max_queries is not None:
        query_rows = rows[: args.max_queries]
    else:
        query_rows = rows

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
    candidate_rows = rows if args.candidate_mode == "all" else rows[: len(query_rows)]
    candidate_response_ids = [encode_response(tokenizer, row["response"]) for row in candidate_rows]
    candidate_ids = [row["source_sample_id"] for row in candidate_rows]
    reference_activation = torch.tensor(
        np.asarray([row["activation_vector"] for row in rows], dtype=np.float32).mean(axis=0),
        dtype=torch.float32,
    )

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

    ranks = []
    margins = []
    correct_scores = []
    correct_raw_scores = []
    correct_reference_scores = []
    examples = []
    per_query = []
    unique_label_ranks = []
    all_scores = []
    all_raw_scores = []
    for qi, row in enumerate(query_rows):
        activation = torch.tensor(row["activation_vector"], dtype=torch.float32)
        raw_scores = score_candidates(
            model,
            activation_adapter,
            prompt_ids,
            inj_positions,
            activation,
            candidate_response_ids,
            pad_id,
            args.injection_scale,
            args.batch_size,
            device,
        )
        if args.score_mode == "activation_gain":
            reference_scores = score_candidates(
                model,
                activation_adapter,
                prompt_ids,
                inj_positions,
                reference_activation,
                candidate_response_ids,
                pad_id,
                args.injection_scale,
                args.batch_size,
                device,
            )
            scores = raw_scores - reference_scores
        else:
            reference_scores = None
            scores = raw_scores
        correct_id = row["source_sample_id"]
        correct_index = candidate_ids.index(correct_id)
        order = torch.argsort(scores)
        rank = int((order == correct_index).nonzero(as_tuple=False)[0].item()) + 1
        correct_label = row["response"]
        seen_labels = set()
        unique_rank = None
        for idx in order.tolist():
            label = candidate_rows[int(idx)]["response"]
            if label in seen_labels:
                continue
            seen_labels.add(label)
            if label == correct_label:
                unique_rank = len(seen_labels)
                break
        if unique_rank is None:
            raise RuntimeError(f"correct response label not found for {correct_id}")
        ranks.append(rank)
        unique_label_ranks.append(unique_rank)
        correct_score = float(scores[correct_index].item())
        correct_raw_score = float(raw_scores[correct_index].item())
        best_wrong_score = float(scores[[i for i in range(len(scores)) if i != correct_index]].min().item())
        margins.append(best_wrong_score - correct_score)
        correct_scores.append(correct_score)
        correct_raw_scores.append(correct_raw_score)
        if reference_scores is not None:
            correct_reference_scores.append(float(reference_scores[correct_index].item()))
        all_scores.append(scores.numpy())
        all_raw_scores.append(raw_scores.numpy())
        topk = []
        for r, idx in enumerate(order[:5].tolist()):
            item = {
                "rank": r + 1,
                "candidate_index": int(idx),
                "sample_id": candidate_ids[int(idx)],
                "score": float(scores[int(idx)].item()),
                "raw_nll": float(raw_scores[int(idx)].item()),
                "is_correct": int(idx) == correct_index,
                "same_response_label": candidate_rows[int(idx)]["response"] == correct_label,
                "response": candidate_rows[int(idx)]["response"],
                "source_description": candidate_rows[int(idx)].get("source_description", ""),
            }
            if reference_scores is not None:
                item["reference_nll"] = float(reference_scores[int(idx)].item())
            topk.append(item)
        query_record = {
            "query_index": qi,
            "query_sample_id": correct_id,
            "correct_rank": rank,
            "unique_label_rank": unique_rank,
            "correct_score": correct_score,
            "correct_raw_nll": correct_raw_score,
            "correct_reference_nll": (
                float(reference_scores[correct_index].item()) if reference_scores is not None else None
            ),
            "source_description": row.get("source_description", ""),
            "top5": topk,
        }
        per_query.append(query_record)
        if qi < 8:
            examples.append(query_record)

    ranks_arr = np.asarray(ranks)
    unique_label_ranks_arr = np.asarray(unique_label_ranks)
    margins_arr = np.asarray(margins)
    score_mat = np.stack(all_scores, axis=0)
    raw_score_mat = np.stack(all_raw_scores, axis=0)
    summary = {
        "av_parquet": args.av_parquet,
        "adapter": args.adapter,
        "activation_adapter": args.activation_adapter,
        "model_id": args.model_id,
        "num_queries": len(query_rows),
        "num_candidates": len(candidate_rows),
        "candidate_mode": args.candidate_mode,
        "score_mode": args.score_mode,
        "injection_scale": args.injection_scale,
        "num_injection_tokens": args.num_injection_tokens,
        "prompt_token_count": len(prompt_ids),
        "injection_positions": [int(x) for x in inj_positions],
        "mean_rank": float(ranks_arr.mean()),
        "median_rank": float(np.median(ranks_arr)),
        "top1_accuracy": float((ranks_arr == 1).mean()),
        "top3_accuracy": float((ranks_arr <= 3).mean()),
        "top5_accuracy": float((ranks_arr <= 5).mean()),
        "unique_label_mean_rank": float(unique_label_ranks_arr.mean()),
        "unique_label_median_rank": float(np.median(unique_label_ranks_arr)),
        "unique_label_top1_accuracy": float((unique_label_ranks_arr == 1).mean()),
        "unique_label_top3_accuracy": float((unique_label_ranks_arr <= 3).mean()),
        "unique_label_top5_accuracy": float((unique_label_ranks_arr <= 5).mean()),
        "mean_margin_best_wrong_minus_correct": float(margins_arr.mean()),
        "median_margin_best_wrong_minus_correct": float(np.median(margins_arr)),
        "mean_correct_nll": float(np.mean(correct_scores)),
        "mean_correct_raw_nll": float(np.mean(correct_raw_scores)),
        "mean_correct_reference_nll": (
            float(np.mean(correct_reference_scores)) if correct_reference_scores else None
        ),
        "mean_all_nll": float(score_mat.mean()),
        "mean_all_raw_nll": float(raw_score_mat.mean()),
        "examples": examples,
        "per_query": per_query,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
