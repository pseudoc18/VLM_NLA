#!/usr/bin/env python3
"""Counterfactual mask pilot for Qwen3-VL VLM-NLA.

For each COCO object-token row, this script masks the object's bbox in the
image, extracts the same target-region activation from the edited image, and
checks whether the trained AV still ranks the original object explanation.

This is an H5 pilot: if explanations are grounded in visual evidence, masking
the target object should make the original label less preferred.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image, ImageFilter, ImageStat
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qwen3vl.eval_qwen3vl_av_candidate_ranking import (
    encode_prompt,
    encode_response,
    load_activation_adapter,
    score_candidates,
    sidecar_path,
)
from scripts.qwen3vl.extract_qwen3vl_coco_object_tokens import bbox_token_indices, center_token_index


def load_rows(path: str, max_rows: int | None = None) -> list[dict]:
    table = pq.read_table(path)
    if max_rows is not None:
        table = table.slice(0, max_rows)
    return table.to_pylist()


def mask_image(image: Image.Image, bbox: list[float], mode: str) -> Image.Image:
    edited = image.copy()
    x, y, w, h = [int(round(v)) for v in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(edited.width, x + w), min(edited.height, y + h)
    if x1 <= x0 or y1 <= y0:
        return edited
    if mode == "blur":
        patch = edited.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=16))
        edited.paste(patch, (x0, y0))
    else:
        stat = ImageStat.Stat(edited)
        color = tuple(int(v) for v in stat.mean[:3])
        edited.paste(color, (x0, y0, x1, y1))
    return edited


def prompt_for_image(processor, image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one concise sentence."},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def extract_edited_activations(
    *,
    model,
    processor,
    tokenizer,
    rows: list[dict],
    layer_index: int,
    target_token: str,
    max_bbox_tokens: int,
    mask_mode: str,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    image_pad_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    merge = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    captured: dict[str, torch.Tensor] = {}

    def layer_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["layer_hidden"] = hidden.detach().float().cpu()

    handle = model.model.language_model.layers[layer_index].register_forward_hook(layer_hook)
    activations: list[np.ndarray] = []
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            images = []
            texts = []
            for row in batch_rows:
                image = Image.open(row["source_image_path"]).convert("RGB")
                edited = mask_image(image, row["bbox_xywh"], mask_mode)
                images.append(edited)
                texts.append(prompt_for_image(processor, edited))
            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            captured.clear()
            model(**inputs, use_cache=False, return_dict=True)
            hidden = captured["layer_hidden"]
            input_ids = inputs["input_ids"].detach().cpu()
            grids = inputs["image_grid_thw"].detach().cpu().tolist()
            for i, row in enumerate(batch_rows):
                ids = input_ids[i]
                image_positions = (ids == image_pad_id).nonzero(as_tuple=False).flatten()
                grid_t, raw_h, raw_w = [int(x) for x in grids[i]]
                grid_h = raw_h // merge
                grid_w = raw_w // merge
                sample = type(
                    "Sample",
                    (),
                    {
                        "bbox": tuple(float(x) for x in row["bbox_xywh"]),
                        "width": int(Image.open(row["source_image_path"]).size[0]),
                        "height": int(Image.open(row["source_image_path"]).size[1]),
                    },
                )()
                center_idx = center_token_index(sample, grid_h, grid_w)
                if target_token == "object_center":
                    selected_indices = [center_idx]
                else:
                    selected_indices = bbox_token_indices(sample, grid_h, grid_w, max_bbox_tokens)
                selected_positions = image_positions[torch.tensor(selected_indices, dtype=torch.long)]
                activation = hidden[i, selected_positions].mean(dim=0).numpy().astype(np.float32)
                activations.append(activation)
    finally:
        handle.remove()
    return activations


def rank_correct(scores: torch.Tensor, correct_index: int) -> int:
    order = torch.argsort(scores)
    return int((order == correct_index).nonzero(as_tuple=False)[0].item()) + 1


def top_response(scores: torch.Tensor, candidate_rows: list[dict]) -> str:
    idx = int(torch.argmin(scores).item())
    return candidate_rows[idx]["response"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av-parquet", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--activation-adapter", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--candidate-rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--injection-scale", type=float, default=57.75)
    parser.add_argument("--num-injection-tokens", type=int, default=8)
    parser.add_argument("--mask-mode", choices=["mean", "blur"], default="mean")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows_all = load_rows(args.av_parquet)
    rows = rows_all[: args.max_rows]
    candidate_rows = rows_all[: args.candidate_rows]
    meta = yaml.safe_load(sidecar_path(args.av_parquet).read_text(encoding="utf-8"))
    token_meta = meta["tokens"]
    layer_index = int(rows[0]["activation_layer"])
    target_token = str(rows[0]["target_token"])
    max_bbox_tokens = max(int(row["num_selected_image_tokens"]) for row in rows)

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
    candidate_response_ids = [encode_response(tokenizer, row["response"]) for row in candidate_rows]
    candidate_ids = [row["source_sample_id"] for row in candidate_rows]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    edited_activations = extract_edited_activations(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        rows=rows,
        layer_index=layer_index,
        target_token=target_token,
        max_bbox_tokens=max_bbox_tokens,
        mask_mode=args.mask_mode,
        batch_size=args.batch_size,
        device=device,
    )

    model = PeftModel.from_pretrained(model, args.adapter).to(device)
    model.eval()
    activation_adapter = load_activation_adapter(
        args.activation_adapter,
        int(len(rows[0]["activation_vector"])),
        args.num_injection_tokens,
        device,
    )

    records = []
    original_ranks = []
    edited_ranks = []
    original_correct_nll = []
    edited_correct_nll = []
    top_changed = []
    for row, edited_activation in zip(rows, edited_activations, strict=True):
        correct_index = candidate_ids.index(row["source_sample_id"])
        original_activation = torch.tensor(row["activation_vector"], dtype=torch.float32)
        edited_activation_tensor = torch.tensor(edited_activation, dtype=torch.float32)
        original_scores = score_candidates(
            model,
            activation_adapter,
            prompt_ids,
            inj_positions,
            original_activation,
            candidate_response_ids,
            pad_id,
            args.injection_scale,
            args.score_batch_size,
            device,
        )
        edited_scores = score_candidates(
            model,
            activation_adapter,
            prompt_ids,
            inj_positions,
            edited_activation_tensor,
            candidate_response_ids,
            pad_id,
            args.injection_scale,
            args.score_batch_size,
            device,
        )
        original_rank = rank_correct(original_scores, correct_index)
        edited_rank = rank_correct(edited_scores, correct_index)
        original_top = top_response(original_scores, candidate_rows)
        edited_top = top_response(edited_scores, candidate_rows)
        original_nll = float(original_scores[correct_index].item())
        edited_nll = float(edited_scores[correct_index].item())
        original_ranks.append(original_rank)
        edited_ranks.append(edited_rank)
        original_correct_nll.append(original_nll)
        edited_correct_nll.append(edited_nll)
        top_changed.append(original_top != edited_top)
        records.append(
            {
                "sample_id": row["source_sample_id"],
                "source_description": row.get("source_description", ""),
                "bbox_xywh": row["bbox_xywh"],
                "original_rank": original_rank,
                "edited_rank": edited_rank,
                "rank_delta": edited_rank - original_rank,
                "original_correct_nll": original_nll,
                "edited_correct_nll": edited_nll,
                "correct_nll_increase": edited_nll - original_nll,
                "original_top_response": original_top,
                "edited_top_response": edited_top,
                "top_response_changed": original_top != edited_top,
            }
        )

    original_ranks_arr = np.asarray(original_ranks)
    edited_ranks_arr = np.asarray(edited_ranks)
    nll_increase = np.asarray(edited_correct_nll) - np.asarray(original_correct_nll)
    summary = {
        "av_parquet": args.av_parquet,
        "adapter": args.adapter,
        "activation_adapter": args.activation_adapter,
        "model_id": args.model_id,
        "num_rows": len(rows),
        "num_candidates": len(candidate_rows),
        "layer_index": layer_index,
        "target_token": target_token,
        "mask_mode": args.mask_mode,
        "original_mean_rank": float(original_ranks_arr.mean()),
        "edited_mean_rank": float(edited_ranks_arr.mean()),
        "mean_rank_delta": float((edited_ranks_arr - original_ranks_arr).mean()),
        "original_top1": float((original_ranks_arr == 1).mean()),
        "edited_same_label_top1": float((edited_ranks_arr == 1).mean()),
        "original_top5": float((original_ranks_arr <= 5).mean()),
        "edited_same_label_top5": float((edited_ranks_arr <= 5).mean()),
        "mean_correct_nll_increase": float(nll_increase.mean()),
        "fraction_correct_nll_increased": float((nll_increase > 0).mean()),
        "top_response_changed_fraction": float(np.mean(top_changed)),
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
