#!/usr/bin/env python3
"""H4 pilot: absent-object hallucination probes and local-token attribution.

This script asks Qwen3-VL forced-choice object-presence questions for COCO
categories that are absent from each image. When the base model prefers "Yes"
over "No" for an absent category, the script masks visible object regions to
test which region reduces the false-positive margin most. It then scores two
NLA explanations for that high-impact region:

  1. the true COCO object category for the region
  2. the hallucinated absent category

The result is a small, traceable pilot for the question: can local image-token
NLA explanations help diagnose visual shortcuts behind LVLM hallucinations?
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from PIL import Image, ImageStat
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qwen3vl.eval_qwen3vl_av_candidate_ranking import (  # noqa: E402
    encode_prompt,
    encode_response,
    load_activation_adapter,
    score_candidates,
    sidecar_path,
)
from scripts.qwen3vl.extract_qwen3vl_coco_object_tokens import (  # noqa: E402
    bbox_token_indices,
    center_token_index,
    coarse_position,
    ensure_coco,
)


@dataclass(frozen=True)
class RegionProbe:
    image_id: int
    image_path: str
    width: int
    height: int
    ann_id: int
    category: str
    bbox_xywh: tuple[float, float, float, float]
    area: float
    coarse_position: str


def load_rows(path: str, max_rows: int | None = None) -> list[dict]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    return rows[:max_rows] if max_rows is not None else rows


def article(category: str) -> str:
    return "an" if category[:1].lower() in "aeiou" else "a"


def region_description(category: str, position: str) -> str:
    return f"These image tokens represent {article(category)} {category} in the {position} region of the image."


def response_for(category: str, position: str) -> str:
    return f"<explanation>\n{region_description(category, position)}\n</explanation>"


def load_coco_regions(
    *,
    coco_root: Path,
    image_rows: list[dict],
    min_area_frac: float,
    max_objects_per_image: int,
) -> tuple[list[str], dict[int, set[str]], dict[int, list[RegionProbe]], dict[str, dict[str, int]]]:
    instances_path, _captions_path, _image_dir = ensure_coco(coco_root)
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]) for cat in instances["categories"]}
    all_categories = [categories[int(cat["id"])] for cat in instances["categories"]]
    images = {int(img["id"]): img for img in instances["images"]}
    wanted = {int(row["coco_image_id"]): row for row in image_rows}
    present_by_image: dict[int, set[str]] = {image_id: set() for image_id in wanted}
    regions_by_image: dict[int, list[RegionProbe]] = {image_id: [] for image_id in wanted}
    full_present_by_image: dict[int, set[str]] = defaultdict(set)

    for ann in instances["annotations"]:
        image_id = int(ann["image_id"])
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        category = categories[int(ann["category_id"])]
        full_present_by_image[image_id].add(category)
        if image_id not in wanted:
            continue
        image = images[image_id]
        present_by_image[image_id].add(category)
        area = float(ann.get("area", 0.0))
        min_area = min_area_frac * float(image["width"] * image["height"])
        if area < min_area:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        position = coarse_position(x + w / 2, y + h / 2, float(image["width"]), float(image["height"]))
        row = wanted[image_id]
        regions_by_image[image_id].append(
            RegionProbe(
                image_id=image_id,
                image_path=str(row["source_image_path"]),
                width=int(image["width"]),
                height=int(image["height"]),
                ann_id=int(ann["id"]),
                category=category,
                bbox_xywh=(x, y, w, h),
                area=area,
                coarse_position=position,
            )
        )

    for image_id, regions in regions_by_image.items():
        regions.sort(key=lambda region: region.area, reverse=True)
        regions_by_image[image_id] = regions[:max_objects_per_image]

    cooccurrence: dict[str, Counter[str]] = {category: Counter() for category in all_categories}
    for cats in full_present_by_image.values():
        for category in cats:
            cooccurrence[category].update(other for other in cats if other != category)
    cooccurrence_json = {category: dict(counter) for category, counter in cooccurrence.items()}
    return all_categories, present_by_image, regions_by_image, cooccurrence_json


def choose_absent_categories(
    *,
    all_categories: list[str],
    present: set[str],
    cooccurrence: dict[str, dict[str, int]],
    sampling: str,
    count: int,
    rng: random.Random,
) -> list[tuple[str, int]]:
    absent = [category for category in all_categories if category not in present]
    if sampling == "random":
        rng.shuffle(absent)
        return [(category, 0) for category in absent[:count]]
    scored = []
    for category in absent:
        score = sum(int(cooccurrence.get(present_category, {}).get(category, 0)) for present_category in present)
        scored.append((category, score, rng.random()))
    scored.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [(category, score) for category, score, _tie in scored[:count]]


def presence_prompt(processor, image: Image.Image, category: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f'Answer only "Yes" or "No". Is there any object of the category '
                        f'"{category}" in this image?'
                    ),
                },
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_yes_no(
    *,
    model,
    processor,
    tokenizer,
    image: Image.Image,
    category: str,
    device: torch.device,
) -> dict[str, float]:
    prompt = presence_prompt(processor, image, category)
    candidates = ["Yes", "No"]
    texts = [prompt + candidate for candidate in candidates]
    images = [image.copy(), image.copy()]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[inputs["attention_mask"] == 0] = -100
    out = model(**inputs, use_cache=False, return_dict=True)
    logits = out.logits.float()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    counts = (shift_labels != -100).sum(dim=1).clamp_min(1)
    scores = (losses.sum(dim=1) / counts).detach().cpu().tolist()
    yes_nll = float(scores[0])
    no_nll = float(scores[1])
    return {
        "yes_nll": yes_nll,
        "no_nll": no_nll,
        "yes_margin": no_nll - yes_nll,
        "preferred_answer": "Yes" if yes_nll < no_nll else "No",
    }


def mask_bbox(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    edited = image.copy()
    x, y, w, h = [int(round(v)) for v in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(edited.width, x + w), min(edited.height, y + h)
    if x1 <= x0 or y1 <= y0:
        return edited
    color = tuple(int(v) for v in ImageStat.Stat(edited).mean[:3])
    edited.paste(color, (x0, y0, x1, y1))
    return edited


@torch.no_grad()
def extract_region_activations(
    *,
    model,
    processor,
    tokenizer,
    regions: list[RegionProbe],
    layer_index: int,
    max_bbox_tokens: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    if not regions:
        return []
    image_pad_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    merge = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    captured: dict[str, torch.Tensor] = {}

    def layer_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["layer_hidden"] = hidden.detach().float().cpu()

    handle = model.model.language_model.layers[layer_index].register_forward_hook(layer_hook)
    activations: list[np.ndarray] = []
    try:
        for start in range(0, len(regions), batch_size):
            batch_regions = regions[start : start + batch_size]
            images = [Image.open(region.image_path).convert("RGB") for region in batch_regions]
            texts = []
            for image in images:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Describe this image in one concise sentence."},
                        ],
                    }
                ]
                texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            captured.clear()
            model(**inputs, use_cache=False, return_dict=True)
            hidden = captured["layer_hidden"]
            input_ids = inputs["input_ids"].detach().cpu()
            grids = inputs["image_grid_thw"].detach().cpu().tolist()
            for i, region in enumerate(batch_regions):
                ids = input_ids[i]
                image_positions = (ids == image_pad_id).nonzero(as_tuple=False).flatten()
                grid_t, raw_h, raw_w = [int(x) for x in grids[i]]
                grid_h = raw_h // merge
                grid_w = raw_w // merge
                expected_tokens = grid_t * grid_h * grid_w
                if expected_tokens != len(image_positions):
                    raise RuntimeError(
                        f"image {region.image_id}: grid says {expected_tokens} tokens, got {len(image_positions)}"
                    )
                sample = SimpleNamespace(
                    bbox=region.bbox_xywh,
                    width=region.width,
                    height=region.height,
                )
                selected_indices = bbox_token_indices(sample, grid_h, grid_w, max_bbox_tokens)
                if not selected_indices:
                    selected_indices = [center_token_index(sample, grid_h, grid_w)]
                selected_positions = image_positions[torch.tensor(selected_indices, dtype=torch.long)]
                activation = hidden[i, selected_positions].mean(dim=0).numpy().astype(np.float32)
                activations.append(activation)
    finally:
        handle.remove()
    return activations


def rank_best_region(
    *,
    model,
    processor,
    tokenizer,
    image: Image.Image,
    category: str,
    regions: list[RegionProbe],
    original_margin: float,
    device: torch.device,
) -> tuple[dict | None, list[dict]]:
    masked_records = []
    for region in regions:
        masked = mask_bbox(image, region.bbox_xywh)
        masked_score = score_yes_no(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image=masked,
            category=category,
            device=device,
        )
        margin_drop = original_margin - masked_score["yes_margin"]
        record = {
            "ann_id": region.ann_id,
            "region_category": region.category,
            "bbox_xywh": [float(x) for x in region.bbox_xywh],
            "coarse_position": region.coarse_position,
            "masked_yes_nll": masked_score["yes_nll"],
            "masked_no_nll": masked_score["no_nll"],
            "masked_yes_margin": masked_score["yes_margin"],
            "margin_drop": margin_drop,
        }
        masked_records.append(record)
    if not masked_records:
        return None, []
    best = max(masked_records, key=lambda record: float(record["margin_drop"]))
    return best, masked_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av-parquet", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--activation-adapter", required=True)
    parser.add_argument("--coco-root", default="data/coco2017")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--absent-per-image", type=int, default=3)
    parser.add_argument("--absent-sampling", choices=["random", "cooccurrence"], default="random")
    parser.add_argument("--max-objects-per-image", type=int, default=3)
    parser.add_argument("--max-hallucinations", type=int, default=24)
    parser.add_argument("--min-area-frac", type=float, default=0.015)
    parser.add_argument("--layer-index", type=int, default=None)
    parser.add_argument("--max-bbox-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--injection-scale", type=float, default=57.75)
    parser.add_argument("--num-injection-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4701)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_rows(args.av_parquet)
    seen_images = set()
    image_rows = []
    for row in rows:
        image_id = int(row["coco_image_id"])
        if image_id in seen_images:
            continue
        seen_images.add(image_id)
        image_rows.append(row)
        if len(image_rows) >= args.max_images:
            break

    all_categories, present_by_image, regions_by_image, cooccurrence = load_coco_regions(
        coco_root=Path(args.coco_root),
        image_rows=image_rows,
        min_area_frac=args.min_area_frac,
        max_objects_per_image=args.max_objects_per_image,
    )
    meta = yaml.safe_load(sidecar_path(args.av_parquet).read_text(encoding="utf-8"))
    token_meta = meta["tokens"]
    layer_index = args.layer_index if args.layer_index is not None else int(rows[0]["activation_layer"])

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    absent_probe_records = []
    hallucination_jobs = []
    for row in image_rows:
        image_id = int(row["coco_image_id"])
        present = present_by_image.get(image_id, set())
        absent = choose_absent_categories(
            all_categories=all_categories,
            present=present,
            cooccurrence=cooccurrence,
            sampling=args.absent_sampling,
            count=args.absent_per_image,
            rng=rng,
        )
        image = Image.open(row["source_image_path"]).convert("RGB")
        for category, cooccurrence_score in absent:
            score = score_yes_no(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=image,
                category=category,
                device=device,
            )
            record = {
                "image_id": image_id,
                "image_path": row["source_image_path"],
                "absent_category": category,
                "absent_sampling": args.absent_sampling,
                "cooccurrence_score": int(cooccurrence_score),
                "present_categories": sorted(present),
                "yes_nll": score["yes_nll"],
                "no_nll": score["no_nll"],
                "yes_margin": score["yes_margin"],
                "preferred_answer": score["preferred_answer"],
                "hallucinated_absent": score["yes_margin"] > 0,
            }
            absent_probe_records.append(record)
            if score["yes_margin"] > 0:
                hallucination_jobs.append((row, category, score["yes_margin"]))

    hallucination_jobs.sort(key=lambda item: item[2], reverse=True)
    hallucination_jobs = hallucination_jobs[: args.max_hallucinations]

    attribution_records = []
    top_regions: list[RegionProbe] = []
    top_region_keys: list[tuple[int, int, str, float, dict, list[dict]]] = []
    for row, hallucinated_category, original_margin in hallucination_jobs:
        image_id = int(row["coco_image_id"])
        regions = regions_by_image.get(image_id, [])
        image = Image.open(row["source_image_path"]).convert("RGB")
        best, masked_records = rank_best_region(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image=image,
            category=hallucinated_category,
            regions=regions,
            original_margin=original_margin,
            device=device,
        )
        if best is None:
            continue
        matching_regions = [region for region in regions if region.ann_id == int(best["ann_id"])]
        if not matching_regions:
            continue
        top_region = matching_regions[0]
        top_regions.append(top_region)
        top_region_keys.append((image_id, top_region.ann_id, hallucinated_category, original_margin, best, masked_records))

    top_activations = extract_region_activations(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        regions=top_regions,
        layer_index=layer_index,
        max_bbox_tokens=args.max_bbox_tokens,
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

    for (region, activation, key) in zip(top_regions, top_activations, top_region_keys, strict=True):
        image_id, ann_id, hallucinated_category, original_margin, best, masked_records = key
        actual_response = response_for(region.category, region.coarse_position)
        hallucinated_response = response_for(hallucinated_category, region.coarse_position)
        candidate_response_ids = [encode_response(tokenizer, actual_response), encode_response(tokenizer, hallucinated_response)]
        scores = score_candidates(
            model,
            activation_adapter,
            prompt_ids,
            inj_positions,
            torch.tensor(activation, dtype=torch.float32),
            candidate_response_ids,
            pad_id,
            args.injection_scale,
            args.score_batch_size,
            device,
        )
        actual_nll = float(scores[0].item())
        hallucinated_nll = float(scores[1].item())
        attribution_records.append(
            {
                "image_id": image_id,
                "ann_id": ann_id,
                "hallucinated_category": hallucinated_category,
                "original_yes_margin": original_margin,
                "top_region_category": region.category,
                "top_region_position": region.coarse_position,
                "top_region_bbox_xywh": [float(x) for x in region.bbox_xywh],
                "top_region_margin_drop": float(best["margin_drop"]),
                "masked_region_records": masked_records,
                "nla_actual_response": actual_response,
                "nla_hallucinated_response": hallucinated_response,
                "nla_actual_nll": actual_nll,
                "nla_hallucinated_nll": hallucinated_nll,
                "nla_hallucinated_minus_actual_nll": hallucinated_nll - actual_nll,
                "nla_prefers_hallucinated": hallucinated_nll < actual_nll,
            }
        )

    absent_margins = np.asarray([record["yes_margin"] for record in absent_probe_records], dtype=np.float32)
    hallucinated = [record for record in absent_probe_records if record["hallucinated_absent"]]
    top_drops = np.asarray([record["top_region_margin_drop"] for record in attribution_records], dtype=np.float32)
    nla_deltas = np.asarray(
        [record["nla_hallucinated_minus_actual_nll"] for record in attribution_records],
        dtype=np.float32,
    )
    summary = {
        "av_parquet": args.av_parquet,
        "adapter": args.adapter,
        "activation_adapter": args.activation_adapter,
        "model_id": args.model_id,
        "num_images": len(image_rows),
        "num_absent_probes": len(absent_probe_records),
        "absent_per_image": args.absent_per_image,
        "absent_sampling": args.absent_sampling,
        "max_objects_per_image": args.max_objects_per_image,
        "num_hallucinated_absent_probes": len(hallucinated),
        "hallucination_rate": float(len(hallucinated) / max(1, len(absent_probe_records))),
        "mean_absent_yes_margin": float(absent_margins.mean()) if len(absent_margins) else None,
        "mean_hallucinated_yes_margin": (
            float(np.mean([record["yes_margin"] for record in hallucinated])) if hallucinated else None
        ),
        "num_attributed_hallucinations": len(attribution_records),
        "mean_top_region_margin_drop": float(top_drops.mean()) if len(top_drops) else None,
        "fraction_positive_top_region_margin_drop": float((top_drops > 0).mean()) if len(top_drops) else None,
        "mean_nla_hallucinated_minus_actual_nll": float(nla_deltas.mean()) if len(nla_deltas) else None,
        "fraction_nla_prefers_hallucinated": (
            float(np.mean([record["nla_prefers_hallucinated"] for record in attribution_records]))
            if attribution_records
            else None
        ),
        "absent_probe_records": absent_probe_records,
        "attribution_records": attribution_records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
