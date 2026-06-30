#!/usr/bin/env python3
"""Extract Qwen3-VL activations for specific COCO image tokens.

This dataset is meant to test a more local NLA question than whole-image
`image_mean`: can an AV explain one selected image token, or a small group of
image tokens covering a COCO object?
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VAL_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"


@dataclass(frozen=True)
class CocoObjectSample:
    sample_id: str
    image_id: int
    ann_id: int
    file_name: str
    image_path: str
    width: int
    height: int
    category: str
    bbox: tuple[float, float, float, float]
    area: float
    coarse_position: str
    caption: str
    description: str


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"downloading {url} -> {path}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def ensure_coco(root: Path) -> tuple[Path, Path, Path]:
    ann_dir = root / "annotations"
    instances = ann_dir / "instances_val2017.json"
    captions = ann_dir / "captions_val2017.json"
    if instances.exists() and captions.exists():
        return instances, captions, root / "val2017"

    zip_path = root / "annotations_trainval2017.zip"
    download_file(ANNOTATIONS_URL, zip_path)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in ["annotations/instances_val2017.json", "annotations/captions_val2017.json"]:
            zf.extract(member, root)
    return instances, captions, root / "val2017"


def coarse_position(cx: float, cy: float, width: float, height: float) -> str:
    x_band = "left" if cx < width / 3 else "right" if cx > 2 * width / 3 else "center"
    y_band = "top" if cy < height / 3 else "bottom" if cy > 2 * height / 3 else "middle"
    if x_band == "center" and y_band == "middle":
        return "center"
    if x_band == "center":
        return y_band
    if y_band == "middle":
        return x_band
    return f"{y_band}-{x_band}"


def build_description(mode: str, category: str, position: str, caption: str) -> str:
    if mode == "object_center":
        return (
            f"This image token represents a {category} in the {position} region of the image. "
            f"The full COCO caption is: {caption}"
        )
    return (
        f"These image tokens represent a {category} in the {position} region of the image. "
        f"The full COCO caption is: {caption}"
    )


def load_image_id_list(path: Path, key: str) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        if key in payload:
            values = payload[key]
        elif "splits" in payload and key in payload["splits"]:
            values = payload["splits"][key]
        else:
            raise KeyError(f"{path} does not contain key {key!r} or splits.{key}")
    else:
        raise TypeError(f"unsupported image-id manifest format: {type(payload).__name__}")
    return [int(x) for x in values]


def load_samples(
    *,
    coco_root: Path,
    out_dir: Path,
    num_samples: int,
    seed: int,
    min_area_frac: float,
    target_mode: str,
    image_ids: list[int] | None = None,
) -> list[CocoObjectSample]:
    instances_path, captions_path, image_dir = ensure_coco(coco_root)
    image_dir.mkdir(parents=True, exist_ok=True)
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    captions = json.loads(captions_path.read_text(encoding="utf-8"))

    categories = {int(cat["id"]): cat["name"] for cat in instances["categories"]}
    images = {int(img["id"]): img for img in instances["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in instances["annotations"]:
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        image = images.get(int(ann["image_id"]))
        if not image:
            continue
        if float(ann.get("area", 0.0)) < min_area_frac * float(image["width"] * image["height"]):
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for cap in captions["annotations"]:
        captions_by_image[int(cap["image_id"])].append(" ".join(str(cap["caption"]).split()))

    rng = random.Random(seed)
    if image_ids is None:
        candidate_image_ids = list(anns_by_image)
    else:
        candidate_image_ids = [image_id for image_id in image_ids if image_id in anns_by_image]
    rng.shuffle(candidate_image_ids)
    samples: list[CocoObjectSample] = []
    used_categories = Counter()
    for image_id in candidate_image_ids:
        image = images[image_id]
        file_name = image["file_name"]
        image_path = image_dir / file_name
        try:
            download_file(VAL_IMAGE_URL.format(file_name=file_name), image_path)
            with Image.open(image_path) as img:
                img.verify()
        except Exception as exc:
            print(f"skip image {image_id}: {exc}")
            continue

        anns = sorted(anns_by_image[image_id], key=lambda ann: float(ann.get("area", 0.0)), reverse=True)
        # Prefer a prominent object, while avoiding a subset dominated by one class.
        chosen = None
        for ann in anns[:5]:
            cat = categories[int(ann["category_id"])]
            if used_categories[cat] < max(3, num_samples // 20):
                chosen = ann
                break
        if chosen is None:
            chosen = anns[0]
        category = categories[int(chosen["category_id"])]
        used_categories[category] += 1
        x, y, w, h = [float(v) for v in chosen["bbox"]]
        cx, cy = x + w / 2, y + h / 2
        position = coarse_position(cx, cy, float(image["width"]), float(image["height"]))
        caption = captions_by_image[image_id][0] if captions_by_image[image_id] else f"A photo containing a {category}."
        description = build_description(target_mode, category, position, caption)
        samples.append(
            CocoObjectSample(
                sample_id=f"coco_{image_id:012d}_{int(chosen['id'])}",
                image_id=image_id,
                ann_id=int(chosen["id"]),
                file_name=file_name,
                image_path=str(image_path),
                width=int(image["width"]),
                height=int(image["height"]),
                category=category,
                bbox=(x, y, w, h),
                area=float(chosen["area"]),
                coarse_position=position,
                caption=caption,
                description=description,
            )
        )
        if len(samples) >= num_samples:
            break

    if len(samples) < num_samples:
        raise RuntimeError(f"only built {len(samples)} samples, requested {num_samples}")

    selected_path = out_dir / "selected_coco_objects.json"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(
        json.dumps([sample.__dict__ for sample in samples], indent=2),
        encoding="utf-8",
    )
    return samples


def av_prompt(num_injection_tokens: int = 1) -> str:
    return (
        "<|im_start|>user\n"
        "You are interpreting selected Qwen3-VL image-token activations from a real COCO image.\n\n"
        "The activation is inserted here: <|vision_start|>"
        + ("<|image_pad|>" * num_injection_tokens)
        + "<|vision_end|>\n"
        "Explain what the selected image token or local token group represents inside <explanation> tags."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def schema(d_model: int) -> pa.Schema:
    return pa.schema(
        [
            ("prompt", pa.string()),
            ("response", pa.string()),
            ("activation_vector", pa.list_(pa.float32(), d_model)),
            ("source_sample_id", pa.string()),
            ("source_image_path", pa.string()),
            ("source_description", pa.string()),
            ("coco_image_id", pa.int64()),
            ("coco_ann_id", pa.int64()),
            ("coco_category", pa.string()),
            ("coco_caption", pa.string()),
            ("bbox_xywh", pa.list_(pa.float32(), 4)),
            ("coarse_position", pa.string()),
            ("activation_layer", pa.int64()),
            ("target_token", pa.string()),
            ("target_pos", pa.int64()),
            ("target_image_token_index", pa.int64()),
            ("selected_image_token_indices", pa.string()),
            ("num_selected_image_tokens", pa.int64()),
            ("num_image_tokens", pa.int64()),
            ("image_token_first_pos", pa.int64()),
            ("image_token_last_pos", pa.int64()),
            ("image_grid_thw", pa.list_(pa.int64(), 3)),
            ("token_grid_hw", pa.list_(pa.int64(), 2)),
        ]
    )


def center_token_index(sample: CocoObjectSample, grid_h: int, grid_w: int) -> int:
    x, y, w, h = sample.bbox
    cx, cy = x + w / 2, y + h / 2
    gx = min(grid_w - 1, max(0, int(cx / sample.width * grid_w)))
    gy = min(grid_h - 1, max(0, int(cy / sample.height * grid_h)))
    return gy * grid_w + gx


def bbox_token_indices(sample: CocoObjectSample, grid_h: int, grid_w: int, max_tokens: int) -> list[int]:
    x, y, w, h = sample.bbox
    x1, y1 = x + w, y + h
    indices: list[int] = []
    for gy in range(grid_h):
        cy = (gy + 0.5) / grid_h * sample.height
        for gx in range(grid_w):
            cx = (gx + 0.5) / grid_w * sample.width
            if x <= cx <= x1 and y <= cy <= y1:
                indices.append(gy * grid_w + gx)
    center = center_token_index(sample, grid_h, grid_w)
    if not indices:
        return [center]
    if len(indices) <= max_tokens:
        return indices
    center_y, center_x = divmod(center, grid_w)
    indices.sort(key=lambda idx: (divmod(idx, grid_w)[0] - center_y) ** 2 + (divmod(idx, grid_w)[1] - center_x) ** 2)
    return indices[:max_tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--coco-root", default="data/coco2017")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--layer-index", type=int, default=15)
    parser.add_argument("--target-token", choices=["object_center", "object_bbox_mean"], default="object_center")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--min-area-frac", type=float, default=0.015)
    parser.add_argument("--max-bbox-tokens", type=int, default=8)
    parser.add_argument("--image-ids-json", default=None, help="Optional JSON list or split manifest for fixed image IDs.")
    parser.add_argument("--image-ids-key", default="train", help="Key to read when --image-ids-json is a split manifest.")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_ids = load_image_id_list(Path(args.image_ids_json), args.image_ids_key) if args.image_ids_json else None
    samples = load_samples(
        coco_root=Path(args.coco_root),
        out_dir=out_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        min_area_frac=args.min_area_frac,
        target_mode=args.target_token,
        image_ids=image_ids,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    image_pad_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    vision_start_id = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    vision_end_id = int(tokenizer.convert_tokens_to_ids("<|vision_end|>"))

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    d_model = int(model.config.text_config.hidden_size)
    merge = int(getattr(model.config.vision_config, "spatial_merge_size", 2))

    rows: dict[str, list] = {name: [] for name in schema(d_model).names}
    captured: dict[str, torch.Tensor] = {}

    def layer_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["layer_hidden"] = hidden.detach().float().cpu()

    handle = model.model.language_model.layers[args.layer_index].register_forward_hook(layer_hook)
    try:
        for start in range(0, len(samples), args.batch_size):
            batch_samples = samples[start : start + args.batch_size]
            images = [Image.open(sample.image_path).convert("RGB") for sample in batch_samples]
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
            with torch.no_grad():
                model(**inputs, use_cache=False, return_dict=True)
            hidden = captured["layer_hidden"]
            input_ids = inputs["input_ids"].detach().cpu()
            grids = inputs["image_grid_thw"].detach().cpu().tolist()

            for i, sample in enumerate(batch_samples):
                ids = input_ids[i]
                image_positions = (ids == image_pad_id).nonzero(as_tuple=False).flatten()
                if len(image_positions) == 0:
                    raise RuntimeError(f"{sample.sample_id}: no image placeholders")
                grid_t, raw_h, raw_w = [int(x) for x in grids[i]]
                grid_h = raw_h // merge
                grid_w = raw_w // merge
                expected_tokens = grid_t * grid_h * grid_w
                if expected_tokens != len(image_positions):
                    raise RuntimeError(
                        f"{sample.sample_id}: grid says {expected_tokens} tokens, got {len(image_positions)}"
                    )

                center_idx = center_token_index(sample, grid_h, grid_w)
                if args.target_token == "object_center":
                    selected_indices = [center_idx]
                else:
                    selected_indices = bbox_token_indices(sample, grid_h, grid_w, args.max_bbox_tokens)
                selected_positions = image_positions[torch.tensor(selected_indices, dtype=torch.long)]
                activation = hidden[i, selected_positions].mean(dim=0).numpy().astype(np.float32)
                target_pos = int(selected_positions[0])

                rows["prompt"].append(av_prompt(num_injection_tokens=1))
                rows["response"].append(f"<explanation>\n{sample.description}\n</explanation>")
                rows["activation_vector"].append(activation.tolist())
                rows["source_sample_id"].append(sample.sample_id)
                rows["source_image_path"].append(sample.image_path)
                rows["source_description"].append(sample.description)
                rows["coco_image_id"].append(sample.image_id)
                rows["coco_ann_id"].append(sample.ann_id)
                rows["coco_category"].append(sample.category)
                rows["coco_caption"].append(sample.caption)
                rows["bbox_xywh"].append([float(x) for x in sample.bbox])
                rows["coarse_position"].append(sample.coarse_position)
                rows["activation_layer"].append(args.layer_index)
                rows["target_token"].append(args.target_token)
                rows["target_pos"].append(target_pos)
                rows["target_image_token_index"].append(int(center_idx))
                rows["selected_image_token_indices"].append(json.dumps([int(x) for x in selected_indices]))
                rows["num_selected_image_tokens"].append(int(len(selected_indices)))
                rows["num_image_tokens"].append(int(len(image_positions)))
                rows["image_token_first_pos"].append(int(image_positions[0]))
                rows["image_token_last_pos"].append(int(image_positions[-1]))
                rows["image_grid_thw"].append([int(x) for x in grids[i]])
                rows["token_grid_hw"].append([int(grid_h), int(grid_w)])
    finally:
        handle.remove()

    out_path = out_dir / f"qwen3vl_coco_L{args.layer_index}_{args.target_token}_av_sft.parquet"
    table = pa.table(rows, schema=schema(d_model))
    pq.write_table(table, out_path)
    activations = np.asarray(rows["activation_vector"], dtype=np.float32)
    norms = np.linalg.norm(activations, axis=1)

    sidecar = {
        "format": "nla_av_sft",
        "source": "mscoco_val2017_object_tokens",
        "model_id": args.model_id,
        "layer_index": args.layer_index,
        "target_token": args.target_token,
        "image_ids_json": args.image_ids_json,
        "image_ids_key": args.image_ids_key if args.image_ids_json else None,
        "tokens": {
            "injection_token": "<|image_pad|>",
            "injection_token_id": image_pad_id,
            "vision_start_token_id": vision_start_id,
            "vision_end_token_id": vision_end_id,
        },
        "activation": {
            "d_model": d_model,
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
        },
    }
    sidecar_path = Path(str(out_path) + ".nla_meta.yaml")
    sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False), encoding="utf-8")

    summary = {
        "model_id": args.model_id,
        "num_samples": len(samples),
        "d_model": d_model,
        "layer_index": args.layer_index,
        "target_token": args.target_token,
        "image_ids_json": args.image_ids_json,
        "image_ids_key": args.image_ids_key if args.image_ids_json else None,
        "parquet_path": str(out_path),
        "activation_norm_mean": float(norms.mean()),
        "activation_norm_std": float(norms.std()),
        "num_image_tokens_unique": sorted(set(int(x) for x in rows["num_image_tokens"])),
        "num_selected_image_tokens_unique": sorted(set(int(x) for x in rows["num_selected_image_tokens"])),
        "token_grid_hw_first10": rows["token_grid_hw"][:10],
        "category_counts_top20": Counter(rows["coco_category"]).most_common(20),
        "first_response": rows["response"][0],
    }
    (out_dir / f"qwen3vl_coco_L{args.layer_index}_{args.target_token}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
