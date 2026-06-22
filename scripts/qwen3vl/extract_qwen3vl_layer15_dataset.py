#!/usr/bin/env python3
"""Extract a small Qwen3-VL layer activation dataset for NLA experiments."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 45, 45),
    "green": (45, 170, 75),
    "blue": (45, 105, 220),
    "yellow": (245, 205, 45),
    "purple": (145, 75, 210),
    "orange": (235, 135, 35),
}
SHAPES = ("square", "circle", "triangle")
POSITIONS = ("left", "right", "top", "bottom", "center")


@dataclass(frozen=True)
class ObjectSpec:
    color: str
    shape: str
    position: str


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    objects: tuple[ObjectSpec, ...]
    description: str
    image_path: str


def box_for(position: str) -> tuple[int, int, int, int]:
    return {
        "left": (48, 118, 138, 208),
        "right": (198, 118, 288, 208),
        "top": (123, 48, 213, 138),
        "bottom": (123, 198, 213, 288),
        "center": (123, 123, 213, 213),
    }[position]


def draw_shape(draw: ImageDraw.ImageDraw, obj: ObjectSpec) -> None:
    box = box_for(obj.position)
    fill = COLORS[obj.color]
    outline = tuple(max(0, c - 80) for c in fill)
    if obj.shape == "square":
        draw.rectangle(box, fill=fill, outline=outline, width=5)
    elif obj.shape == "circle":
        draw.ellipse(box, fill=fill, outline=outline, width=5)
    elif obj.shape == "triangle":
        x0, y0, x1, y1 = box
        points = [(x0 + x1) // 2, y0], [x0, y1], [x1, y1]
        draw.polygon(points, fill=fill, outline=outline)
        draw.line([points[0], points[1], points[2], points[0]], fill=outline, width=5)
    else:
        raise AssertionError(obj.shape)


def make_description(objects: tuple[ObjectSpec, ...]) -> str:
    parts = [f"a {obj.color} {obj.shape} on the {obj.position}" for obj in objects]
    if len(parts) == 1:
        return f"The image shows {parts[0]}."
    return f"The image shows {parts[0]} and {parts[1]}."


def build_specs(out_dir: Path, num_samples: int, seed: int) -> list[SampleSpec]:
    rng = random.Random(seed)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    specs: list[SampleSpec] = []
    used: set[tuple[tuple[str, str, str], ...]] = set()
    attempts = 0
    while len(specs) < num_samples:
        attempts += 1
        if attempts > num_samples * 100:
            raise RuntimeError("could not generate enough unique samples")
        n_objects = 1 if rng.random() < 0.35 else 2
        positions = rng.sample(list(POSITIONS), k=n_objects)
        objects = tuple(
            ObjectSpec(
                color=rng.choice(list(COLORS)),
                shape=rng.choice(list(SHAPES)),
                position=positions[i],
            )
            for i in range(n_objects)
        )
        key = tuple((obj.color, obj.shape, obj.position) for obj in objects)
        if key in used:
            continue
        used.add(key)
        sample_id = f"synthetic_{len(specs):05d}"
        image_path = images_dir / f"{sample_id}.png"
        image = Image.new("RGB", (336, 336), "white")
        draw = ImageDraw.Draw(image)
        for obj in objects:
            draw_shape(draw, obj)
        draw.text((12, 310), make_description(objects)[:65], fill=(0, 0, 0))
        image.save(image_path)
        specs.append(SampleSpec(sample_id, objects, make_description(objects), str(image_path)))
    return specs


def schema(d_model: int) -> pa.Schema:
    return pa.schema(
        [
            ("prompt", pa.string()),
            ("response", pa.string()),
            ("activation_vector", pa.list_(pa.float32(), d_model)),
            ("source_sample_id", pa.string()),
            ("source_image_path", pa.string()),
            ("source_description", pa.string()),
            ("objects_json", pa.string()),
            ("activation_layer", pa.int64()),
            ("target_token", pa.string()),
            ("target_pos", pa.int64()),
            ("num_image_tokens", pa.int64()),
            ("image_token_first_pos", pa.int64()),
            ("image_token_last_pos", pa.int64()),
            ("image_grid_thw", pa.list_(pa.int64(), 3)),
        ]
    )


def av_prompt(num_injection_tokens: int = 1) -> str:
    return (
        "<|im_start|>user\n"
        "You are a careful interpreter of Qwen3-VL internal activations.\n\n"
        "The activation is inserted here: <|vision_start|>"
        + ("<|image_pad|>" * num_injection_tokens)
        + "<|vision_end|>\n"
        "Explain the visual concept encoded by this activation inside <explanation> tags."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layer-index", type=int, default=15)
    parser.add_argument("--target-token", choices=["last_prompt", "image", "image_mean"], default="last_prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(out_dir, args.num_samples, args.seed)

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

    rows: dict[str, list] = {name: [] for name in schema(d_model).names}
    captured: dict[str, torch.Tensor] = {}

    def layer_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["layer_hidden"] = hidden.detach().float().cpu()

    handle = model.model.language_model.layers[args.layer_index].register_forward_hook(layer_hook)
    try:
        for start in range(0, len(specs), args.batch_size):
            batch_specs = specs[start : start + args.batch_size]
            images = [Image.open(spec.image_path).convert("RGB") for spec in batch_specs]
            texts = []
            for image in images:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Describe the visible shapes, colors, and positions."},
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
            attention_mask = inputs["attention_mask"].detach().cpu()
            grids = inputs["image_grid_thw"].detach().cpu().tolist()

            for i, spec in enumerate(batch_specs):
                ids = input_ids[i]
                image_positions = (ids == image_pad_id).nonzero(as_tuple=False).flatten()
                if len(image_positions) == 0:
                    raise RuntimeError(f"{spec.sample_id}: no image placeholders")
                if args.target_token == "image":
                    target_pos = int(image_positions[len(image_positions) // 2])
                    activation = hidden[i, target_pos].numpy().astype(np.float32)
                elif args.target_token == "image_mean":
                    target_pos = int(image_positions[len(image_positions) // 2])
                    activation = hidden[i, image_positions].mean(dim=0).numpy().astype(np.float32)
                else:
                    target_pos = int(attention_mask[i].sum().item() - 1)
                    activation = hidden[i, target_pos].numpy().astype(np.float32)

                rows["prompt"].append(av_prompt(num_injection_tokens=1))
                rows["response"].append(f"<explanation>\n{spec.description}\n</explanation>")
                rows["activation_vector"].append(activation.tolist())
                rows["source_sample_id"].append(spec.sample_id)
                rows["source_image_path"].append(spec.image_path)
                rows["source_description"].append(spec.description)
                rows["objects_json"].append(json.dumps([obj.__dict__ for obj in spec.objects], sort_keys=True))
                rows["activation_layer"].append(args.layer_index)
                rows["target_token"].append(args.target_token)
                rows["target_pos"].append(target_pos)
                rows["num_image_tokens"].append(int(len(image_positions)))
                rows["image_token_first_pos"].append(int(image_positions[0]))
                rows["image_token_last_pos"].append(int(image_positions[-1]))
                rows["image_grid_thw"].append([int(x) for x in grids[i]])
    finally:
        handle.remove()

    out_path = out_dir / f"qwen3vl_L{args.layer_index}_{args.target_token}_av_sft.parquet"
    table = pa.table(rows, schema=schema(d_model))
    pq.write_table(table, out_path)
    activations = np.asarray(rows["activation_vector"], dtype=np.float32)
    summary = {
        "model_id": args.model_id,
        "num_samples": len(specs),
        "d_model": d_model,
        "layer_index": args.layer_index,
        "target_token": args.target_token,
        "parquet_path": str(out_path),
        "activation_norm_mean": float(np.linalg.norm(activations, axis=1).mean()),
        "activation_norm_std": float(np.linalg.norm(activations, axis=1).std()),
        "num_image_tokens_unique": sorted(set(rows["num_image_tokens"])),
        "target_pos_unique_first10": sorted(set(rows["target_pos"]))[:10],
        "image_pad_token_id": image_pad_id,
        "vision_start_token_id": vision_start_id,
        "vision_end_token_id": vision_end_id,
        "first_response": rows["response"][0],
    }
    (out_dir / f"qwen3vl_L{args.layer_index}_{args.target_token}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    sidecar = {
        "kind": "qwen3vl_nla_av_sft",
        "schema_version": 1,
        "base_model": args.model_id,
        "d_model": d_model,
        "activation_layer": args.layer_index,
        "target_token": args.target_token,
        "row_count": len(specs),
        "tokens": {
            "injection_token": "<|image_pad|>",
            "injection_token_id": image_pad_id,
            "vision_start_token_id": vision_start_id,
            "vision_end_token_id": vision_end_id,
        },
        "note": "Prompt stores one canonical <|image_pad|>; AV training should expand it to N placeholders.",
    }
    (Path(str(out_path) + ".nla_meta.yaml")).write_text(yaml.safe_dump(sidecar, sort_keys=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
