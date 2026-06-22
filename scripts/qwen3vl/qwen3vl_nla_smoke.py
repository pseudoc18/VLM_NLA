#!/usr/bin/env python3
"""Qwen3-VL NLA mechanism smoke test.

This checks whether Qwen3-VL exposes the same core path needed by NLA:

1. Run a normal image prompt and capture a language-layer activation.
2. Build a text-only AV prompt containing repeated image placeholder tokens.
3. Replace those placeholder embeddings with the captured activation.
4. Run the language model forward through `inputs_embeds`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def make_synthetic_image(path: Path) -> Image.Image:
    image = Image.new("RGB", (192, 192), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((38, 58, 98, 118), fill=(35, 160, 70), outline=(0, 95, 35), width=4)
    draw.polygon([(130, 52), (96, 130), (164, 130)], fill=(50, 100, 220), outline=(15, 45, 150))
    draw.text((18, 160), "green square and blue triangle", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return image


def finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layer-index", type=int, default=15)
    parser.add_argument("--num-injection-tokens", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "qwen3vl_nla_smoke_image.png"
    image = make_synthetic_image(image_path)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer = processor.tokenizer
    image_pad_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    vision_start_id = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    vision_end_id = int(tokenizer.convert_tokens_to_ids("<|vision_end|>"))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the image in one concise sentence."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_trimmed = generated[:, inputs["input_ids"].shape[1] :]
        generated_text = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Qwen3-VL did not return hidden_states")
    # hidden_states[0] is embedding output; layer 15 output is index 16.
    layer_hidden = hidden_states[args.layer_index + 1]
    target_pos = int(inputs["input_ids"].shape[1] - 1)
    activation = layer_hidden[0, target_pos].detach().float().cpu()
    activation_norm = float(activation.norm().item())

    input_ids_list = inputs["input_ids"][0].detach().cpu().tolist()
    image_positions = [i for i, token_id in enumerate(input_ids_list) if int(token_id) == image_pad_id]

    av_prompt = (
        "<|im_start|>user\n"
        "You are interpreting an internal Qwen3-VL activation. "
        "The activation is inserted in the visual placeholder positions: "
        "<|vision_start|>"
        + ("<|image_pad|>" * args.num_injection_tokens)
        + "<|vision_end|>"
        " Explain the visual concept in one concise sentence."
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    av_ids = tokenizer.encode(av_prompt, add_special_tokens=False)
    av_positions = [i for i, token_id in enumerate(av_ids) if int(token_id) == image_pad_id]
    if len(av_positions) != args.num_injection_tokens:
        raise RuntimeError(f"expected {args.num_injection_tokens} AV placeholders, got {len(av_positions)}")

    av_input_ids = torch.tensor([av_ids], dtype=torch.long, device=device)
    av_attention_mask = torch.ones_like(av_input_ids, device=device)
    base = model.model
    with torch.no_grad():
        av_embeds = base.get_input_embeddings()(av_input_ids)
        mapped = activation.to(device=device, dtype=av_embeds.dtype).view(1, 1, -1)
        mapped = mapped.expand(1, args.num_injection_tokens, -1) / math.sqrt(args.num_injection_tokens)
        av_embeds = av_embeds.clone()
        av_embeds[:, av_positions, :] = mapped
        av_outputs = base(
            input_ids=None,
            inputs_embeds=av_embeds,
            attention_mask=av_attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        logits = model.lm_head(av_outputs.last_hidden_state[:, -1:, :])
        next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())

    summary = {
        "model_id": args.model_id,
        "model_class": type(model).__name__,
        "text_num_hidden_layers": int(model.config.text_config.num_hidden_layers),
        "text_hidden_size": int(model.config.text_config.hidden_size),
        "layer_index": args.layer_index,
        "target_pos": target_pos,
        "activation_shape": list(activation.shape),
        "activation_norm": activation_norm,
        "image_path": str(image_path),
        "generated_text": generated_text,
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "image_pad_token_id": image_pad_id,
        "vision_start_token_id": vision_start_id,
        "vision_end_token_id": vision_end_id,
        "image_pad_count": len(image_positions),
        "first_image_positions": image_positions[:16],
        "image_grid_thw": inputs.get("image_grid_thw").detach().cpu().tolist() if "image_grid_thw" in inputs else None,
        "pixel_values_shape": list(inputs["pixel_values"].shape) if "pixel_values" in inputs else None,
        "av_prompt_token_count": len(av_ids),
        "num_injection_tokens": args.num_injection_tokens,
        "av_injection_positions": av_positions,
        "av_forward_last_hidden_shape": list(av_outputs.last_hidden_state.shape),
        "finite_original_logits": finite_tensor(outputs.logits),
        "finite_av_logits": finite_tensor(logits),
        "av_next_token_id": next_token,
        "av_next_token": tokenizer.decode([next_token], skip_special_tokens=False),
    }
    (out_dir / "qwen3vl_nla_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(activation, out_dir / f"qwen3vl_L{args.layer_index}_last_prompt_activation.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
