#!/usr/bin/env python3
"""Build LLaVA-specific NLA-style AV/AR parquets from extracted activations.

The upstream NLA repo assumes chat-template-driven text-only models and a rare
single-token CJK marker. LLaVA-1.5 uses a Llama tokenizer where those CJK chars
split into byte tokens, but it already has a perfect single-token placeholder:
`<image>`. For LLaVA-NLA we reuse `<image>` as the activation injection marker
inside a text-only prompt. The training/inference hook should overwrite that
token's embedding with the target activation vector, and no `pixel_values` are
passed for the AV prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoProcessor

EXPLANATION_OPEN = "<explanation>"
EXPLANATION_CLOSE = "</explanation>"

DEFAULT_AV_TEMPLATE = """USER: You are a careful interpreter of LLaVA-1.5 internal activations.

The activation vector is inserted at the marker inside <concept> tags:

<concept>{injection_token}</concept>

Describe the visual concept, objects, colors, and spatial information represented by this activation.
Answer only inside <explanation> tags.
ASSISTANT:"""

DEFAULT_AR_TEMPLATE = "Summary of the following visual concept text: <text>{explanation}</text> <summary>"


def wrap_explanation(text: str) -> str:
    return f"{EXPLANATION_OPEN}\n{text}\n{EXPLANATION_CLOSE}"


def _read_activation_table(path: str) -> pa.Table:
    return pq.read_table(path)


def _fixed_list_width(table: pa.Table, col: str = "activation_vector") -> int:
    typ = table.schema.field(col).type
    if hasattr(typ, "list_size"):
        return int(typ.list_size)
    first = table.column(col)[0].as_py()
    return len(first)


def _schema_av(d_model: int) -> pa.Schema:
    return pa.schema(
        [
            ("prompt", pa.string()),
            ("response", pa.string()),
            ("activation_vector", pa.list_(pa.float32(), d_model)),
            ("source_sample_id", pa.string()),
            ("source_image_path", pa.string()),
            ("source_description", pa.string()),
            ("activation_layer", pa.int64()),
            ("target_token", pa.string()),
            ("target_pos", pa.int64()),
        ]
    )


def _schema_ar(d_model: int) -> pa.Schema:
    return pa.schema(
        [
            ("prompt", pa.string()),
            ("activation_vector", pa.list_(pa.float32(), d_model)),
            ("source_sample_id", pa.string()),
            ("source_image_path", pa.string()),
            ("source_description", pa.string()),
            ("activation_layer", pa.int64()),
            ("target_token", pa.string()),
            ("target_pos", pa.int64()),
        ]
    )


def _token_meta(tokenizer, av_prompt: str, injection_token: str) -> dict:
    inj_ids = tokenizer.encode(injection_token, add_special_tokens=False)
    if len(inj_ids) != 1:
        raise RuntimeError(f"injection token {injection_token!r} must tokenize to one id, got {inj_ids}")
    ids = tokenizer.encode(av_prompt, add_special_tokens=True)
    matches = [i for i, tid in enumerate(ids) if tid == inj_ids[0]]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one injection token in AV prompt, found {len(matches)}")
    p = matches[0]
    if not (0 < p < len(ids) - 1):
        raise RuntimeError(f"injection position {p} is at prompt edge")
    return {
        "injection_token": injection_token,
        "injection_token_id": int(inj_ids[0]),
        "injection_left_neighbor_id": int(ids[p - 1]),
        "injection_right_neighbor_id": int(ids[p + 1]),
        "injection_position_in_canonical_prompt": int(p),
        "canonical_prompt_token_count": len(ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--injection-token", default="<image>")
    parser.add_argument("--av-template", default=DEFAULT_AV_TEMPLATE)
    parser.add_argument("--ar-template", default=DEFAULT_AR_TEMPLATE)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = _read_activation_table(args.input_parquet)
    d_model = _fixed_list_width(table)
    rows = table.to_pydict()
    n = table.num_rows
    layer_values = sorted(set(rows["activation_layer"]))
    target_values = sorted(set(rows["target_token"]))
    if len(layer_values) != 1:
        raise RuntimeError(f"expected one activation layer, got {layer_values}")
    if len(target_values) != 1:
        raise RuntimeError(f"expected one target token type, got {target_values}")

    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    tokenizer = processor.tokenizer
    av_prompt = args.av_template.format(injection_token=args.injection_token)
    token_meta = _token_meta(tokenizer, av_prompt, args.injection_token)

    av_cols = {
        "prompt": [av_prompt] * n,
        "response": [wrap_explanation(x) for x in rows["description"]],
        "activation_vector": rows["activation_vector"],
        "source_sample_id": rows["sample_id"],
        "source_image_path": rows["image_path"],
        "source_description": rows["description"],
        "activation_layer": rows["activation_layer"],
        "target_token": rows["target_token"],
        "target_pos": rows["target_pos"],
    }
    ar_cols = {
        "prompt": [args.ar_template.format(explanation=x) for x in rows["description"]],
        "activation_vector": rows["activation_vector"],
        "source_sample_id": rows["sample_id"],
        "source_image_path": rows["image_path"],
        "source_description": rows["description"],
        "activation_layer": rows["activation_layer"],
        "target_token": rows["target_token"],
        "target_pos": rows["target_pos"],
    }

    av_path = out_dir / "av_sft.parquet"
    ar_path = out_dir / "ar_sft.parquet"
    pq.write_table(pa.table(av_cols, schema=_schema_av(d_model)), av_path)
    pq.write_table(pa.table(ar_cols, schema=_schema_ar(d_model)), ar_path)

    sidecar = {
        "kind": "llava_nla_dataset",
        "schema_version": 1,
        "base_model": args.model_id,
        "d_model": d_model,
        "activation_layer": int(layer_values[0]),
        "target_token": target_values[0],
        "row_count": n,
        "activation_norm": "none",
        "tokens": token_meta,
        "prompt_templates": {
            "av": args.av_template,
            "ar": args.ar_template,
        },
        "source_parquet": args.input_parquet,
    }
    for path in (av_path, ar_path):
        Path(str(path) + ".nla_meta.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False), encoding="utf-8")

    summary = {
        "input_parquet": args.input_parquet,
        "av_path": str(av_path),
        "ar_path": str(ar_path),
        "row_count": n,
        "d_model": d_model,
        "activation_layer": int(layer_values[0]),
        "target_token": target_values[0],
        "token_meta": token_meta,
        "first_response": av_cols["response"][0],
        "first_ar_prompt": ar_cols["prompt"][0],
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

