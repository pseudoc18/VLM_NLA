#!/usr/bin/env python3
"""Build a multi-layer Qwen3-VL AV parquet by concatenating aligned layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def sidecar_path(path: str | Path) -> Path:
    return Path(str(path) + ".nla_meta.yaml")


def layer_tag(layers: list[int]) -> str:
    return "L" + "-L".join(str(layer) for layer in layers)


def infer_layer(path: Path, rows: list[dict]) -> int:
    meta_path = sidecar_path(path)
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        if "layer_index" in meta:
            return int(meta["layer_index"])
    return int(rows[0]["activation_layer"])


def replace_schema_activation_dim(schema: pa.Schema, activation_dim: int) -> pa.Schema:
    fields = []
    for field in schema:
        if field.name == "activation_vector":
            fields.append(pa.field("activation_vector", pa.list_(pa.float32(), activation_dim)))
        else:
            fields.append(field)
    return pa.schema(fields, metadata=schema.metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", nargs="+", required=True, help="Layer parquets to concatenate in the given order.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--require-identical-responses", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    src_paths = [Path(path) for path in args.src]
    tables = [pq.read_table(path) for path in src_paths]
    rows_by_layer = [table.to_pylist() for table in tables]
    layers = [infer_layer(path, rows) for path, rows in zip(src_paths, rows_by_layer, strict=True)]
    indexed = [{row["source_sample_id"]: row for row in rows} for rows in rows_by_layer]

    base_rows = rows_by_layer[0]
    out_rows = []
    missing = []
    mismatched_response = []
    for base in base_rows:
        sample_id = base["source_sample_id"]
        aligned = []
        for layer_map in indexed:
            row = layer_map.get(sample_id)
            if row is None:
                missing.append(sample_id)
                break
            aligned.append(row)
        if len(aligned) != len(indexed):
            continue
        if args.require_identical_responses:
            responses = {row["response"] for row in aligned}
            if len(responses) != 1:
                mismatched_response.append(sample_id)
                continue
        row = dict(base)
        vectors = [np.asarray(layer_row["activation_vector"], dtype=np.float32) for layer_row in aligned]
        row["activation_vector"] = np.concatenate(vectors, axis=0).astype(np.float32).tolist()
        row["activation_layer"] = -1
        row["source_description"] = base.get("source_description", "")
        out_rows.append(row)

    if missing:
        raise RuntimeError(f"{len(missing)} sample ids missing from at least one layer; first: {missing[:5]}")
    if mismatched_response:
        raise RuntimeError(f"{len(mismatched_response)} responses differ across layers; first: {mismatched_response[:5]}")
    if not out_rows:
        raise RuntimeError("no aligned rows written")

    activation_dim = len(out_rows[0]["activation_vector"])
    hidden_size = len(rows_by_layer[0][0]["activation_vector"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = str(out_rows[0]["target_token"])
    if args.out_name:
        out_name = args.out_name
    else:
        out_name = f"qwen3vl_coco_{layer_tag(layers)}_{target}_multilayer_av_sft.parquet"
    out_path = out_dir / out_name
    schema = replace_schema_activation_dim(tables[0].schema, activation_dim)
    pq.write_table(pa.Table.from_pylist(out_rows, schema=schema), out_path)

    norms = np.linalg.norm(np.asarray([row["activation_vector"] for row in out_rows], dtype=np.float32), axis=1)
    first_meta = yaml.safe_load(sidecar_path(src_paths[0]).read_text(encoding="utf-8"))
    sidecar = dict(first_meta)
    sidecar.update(
        {
            "format": "nla_av_sft_multilayer_concat",
            "source_parquets": [str(path) for path in src_paths],
            "layer_indices": layers,
            "layer_tag": layer_tag(layers),
            "activation": {
                "d_model": activation_dim,
                "hidden_size": hidden_size,
                "num_layers": len(layers),
                "norm_mean": float(norms.mean()),
                "norm_std": float(norms.std()),
            },
        }
    )
    sidecar_path(out_path).write_text(yaml.safe_dump(sidecar, sort_keys=False), encoding="utf-8")
    summary = {
        "source_parquets": [str(path) for path in src_paths],
        "parquet_path": str(out_path),
        "num_rows": len(out_rows),
        "layers": layers,
        "target_token": target,
        "activation_dim": activation_dim,
        "hidden_size": hidden_size,
        "first_response": out_rows[0]["response"],
    }
    (out_dir / (out_path.stem + "_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
