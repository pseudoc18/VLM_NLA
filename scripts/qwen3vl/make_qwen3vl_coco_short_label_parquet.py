#!/usr/bin/env python3
"""Rewrite COCO object-token AV parquets to category+region-only labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def sidecar_path(path: str | Path) -> Path:
    return Path(str(path) + ".nla_meta.yaml")


def short_description(row: dict) -> str:
    subject = "This image token" if row["target_token"] == "object_center" else "These image tokens"
    verb = "represents" if row["target_token"] == "object_center" else "represent"
    article = "an" if row["coco_category"][0].lower() in "aeiou" else "a"
    return f"{subject} {verb} {article} {row['coco_category']} in the {row['coarse_position']} region of the image."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    table = pq.read_table(src)
    rows = table.to_pylist()
    for row in rows:
        desc = short_description(row)
        row["source_description"] = desc
        row["response"] = f"<explanation>\n{desc}\n</explanation>"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = rows[0]["target_token"]
    layer = int(rows[0]["activation_layer"])
    out_path = out_dir / f"qwen3vl_coco_L{layer}_{target}_short_av_sft.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), out_path)

    sidecar = yaml.safe_load(sidecar_path(src).read_text(encoding="utf-8"))
    sidecar["label_variant"] = "category_region_short"
    sidecar_path(out_path).write_text(yaml.safe_dump(sidecar, sort_keys=False), encoding="utf-8")

    summary = {
        "source_parquet": str(src),
        "parquet_path": str(out_path),
        "num_rows": len(rows),
        "target_token": target,
        "label_variant": "category_region_short",
        "first_response": rows[0]["response"],
    }
    (out_dir / f"qwen3vl_coco_L{layer}_{target}_short_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
