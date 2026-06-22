#!/usr/bin/env python3
"""Make visual panels for Qwen3-VL COCO object-token NLA experiments."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


TAG_RE = re.compile(r"</?explanation>")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(TAG_RE.sub("", text).split())


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def wrap_lines(text: str, width: int) -> list[str]:
    out: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        out.extend(textwrap.wrap(paragraph, width=width, break_long_words=False) or [""])
    return out


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    max_chars: int,
    font_obj: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_spacing: int = 5,
    max_lines: int | None = None,
) -> int:
    x, y = xy
    lines = wrap_lines(text, max_chars)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "..."
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font_obj)
        bbox = draw.textbbox((x, y), line or " ", font=font_obj)
        y += bbox[3] - bbox[1] + line_spacing
    return y


def rows_by_id(parquet_path: str | Path) -> dict[str, dict]:
    rows = pq.read_table(parquet_path).to_pylist()
    return {row["source_sample_id"]: row for row in rows}


def metric_row(label: str, train_summary: dict, sens: dict, rank: dict, split_note: str) -> dict:
    return {
        "label": label,
        "eval_loss": train_summary.get("final_eval_loss_mean"),
        "sens_delta": sens.get("shuffled_minus_matched_mean"),
        "match_better": sens.get("shuffled_match_better_fraction"),
        "mean_rank": rank.get("mean_rank"),
        "top1": rank.get("top1_accuracy"),
        "top5": rank.get("top5_accuracy"),
        "note": split_note,
    }


def metric_row_eval_only(label: str, sens: dict, rank: dict, split_note: str) -> dict:
    return {
        "label": label,
        "eval_loss": None,
        "sens_delta": sens.get("shuffled_minus_matched_mean"),
        "match_better": sens.get("shuffled_match_better_fraction"),
        "mean_rank": rank.get("mean_rank"),
        "top1": rank.get("top1_accuracy"),
        "top5": rank.get("top5_accuracy"),
        "note": split_note,
    }


def make_metric_panel(rows: list[dict], out_path: Path) -> None:
    width, height = 1640, 620
    panel = Image.new("RGB", (width, height), (248, 248, 245))
    draw = ImageDraw.Draw(panel)
    f_title = font(31, bold=True)
    f_head = font(18, bold=True)
    f_body = font(17)
    f_note = font(15)
    draw.text((36, 30), "Qwen3-VL COCO object-token AV results", fill=(24, 28, 34), font=f_title)
    columns = [
        ("run", 36, 360),
        ("eval loss", 430, 110),
        ("sens delta", 575, 115),
        ("match better", 725, 130),
        ("mean rank", 895, 130),
        ("top1", 1060, 90),
        ("top5", 1180, 90),
        ("note", 1310, 260),
    ]
    y = 104
    for label, x, _ in columns:
        draw.text((x, y), label, fill=(71, 78, 88), font=f_head)
    y += 44
    for row in rows:
        draw.line((36, y - 14, width - 36, y - 14), fill=(220, 224, 229), width=1)
        values = [
            row["label"],
            "n/a" if row["eval_loss"] is None else f"{row['eval_loss']:.3f}",
            f"{row['sens_delta']:.3f}",
            f"{100 * row['match_better']:.1f}%",
            f"{row['mean_rank']:.2f}",
            f"{100 * row['top1']:.1f}%",
            f"{100 * row['top5']:.1f}%",
            row["note"],
        ]
        for value, (_, x, max_px) in zip(values, columns):
            clipped = str(value)
            while draw.textlength(clipped, font=f_body) > max_px and len(clipped) > 8:
                clipped = clipped[:-4] + "..."
            draw.text((x, y), clipped, fill=(29, 34, 41), font=f_body)
        y += 58
    notes = [
        "Single token = activation at the COCO object-center image token.",
        "BBox group = mean of 4-8 local image tokens whose cell centers fall inside the COCO object bbox.",
    ]
    note_y = height - 92
    for line in notes:
        draw.text((36, note_y), line, fill=(85, 91, 101), font=f_note)
        note_y += 26
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def draw_image_with_token_overlay(row: dict, box: tuple[int, int, int, int]) -> Image.Image:
    img = Image.open(row["source_image_path"]).convert("RGB")
    img.thumbnail((box[2] - box[0], box[3] - box[1]))
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255))
    ox = (canvas.width - img.width) // 2
    oy = (canvas.height - img.height) // 2
    canvas.paste(img, (ox, oy))
    draw = ImageDraw.Draw(canvas, "RGBA")
    scale_x = img.width / max(1, Image.open(row["source_image_path"]).width)
    scale_y = img.height / max(1, Image.open(row["source_image_path"]).height)

    bx, by, bw, bh = [float(v) for v in row["bbox_xywh"]]
    bbox = (
        ox + bx * scale_x,
        oy + by * scale_y,
        ox + (bx + bw) * scale_x,
        oy + (by + bh) * scale_y,
    )
    draw.rectangle(bbox, outline=(220, 60, 55, 230), width=4)

    grid_h, grid_w = [int(v) for v in row["token_grid_hw"]]
    selected = json.loads(row["selected_image_token_indices"])
    cell_w = img.width / grid_w
    cell_h = img.height / grid_h
    for idx in selected:
        gy, gx = divmod(int(idx), grid_w)
        rect = (
            ox + gx * cell_w,
            oy + gy * cell_h,
            ox + (gx + 1) * cell_w,
            oy + (gy + 1) * cell_h,
        )
        draw.rectangle(rect, fill=(45, 120, 230, 80), outline=(25, 85, 190, 240), width=3)
    return canvas


def make_example_panel(
    *,
    title: str,
    ranking_example: dict,
    sensitivity_examples: dict[str, dict],
    row_lookup: dict[str, dict],
    out_path: Path,
) -> None:
    row = row_lookup[ranking_example["query_sample_id"]]
    width, height = 1660, 940
    margin = 36
    image_box = (margin, 125, 500, 590)
    text_x = 560
    panel = Image.new("RGB", (width, height), (248, 248, 245))
    draw = ImageDraw.Draw(panel)
    f_title = font(31, bold=True)
    f_head = font(21, bold=True)
    f_body = font(19)
    f_small = font(16)

    draw.text((margin, 30), title, fill=(24, 28, 34), font=f_title)
    subtitle = f"{ranking_example['query_sample_id']} | rank {ranking_example['correct_rank']} / {len(row_lookup)}"
    draw.text((margin, 78), subtitle, fill=(83, 89, 98), font=f_body)

    overlay = draw_image_with_token_overlay(row, image_box)
    panel.paste(overlay, (image_box[0], image_box[1]))
    draw.rectangle((image_box[0], image_box[1], image_box[2], image_box[3]), outline=(205, 210, 216), width=2)
    draw.text((margin, 603), "red = COCO bbox, blue = selected image token(s)", fill=(83, 89, 98), font=f_small)

    y = 636
    draw.text((margin, y), "Target explanation", fill=(24, 28, 34), font=f_head)
    y = draw_wrapped(
        draw,
        (margin, y + 32),
        ranking_example["source_description"],
        max_chars=50,
        font_obj=f_body,
        fill=(35, 39, 45),
        max_lines=7,
    )

    sens = sensitivity_examples.get(ranking_example["query_sample_id"])
    if sens:
        y += 18
        draw.rounded_rectangle((margin, y, 500, y + 100), radius=6, fill=(236, 242, 238))
        for i, line in enumerate(
            [
                f"matched NLL:  {sens['matched_nll']:.4f}",
                f"shuffled NLL: {sens['shuffled_nll']:.4f}",
                f"delta:        {sens['shuffled_minus_matched']:+.4f}",
            ]
        ):
            draw.text((margin + 18, y + 14 + i * 28), line, fill=(27, 74, 52), font=f_small)

    draw.text((text_x, 125), "Top-5 candidate explanations by raw NLL", fill=(24, 28, 34), font=f_head)
    y = 170
    for item in ranking_example["top5"]:
        is_correct = item["is_correct"]
        bg = (231, 246, 238) if is_correct else (255, 255, 255)
        border = (80, 160, 112) if is_correct else (213, 217, 222)
        draw.rounded_rectangle((text_x, y, width - margin, y + 130), radius=6, fill=bg, outline=border, width=2)
        prefix = f"#{item['rank']}  {item['sample_id']}  NLL={item['score']:.4f}"
        if is_correct:
            prefix += "  CORRECT"
        draw.text((text_x + 18, y + 14), prefix, fill=(23, 28, 35), font=f_small)
        draw_wrapped(
            draw,
            (text_x + 18, y + 46),
            clean(item["response"]),
            max_chars=92,
            font_obj=f_small,
            fill=(48, 53, 61),
            max_lines=3,
        )
        y += 142
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--center-parquet", required=True)
    parser.add_argument("--bbox-parquet", required=True)
    parser.add_argument("--center-summary", required=True)
    parser.add_argument("--bbox-summary", required=True)
    parser.add_argument("--center-sensitivity", required=True)
    parser.add_argument("--bbox-sensitivity", required=True)
    parser.add_argument("--center-ranking", required=True)
    parser.add_argument("--bbox-ranking", required=True)
    parser.add_argument("--center-heldout-sensitivity", required=True)
    parser.add_argument("--bbox-heldout-sensitivity", required=True)
    parser.add_argument("--center-heldout-ranking", required=True)
    parser.add_argument("--bbox-heldout-ranking", required=True)
    parser.add_argument("--center-heldout-parquet", required=True)
    parser.add_argument("--bbox-heldout-parquet", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    center_summary = load_json(args.center_summary)
    bbox_summary = load_json(args.bbox_summary)
    center_sens = load_json(args.center_sensitivity)
    bbox_sens = load_json(args.bbox_sensitivity)
    center_rank = load_json(args.center_ranking)
    bbox_rank = load_json(args.bbox_ranking)
    center_h_sens = load_json(args.center_heldout_sensitivity)
    bbox_h_sens = load_json(args.bbox_heldout_sensitivity)
    center_h_rank = load_json(args.center_heldout_ranking)
    bbox_h_rank = load_json(args.bbox_heldout_ranking)

    make_metric_panel(
        [
            metric_row("single object-center token", center_summary, center_sens, center_rank, "train 64q/128c"),
            metric_row("bbox local token group", bbox_summary, bbox_sens, bbox_rank, "train 64q/128c"),
            metric_row_eval_only("single token held-out", center_h_sens, center_h_rank, "held-out 32q/64c"),
            metric_row_eval_only("bbox group held-out", bbox_h_sens, bbox_h_rank, "held-out 32q/64c"),
        ],
        out_dir / "qwen3vl_coco_token_metrics.png",
    )

    center_rows = rows_by_id(args.center_parquet)
    bbox_rows = rows_by_id(args.bbox_parquet)
    center_h_rows = rows_by_id(args.center_heldout_parquet)
    bbox_h_rows = rows_by_id(args.bbox_heldout_parquet)
    center_sens_examples = {item["sample_id"]: item for item in center_sens.get("examples", [])}
    bbox_sens_examples = {item["sample_id"]: item for item in bbox_sens.get("examples", [])}
    center_h_sens_examples = {item["sample_id"]: item for item in center_h_sens.get("examples", [])}
    bbox_h_sens_examples = {item["sample_id"]: item for item in bbox_h_sens.get("examples", [])}

    for i, ex in enumerate(center_rank.get("examples", [])[:2]):
        make_example_panel(
            title="Qwen3-VL COCO: single object-center image token",
            ranking_example=ex,
            sensitivity_examples=center_sens_examples,
            row_lookup=center_rows,
            out_path=out_dir / f"center_train_{i}_{ex['query_sample_id']}.png",
        )
    for i, ex in enumerate(bbox_rank.get("examples", [])[:2]):
        make_example_panel(
            title="Qwen3-VL COCO: local bbox image-token group",
            ranking_example=ex,
            sensitivity_examples=bbox_sens_examples,
            row_lookup=bbox_rows,
            out_path=out_dir / f"bbox_train_{i}_{ex['query_sample_id']}.png",
        )
    for i, ex in enumerate(center_h_rank.get("examples", [])[:1]):
        make_example_panel(
            title="Qwen3-VL COCO held-out: single object-center image token",
            ranking_example=ex,
            sensitivity_examples=center_h_sens_examples,
            row_lookup=center_h_rows,
            out_path=out_dir / f"center_heldout_{i}_{ex['query_sample_id']}.png",
        )
    for i, ex in enumerate(bbox_h_rank.get("examples", [])[:1]):
        make_example_panel(
            title="Qwen3-VL COCO held-out: local bbox image-token group",
            ranking_example=ex,
            sensitivity_examples=bbox_h_sens_examples,
            row_lookup=bbox_h_rows,
            out_path=out_dir / f"bbox_heldout_{i}_{ex['query_sample_id']}.png",
        )


if __name__ == "__main__":
    main()
