#!/usr/bin/env python3
"""Create qualitative visualization panels for the LLaVA-1.5 NLA experiments."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TAG_RE = re.compile(r"</?explanation>")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_response(text: str | None) -> str:
    if not text:
        return ""
    text = TAG_RE.sub("", text)
    return " ".join(text.split())


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
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
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(paragraph, width=width, break_long_words=False, replace_whitespace=False)
        lines.extend(wrapped or [""])
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    max_chars: int,
    line_spacing: int,
    fill: tuple[int, int, int],
    font_obj: ImageFont.ImageFont,
) -> int:
    x, y = xy
    for line in wrap_lines(text, max_chars):
        draw.text((x, y), line, fill=fill, font=font_obj)
        bbox = draw.textbbox((x, y), line or " ", font=font_obj)
        y += (bbox[3] - bbox[1]) + line_spacing
    return y


def make_example_panel(
    *,
    ranking_example: dict,
    sensitivity_example: dict | None,
    generation: str | None,
    image_dir: Path,
    title: str,
    out_path: Path,
) -> None:
    width, height = 1500, 860
    margin = 36
    image_box = (margin, 122, 420, 506)
    text_x = 500
    panel = Image.new("RGB", (width, height), (248, 248, 245))
    draw = ImageDraw.Draw(panel)
    f_title = font(31, bold=True)
    f_head = font(21, bold=True)
    f_body = font(20)
    f_small = font(17)

    draw.text((margin, 30), title, fill=(25, 28, 32), font=f_title)
    subtitle = f"{ranking_example['query_sample_id']} | correct rank {ranking_example['correct_rank']} / 128"
    draw.text((margin, 78), subtitle, fill=(83, 89, 98), font=f_body)

    img_path = image_dir / f"{ranking_example['query_sample_id']}.png"
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((image_box[2] - image_box[0], image_box[3] - image_box[1]))
    img_x = image_box[0] + ((image_box[2] - image_box[0]) - img.width) // 2
    img_y = image_box[1] + ((image_box[3] - image_box[1]) - img.height) // 2
    draw.rounded_rectangle(
        (image_box[0] - 6, image_box[1] - 6, image_box[2] + 6, image_box[3] + 6),
        radius=6,
        fill=(255, 255, 255),
        outline=(210, 214, 219),
        width=2,
    )
    panel.paste(img, (img_x, img_y))

    y = 536
    draw.text((margin, y), "GT explanation", fill=(25, 28, 32), font=f_head)
    y = draw_wrapped(
        draw,
        (margin, y + 32),
        ranking_example["source_description"],
        max_chars=42,
        line_spacing=6,
        fill=(35, 39, 45),
        font_obj=f_body,
    )

    y += 24
    draw.text((margin, y), "Greedy generation", fill=(25, 28, 32), font=f_head)
    generated = clean_response(generation) if generation else "not logged for this sample"
    y = draw_wrapped(
        draw,
        (margin, y + 32),
        generated,
        max_chars=42,
        line_spacing=6,
        fill=(45, 49, 56),
        font_obj=f_body,
    )

    metric_y = y + 28
    if sensitivity_example is not None:
        metric_lines = [
            f"matched NLL:  {sensitivity_example['matched_nll']:.4f}",
            f"shuffled NLL: {sensitivity_example['shuffled_nll']:.4f}",
            f"delta:        {sensitivity_example['shuffled_minus_matched']:+.4f}",
        ]
        draw.rounded_rectangle((margin, metric_y, 430, metric_y + 116), radius=6, fill=(236, 242, 238))
        for i, line in enumerate(metric_lines):
            draw.text((margin + 18, metric_y + 16 + i * 31), line, fill=(27, 74, 52), font=f_small)

    draw.text((text_x, 122), "Top-5 candidate responses by raw NLL", fill=(25, 28, 32), font=f_head)
    y = 166
    for item in ranking_example["top5"]:
        is_correct = item["is_correct"]
        bg = (231, 246, 238) if is_correct else (255, 255, 255)
        border = (80, 160, 112) if is_correct else (213, 217, 222)
        draw.rounded_rectangle((text_x, y, width - margin, y + 118), radius=6, fill=bg, outline=border, width=2)
        prefix = f"#{item['rank']}  {item['sample_id']}  NLL={item['score']:.4f}"
        if is_correct:
            prefix += "  CORRECT"
        draw.text((text_x + 18, y + 14), prefix, fill=(23, 28, 35), font=f_small)
        draw_wrapped(
            draw,
            (text_x + 18, y + 46),
            clean_response(item["response"]),
            max_chars=82,
            line_spacing=4,
            fill=(48, 53, 61),
            font_obj=f_small,
        )
        y += 132

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def make_metric_panel(*, rows: list[dict], out_path: Path) -> None:
    width, height = 1320, 560
    panel = Image.new("RGB", (width, height), (248, 248, 245))
    draw = ImageDraw.Draw(panel)
    f_title = font(30, bold=True)
    f_head = font(19, bold=True)
    f_body = font(18)
    draw.text((36, 30), "LLaVA-1.5 Layer-15 AV: multi-token progress", fill=(25, 28, 32), font=f_title)

    columns = [
        ("run", 36, 310),
        ("sens delta", 360, 135),
        ("match better", 520, 150),
        ("mean rank", 700, 135),
        ("top5", 860, 95),
        ("eval loss", 990, 120),
    ]
    y = 100
    for label, x, _ in columns:
        draw.text((x, y), label, fill=(71, 78, 88), font=f_head)
    y += 42
    for row in rows:
        draw.line((36, y - 12, width - 36, y - 12), fill=(220, 224, 229), width=1)
        values = [
            row["run"],
            f"{row['sens_delta']:.5f}" if row.get("sens_delta") is not None else "n/a",
            f"{100 * row['match_better']:.1f}%" if row.get("match_better") is not None else "n/a",
            f"{row['mean_rank']:.2f}/128" if row.get("mean_rank") is not None else "n/a",
            f"{100 * row['top5']:.1f}%" if row.get("top5") is not None else "n/a",
            f"{row['eval_loss']:.4f}" if row.get("eval_loss") is not None else "n/a",
        ]
        for value, (_, x, max_px) in zip(values, columns):
            clipped = value
            while draw.textlength(clipped, font=f_body) > max_px and len(clipped) > 8:
                clipped = clipped[:-4] + "..."
            draw.text((x, y), clipped, fill=(29, 34, 41), font=f_body)
        y += 54

    note = "Lower mean rank is better. Sens delta = shuffled NLL - matched NLL for the same target response."
    draw.text((36, height - 62), note, fill=(85, 91, 101), font=f_body)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def generation_map(summary: dict) -> dict[int, str]:
    epochs = summary.get("epoch_summaries") or []
    if not epochs:
        return {}
    out: dict[int, str] = {}
    for item in epochs[-1].get("generations", []):
        out[int(item["sample_index"])] = item.get("generated", "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-json", required=True)
    parser.add_argument("--sensitivity-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()

    ranking = load_json(Path(args.ranking_json))
    sensitivity = load_json(Path(args.sensitivity_json))
    train_summary = load_json(Path(args.summary_json))
    sens_by_id = {item["sample_id"]: item for item in sensitivity.get("examples", [])}
    gen_by_index = generation_map(train_summary)

    out_dir = Path(args.out_dir)
    metric_rows = [
        {
            "run": args.run_label,
            "sens_delta": sensitivity.get("shuffled_minus_matched_mean"),
            "match_better": sensitivity.get("shuffled_match_better_fraction"),
            "mean_rank": ranking.get("mean_rank"),
            "top5": ranking.get("top5_accuracy"),
            "eval_loss": train_summary.get("final_eval_loss_mean"),
        }
    ]
    make_metric_panel(rows=metric_rows, out_path=out_dir / f"{args.run_label}_metrics.png")

    for ex in ranking.get("examples", [])[:5]:
        query_index = int(ex["query_index"])
        sample_id = ex["query_sample_id"]
        title = f"{args.run_label}: activation-conditioned explanation"
        make_example_panel(
            ranking_example=ex,
            sensitivity_example=sens_by_id.get(sample_id),
            generation=gen_by_index.get(query_index),
            image_dir=Path(args.image_dir),
            title=title,
            out_path=out_dir / f"{args.run_label}_{sample_id}.png",
        )


if __name__ == "__main__":
    main()
