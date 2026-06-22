#!/usr/bin/env python3
"""Create visualization panels for the Qwen3-VL NLA/AV experiments."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TAG_RE = re.compile(r"</?explanation>")


def load_json(path: str | Path | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        y += (bbox[3] - bbox[1]) + line_spacing
    return y


def generation_map(summary: dict | None) -> dict[int, str]:
    if not summary:
        return {}
    epochs = summary.get("epoch_summaries") or []
    if not epochs:
        return {}
    out: dict[int, str] = {}
    for item in epochs[-1].get("generations", []):
        out[int(item["sample_index"])] = item.get("generated", "")
    return out


def metric_row(
    *,
    run: str,
    sensitivity: dict | None,
    ranking: dict | None,
    summary: dict | None = None,
    note: str = "",
) -> dict:
    return {
        "run": run,
        "sens_delta": None if sensitivity is None else sensitivity.get("shuffled_minus_matched_mean"),
        "match_better": None if sensitivity is None else sensitivity.get("shuffled_match_better_fraction"),
        "mean_rank": None if ranking is None else ranking.get("mean_rank"),
        "top1": None if ranking is None else ranking.get("top1_accuracy"),
        "top5": None if ranking is None else ranking.get("top5_accuracy"),
        "eval_loss": None if summary is None else summary.get("final_eval_loss_mean"),
        "note": note,
    }


def make_metric_panel(*, rows: list[dict], out_path: Path) -> None:
    width, height = 1600, 620
    panel = Image.new("RGB", (width, height), (248, 248, 245))
    draw = ImageDraw.Draw(panel)
    f_title = font(32, bold=True)
    f_head = font(18, bold=True)
    f_body = font(17)
    f_note = font(15)

    draw.text((36, 30), "Layer-15 AV comparison: LLaVA-1.5 vs Qwen3-VL", fill=(25, 28, 32), font=f_title)
    columns = [
        ("run", 36, 360),
        ("sens delta", 430, 120),
        ("match better", 580, 130),
        ("mean rank", 750, 130),
        ("top1", 910, 90),
        ("top5", 1030, 90),
        ("eval loss", 1150, 110),
        ("note", 1290, 260),
    ]
    y = 102
    for label, x, _ in columns:
        draw.text((x, y), label, fill=(71, 78, 88), font=f_head)
    y += 44

    for row in rows:
        draw.line((36, y - 14, width - 36, y - 14), fill=(220, 224, 229), width=1)
        values = [
            row["run"],
            f"{row['sens_delta']:.3f}" if row.get("sens_delta") is not None else "n/a",
            f"{100 * row['match_better']:.1f}%" if row.get("match_better") is not None else "n/a",
            f"{row['mean_rank']:.2f}/128" if row.get("mean_rank") is not None else "n/a",
            f"{100 * row['top1']:.1f}%" if row.get("top1") is not None else "n/a",
            f"{100 * row['top5']:.1f}%" if row.get("top5") is not None else "n/a",
            f"{row['eval_loss']:.4f}" if row.get("eval_loss") is not None else "n/a",
            row.get("note", ""),
        ]
        for value, (_, x, max_px) in zip(values, columns):
            clipped = str(value)
            while draw.textlength(clipped, font=f_body) > max_px and len(clipped) > 8:
                clipped = clipped[:-4] + "..."
            draw.text((x, y), clipped, fill=(29, 34, 41), font=f_body)
        y += 58

    notes = [
        "Lower mean rank is better. Sens delta = shuffled target NLL - matched target NLL.",
        "Qwen3 uses the mean activation over all image tokens at layer 15; LLaVA uses the layer-15 last-prompt activation.",
    ]
    note_y = height - 92
    for line in notes:
        draw.text((36, note_y), line, fill=(85, 91, 101), font=f_note)
        note_y += 26
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def make_probe_panel(*, probes: list[tuple[str, dict]], out_path: Path) -> None:
    width, height = 1180, 560
    panel = Image.new("RGB", (width, height), (248, 248, 245))
    draw = ImageDraw.Draw(panel)
    f_title = font(31, bold=True)
    f_head = font(18, bold=True)
    f_body = font(17)
    draw.text((36, 30), "Qwen3-VL layer-15 target choice matters", fill=(25, 28, 32), font=f_title)

    x0, y0 = 320, 140
    bar_w, row_h = 500, 78
    f1_x = 850
    exact_x = 1035
    draw.text((36, 104), "target activation", fill=(71, 78, 88), font=f_head)
    draw.text((x0, 104), "micro-F1", fill=(71, 78, 88), font=f_head)
    draw.text((exact_x, 104), "exact set", fill=(71, 78, 88), font=f_head)
    for i, (label, data) in enumerate(probes):
        y = y0 + i * row_h
        f1 = float(data["micro_f1"])
        exact = float(data["exact_set_accuracy"])
        draw.text((36, y + 7), label, fill=(29, 34, 41), font=f_body)
        draw.rounded_rectangle((x0, y, x0 + bar_w, y + 32), radius=5, fill=(229, 233, 238))
        color = (35, 125, 94) if "mean" in label else (83, 115, 170)
        draw.rounded_rectangle((x0, y, x0 + int(bar_w * f1), y + 32), radius=5, fill=color)
        draw.text((f1_x, y + 6), f"{f1:.3f}", fill=(29, 34, 41), font=f_body)
        draw.text((exact_x, y + 6), f"{100 * exact:.1f}%", fill=(29, 34, 41), font=f_body)

    note = "The mean over 100 image tokens carries much cleaner color/shape/position information than a single prompt or image token."
    draw_wrapped(draw, (36, height - 94), note, max_chars=120, line_spacing=6, fill=(85, 91, 101), font_obj=f_body)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def make_example_panel(
    *,
    ranking_example: dict,
    sensitivity_example: dict | None,
    generation: str | None,
    image_dir: Path,
    title: str,
    score_label: str,
    out_path: Path,
) -> None:
    width, height = 1540, 900
    margin = 36
    image_box = (margin, 124, 420, 508)
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

    y = 538
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
        max_lines=4,
    )

    metric_y = min(y + 28, 752)
    if sensitivity_example is not None:
        metric_lines = [
            f"matched NLL:  {sensitivity_example['matched_nll']:.4f}",
            f"shuffled NLL: {sensitivity_example['shuffled_nll']:.4f}",
            f"delta:        {sensitivity_example['shuffled_minus_matched']:+.4f}",
        ]
        draw.rounded_rectangle((margin, metric_y, 430, metric_y + 116), radius=6, fill=(236, 242, 238))
        for i, line in enumerate(metric_lines):
            draw.text((margin + 18, metric_y + 16 + i * 31), line, fill=(27, 74, 52), font=f_small)

    draw.text((text_x, 124), f"Top-5 candidate responses by {score_label}", fill=(25, 28, 32), font=f_head)
    y = 168
    for item in ranking_example["top5"]:
        is_correct = item["is_correct"]
        bg = (231, 246, 238) if is_correct else (255, 255, 255)
        border = (80, 160, 112) if is_correct else (213, 217, 222)
        draw.rounded_rectangle((text_x, y, width - margin, y + 124), radius=6, fill=bg, outline=border, width=2)
        score = item.get("score")
        raw = item.get("raw_nll")
        prefix = f"#{item['rank']}  {item['sample_id']}  score={score:.4f}"
        if raw is not None and abs(raw - score) > 1e-6:
            prefix += f"  raw={raw:.4f}"
        if is_correct:
            prefix += "  CORRECT"
        draw.text((text_x + 18, y + 14), prefix, fill=(23, 28, 35), font=f_small)
        draw_wrapped(
            draw,
            (text_x + 18, y + 48),
            clean_response(item["response"]),
            max_chars=85,
            line_spacing=4,
            fill=(48, 53, 61),
            font_obj=f_small,
            max_lines=3,
        )
        y += 136

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def examples_by_id(ranking: dict) -> dict[str, dict]:
    return {ex["query_sample_id"]: ex for ex in ranking.get("examples", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-ranking-json", required=True)
    parser.add_argument("--qwen-gain-ranking-json", required=True)
    parser.add_argument("--qwen-sensitivity-json", required=True)
    parser.add_argument("--qwen-summary-json", required=True)
    parser.add_argument("--qwen-image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--llava-ranking-json")
    parser.add_argument("--llava-sensitivity-json")
    parser.add_argument("--llava-summary-json")
    parser.add_argument("--heldout-ranking-json")
    parser.add_argument("--heldout-sensitivity-json")
    parser.add_argument("--heldout-image-dir")
    parser.add_argument("--probe-last-prompt-json")
    parser.add_argument("--probe-image-json")
    parser.add_argument("--probe-image-mean-json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    qwen_ranking = load_json(args.qwen_ranking_json)
    qwen_gain = load_json(args.qwen_gain_ranking_json)
    qwen_sensitivity = load_json(args.qwen_sensitivity_json)
    qwen_summary = load_json(args.qwen_summary_json)
    llava_ranking = load_json(args.llava_ranking_json)
    llava_sensitivity = load_json(args.llava_sensitivity_json)
    llava_summary = load_json(args.llava_summary_json)
    heldout_ranking = load_json(args.heldout_ranking_json)
    heldout_sensitivity = load_json(args.heldout_sensitivity_json)

    rows = []
    if llava_ranking and llava_sensitivity:
        rows.append(
            metric_row(
                run="LLaVA-1.5 L15 last_prompt",
                sensitivity=llava_sensitivity,
                ranking=llava_ranking,
                summary=llava_summary,
                note="512 train pool",
            )
        )
    rows.append(
        metric_row(
            run="Qwen3-VL L15 image_mean",
            sensitivity=qwen_sensitivity,
            ranking=qwen_ranking,
            summary=qwen_summary,
            note="512 train pool",
        )
    )
    if heldout_ranking and heldout_sensitivity:
        rows.append(
            metric_row(
                run="Qwen3-VL held-out seed2026",
                sensitivity=heldout_sensitivity,
                ranking=heldout_ranking,
                summary=None,
                note="32q/128c",
            )
        )
    make_metric_panel(rows=rows, out_path=out_dir / "qwen3vl_llava_metric_comparison.png")

    probes: list[tuple[str, dict]] = []
    for label, path in [
        ("last_prompt", args.probe_last_prompt_json),
        ("single image token", args.probe_image_json),
        ("image_mean", args.probe_image_mean_json),
    ]:
        data = load_json(path)
        if data:
            probes.append((label, data))
    if probes:
        make_probe_panel(probes=probes, out_path=out_dir / "qwen3vl_layer15_probe_targets.png")

    sens_by_id = {item["sample_id"]: item for item in qwen_sensitivity.get("examples", [])}
    gen_by_index = generation_map(qwen_summary)
    image_dir = Path(args.qwen_image_dir)
    raw_by_id = examples_by_id(qwen_ranking)
    gain_by_id = examples_by_id(qwen_gain)

    for sample_id in ["synthetic_00000", "synthetic_00001", "synthetic_00002", "synthetic_00005"]:
        ex = raw_by_id.get(sample_id)
        if not ex:
            continue
        make_example_panel(
            ranking_example=ex,
            sensitivity_example=sens_by_id.get(sample_id),
            generation=gen_by_index.get(int(ex["query_index"])),
            image_dir=image_dir,
            title="Qwen3-VL: layer-15 image_mean -> explanation",
            score_label="raw NLL",
            out_path=out_dir / f"qwen3vl_raw_{sample_id}.png",
        )

    gain_ex = gain_by_id.get("synthetic_00005")
    if gain_ex:
        make_example_panel(
            ranking_example=gain_ex,
            sensitivity_example=sens_by_id.get("synthetic_00005"),
            generation=gen_by_index.get(int(gain_ex["query_index"])),
            image_dir=image_dir,
            title="Qwen3-VL: activation-gain diagnostic example",
            score_label="activation-gain score",
            out_path=out_dir / "qwen3vl_gain_synthetic_00005.png",
        )

    if heldout_ranking and heldout_sensitivity and args.heldout_image_dir:
        heldout_sens_by_id = {item["sample_id"]: item for item in heldout_sensitivity.get("examples", [])}
        heldout_by_id = examples_by_id(heldout_ranking)
        heldout_image_dir = Path(args.heldout_image_dir)
        for sample_id in ["synthetic_00000", "synthetic_00001"]:
            ex = heldout_by_id.get(sample_id)
            if not ex:
                continue
            make_example_panel(
                ranking_example=ex,
                sensitivity_example=heldout_sens_by_id.get(sample_id),
                generation=None,
                image_dir=heldout_image_dir,
                title="Qwen3-VL held-out: layer-15 image_mean -> explanation",
                score_label="raw NLL",
                out_path=out_dir / f"qwen3vl_heldout_raw_{sample_id}.png",
            )


if __name__ == "__main__":
    main()
