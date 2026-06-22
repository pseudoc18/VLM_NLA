#!/usr/bin/env python3
"""AV-lite semantic probe: activation -> color/shape/position attributes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

COLORS = ("red", "green", "blue", "yellow", "purple", "orange")
SHAPES = ("square", "circle", "triangle")
POSITIONS = ("left", "right", "top", "bottom", "center")
ATTRS = tuple(f"color:{x}" for x in COLORS) + tuple(f"shape:{x}" for x in SHAPES) + tuple(f"position:{x}" for x in POSITIONS)


def load_av_parquet(path: str) -> tuple[list[str], torch.Tensor, list[str]]:
    table = pq.read_table(path, columns=["source_description", "activation_vector", "source_sample_id"])
    descriptions = table.column("source_description").to_pylist()
    sample_ids = table.column("source_sample_id").to_pylist()
    activation_col = table.column("activation_vector").combine_chunks()
    flat = activation_col.values.to_numpy(zero_copy_only=False).astype(np.float32)
    activations = torch.from_numpy(flat.reshape(len(descriptions), -1))
    return descriptions, activations, sample_ids


def labels_from_descriptions(descriptions: list[str]) -> torch.Tensor:
    labels = torch.zeros(len(descriptions), len(ATTRS), dtype=torch.float32)
    for i, desc in enumerate(descriptions):
        low = desc.lower()
        for j, attr in enumerate(ATTRS):
            _, value = attr.split(":", 1)
            if value in low:
                labels[i, j] = 1.0
    return labels


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)


def ridge_scores(x_train: torch.Tensor, y_train: torch.Tensor, x_eval: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    x_train = normalize_features(x_train)
    x_eval = normalize_features(x_eval)
    ones_train = torch.ones(x_train.shape[0], 1)
    ones_eval = torch.ones(x_eval.shape[0], 1)
    x_train = torch.cat([x_train, ones_train], dim=1)
    x_eval = torch.cat([x_eval, ones_eval], dim=1)
    gram = x_train @ x_train.T + ridge_lambda * torch.eye(x_train.shape[0])
    alpha = torch.linalg.solve(gram, y_train)
    return x_eval @ x_train.T @ alpha


def f1_counts(pred: torch.Tensor, gold: torch.Tensor) -> tuple[float, float, float, float]:
    pred_b = pred.bool()
    gold_b = gold.bool()
    tp = (pred_b & gold_b).sum().item()
    fp = (pred_b & ~gold_b).sum().item()
    fn = (~pred_b & gold_b).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    acc = (pred_b == gold_b).float().mean().item()
    return precision, recall, f1, acc


def exact_set_accuracy(pred: torch.Tensor, gold: torch.Tensor) -> float:
    return (pred.bool() == gold.bool()).all(dim=1).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av-parquet", required=True)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    descriptions, activations, sample_ids = load_av_parquet(args.av_parquet)
    labels = labels_from_descriptions(descriptions)
    n = len(descriptions)
    rng = np.random.default_rng(args.seed)
    perm = torch.from_numpy(rng.permutation(n))
    n_train = max(4, int(n * args.train_frac))
    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]

    x_train, y_train = activations[train_idx], labels[train_idx]
    x_eval, y_eval = activations[eval_idx], labels[eval_idx]
    scores = ridge_scores(x_train, y_train, x_eval, args.ridge_lambda)
    pred = scores > 0.5

    train_rate = y_train.mean(dim=0, keepdim=True)
    majority = (train_rate > 0.5).expand_as(y_eval)
    top_rate = train_rate.expand_as(y_eval)

    precision, recall, micro_f1, attr_acc = f1_counts(pred, y_eval)
    b_precision, b_recall, b_micro_f1, b_attr_acc = f1_counts(majority, y_eval)

    per_attr = []
    for j, attr in enumerate(ATTRS):
        _, _, f1, acc = f1_counts(pred[:, j : j + 1], y_eval[:, j : j + 1])
        _, _, bf1, bacc = f1_counts(majority[:, j : j + 1], y_eval[:, j : j + 1])
        per_attr.append(
            {
                "attr": attr,
                "eval_positive_rate": float(y_eval[:, j].mean().item()),
                "f1": f1,
                "accuracy": acc,
                "majority_f1": bf1,
                "majority_accuracy": bacc,
            }
        )

    examples = []
    for row, idx in enumerate(eval_idx[:8].tolist()):
        gold_attrs = [ATTRS[j] for j in torch.where(y_eval[row] > 0.5)[0].tolist()]
        pred_attrs = [ATTRS[j] for j in torch.where(pred[row])[0].tolist()]
        examples.append(
            {
                "sample_id": sample_ids[idx],
                "description": descriptions[idx],
                "gold_attrs": gold_attrs,
                "pred_attrs": pred_attrs,
            }
        )

    summary = {
        "av_parquet": args.av_parquet,
        "n_total": n,
        "n_train": int(len(train_idx)),
        "n_eval": int(len(eval_idx)),
        "num_attrs": len(ATTRS),
        "ridge_lambda": args.ridge_lambda,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "attribute_accuracy": attr_acc,
        "exact_set_accuracy": exact_set_accuracy(pred, y_eval),
        "majority_micro_precision": b_precision,
        "majority_micro_recall": b_recall,
        "majority_micro_f1": b_micro_f1,
        "majority_attribute_accuracy": b_attr_acc,
        "majority_exact_set_accuracy": exact_set_accuracy(majority, y_eval),
        "mean_abs_score_error": float((scores - y_eval).abs().mean().item()),
        "mean_abs_majority_score_error": float((top_rate - y_eval).abs().mean().item()),
        "per_attr": per_attr,
        "examples": examples,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

