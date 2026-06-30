#!/usr/bin/env python3
"""Ridge AR baseline for Qwen3-VL VLM-NLA activations.

This is a lightweight text -> activation reconstructor. It uses hashed word
n-gram features from explanation text and dual ridge regression to predict the
target activation vector. It is meant as a fast H3 baseline before training a
full neural AR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")


def load_rows(path: str, max_rows: int | None = None) -> list[dict]:
    table = pq.read_table(path)
    if max_rows is not None:
        table = table.slice(0, max_rows)
    return table.to_pylist()


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def text_features(texts: list[str], dim: int) -> np.ndarray:
    x = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        words = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
        feats = words[:]
        feats.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
        for feat in feats:
            h = stable_hash(feat)
            idx = h % dim
            sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
            x[i, idx] += sign
        norm = np.linalg.norm(x[i])
        if norm > 0:
            x[i] /= norm
    return x


def activations(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["activation_vector"] for row in rows], dtype=np.float32)


def no_fixed_point_permutation(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    if n > 1:
        for i in range(n):
            if perm[i] == i:
                j = (i + 1) % n
                perm[i], perm[j] = perm[j], perm[i]
    return perm


def fit_dual_ridge(x_train: np.ndarray, y_train: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    k_train = x_train @ x_train.T
    k_train.flat[:: k_train.shape[0] + 1] += ridge
    alpha = np.linalg.solve(k_train.astype(np.float64), y_train.astype(np.float64)).astype(np.float32)
    return x_train, alpha


def predict(x_ref: np.ndarray, alpha: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (x @ x_ref.T) @ alpha


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.clip(den, 1e-12, None)


def retrieval_metrics(pred: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    pred_n = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12, None)
    gold_n = gold / np.clip(np.linalg.norm(gold, axis=1, keepdims=True), 1e-12, None)
    scores = pred_n @ gold_n.T
    ranks = []
    for i in range(scores.shape[0]):
        order = np.argsort(-scores[i])
        ranks.append(int(np.where(order == i)[0][0]) + 1)
    ranks_arr = np.asarray(ranks)
    return {
        "retrieval_mean_rank": float(ranks_arr.mean()),
        "retrieval_top1": float((ranks_arr == 1).mean()),
        "retrieval_top5": float((ranks_arr <= 5).mean()),
    }


def score_prediction(name: str, pred: np.ndarray, gold: np.ndarray, mean_pred: np.ndarray) -> dict[str, float | str]:
    sse = float(np.sum((pred - gold) ** 2))
    sse_mean = float(np.sum((mean_pred - gold) ** 2))
    mse = float(np.mean((pred - gold) ** 2))
    norm_mse = float(np.mean(np.sum((pred - gold) ** 2, axis=1) / np.clip(np.sum(gold**2, axis=1), 1e-12, None)))
    out: dict[str, float | str] = {
        "name": name,
        "mse": mse,
        "normalized_mse": norm_mse,
        "cosine_similarity": float(cosine_rows(pred, gold).mean()),
        "fve_vs_mean": float(1.0 - sse / sse_mean) if sse_mean > 0 else 0.0,
    }
    out.update(retrieval_metrics(pred, gold))
    return out


def top1_texts_from_ranking(path: str, test_rows: list[dict]) -> list[str]:
    ranking = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {item["query_sample_id"]: item["top5"][0]["response"] for item in ranking["per_query"]}
    return [by_id[row["source_sample_id"]] for row in test_rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--feature-dim", type=int, default=4096)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4401)
    parser.add_argument("--ranking-json", default=None)
    args = parser.parse_args()

    train_rows = load_rows(args.train_parquet, args.max_train_rows)
    test_rows = load_rows(args.test_parquet, args.max_test_rows)
    y_train = activations(train_rows)
    y_test = activations(test_rows)
    train_texts = [row["response"] for row in train_rows]
    test_texts = [row["response"] for row in test_rows]

    x_train = text_features(train_texts, args.feature_dim)
    x_test = text_features(test_texts, args.feature_dim)
    x_ref, alpha = fit_dual_ridge(x_train, y_train, args.ridge)

    mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0)
    matched_pred = predict(x_ref, alpha, x_test)
    perm = no_fixed_point_permutation(len(test_texts), args.seed)
    shuffled_pred = predict(x_ref, alpha, x_test[perm])

    results = [
        score_prediction("mean_activation_baseline", mean_pred, y_test, mean_pred),
        score_prediction("matched_gold_text", matched_pred, y_test, mean_pred),
        score_prediction("shuffled_gold_text", shuffled_pred, y_test, mean_pred),
    ]
    results[1]["fve_vs_shuffled_text"] = float(
        1.0
        - np.sum((matched_pred - y_test) ** 2)
        / max(float(np.sum((shuffled_pred - y_test) ** 2)), 1e-12)
    )

    if args.ranking_json:
        top1_texts = top1_texts_from_ranking(args.ranking_json, test_rows)
        x_top1 = text_features(top1_texts, args.feature_dim)
        top1_pred = predict(x_ref, alpha, x_top1)
        results.append(score_prediction("av_top1_text", top1_pred, y_test, mean_pred))

    summary = {
        "train_parquet": args.train_parquet,
        "test_parquet": args.test_parquet,
        "ranking_json": args.ranking_json,
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "feature_dim": args.feature_dim,
        "ridge": args.ridge,
        "seed": args.seed,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
