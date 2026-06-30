#!/usr/bin/env python3
"""Build image-disjoint COCO split manifests for object-token VLM-NLA runs."""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"downloading {url} -> {path}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def ensure_instances(coco_root: Path) -> Path:
    instances = coco_root / "annotations" / "instances_val2017.json"
    if instances.exists():
        return instances
    zip_path = coco_root / "annotations_trainval2017.zip"
    download_file(ANNOTATIONS_URL, zip_path)
    coco_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("annotations/instances_val2017.json", coco_root)
    return instances


def primary_category_by_image(instances: dict, min_area_frac: float) -> dict[int, str]:
    categories = {int(cat["id"]): cat["name"] for cat in instances["categories"]}
    images = {int(img["id"]): img for img in instances["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in instances["annotations"]:
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        image = images.get(int(ann["image_id"]))
        if not image:
            continue
        min_area = min_area_frac * float(image["width"] * image["height"])
        if float(ann.get("area", 0.0)) < min_area:
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    primary: dict[int, str] = {}
    for image_id, anns in anns_by_image.items():
        chosen = max(anns, key=lambda ann: float(ann.get("area", 0.0)))
        primary[image_id] = categories[int(chosen["category_id"])]
    return primary


def allocate_stratified(
    primary: dict[int, str],
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> dict[str, list[int]]:
    desired = {"train": train_size, "val": val_size, "test": test_size}
    total_needed = sum(desired.values())
    if len(primary) < total_needed:
        raise RuntimeError(f"only {len(primary)} eligible images, need {total_needed}")

    rng = random.Random(seed)
    by_category: dict[str, list[int]] = defaultdict(list)
    for image_id, category in primary.items():
        by_category[category].append(image_id)
    for ids in by_category.values():
        rng.shuffle(ids)

    def take_balanced(size: int) -> list[int]:
        selected: list[int] = []
        categories = list(by_category)
        while len(selected) < size:
            active = [category for category in categories if by_category[category]]
            if not active:
                break
            rng.shuffle(active)
            for category in active:
                selected.append(by_category[category].pop())
                if len(selected) >= size:
                    break
        return selected

    splits: dict[str, list[int]] = {}
    for name, size in desired.items():
        splits[name] = take_balanced(size)

    for name, size in desired.items():
        if len(splits[name]) != size:
            raise RuntimeError(f"split {name} has {len(splits[name])} images, expected {size}")
        rng.shuffle(splits[name])
    return splits


def category_counts(image_ids: list[int], primary: dict[int, str]) -> list[tuple[str, int]]:
    return Counter(primary[image_id] for image_id in image_ids).most_common()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", default="data/coco2017")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--min-area-frac", type=float, default=0.015)
    args = parser.parse_args()

    instances_path = ensure_instances(Path(args.coco_root))
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    primary = primary_category_by_image(instances, args.min_area_frac)
    splits = allocate_stratified(
        primary,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    payload = {
        "dataset": "mscoco_val2017",
        "source_instances": str(instances_path),
        "seed": args.seed,
        "min_area_frac": args.min_area_frac,
        "sizes": {name: len(ids) for name, ids in splits.items()},
        "splits": splits,
        "category_counts": {name: category_counts(ids, primary) for name, ids in splits.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["dataset", "seed", "sizes"]}, indent=2))


if __name__ == "__main__":
    main()
