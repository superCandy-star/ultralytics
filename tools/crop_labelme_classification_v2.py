#!/usr/bin/env python3
"""Crop LabelMe object annotations into ImageFolder-style classification data.

Crop strategy: tight bbox + fixed bias expansion (default 5 px) -> resize to 224x224.
No adaptive square crop, no hand union, no minimum-dim expansion.
Focuses on keeping the object itself dominant in the crop.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_EXCLUDE_CLASSES = {"hand", "obj"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")


def load_labelme_json(json_path: Path) -> dict:
    """Load a LabelMe JSON file with fallback encodings."""
    for encoding in ("utf-8", "gb2312", "latin1"):
        try:
            with open(json_path, "r", encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Cannot decode JSON file: {json_path}")


def find_image_file(json_path: Path, image_path_value: str | None = None) -> Path | None:
    """Find the image corresponding to a LabelMe JSON file."""
    candidates = []
    if image_path_value:
        image_path = Path(image_path_value)
        candidates.append(json_path.parent / image_path.name)
        if image_path.is_absolute():
            candidates.append(image_path)
    candidates.extend(json_path.with_suffix(ext) for ext in IMAGE_EXTS)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def shape_to_bbox(shape: dict) -> tuple[float, float, float, float] | None:
    """Convert a LabelMe shape to an xyxy bounding box."""
    points = shape.get("points") or []
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_bbox_with_bias(
    bbox: tuple[float, float, float, float],
    bias: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Expand tight xyxy bbox by a fixed bias on each side, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - bias)
    y1 = max(0, y1 - bias)
    x2 = min(img_w - 1, x2 + bias)
    y2 = min(img_h - 1, y2 + bias)
    return x1, y1, x2, y2


def safe_class_name(label: str) -> str:
    """Make a label safe as a folder name while keeping it readable."""
    return label.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def collect_annotations(source_dir: Path, exclude_classes: set[str]) -> list[dict]:
    """Collect all crop annotations from LabelMe JSON files."""
    items = []
    json_files = [p for p in source_dir.rglob("*.json") if not p.name.startswith("._")]
    for json_path in json_files:
        data = load_labelme_json(json_path)
        image_path = find_image_file(json_path, data.get("imagePath"))
        if image_path is None:
            continue
        image_width = int(data.get("imageWidth") or 0)
        image_height = int(data.get("imageHeight") or 0)
        for shape_idx, shape in enumerate(data.get("shapes") or []):
            label = str(shape.get("label", "")).strip()
            if not label or label in exclude_classes:
                continue
            bbox = shape_to_bbox(shape)
            if bbox is None:
                continue
            items.append(
                {
                    "json_path": json_path,
                    "image_path": image_path,
                    "image_width": image_width,
                    "image_height": image_height,
                    "shape_idx": shape_idx,
                    "label": label,
                    "bbox": bbox,
                }
            )
    return items


def split_by_class(items: list[dict], train_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split annotations by class so each class roughly follows train_ratio."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for item in items:
        by_class[item["label"]].append(item)

    train, val = [], []
    for label, class_items in by_class.items():
        rng.shuffle(class_items)
        split_idx = int(len(class_items) * train_ratio)
        if len(class_items) > 1:
            split_idx = max(1, min(split_idx, len(class_items) - 1))
        train.extend(class_items[:split_idx])
        val.extend(class_items[split_idx:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def save_crops(
    items: list[dict],
    split: str,
    output_dir: Path,
    bias: float,
    input_size: int,
) -> Counter:
    """Save bias-expanded crop images for one split and return per-class counts."""
    counts = Counter()
    image_cache: dict[Path, Image.Image] = {}
    for item in items:
        image_path = item["image_path"]
        if image_path not in image_cache:
            image_cache[image_path] = Image.open(image_path).convert("RGB")
        image = image_cache[image_path]
        img_w, img_h = image.size

        bbox = item["bbox"]
        x1, y1, x2, y2 = expand_bbox_with_bias(bbox, bias, img_w, img_h)

        # Ensure at least 1x1
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1

        crop = image.crop((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
        crop = crop.resize((input_size, input_size), Image.Resampling.BILINEAR)

        label_dir_name = safe_class_name(item["label"])
        class_dir = output_dir / split / label_dir_name
        class_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{image_path.stem}_{item['shape_idx']:03d}.jpg"
        crop.save(class_dir / out_name, quality=95)
        counts[item["label"]] += 1
    return counts


def write_class_index(output_dir: Path, labels: list[str]) -> None:
    """Write deterministic class index files."""
    labels = sorted(labels)
    with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
        for i, label in enumerate(labels):
            f.write(f"{i} {label}\n")
    with open(output_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump({label: i for i, label in enumerate(labels)}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Crop LabelMe annotations into classification folders (bias expansion)")
    parser.add_argument("--source", type=str, default="/root/taojianwei/datasets/IRIS/20260703_datasets", help="LabelMe JSON/image directory")
    parser.add_argument("--output", type=str, default="/root/taojianwei/datasets/IRIS/20260703_datasets_cls_crop_bias5", help="Output ImageFolder dataset directory")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio per class")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    parser.add_argument("--bias", type=float, default=5.0, help="Fixed pixel bias expansion around tight bbox on each side")
    parser.add_argument("--input_size", type=int, default=224, help="Resize saved crops to this square size")
    parser.add_argument("--exclude", type=str, default="hand,obj", help="Comma-separated labels to exclude")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    exclude_classes = {label.strip() for label in args.exclude.split(",") if label.strip()}

    items = collect_annotations(source_dir, exclude_classes)
    if not items:
        raise ValueError(f"No classification annotations found in {source_dir} after excluding {sorted(exclude_classes)}")

    train_items, val_items = split_by_class(items, args.train_ratio, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_class_index(output_dir, sorted({item["label"] for item in items}))

    train_counts = save_crops(train_items, "train", output_dir, args.bias, args.input_size)
    val_counts = save_crops(val_items, "val", output_dir, args.bias, args.input_size)

    print(f"Train: {sum(train_counts.values())} crops, {len(train_counts)} classes")
    print(f"Val:   {sum(val_counts.values())} crops, {len(val_counts)} classes")
    for label in sorted({item["label"] for item in items}):
        print(f"  {label}: train={train_counts.get(label, 0)} val={val_counts.get(label, 0)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
