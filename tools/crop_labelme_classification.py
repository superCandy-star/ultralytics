#!/usr/bin/env python3
"""Crop LabelMe object annotations into ImageFolder-style classification data.

The script reads LabelMe JSON files, excludes hand/obj superclass annotations, and saves object-centered adaptive square
crops into class subdirectories that can be consumed by torchvision.datasets.ImageFolder or MobileNetV3 training scripts.
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


def build_obj_centered_crop_box(
    bbox: tuple[float, float, float, float],
    target_min_dim: float,
    context_scale: float,
    max_context_ratio: float,
) -> tuple[float, float, float, float, float]:
    """Build an obj-centered adaptive square crop box.

    Crop center always stays at the object center. Crop side length is decided by object long side, a minimum crop size,
    and a maximum context ratio.
    """
    x1, y1, x2, y2 = bbox
    obj_w, obj_h = x2 - x1, y2 - y1
    obj_long_side = max(obj_w, obj_h)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    crop_size = max(obj_long_side * context_scale, target_min_dim)
    max_crop_size = max(obj_long_side * max_context_ratio, target_min_dim)
    crop_size = min(crop_size, max_crop_size)

    half = crop_size / 2.0
    return cx - half, cy - half, cx + half, cy + half, crop_size


def crop_with_padding(image: Image.Image, crop_box: tuple[float, float, float, float], fill=(114, 114, 114)) -> Image.Image:
    """Crop an image by xyxy box, padding outside-image regions with a constant color."""
    x1, y1, x2, y2 = crop_box
    left, top = int(round(x1)), int(round(y1))
    right, bottom = int(round(x2)), int(round(y2))

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image.width)
    pad_bottom = max(0, bottom - image.height)

    clipped = image.crop((max(0, left), max(0, top), min(image.width, right), min(image.height, bottom)))
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        clipped = ImageOps.expand(clipped, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)
    return clipped


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
    target_min_dim: int,
    context_scale: float,
    max_context_ratio: float,
    input_size: int | None,
    min_obj_long_side: float,
    low_quality_subdir: bool,
) -> Counter:
    """Save crop images for one split and return per-class counts."""
    counts = Counter()
    image_cache: dict[Path, Image.Image] = {}
    for item in items:
        image_path = item["image_path"]
        if image_path not in image_cache:
            image_cache[image_path] = Image.open(image_path).convert("RGB")
        image = image_cache[image_path]
        bbox = item["bbox"]
        x1, y1, x2, y2 = bbox
        obj_long_side = max(x2 - x1, y2 - y1)
        crop_x1, crop_y1, crop_x2, crop_y2, crop_size = build_obj_centered_crop_box(
            bbox, target_min_dim, context_scale, max_context_ratio
        )
        crop = crop_with_padding(image, (crop_x1, crop_y1, crop_x2, crop_y2))
        if input_size:
            crop = crop.resize((input_size, input_size), Image.Resampling.BILINEAR)

        label_dir_name = safe_class_name(item["label"])
        if low_quality_subdir and obj_long_side < min_obj_long_side:
            class_dir = output_dir / split / label_dir_name / "_low_quality_small_obj"
        else:
            class_dir = output_dir / split / label_dir_name
        class_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{image_path.stem}_{item['shape_idx']:03d}_long{obj_long_side:.0f}_crop{crop_size:.0f}.jpg"
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
    parser = argparse.ArgumentParser(description="Crop LabelMe annotations into classification folders")
    parser.add_argument("--source", type=str, default="/root/taojianwei/datasets/IRIS/20260703_datasets", help="LabelMe JSON/image directory")
    parser.add_argument("--output", type=str, default="/root/taojianwei/datasets/IRIS/20260703_datasets_cls_crop", help="Output ImageFolder dataset directory")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio per class")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    parser.add_argument("--target_min_dim", type=int, default=160, help="Minimum square crop side before resizing")
    parser.add_argument("--context_scale", type=float, default=2.0, help="Crop side = max(obj_long_side * scale, target_min_dim)")
    parser.add_argument("--max_context_ratio", type=float, default=4.0, help="Maximum crop side as obj_long_side * ratio, lower-bounded by target_min_dim")
    parser.add_argument("--input_size", type=int, default=224, help="Resize saved crops to this square size; <=0 keeps crop size")
    parser.add_argument("--min_obj_long_side", type=float, default=32.0, help="Long side threshold for optional low-quality subfolder")
    parser.add_argument("--low_quality_subdir", action="store_true", help="Place very small objects under class/_low_quality_small_obj")
    parser.add_argument("--exclude", type=str, default="hand,obj", help="Comma-separated labels to exclude")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    exclude_classes = {label.strip() for label in args.exclude.split(",") if label.strip()}
    input_size = args.input_size if args.input_size and args.input_size > 0 else None

    items = collect_annotations(source_dir, exclude_classes)
    if not items:
        raise ValueError(f"No classification annotations found in {source_dir} after excluding {sorted(exclude_classes)}")

    train_items, val_items = split_by_class(items, args.train_ratio, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_class_index(output_dir, sorted({item["label"] for item in items}))

    train_counts = save_crops(
        train_items,
        "train",
        output_dir,
        args.target_min_dim,
        args.context_scale,
        args.max_context_ratio,
        input_size,
        args.min_obj_long_side,
        args.low_quality_subdir,
    )
    val_counts = save_crops(
        val_items,
        "val",
        output_dir,
        args.target_min_dim,
        args.context_scale,
        args.max_context_ratio,
        input_size,
        args.min_obj_long_side,
        args.low_quality_subdir,
    )

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Excluded labels: {sorted(exclude_classes)}")
    print(f"Total crops: {len(items)} | train: {len(train_items)} | val: {len(val_items)}")
    print("\nPer-class counts:")
    print(f"{'class':<18} {'train':>8} {'val':>8} {'total':>8}")
    print("-" * 46)
    for label in sorted({item["label"] for item in items}):
        train_n, val_n = train_counts[label], val_counts[label]
        print(f"{label:<18} {train_n:8d} {val_n:8d} {train_n + val_n:8d}")


if __name__ == "__main__":
    main()
