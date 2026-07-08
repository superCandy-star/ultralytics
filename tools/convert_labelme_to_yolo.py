#!/usr/bin/env python3
"""Convert labelme JSON annotations to YOLOv11 format."""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
import yaml


CLASS_NAMES = ['hand', 'obj']


def load_labelme_json(json_path):
    """Load labelme JSON file."""
    encodings = ['utf-8', 'gb2312', 'latin1']
    for encoding in encodings:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Cannot decode {json_path}")


def get_class_names(json_files):
    """Extract unique class names from all JSON files."""
    classes = set()
    for json_file in json_files:
        data = load_labelme_json(json_file)
        for shape in data.get('shapes', []):
            classes.add(shape['label'])
    return sorted(list(classes))


def convert_labelme_to_yolo(json_path, img_width, img_height, class_names):
    """Convert labelme JSON to YOLO format annotations."""
    data = load_labelme_json(json_path)
    yolo_annotations = []

    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

    for shape in data.get('shapes', []):
        label = str(shape.get('label', '')).strip()
        if not label:
            continue

        mapped_label = 'hand' if label == 'hand' else 'obj'
        class_id = class_name_to_id[mapped_label]
        points = shape['points']

        if shape['shape_type'] == 'rectangle':
            # Get bounding box from two points
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
        elif shape['shape_type'] == 'polygon':
            # Get bounding box from polygon
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
        else:
            continue

        # Convert to YOLO format (normalized)
        center_x = (x_min + x_max) / 2 / img_width
        center_y = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # Clamp values to [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    return yolo_annotations


def process_dataset(source_dir, output_dir, train_ratio=0.9):
    """Process dataset: convert annotations and split into train/val."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Create directory structure
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    train_images = images_dir / 'train'
    val_images = images_dir / 'val'
    train_labels = labels_dir / 'train'
    val_labels = labels_dir / 'val'

    for d in [train_images, val_images, train_labels, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # Find JSON files (skip macOS system files)
    json_files = [f for f in source_path.glob('*.json') if not f.name.startswith('._')]
    if not json_files:
        raise ValueError(f"No JSON files found in {source_dir}")

    print(f"Found {len(json_files)} JSON files")

    # Fixed class mapping: hand -> 0, all other labels -> obj -> 1
    class_names = CLASS_NAMES
    print(f"Classes: {class_names}")

    # Prepare file pairs
    file_pairs = []
    for json_file in json_files:
        # Find corresponding image file
        base_name = json_file.stem
        img_file = None
        for ext in ['.jpg', '.png', '.JPG', '.PNG']:
            candidate = source_path / (base_name + ext)
            if candidate.exists():
                img_file = candidate
                break

        if img_file:
            file_pairs.append((json_file, img_file))

    print(f"Found {len(file_pairs)} image-annotation pairs")

    # Split into train/val
    random.shuffle(file_pairs)
    split_idx = int(len(file_pairs) * train_ratio)
    train_pairs = file_pairs[:split_idx]
    val_pairs = file_pairs[split_idx:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Process each pair
    for json_file, img_file in train_pairs:
        _process_pair(json_file, img_file, train_images, train_labels, class_names)

    for json_file, img_file in val_pairs:
        _process_pair(json_file, img_file, val_images, val_labels, class_names)

    # Create data.yaml
    data_yaml = {
        'path': str(output_path),
        'train': str(train_images),
        'val': str(val_images),
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"Data.yaml saved to {yaml_path}")
    print("Conversion complete!")


def _process_pair(json_file, img_file, img_output, label_output, class_names):
    """Process a single image-annotation pair."""
    from PIL import Image

    # Get image dimensions
    with Image.open(img_file) as img:
        img_width, img_height = img.size

    # Convert annotations
    yolo_annotations = convert_labelme_to_yolo(json_file, img_width, img_height, class_names)

    # Copy image
    new_img_name = f"{img_file.stem}{img_file.suffix}"
    shutil.copy(img_file, img_output / new_img_name)

    # Save label
    label_file = label_output / f"{img_file.stem}.txt"
    with open(label_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))


if __name__ == '__main__':
    source = '/root/taojianwei/datasets/IRIS/20260708_datasets'
    output = '/root/taojianwei/datasets/IRIS/20260708_datasets_full_yolo'

    print(f"Converting dataset from {source}")
    print(f"Output directory: {output}")

    process_dataset(source, output, train_ratio=0.9)
