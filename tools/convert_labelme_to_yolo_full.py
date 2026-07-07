#!/usr/bin/env python3
"""Convert labelme JSON annotations to YOLO format.
Auto-detect all classes from JSON files. Usage:
  python tools/convert_labelme_to_yolo_full.py --source /path/to/labelme_folder --output /path/to/yolo_folder
"""

import json
import shutil
import random
import argparse
from pathlib import Path

import yaml


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
            label = str(shape.get('label', '')).strip()
            if label:
                classes.add(label)
    return sorted(list(classes))


def convert_labelme_to_yolo(json_path, img_width, img_height, class_names):
    """Convert labelme JSON to YOLO format annotations."""
    data = load_labelme_json(json_path)
    yolo_annotations = []

    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

    for shape in data.get('shapes', []):
        label = str(shape.get('label', '')).strip()
        if not label or label not in class_name_to_id:
            continue

        class_id = class_name_to_id[label]
        points = shape['points']

        if shape['shape_type'] == 'rectangle':
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
        elif shape['shape_type'] == 'polygon':
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
        else:
            continue

        center_x = (x_min + x_max) / 2 / img_width
        center_y = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    return yolo_annotations


def process_dataset(source_dir, output_dir, train_ratio=0.9, seed=42):
    """Process dataset: convert annotations and split into train/val."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    train_images = images_dir / 'train'
    val_images = images_dir / 'val'
    train_labels = labels_dir / 'train'
    val_labels = labels_dir / 'val'

    for d in [train_images, val_images, train_labels, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    json_files = [f for f in source_path.glob('*.json') if not f.name.startswith('._')]
    if not json_files:
        raise ValueError(f"No JSON files found in {source_dir}")
    print(f"Found {len(json_files)} JSON files")

    class_names = get_class_names(json_files)
    print(f"Detected classes ({len(class_names)}): {class_names}")

    file_pairs = []
    for json_file in json_files:
        base_name = json_file.stem
        img_file = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP', '.webp', '.WEBP']:
            candidate = source_path / (base_name + ext)
            if candidate.exists():
                img_file = candidate
                break
        if img_file:
            file_pairs.append((json_file, img_file))

    print(f"Found {len(file_pairs)} image-annotation pairs")

    random.seed(seed)
    random.shuffle(file_pairs)
    split_idx = int(len(file_pairs) * train_ratio)
    train_pairs = file_pairs[:split_idx]
    val_pairs = file_pairs[split_idx:]
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    for json_file, img_file in train_pairs:
        _process_pair(json_file, img_file, train_images, train_labels, class_names)

    for json_file, img_file in val_pairs:
        _process_pair(json_file, img_file, val_images, val_labels, class_names)

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

    with Image.open(img_file) as img:
        img_width, img_height = img.size

    yolo_annotations = convert_labelme_to_yolo(json_file, img_width, img_height, class_names)

    new_img_name = f"{img_file.stem}{img_file.suffix}"
    shutil.copy(img_file, img_output / new_img_name)

    label_file = label_output / f"{img_file.stem}.txt"
    with open(label_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将 LabelMe JSON 转换为 YOLO 格式，自动检测所有类别')
    parser.add_argument('--source', required=True, help='LabelMe 数据集文件夹（包含 json 和图片）')
    parser.add_argument('--output', required=True, help='YOLO 数据集输出文件夹')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='训练集比例（默认 0.9）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认 42）')
    args = parser.parse_args()

    print(f"Converting dataset from {args.source}")
    print(f"Output directory: {args.output}")

    process_dataset(args.source, args.output, train_ratio=args.train_ratio, seed=args.seed)
