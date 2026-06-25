#!/usr/bin/env python3
"""
使用现有data.yaml合并新的labelme数据集
支持：
  1. 读取现有data.yaml中的类别索引
  2. 扫描新数据中的标签，自动合并新类别
  3. 按统一的类别索引转换为YOLO格式
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Set, Tuple
from PIL import Image
import shutil
import random


def load_data_yaml(yaml_path: str) -> Dict:
    """加载现有data.yaml文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    names = data.get('names', {})
    if isinstance(names, dict):
        class_id_to_name = names
    else:
        class_id_to_name = {i: n for i, n in enumerate(names)}

    return class_id_to_name


def load_labelme_json(json_path: str) -> dict:
    """加载labelme JSON文件，处理多种编码"""
    encodings = ['utf-8', 'gb2312', 'latin1']
    for encoding in encodings:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Cannot decode {json_path}")


def extract_all_classes(data_dir: str) -> Set[str]:
    """扫描文件夹，提取所有类别"""
    classes = set()
    data_path = Path(data_dir)
    json_files = [f for f in data_path.glob('*.json') if not f.name.startswith('._')]

    for json_file in json_files:
        try:
            data = load_labelme_json(str(json_file))
            for shape in data.get('shapes', []):
                classes.add(shape['label'])
        except Exception as e:
            print(f"  跳过 {json_file.name}: {e}")

    return classes


def merge_classes(existing_classes: Dict, new_classes: Set[str]) -> Dict:
    """合并现有和新类别"""
    existing_names = set(existing_classes.values())
    merged = existing_classes.copy()

    max_id = max(merged.keys()) if merged else -1

    for class_name in sorted(new_classes):
        if class_name not in existing_names:
            max_id += 1
            merged[max_id] = class_name
            print(f"    新增类别: {class_name} → ID {max_id}")

    return merged


def convert_to_yolo(json_path: str, img_width: int, img_height: int,
                    class_name_to_id: Dict) -> list:
    """转换labelme JSON到YOLO格式"""
    data = load_labelme_json(json_path)
    annotations = []

    for shape in data.get('shapes', []):
        label = shape['label']
        if label not in class_name_to_id:
            continue

        class_id = class_name_to_id[label]
        points = shape['points']

        if shape['shape_type'] in ['rectangle', 'polygon']:
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

        annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    return annotations


def process_dataset(data_dir: str, class_name_to_id: Dict, output_dir: str, include_neg_samples: bool = False):
    """处理数据集

    Args:
        data_dir: 数据文件夹
        class_name_to_id: 类别名到ID的映射
        output_dir: 输出目录
        include_neg_samples: 是否包含负样本（无标注图片）
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    train_img = output_path / 'images' / 'train'
    val_img = output_path / 'images' / 'val'
    train_label = output_path / 'labels' / 'train'
    val_label = output_path / 'labels' / 'val'

    for d in [train_img, val_img, train_label, val_label]:
        d.mkdir(parents=True, exist_ok=True)

    # 找到所有图片文件（排除 macOS 临时文件）
    img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    img_files = [f for f in data_path.glob('*') if f.suffix in img_extensions and not f.name.startswith('._')]

    print(f"  找到 {len(img_files)} 个图片文件")

    file_pairs = []
    neg_samples = []

    for img_file in img_files:
        json_file = data_path / (img_file.stem + '.json')
        if json_file.exists():
            file_pairs.append((json_file, img_file))
        else:
            if include_neg_samples:
                neg_samples.append(img_file)

    print(f"  正样本（有标注）: {len(file_pairs)}")
    if neg_samples:
        print(f"  负样本（无标注）: {len(neg_samples)}")

    # 分割为训练集和验证集
    random.shuffle(file_pairs)
    split_idx = int(len(file_pairs) * 0.9)
    train_pairs = file_pairs[:split_idx]
    val_pairs = file_pairs[split_idx:]

    print(f"  训练集: {len(train_pairs)}, 验证集: {len(val_pairs)}")

    # 处理训练集
    for json_file, img_file in train_pairs:
        with Image.open(img_file) as img:
            w, h = img.size
        annotations = convert_to_yolo(str(json_file), w, h, class_name_to_id)
        if annotations:
            shutil.copy(img_file, train_img / img_file.name)
            with open(train_label / f"{img_file.stem}.txt", 'w') as f:
                f.write('\n'.join(annotations))

    # 处理验证集
    for json_file, img_file in val_pairs:
        with Image.open(img_file) as img:
            w, h = img.size
        annotations = convert_to_yolo(str(json_file), w, h, class_name_to_id)
        if annotations:
            shutil.copy(img_file, val_img / img_file.name)
            with open(val_label / f"{img_file.stem}.txt", 'w') as f:
                f.write('\n'.join(annotations))

    # 处理负样本
    if include_neg_samples and neg_samples:
        random.shuffle(neg_samples)
        neg_split = int(len(neg_samples) * 0.9)
        for img_file in neg_samples[:neg_split]:
            shutil.copy(img_file, train_img / img_file.name)
            # 创建空标注文件
            with open(train_label / f"{img_file.stem}.txt", 'w') as f:
                f.write('')
        for img_file in neg_samples[neg_split:]:
            shutil.copy(img_file, val_img / img_file.name)
            with open(val_label / f"{img_file.stem}.txt", 'w') as f:
                f.write('')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='合并新数据集到现有data.yaml')
    parser.add_argument('--dir', type=str, required=True, help='新数据文件夹路径')
    parser.add_argument('--data_yaml', type=str, required=True, help='现有data.yaml文件路径')
    parser.add_argument('--save_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--include_neg', action='store_true', help='包含负样本（无标注图片）')
    args = parser.parse_args()

    print(f"加载现有data.yaml: {args.data_yaml}")
    existing_classes = load_data_yaml(args.data_yaml)
    print(f"  现有类别数: {len(existing_classes)}")

    print(f"\n扫描新数据: {args.dir}")
    new_classes = extract_all_classes(args.dir)
    print(f"  新数据类别数: {len(new_classes)}")

    print(f"\n合并类别")
    merged_classes = merge_classes(existing_classes, new_classes)
    print(f"  合并后类别数: {len(merged_classes)}")

    print(f"\n转换数据集")
    process_dataset(args.dir, {v: k for k, v in merged_classes.items()}, args.save_dir, args.include_neg)

    # 保存更新的data.yaml
    output_path = Path(args.save_dir)
    data_yaml = {
        'path': str(output_path),
        'train': str(output_path / 'images' / 'train'),
        'val': str(output_path / 'images' / 'val'),
        'nc': len(merged_classes),
        'names': merged_classes
    }

    yaml_out = output_path / 'data.yaml'
    with open(yaml_out, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n完成! 已保存到 {args.save_dir}")
    print(f"  data.yaml: {yaml_out}")


if __name__ == '__main__':
    main()

