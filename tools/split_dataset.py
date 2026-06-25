#!/usr/bin/env python3
"""
将json和图片文件按指定数量切分到不同的文件夹。

用法:
    python tools/split_dataset.py --json_dir labels/ --image_dir images/ --output_dir splits/ --num_splits 3
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple


def get_file_pairs(json_dir: str, image_dir: str) -> List[Tuple[Path, Path]]:
    """获取json和对应的图片文件对

    Returns:
        [(json_file, image_file), ...]
    """
    json_path = Path(json_dir)
    image_path = Path(image_dir)

    json_files = sorted([f for f in json_path.glob('*.json') if not f.name.startswith('._')])
    file_pairs = []

    for json_file in json_files:
        # 查找对应的图片文件
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            img_file = image_path / (json_file.stem + ext)
            if img_file.exists():
                file_pairs.append((json_file, img_file))
                break

    return file_pairs


def split_dataset(file_pairs: List[Tuple[Path, Path]], num_splits: int) -> dict:
    """按顺序均分文件对到不同的split"""
    splits = {f'split_{i+1}': [] for i in range(num_splits)}

    total = len(file_pairs)
    split_size = total // num_splits

    for i in range(num_splits):
        start = i * split_size
        end = start + split_size if i < num_splits - 1 else total
        splits[f'split_{i+1}'] = file_pairs[start:end]

    return splits


def main():
    parser = argparse.ArgumentParser(description='切分json和图片文件')
    parser.add_argument('--json_dir', type=str, required=True, help='json文件夹路径')
    parser.add_argument('--image_dir', type=str, required=True, help='图片文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--num_splits', type=int, default=3, help='切分数量 (默认 3)')
    args = parser.parse_args()

    # 获取文件对
    file_pairs = get_file_pairs(args.json_dir, args.image_dir)
    print(f"找到 {len(file_pairs)} 个文件对")

    if not file_pairs:
        print("未找到任何文件对")
        return

    # 切分数据
    splits = split_dataset(file_pairs, args.num_splits)

    # 创建输出目录并复制文件
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, pairs in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)

        json_split_dir = split_dir / 'json'
        image_split_dir = split_dir / 'images'
        json_split_dir.mkdir(exist_ok=True)
        image_split_dir.mkdir(exist_ok=True)

        for json_file, img_file in pairs:
            shutil.copy(str(json_file), str(json_split_dir / json_file.name))
            shutil.copy(str(img_file), str(image_split_dir / img_file.name))

        print(f"✓ {split_name}: {len(pairs)} 个文件对")

    print(f"\n完成! 文件保存到: {output_path}")


if __name__ == '__main__':
    main()
