#!/usr/bin/env python3
"""
统计data.yaml中train和val的不同类别数量。

用法:
    python tools/count_classes.py --data_yaml data.yaml
"""

import argparse
import yaml
from pathlib import Path
from collections import defaultdict


def count_classes_in_dir(labels_dir: str) -> dict:
    """统计目录中所有txt文件的类别数量"""
    class_counts = defaultdict(int)
    labels_path = Path(labels_dir)

    if not labels_path.exists():
        print(f"  ⚠ 目录不存在: {labels_dir}")
        return class_counts

    txt_files = list(labels_path.glob('*.txt'))

    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

    return class_counts


def main():
    parser = argparse.ArgumentParser(description='统计data.yaml中的类别数量')
    parser.add_argument('--data_yaml', type=str, required=True, help='data.yaml文件路径')
    args = parser.parse_args()

    # 读取yaml文件
    with open(args.data_yaml) as f:
        data = yaml.safe_load(f)

    # 获取类别名称
    names = data.get('names', {})
    if isinstance(names, list):
        id_to_name = {i: n for i, n in enumerate(names)}
    else:
        id_to_name = names

    print(f"类别映射: {id_to_name}\n")

    # 统计train集
    train_dirs = data.get('train', [])
    if isinstance(train_dirs, str):
        train_dirs = [train_dirs]

    print("=" * 60)
    print("TRAIN 集统计")
    print("=" * 60)

    train_total = defaultdict(int)
    for img_dir in train_dirs:
        labels_dir = img_dir.replace('/images/', '/labels/')
        print(f"\n目录: {labels_dir}")
        counts = count_classes_in_dir(labels_dir)

        for class_id, count in sorted(counts.items()):
            class_name = id_to_name.get(class_id, f"Class_{class_id}")
            print(f"  {class_id}: {class_name:20s} → {count:6d}")
            train_total[class_id] += count

    print(f"\nTRAIN 总计:")
    for class_id in sorted(train_total.keys()):
        class_name = id_to_name.get(class_id, f"Class_{class_id}")
        print(f"  {class_id}: {class_name:20s} → {train_total[class_id]:6d}")

    # 统计val集
    val_dirs = data.get('val', [])
    if isinstance(val_dirs, str):
        val_dirs = [val_dirs]

    print("\n" + "=" * 60)
    print("VAL 集统计")
    print("=" * 60)

    val_total = defaultdict(int)
    for img_dir in val_dirs:
        labels_dir = img_dir.replace('/images/', '/labels/')
        print(f"\n目录: {labels_dir}")
        counts = count_classes_in_dir(labels_dir)

        for class_id, count in sorted(counts.items()):
            class_name = id_to_name.get(class_id, f"Class_{class_id}")
            print(f"  {class_id}: {class_name:20s} → {count:6d}")
            val_total[class_id] += count

    print(f"\nVAL 总计:")
    for class_id in sorted(val_total.keys()):
        class_name = id_to_name.get(class_id, f"Class_{class_id}")
        print(f"  {class_id}: {class_name:20s} → {val_total[class_id]:6d}")

    # 总体统计
    print("\n" + "=" * 60)
    print("总体统计")
    print("=" * 60)
    all_classes = set(train_total.keys()) | set(val_total.keys())
    for class_id in sorted(all_classes):
        class_name = id_to_name.get(class_id, f"Class_{class_id}")
        train_count = train_total.get(class_id, 0)
        val_count = val_total.get(class_id, 0)
        total = train_count + val_count
        print(f"  {class_id}: {class_name:20s} → Train: {train_count:6d}  Val: {val_count:6d}  Total: {total:6d}")

    train_sum = sum(train_total.values())
    val_sum = sum(val_total.values())
    print(f"\n总数: Train={train_sum}  Val={val_sum}  Total={train_sum + val_sum}")


if __name__ == '__main__':
    main()
