#!/usr/bin/env python3
"""
统计 tao_worker_collect 文件夹中 JSON 文件内不同标签的数量。
用法:
    python count_labels.py                              # 默认统计桌面上的 tao_worker_collect
    python count_labels.py --dir /path/to/folder        # 统计指定文件夹
"""

import argparse
import json
import sys
import os
from collections import Counter
from glob import glob


def count_labels(directory):
    json_files = glob(os.path.join(directory, "*.json"))
    if not json_files:
        print(f"错误: 在 {directory} 中未找到 JSON 文件")
        return

    total_counter = Counter()

    for fp in sorted(json_files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 无法读取 {os.path.basename(fp)} — {e}", file=sys.stderr)
            continue

        shapes = data.get("shapes", [])
        labels = [s.get("label", "unknown") for s in shapes]
        if not labels:
            continue

        for label in labels:
            total_counter[label] += 1

    if not total_counter:
        print("没有找到有效标签。")
        return

    total = sum(total_counter.values())

    # 按数量降序排列
    sorted_labels = total_counter.most_common()

    max_label_len = max(len(k) for k, _ in sorted_labels)
    print("=" * (max_label_len + 30))
    print(f"{'标签':<{max_label_len}}  {'数量':>6}  {'占比':>7}")
    print("=" * (max_label_len + 30))
    for label, cnt in sorted_labels:
        pct = cnt / total * 100
        print(f"{label:<{max_label_len}}  {cnt:>6}  {pct:>6.2f}%")
    print("=" * (max_label_len + 30))
    print(f"{'合计':<{max_label_len}}  {total:>6}")



def main():
    parser = argparse.ArgumentParser(
        description="统计 JSON 标注文件中各标签的数量"
    )
    parser.add_argument(
        "--dir",
        default=os.path.expanduser("~/Desktop/tao_worker_collect"),
        help="目标文件夹路径 (默认: ~/Desktop/tao_worker_collect)",
    )
    args = parser.parse_args()

    target_dir = os.path.abspath(os.path.expanduser(args.dir))
    if not os.path.isdir(target_dir):
        print(f"错误: 目录不存在 — {target_dir}")
        sys.exit(1)

    count_labels(target_dir)


if __name__ == "__main__":
    main()
