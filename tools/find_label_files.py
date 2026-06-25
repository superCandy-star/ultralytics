#!/usr/bin/env python3
"""
根据标签名称查找包含该标签的所有 JSON 文件。
用法:
    python find_label_files.py <标签名>                              # 默认在桌面 tao_worker_collect 中查找
    python find_label_files.py <标签名> --dir /path/to/folder        # 在指定文件夹中查找
    python find_label_files.py <标签名> --list                       # 只列出文件名，不显示详情
"""

import argparse
import json
import sys
import os
from glob import glob


def normalize(name):
    """统一化标签名：转小写，下划线/连字符替换为空格，去除首尾空格"""
    return name.lower().replace("_", " ").replace("-", " ").strip()


def find_label(directory, label_name, list_only=False):
    json_files = glob(os.path.join(directory, "*.json"))
    if not json_files:
        print(f"错误: 在 {directory} 中未找到 JSON 文件")
        return

    search_key = normalize(label_name)
    matched = []

    for fp in sorted(json_files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 无法读取 {os.path.basename(fp)} — {e}", file=sys.stderr)
            continue

        shapes = data.get("shapes", [])
        labels_in_file = [s.get("label", "") for s in shapes]

        if any(normalize(l) == search_key for l in labels_in_file):
            matched.append((os.path.basename(fp), labels_in_file))

    if not matched:
        print(f"未找到包含标签 \"{label_name}\" 的文件")
        return

    # 显示匹配到的实际标签名（取第一个文件中的原始写法）
    actual_label = None
    for _, labels in matched:
        for l in labels:
            if normalize(l) == search_key:
                actual_label = l
                break
        if actual_label:
            break

    display_name = f"{label_name} → {actual_label}" if actual_label != label_name else label_name
    print(f"找到 {len(matched)} 个包含标签 \"{display_name}\" 的文件:\n")

    if list_only:
        for fname, _ in matched:
            print(f"  {fname}")
        return

    for fname, labels in matched:
        cnt = sum(1 for l in labels if normalize(l) == search_key)
        all_labels = ", ".join(sorted(set(labels)))
        print(f"  {fname}  (该标签出现 {cnt} 次, 所有标签: {all_labels})")


def main():
    parser = argparse.ArgumentParser(
        description="查找包含指定标签的 JSON 标注文件"
    )
    parser.add_argument("label", help="要查找的标签名称")
    parser.add_argument(
        "--dir",
        default=os.path.expanduser("~/Desktop/tao_worker_collect"),
        help="目标文件夹路径 (默认: ~/Desktop/tao_worker_collect)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只列出文件名，不显示详情",
    )
    args = parser.parse_args()

    target_dir = os.path.abspath(os.path.expanduser(args.dir))
    if not os.path.isdir(target_dir):
        print(f"错误: 目录不存在 — {target_dir}")
        sys.exit(1)

    find_label(target_dir, args.label, list_only=args.list)


if __name__ == "__main__":
    main()
