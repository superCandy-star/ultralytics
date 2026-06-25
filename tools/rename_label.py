#!/usr/bin/env python3
"""
批量替换 JSON 标注文件中的标签名称。
用法:
    python3 rename_label.py <旧标签> <新标签>                         # 默认操作桌面 tao_worker_collect
    python3 rename_label.py <旧标签> <新标签> --dir /path/to/folder    # 指定文件夹
    python3 rename_label.py <旧标签> <新标签> --dry-run                # 预览模式，不实际修改
"""

import argparse
import json
import os
import sys
from glob import glob


def rename_label(directory, old_label, new_label, dry_run=False):
    json_files = glob(os.path.join(directory, "*.json"))
    if not json_files:
        print(f"错误: 在 {directory} 中未找到 JSON 文件")
        return

    changed_files = 0
    total_replacements = 0

    for fp in sorted(json_files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 无法读取 {os.path.basename(fp)} — {e}", file=sys.stderr)
            continue

        shapes = data.get("shapes", [])
        modified = False
        file_count = 0

        for shape in shapes:
            if shape.get("label") == old_label:
                shape["label"] = new_label
                modified = True
                file_count += 1
                total_replacements += 1

        if modified:
            changed_files += 1
            fname = os.path.basename(fp)
            if dry_run:
                print(f"  [预览] {fname}  — 将替换 {file_count} 处 \"{old_label}\" → \"{new_label}\"")
            else:
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  [已修改] {fname}  — 替换 {file_count} 处 \"{old_label}\" → \"{new_label}\"")

    print(f"\n总计: {changed_files} 个文件, {total_replacements} 处替换", end="")
    if dry_run:
        print(" (预览模式，未实际写入)")
    else:
        print()


def main():
    parser = argparse.ArgumentParser(
        description="批量替换 JSON 标注文件中的标签名称"
    )
    parser.add_argument("old_label", help="要替换的旧标签名称")
    parser.add_argument("new_label", help="新标签名称")
    parser.add_argument(
        "--dir",
        default=os.path.expanduser("~/Desktop/tao_worker_collect"),
        help="目标文件夹路径 (默认: ~/Desktop/tao_worker_collect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，只显示会修改哪些文件，不实际写入",
    )
    args = parser.parse_args()

    if args.old_label == args.new_label:
        print("新旧标签名称相同，无需操作")
        return

    target_dir = os.path.abspath(os.path.expanduser(args.dir))
    if not os.path.isdir(target_dir):
        print(f"错误: 目录不存在 — {target_dir}")
        sys.exit(1)

    rename_label(target_dir, args.old_label, args.new_label, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
