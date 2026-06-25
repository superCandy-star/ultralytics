#!/usr/bin/env python3
"""
将文件夹中的 json 及其对应的 png 配对复制到目标文件夹。

用法:
    python3 pair_copy.py <源文件夹> <目标文件夹> [选项]

选项:
    --img-ext png      图片后缀 (默认: png)
    --json-dir PATH    json 所在的子文件夹 (默认: 与源文件夹相同)
    --img-dir  PATH    图片所在的子文件夹 (默认: 与源文件夹相同)
    -n, --dry-run      只打印，不实际复制
"""

import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="配对复制 json + 图片到目标文件夹")
    parser.add_argument("src", type=Path, help="源文件夹（包含 json 和图片）")
    parser.add_argument("dst", type=Path, help="目标文件夹")
    parser.add_argument("--img-ext", default="png", help="图片后缀 (默认: png)")
    parser.add_argument("--json-dir", type=Path, default=None, help="json 所在子文件夹")
    parser.add_argument("--img-dir", type=Path, default=None, help="图片所在子文件夹")
    parser.add_argument("-n", "--dry-run", action="store_true", help="只预览不复制")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    json_dir = (args.json_dir or src).resolve()
    img_dir = (args.img_dir or src).resolve()
    img_ext = args.img_ext.lstrip(".")

    if not src.is_dir():
        sys.exit(f"错误: 源文件夹不存在: {src}")
    if not json_dir.is_dir():
        sys.exit(f"错误: json 文件夹不存在: {json_dir}")

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        sys.exit(f"没有找到 .json 文件: {json_dir}")

    pairs = []
    missing = []

    for jf in json_files:
        stem = jf.stem  # 去掉 .json 的文件名
        img = img_dir / f"{stem}.{img_ext}"
        if img.is_file():
            pairs.append((jf, img))
        else:
            missing.append((jf, stem))

    if not pairs:
        sys.exit(f"没有找到任何匹配的 .{img_ext} 图片 (已扫描 {len(json_files)} 个 json)")

    # ---------- 输出摘要 ----------
    print(f"源文件夹:   {src}")
    print(f"目标文件夹: {dst}")
    print(f"json 目录:  {json_dir}")
    print(f"图片目录:   {img_dir}")
    print(f"图片后缀:   .{img_ext}")
    print(f"匹配成功:   {len(pairs)} 对")
    print(f"缺失图片:   {len(missing)} 个")
    print(f"模式:       {'dry-run (仅预览)' if args.dry_run else '实际复制'}")
    print("-" * 50)

    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for jf, img in pairs:
        print(f"  {jf.name}  +  {img.name}")
        if not args.dry_run:
            shutil.copy2(jf, dst / jf.name)
            shutil.copy2(img, dst / img.name)
        copied += 2

    if missing:
        print("-" * 50)
        print(f"缺失图片 ({len(missing)} 个):")
        for _, stem in missing:
            print(f"  {stem}.{img_ext}")

    print("-" * 50)
    if args.dry_run:
        print(f"预览完成: 将复制 {copied} 个文件")
    else:
        print(f"完成: 已复制 {copied} 个文件到 {dst}")


if __name__ == "__main__":
    main()
