#!/usr/bin/env python3
"""
给文件夹下的图片及其对应的 LabelMe JSON 标注文件统一添加前缀，
并同步更新 JSON 中的 imagePath 字段。

用法:
    python3 rename_prefix.py --prefix my_prefix_                          # 默认操作当前目录
    python3 rename_prefix.py --prefix proj1_ --dir /path/to/folder        # 指定文件夹
    python3 rename_prefix.py --prefix v2_ --img-ext jpg                   # 指定图片后缀
    python3 rename_prefix.py --prefix test_ --dry-run                     # 预览模式，不实际修改
"""

import argparse
import json
import os
import sys
from glob import glob


def rename_with_prefix(directory, prefix, img_ext="png", dry_run=False):
    """
    重命名图片和 JSON 文件，添加前缀，并更新 JSON 中的 imagePath。

    Args:
        directory:  目标文件夹路径
        prefix:     要添加的前缀字符串
        img_ext:    图片文件后缀 (默认 png)
        dry_run:    True 时只预览，不实际修改文件
    """
    img_pattern = os.path.join(directory, f"*.{img_ext.lstrip('.')}")
    img_files = glob(img_pattern)
    img_files += glob(os.path.join(directory, f"*.{img_ext.lstrip('.').upper()}"))

    if not img_files:
        print(f"错误: 在 {directory} 中未找到 .{img_ext} 图片文件")
        return

    # 构建 stem -> 图片路径 的映射
    stem_to_img = {}
    for fp in img_files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        stem_to_img[stem] = fp

    json_files = glob(os.path.join(directory, "*.json"))
    # 构建 stem -> json 路径 的映射
    stem_to_json = {}
    for fp in json_files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        stem_to_json[stem] = fp

    # 找到配对的文件 (同名 stem)
    paired_stems = sorted(set(stem_to_img.keys()) & set(stem_to_json.keys()))
    unpaired_imgs = sorted(set(stem_to_img.keys()) - set(stem_to_json.keys()))
    unpaired_jsons = sorted(set(stem_to_json.keys()) - set(stem_to_img.keys()))

    if not paired_stems:
        print("错误: 没有找到任何配对 (同名 stem) 的图片和 JSON 文件")
        if unpaired_imgs:
            print(f"  只有图片无 JSON: {unpaired_imgs}")
        if unpaired_jsons:
            print(f"  只有 JSON 无图片: {unpaired_jsons}")
        return

    print(f"目标文件夹: {directory}")
    print(f"前缀:       \"{prefix}\"")
    print(f"图片后缀:   .{img_ext}")
    print(f"配对文件:   {len(paired_stems)} 对")
    if unpaired_imgs:
        print(f"只有图片:   {len(unpaired_imgs)} 个 (将跳过)")
    if unpaired_jsons:
        print(f"只有 JSON:  {len(unpaired_jsons)} 个 (将跳过)")
    print(f"模式:       {'dry-run (仅预览)' if dry_run else '实际重命名'}")
    print("-" * 60)

    renamed_count = 0
    for stem in paired_stems:
        old_img_path = stem_to_img[stem]
        old_json_path = stem_to_json[stem]

        img_ext_actual = os.path.splitext(old_img_path)[1]  # 保留原大小写后缀
        new_stem = f"{prefix}{stem}"
        new_img_name = f"{new_stem}{img_ext_actual}"
        new_json_name = f"{new_stem}.json"

        new_img_path = os.path.join(directory, new_img_name)
        new_json_path = os.path.join(directory, new_json_name)

        # ---------- 读取 JSON，更新 imagePath ----------
        try:
            with open(old_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 无法读取 {os.path.basename(old_json_path)} — {e}", file=sys.stderr)
            continue

        old_image_path_value = data.get("imagePath", "")
        data["imagePath"] = new_img_name

        # ---------- 执行重命名 ----------
        if dry_run:
            print(f"  [预览] {os.path.basename(old_img_path)}  →  {new_img_name}")
            print(f"  [预览] {os.path.basename(old_json_path)} →  {new_json_name}")
            print(f"         imagePath: \"{old_image_path_value}\" → \"{new_img_name}\"")
        else:
            os.rename(old_img_path, new_img_path)
            os.rename(old_json_path, new_json_path)
            with open(new_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  [已重命名] {os.path.basename(old_img_path)}  →  {new_img_name}")
            print(f"             {os.path.basename(old_json_path)} →  {new_json_name}")
            print(f"             imagePath → \"{new_img_name}\"")

        renamed_count += 1

    print("-" * 60)
    print(f"总计: {renamed_count} 对文件", end="")
    if dry_run:
        print(" (预览模式，未实际写入)")
    else:
        print()


def main():
    parser = argparse.ArgumentParser(
        description="给图片和 LabelMe JSON 标注文件统一添加前缀，并同步更新 imagePath"
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="要添加的前缀字符串 (例如: proj1_、batch2_)",
    )
    parser.add_argument(
        "--dir",
        default=os.getcwd(),
        help="目标文件夹路径 (默认: 当前目录)",
    )
    parser.add_argument(
        "--img-ext",
        default="png",
        help="图片文件后缀 (默认: png)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，只显示会修改哪些文件，不实际写入",
    )
    args = parser.parse_args()

    target_dir = os.path.abspath(os.path.expanduser(args.dir))
    if not os.path.isdir(target_dir):
        print(f"错误: 目录不存在 — {target_dir}")
        sys.exit(1)

    img_ext = args.img_ext.strip().lstrip(".")
    rename_with_prefix(target_dir, args.prefix, img_ext, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
