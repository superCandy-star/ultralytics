#!/usr/bin/env python3
"""
按 *_worker_frame 前缀分组，从每组中随机选取指定比例复制到目标文件夹。

用法:
    # 预览模式（不复制）
    python tools/split_worker_frames.py --folder /path/to/datasets --output /path/to/output

    # 实际复制 80%
    python tools/split_worker_frames.py --folder /path/to/datasets --output /path/to/output --ratio 0.8 --copy

    # 自定义比例和随机种子
    python tools/split_worker_frames.py --folder /path/to/datasets --output /path/to/output --ratio 0.7 --seed 123 --copy
"""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


WORKER_FRAME_PATTERN = re.compile(r'^(.+_worker_frame)_\d+\.(jpg|jpeg|png|json)$')


def group_files(folder):
    """按 *_worker_frame 前缀分组，每组返回所有配对帧号及其后缀扩展名。"""
    groups = defaultdict(lambda: defaultdict(dict))  # prefix -> frame_id -> ext -> file_path

    for f in Path(folder).iterdir():
        if not f.is_file():
            continue
        m = WORKER_FRAME_PATTERN.match(f.name)
        if m:
            prefix, ext = m.groups()
            frame_id = f.stem[len(prefix) + 1:]  # 提取帧号部分
            groups[prefix][frame_id][ext] = f

    return groups


def split_and_copy(folder, output_dir, ratio=0.8, seed=42, dry_run=True):
    folder = Path(folder)
    output_dir = Path(output_dir)

    groups = group_files(folder)

    if not groups:
        print("未找到任何 *_worker_frame 文件")
        return

    print(f"文件夹: {folder}")
    print(f"输出: {output_dir}")
    print(f"比例: {ratio*100:.0f}%  随机种子: {seed}  {'[预览模式]' if dry_run else '[实际复制]'}")
    print("=" * 60)

    random.seed(seed)
    total_selected = 0
    total_all = 0

    for prefix in sorted(groups.keys()):
        frame_ids = sorted(groups[prefix].keys())
        total_frames = len(frame_ids)
        total_all += total_frames

        select_count = max(1, int(total_frames * ratio))
        selected = sorted(random.sample(frame_ids, select_count))
        total_selected += select_count

        print(f"\n{prefix}: {total_frames} 帧 → 选取 {select_count} 帧 ({select_count/total_frames*100:.1f}%)")

        for fid in selected:
            for ext, src_file in groups[prefix][fid].items():
                dst = output_dir / src_file.name
                if dry_run:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst)

    print(f"\n{'='*60}")
    print(f"总计: {total_all} 帧 → 选取 {total_selected} 帧 ({total_selected/total_all*100:.1f}%)")
    if dry_run:
        print("预览完成，未实际复制。加 --copy 执行复制。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按 worker_frame 前缀分组随机采样复制")
    parser.add_argument("--folder", required=True, help="源文件夹路径")
    parser.add_argument("--output", required=True, help="目标文件夹路径")
    parser.add_argument("--ratio", type=float, default=0.8, help="采样比例 (默认 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    parser.add_argument("--copy", action="store_true", help="加此参数才实际复制，否则仅预览")
    args = parser.parse_args()

    split_and_copy(args.folder, args.output, args.ratio, args.seed, dry_run=not args.copy)
