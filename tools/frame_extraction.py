#!/usr/bin/env python3
"""
从mp4视频中抽帧，保存为图片。

用法:
    # 单个视频文件
    python tools/frame_extraction.py --video /path/to/video.mp4 --fps 30 --save_dir frames

    # 目录下所有mp4文件
    python tools/frame_extraction.py --video_dir /path/to/videos --fps 10 --save_dir frames

    # fps含义: 每fps帧抽取一帧 (fps=10表示每10帧抽1帧)
"""

import argparse
import cv2
from pathlib import Path


def extract_frames(video_path: str, save_dir: str, fps: int = 30):
    """
    从视频中抽帧。

    Args:
        video_path: 视频文件路径
        save_dir: 图片保存目录
        fps: 抽帧间隔（每fps帧抽取一帧）
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"✗ 视频不存在: {video_path}")
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved_count = 0
    video_name = video_path.stem

    print(f"处理: {video_path.name} ({total_frames} 帧)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % fps == 0:
            img_name = f"{video_name}_{frame_idx:06d}.png"
            cv2.imwrite(str(save_path / img_name), frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"✓ 完成: 抽取 {saved_count} 张图片 → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="视频抽帧工具")
    parser.add_argument("--video", type=str, default=None, help="单个视频文件的绝对路径")
    parser.add_argument("--video_dir", type=str, default=None, help="视频目录（遍历所有mp4）")
    parser.add_argument("--fps", type=int, default=30, help="抽帧间隔：每fps帧抽取1帧 (默认 30)")
    parser.add_argument("--save_dir", type=str, default="frames", help="保存目录 (默认 frames)")
    args = parser.parse_args()

    if not args.video and not args.video_dir:
        parser.error("必须指定 --video 或 --video_dir 之一")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.video:
        extract_frames(args.video, args.save_dir, args.fps)
    else:
        video_dir = Path(args.video_dir)
        video_files = sorted(video_dir.glob("*.mp4"))
        print(f"找到 {len(video_files)} 个mp4文件\n")
        for video in video_files:
            extract_frames(str(video), args.save_dir, args.fps)

    print(f"\n✅ 全部完成，图片保存在: {args.save_dir}")


if __name__ == "__main__":
    main()
