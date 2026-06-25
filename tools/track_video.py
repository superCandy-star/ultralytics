#!/usr/bin/env python3
"""
YOLOv11 视频目标跟踪脚本

用法:
    python tools/track_video.py --weight best.pt --video video.mp4 --output result.mp4
    python tools/track_video.py --weight best.pt --video video.mp4 --output result.mp4 --conf 0.5
"""

import argparse
import shutil
import cv2
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 视频跟踪')
    parser.add_argument('--weight', type=str, required=True, help='模型权重文件')
    parser.add_argument('--video', type=str, required=True, help='输入视频文件')
    parser.add_argument('--output', type=str, default='output.mp4', help='输出视频文件路径')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"错误: 视频文件不存在: {args.video}")
        return

    # 解析输出路径
    output_path = Path(args.output)
    output_dir = output_path.parent if output_path.parent != Path('.') else Path('.')
    output_name = output_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载模型: {args.weight}")
    model = YOLO(args.weight)

    print(f"开始跟踪视频: {args.video}")
    results = model.track(
        source=args.video,
        conf=args.conf,
        persist=True,
        save=True,
        project=str(output_dir),
        name='',
        verbose=False
    )

    # 重命名生成的视频文件
    if results:
        # YOLO 生成的视频通常在 project/name/ 目录下
        video_files = list(output_dir.glob("*.avi")) + list(output_dir.glob("*.mp4"))
        if video_files:
            src_video = video_files[0]
            dst_video = output_dir / output_name
            shutil.move(str(src_video), str(dst_video))
            print(f"✓ 跟踪完成! 结果已保存到: {dst_video}")


if __name__ == '__main__':
    main()

