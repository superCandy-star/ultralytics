#!/usr/bin/env python3
"""将文件夹下的图片帧合成为 H.264 MP4 视频（飞书/微信兼容）。"""

import argparse
import subprocess
import shutil
from pathlib import Path


def images_to_video(input_dir, output, fps=30):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"文件夹不存在: {input_dir}")

    images = sorted([f for f in input_path.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if not images:
        print("没有找到图片")
        return
    print(f"找到 {len(images)} 张图片")

    # 先用 ffmpeg 生成 H.264 编码的 MP4（飞书兼容）
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", str(input_path / "frame_%06d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",
            "-y",
            str(output),
        ]
        print(f"使用 ffmpeg (H.264) 生成: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"视频已保存: {output}")
    else:
        # 回退到 OpenCV
        import cv2
        first = cv2.imread(str(images[0]))
        h, w = first.shape[:2]
        print(f"分辨率: {w}x{h}, 帧率: {fps}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
        for i, img_path in enumerate(images):
            frame = cv2.imread(str(img_path))
            writer.write(frame)
            if (i + 1) % 50 == 0:
                print(f"  已写入 {i + 1}/{len(images)}")
        writer.release()
        print(f"视频已保存: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图片帧 → H.264 MP4 视频（飞书/微信兼容）")
    parser.add_argument("--input", default="/root/taojianwei/projects/deploy/evidence/chen_worker_datasets", help="图片文件夹")
    parser.add_argument("--output", default="/root/taojianwei/projects/deploy/evidence/chen_worker_datasets/output.mp4", help="输出视频路径")
    parser.add_argument("--fps", type=int, default=20, help="帧率 (默认 20)")
    args = parser.parse_args()

    images_to_video(args.input, args.output, args.fps)
