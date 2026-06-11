#!/usr/bin/env python3
"""
YOLO 检测模型验证脚本。

用法:
    python tools/val.py --data_yaml coco.yaml --weight yolo11n.pt
    python tools/val.py --data_yaml coco.yaml --weight runs/train/exp/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO 检测模型验证")
    parser.add_argument("--data_yaml", type=str, required=True, help="数据集配置文件路径")
    parser.add_argument("--weight", type=str, required=True, help="模型权重文件路径")
    args = parser.parse_args()

    # Load a trained YOLO model
    model = YOLO(args.weight)

    # Evaluate the model's performance on the validation set
    metrics = model.val(data=args.data_yaml)
    return metrics


if __name__ == "__main__":
    main()
