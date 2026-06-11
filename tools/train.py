#!/usr/bin/env python3
"""
YOLO 检测模型训练脚本。

用法:
    # 使用预训练权重训练
    python tools/train.py --data_yaml coco.yaml --cfg yolo11.yaml --weights yolo11n.pt

    # 从头训练（不加载权重）
    python tools/train.py --data_yaml coco.yaml --cfg yolo11.yaml

    # 单机多卡训练 (DDP)
    python tools/train.py --data_yaml coco.yaml --cfg yolo11.yaml --weights yolo11n.pt --device 0,1,2,3

    # 恢复训练
    python tools/train.py --data_yaml coco.yaml --cfg yolo11.yaml --weights runs/train/exp/weights/last.pt --resume
"""

import argparse
from ultralytics import YOLO


def parse_device(device_str: str):
    """
    解析 device 参数:
        'cpu'        → 'cpu'
        '0'          → 0 (单卡)
        '0,1,2,3'    → [0, 1, 2, 3] (多卡 DDP)
    """
    device_str = device_str.strip()
    if device_str.lower() == "cpu":
        return "cpu"
    if "," in device_str:
        return [int(d.strip()) for d in device_str.split(",")]
    return int(device_str)


def main():
    parser = argparse.ArgumentParser(description="YOLO 检测模型训练")
    parser.add_argument("--data_yaml", type=str, required=True, help="数据集配置文件路径，如 coco.yaml")
    parser.add_argument("--cfg", type=str, required=True, help="模型配置文件路径，如 yolo11.yaml")
    parser.add_argument("--weights", type=str, default=None, help="预训练权重文件路径，如 yolo11n.pt（可选）")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (默认 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸 (默认 640)")
    parser.add_argument("--batch", type=int, default=16, help="批次大小 (默认 16)")
    parser.add_argument("--device", type=str, default="0", help="训练设备: 'cpu', '0' (单卡), '0,1,2,3' (多卡)")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数 (默认 8)")
    parser.add_argument("--resume", action="store_true", help="从上次中断处恢复训练")
    args = parser.parse_args()

    device = parse_device(args.device)

    print(f"模型配置: {args.cfg}")
    print(f"预训练权重: {args.weights if args.weights else '无 (从头训练)'}")
    print(f"数据: {args.data_yaml}")
    print(f"设备: {args.device}  →  {device}")
    print(f"轮数: {args.epochs}  |  尺寸: {args.imgsz}  |  批次: {args.batch}  |  恢复: {args.resume}")

    # 根据配置文件构建模型
    model = YOLO(args.cfg)

    # 训练
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        resume=args.resume,
        pretrained=args.weights if args.weights else False,
    )
    return results


if __name__ == "__main__":
    main()
