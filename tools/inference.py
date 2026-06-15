#!/usr/bin/env python3
"""
YOLOv11推理脚本
支持：
  1. 加载模型进行推理
  2. 保存推理结果到原图
  3. 以labelme JSON格式保存标注
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from ultralytics import YOLO
import numpy as np


def get_image_files(img_dir: str) -> List[str]:
    """获取目录下所有图片文件"""
    img_dir = Path(img_dir)
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return sorted([
        str(f) for f in img_dir.rglob('*')
        if f.suffix.lower() in img_extensions
    ])


def inference_image(model, img_path: str, conf: float) -> Tuple[np.ndarray, Dict]:
    """
    对单张图片进行推理

    Returns:
        (annotated_img, results_dict): 标注后的图片和结果字典
    """
    # 推理
    results = model.predict(
        source=img_path,
        conf=conf,
        verbose=False,
        save=False
    )

    result = results[0]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 提取检测结果
    detections = []
    if result.boxes is not None:
        for box, conf_score, cls_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy()
        ):
            x1, y1, x2, y2 = [int(v) for v in box]
            class_id = int(cls_id)
            class_name = model.names[class_id]
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': float(conf_score),
                'class': class_name,
                'class_id': class_id
            })

            # 绘制到图片
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{class_name} {conf_score:.2f}"
            cv2.putText(
                img, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

    return img, {
        'image_path': img_path,
        'image_width': w,
        'image_height': h,
        'detections': detections
    }


def to_labelme_format(img_path: str, results: Dict, model) -> Dict:
    """
    将检测结果转换为labelme JSON格式
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    shapes = []
    for det in results['detections']:
        x1, y1, x2, y2 = det['bbox']
        shape = {
            'label': det['class'],
            'points': [[x1, y1], [x2, y2]],
            'group_id': None,
            'description': '',
            'shape_type': 'rectangle',
            'flags': {}
        }
        shapes.append(shape)

    labelme_json = {
        'version': '5.0.1',
        'flags': {},
        'shapes': shapes,
        'imagePath': Path(img_path).name,
        'imageData': None,
        'imageHeight': h,
        'imageWidth': w
    }

    return labelme_json


def main():
    parser = argparse.ArgumentParser(description='YOLOv11推理脚本')
    parser.add_argument('--weight', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--img_dir', type=str, required=True, help='图片目录')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值 (默认 0.5)')
    parser.add_argument('--save', action='store_true', help='保存标注图片')
    parser.add_argument('--labelme', action='store_true', help='保存为labelme JSON格式')
    parser.add_argument('--save_dir', type=str, default='runs/inference', help='保存目录 (默认 runs/inference)')
    args = parser.parse_args()

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img_save_dir = save_dir / 'images' if args.save else None
    json_save_dir = save_dir / 'jsons' if args.labelme else None

    if args.save:
        img_save_dir.mkdir(exist_ok=True)
    if args.labelme:
        json_save_dir.mkdir(exist_ok=True)

    # 加载模型
    print(f"加载模型: {args.weight}")
    model = YOLO(args.weight)

    # 获取图片文件
    img_files = get_image_files(args.img_dir)
    print(f"找到 {len(img_files)} 张图片")

    if not img_files:
        print("未找到任何图片文件")
        return

    # 推理
    print(f"开始推理 (置信度: {args.conf})...")
    for idx, img_path in enumerate(img_files, 1):
        print(f"  [{idx}/{len(img_files)}] 处理: {Path(img_path).name}")

        # 推理
        annotated_img, results = inference_image(model, img_path, args.conf)

        img_name = Path(img_path).stem

        # 保存标注图片
        if args.save:
            img_out_path = img_save_dir / f"{img_name}_pred.jpg"
            cv2.imwrite(str(img_out_path), annotated_img)

        # 保存labelme JSON
        if args.labelme:
            labelme_json = to_labelme_format(img_path, results, model)
            json_out_path = json_save_dir / f"{img_name}.json"
            with open(json_out_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_json, f, indent=2, ensure_ascii=False)

    print(f"推理完成!")
    if args.save:
        print(f"标注图片保存到: {img_save_dir}")
    if args.labelme:
        print(f"标注JSON保存到: {json_save_dir}")


if __name__ == '__main__':
    main()
