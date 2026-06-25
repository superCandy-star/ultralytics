#!/usr/bin/env python3
"""
YOLO 检测模型验证脚本。

用法:
    python tools/val.py --data_yaml coco.yaml --weight yolo11n.pt
    python tools/val.py --data_yaml coco.yaml --weight best.pt --conf 0.3 --labelme --save_dir results
"""

import argparse
import json
import cv2
from pathlib import Path
from ultralytics import YOLO
import yaml


def load_data_yaml(yaml_path: str):
    """加载data.yaml获取验证集路径"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def save_as_labelme(img_path: str, results, model, save_dir: str):
    """将检测结果保存为labelme JSON格式"""
    result = results
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    shapes = []
    if result.boxes is not None:
        for box, conf_score, cls_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy()
        ):
            x1, y1, x2, y2 = [int(v) for v in box]
            class_id = int(cls_id)
            class_name = model.names[class_id]

            shape = {
                'label': class_name,
                'points': [[x1, y1], [x2, y2]],
                'group_id': None,
                'description': f"conf: {float(conf_score):.2f}",
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

    save_path = Path(save_dir) / f"{Path(img_path).stem}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_json, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="YOLO 检测模型验证")
    parser.add_argument("--data_yaml", type=str, required=True, help="数据集配置文件路径")
    parser.add_argument("--weight", type=str, required=True, help="模型权重文件路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值 (默认 0.5)")
    parser.add_argument("--labelme", action="store_true", help="保存结果为labelme JSON格式")
    parser.add_argument("--save_dir", type=str, default="runs/val", help="保存目录 (默认 runs/val)")
    args = parser.parse_args()

    # 加载模型
    model = YOLO(args.weight)

    # 验证模型
    metrics = model.val(data=args.data_yaml, conf=args.conf)

    # 保存为labelme格式 - 直接推理验证集目录
    if args.labelme:
        print(f"\n保存结果为labelme格式...")
        data_info = load_data_yaml(args.data_yaml)
        val_dir = data_info.get('val', '')

        # 处理val_dir可能是列表的情况
        if isinstance(val_dir, list):
            val_dir = val_dir[0] if val_dir else ''

        print(f"  验证集路径: {val_dir}")

        if val_dir and Path(val_dir).exists():
            # 使用predict直接在目录上进行推理，获取所有图片
            results = model.predict(source=val_dir, conf=args.conf, save=False, verbose=False)

            print(f"  找到 {len(results)} 张图片")

            for idx, result in enumerate(results, 1):
                img_path = result.path
                save_as_labelme(img_path, result, model, args.save_dir)
                if idx % 10 == 0:
                    print(f"    已处理 {idx}/{len(results)} 张图片")

            print(f"  结果已保存到: {args.save_dir}")
        else:
            print(f"  验证集路径不存在或为空")

    return metrics


if __name__ == "__main__":
    main()
