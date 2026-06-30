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
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="验证集选择 (默认 val)")
    parser.add_argument("--cache", action="store_true", help="使用缓存（默认禁用缓存）")
    parser.add_argument("--labelme", action="store_true", help="保存结果为labelme JSON格式")
    parser.add_argument("--save_dir", type=str, default="runs/val", help="保存目录 (默认 runs/val)")
    args = parser.parse_args()

    # 加载模型
    model = YOLO(args.weight)

    # 生成单个数据集的yaml文件（只包含选定的split）
    data_info = load_data_yaml(args.data_yaml)
    split_dirs = data_info.get(args.split, [])
    if isinstance(split_dirs, str):
        split_dirs = [split_dirs]

    if split_dirs:
        # 创建临时yaml，使用所有选定的split数据集
        temp_yaml = {
            'names': data_info.get('names', {}),
            'nc': data_info.get('nc', 0),
            'train': [],
            'val': split_dirs
        }

        from pathlib import Path
        temp_yaml_path = Path(args.data_yaml).parent / f'temp_{args.split}.yaml'
        with open(temp_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_yaml, f)

        # 验证模型
        metrics = model.val(data=str(temp_yaml_path), conf=args.conf, cache=args.cache)

        # 删除临时yaml
        temp_yaml_path.unlink()
    else:
        print(f"错误: {args.split}集在data.yaml中不存在")
        return None

    # 保存为labelme格式
    if args.labelme:
        print(f"\n保存结果为labelme格式...")
        split_dir = split_dirs[0]
        print(f"  {args.split}集路径: {split_dir}")

        if split_dir and Path(split_dir).exists():
            results = model.predict(source=split_dir, conf=args.conf, save=False, verbose=False)
            print(f"  找到 {len(results)} 张图片")

            for idx, result in enumerate(results, 1):
                img_path = result.path
                save_as_labelme(img_path, result, model, args.save_dir)
                if idx % 10 == 0:
                    print(f"    已处理 {idx}/{len(results)} 张图片")

            print(f"  结果已保存到: {args.save_dir}")
        else:
            print(f"  {args.split}集路径不存在或为空")

    return metrics


if __name__ == "__main__":
    main()
