#!/usr/bin/env python3
"""
将 COCO 格式的标注文件转换为 YOLO 检测格式。

COCO bbox: [x, y, width, height]（绝对像素坐标）
YOLO 格式: class_id cx cy w h（归一化到 [0, 1]）

输出结构:
    labels/
    ├── train/   # 对应 instances_train2017.json
    └── val/     # 对应 instances_val2017.json
"""

import json
import os
from pathlib import Path


def coco2yolo(
    json_path: str,
    output_dir: str,
    subset: str = "train",
):
    """
    将 COCO JSON 标注文件转换为 YOLO 格式的 txt 标签文件。

    Args:
        json_path: COCO 标注 JSON 文件路径
        output_dir: labels 输出根目录
        subset: 子文件夹名称（"train" 或 "val"）
    """
    with open(json_path) as f:
        coco = json.load(f)

    # 建立 image_id → (width, height, file_name) 的映射
    image_info = {}
    for img in coco["images"]:
        image_info[img["id"]] = {
            "width": img["width"],
            "height": img["height"],
            "file_name": img["file_name"],
        }

    # 建立 category_id → 0-indexed class id 的映射
    # COCO 的 category_id 从 1 开始，且不连续（1-90 有跳跃）
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_class = {cat["id"]: i for i, cat in enumerate(categories)}

    print(f"类别数: {len(categories)}")
    print(f"图片数: {len(coco['images'])}")
    print(f"标注数: {len(coco['annotations'])}")

    # 建立 image_id → [annotations] 的映射
    anns_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # 创建输出目录
    label_dir = Path(output_dir) / subset
    label_dir.mkdir(parents=True, exist_ok=True)

    # 遍历每张图片，生成对应的 txt 标签文件
    skipped = 0
    for img_id, img_meta in image_info.items():
        # 图片文件名（去掉扩展名）
        stem = Path(img_meta["file_name"]).stem
        txt_path = label_dir / f"{stem}.txt"

        anns = anns_by_image.get(img_id, [])
        if not anns:
            skipped += 1
            # 创建空文件（无标注图片）
            txt_path.touch()
            continue

        lines = []
        for ann in anns:
            # COCO bbox: [x, y, w, h] 绝对坐标
            x, y, w, h = ann["bbox"]

            # 过滤无效标注（宽度或高度为 0）
            if w <= 0 or h <= 0:
                continue

            # 转为 YOLO 归一化格式: [cx, cy, w, h]
            cx = (x + w / 2) / img_meta["width"]
            cy = (y + h / 2) / img_meta["height"]
            nw = w / img_meta["width"]
            nh = h / img_meta["height"]

            # 裁剪到 [0, 1] 区间（部分边界框可能略微超出图片边界）
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            class_id = cat_id_to_class[ann["category_id"]]
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

    print(f"已输出到: {label_dir}")
    print(f"含标注图片: {len(image_info) - skipped}")
    if skipped:
        print(f"无标注图片（空文件）: {skipped}")


if __name__ == "__main__":
    # ============ 配置区域 ============
    BASE = "/root/taojianwei/datasets/public/OpenDataLab___COCO_2017/raw/Annotations"

    TASKS = [
        {
            "json_path": os.path.join(BASE, "annotations/instances_train2017.json"),
            "subset": "train",
        },
        {
            "json_path": os.path.join(BASE, "annotations/instances_val2017.json"),
            "subset": "val",
        },
    ]

    OUTPUT_DIR = os.path.join(BASE, "labels")
    # =================================

    for task in TASKS:
        print(f"\n{'='*50}")
        print(f"处理 {task['subset']}: {task['json_path']}")
        print(f"{'='*50}")
        coco2yolo(
            json_path=task["json_path"],
            output_dir=OUTPUT_DIR,
            subset=task["subset"],
        )

    print(f"\n✅ 完成！labels 保存在: {OUTPUT_DIR}")
