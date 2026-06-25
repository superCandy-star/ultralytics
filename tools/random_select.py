import os
import shutil
import random
import argparse
from pathlib import Path


def random_select(image_dir, json_dir, output_dir, count=500):
    """
    从图片和json文件夹中随机筛选文件

    Args:
        image_dir: 图片文件夹路径
        json_dir: json文件夹路径
        output_dir: 输出文件夹路径
        count: 选择的文件数量
    """
    image_dir = Path(image_dir)
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)

    if not image_dir.exists():
        print(f"错误: 图片文件夹 {image_dir} 不存在")
        return
    if not json_dir.exists():
        print(f"错误: json文件夹 {json_dir} 不存在")
        return

    # 获取所有图片文件
    image_files = list(image_dir.glob('*'))
    image_files = [f for f in image_files if f.is_file()]

    if len(image_files) < count:
        print(f"警告: 图片总数({len(image_files)}) 少于选择数量({count}), 将选择全部")
        count = len(image_files)

    # 随机选择图片
    selected_images = random.sample(image_files, count)

    # 创建输出目录
    output_image_dir = output_dir / 'images'
    output_json_dir = output_dir / 'jsons'
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_json_dir.mkdir(parents=True, exist_ok=True)

    # 复制图片和对应的json
    success_count = 0
    for img_file in selected_images:
        # 复制图片
        shutil.copy2(img_file, output_image_dir / img_file.name)

        # 寻找对应的json文件（同名）
        json_name = img_file.stem + '.json'
        json_file = json_dir / json_name

        if json_file.exists():
            shutil.copy2(json_file, output_json_dir / json_name)
            success_count += 1
        else:
            print(f"警告: 找不到json文件 {json_name}")

    print(f"完成: 已复制 {success_count}/{count} 个图片-json对")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从图片和json文件夹中随机筛选文件')
    parser.add_argument('--image-dir', required=True, help='图片文件夹路径')
    parser.add_argument('--json-dir', required=True, help='json文件夹路径')
    parser.add_argument('--output-dir', required=True, help='输出文件夹路径')
    parser.add_argument('--count', type=int, default=500, help='选择的文件数量(默认500)')

    args = parser.parse_args()
    random_select(args.image_dir, args.json_dir, args.output_dir, args.count)
