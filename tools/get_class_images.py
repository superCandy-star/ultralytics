import yaml
import shutil
import random
from pathlib import Path


def get_images_by_class(yaml_path, class_id, count, output_dir):
    """从data.yaml中获取指定类别的指定数量的图片"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    train_dirs = data['train'] if isinstance(data['train'], list) else [data['train']]
    val_dirs = data['val'] if isinstance(data['val'], list) else [data['val']]
    all_dirs = train_dirs + val_dirs

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    matched_images = []

    for img_dir in all_dirs:
        img_path = Path(img_dir)
        if not img_path.exists():
            continue

        for img_file in img_path.glob('*'):
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # 找对应的标签文件
                labels_dir = img_path.parent.parent / 'labels' / img_path.name
                label_file = labels_dir / (img_file.stem + '.txt')

                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line in f:
                            if int(line.split()[0]) == class_id:
                                matched_images.append(img_file)
                                break

    # 随机选择指定数量
    if len(matched_images) < count:
        print(f"警告: 找到{len(matched_images)}张图片，少于指定数量{count}")
        selected = matched_images
    else:
        selected = random.sample(matched_images, count)

    # 复制图片
    for img in selected:
        shutil.copy2(img, output_path / img.name)

    print(f"完成: 已复制 {len(selected)} 张类别{class_id}的图片到 {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='获取指定类别的图片')
    parser.add_argument('--yaml', required=True, help='data.yaml文件路径')
    parser.add_argument('--class-id', type=int, required=True, help='类别ID')
    parser.add_argument('--count', type=int, required=True, help='图片数量')
    parser.add_argument('--output', required=True, help='输出文件夹')
    args = parser.parse_args()

    get_images_by_class(args.yaml, args.class_id, args.count, args.output)
