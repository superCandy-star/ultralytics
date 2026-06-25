import yaml
import shutil
import argparse
from pathlib import Path


def copy_train_images(yaml_path, output_dir):
    """从data.yaml中读取train list中的所有图片并复制到output_dir"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    train_dirs = data['train']
    if isinstance(train_dirs, str):
        train_dirs = [train_dirs]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = 0
    for train_dir in train_dirs:
        dir_path = Path(train_dir)
        if not dir_path.exists():
            print(f"警告: {train_dir} 不存在")
            continue

        for img_file in dir_path.glob('*'):
            if img_file.is_file():
                shutil.copy2(img_file, output_path / img_file.name)
                total += 1

    print(f"完成: 已复制 {total} 张图片到 {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从data.yaml中复制train images')
    parser.add_argument('--yaml', required=True, help='data.yaml文件路径')
    parser.add_argument('--output', required=True, help='输出文件夹路径')
    args = parser.parse_args()

    copy_train_images(args.yaml, args.output)
