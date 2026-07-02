import argparse
import shutil
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def get_json_dir(image_dir):
    """将 *_yolo/images/train(val) 路径转换为保存 json 的原始数据路径。"""
    image_dir = Path(image_dir)
    parts = list(image_dir.parts)

    if len(parts) >= 3 and parts[-3].endswith('_yolo') and parts[-2] == 'images':
        parts[-3] = parts[-3][:-5]
        return Path(*parts[:-2])

    image_dir_str = str(image_dir)
    if '_yolo/images/' in image_dir_str:
        return Path(image_dir_str.replace('_yolo/images/', '/', 1)).parent

    raise ValueError(f'无法转换 json 文件夹路径: {image_dir}')


def load_dirs(data, key):
    dirs = data.get(key, [])
    if isinstance(dirs, str):
        dirs = [dirs]
    return dirs


def copy_jsons_from_split(image_dirs, output_dir, split_name):
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)

    total = 0
    missing = 0
    copied_names = set()

    for image_dir in image_dirs:
        image_dir = Path(image_dir)
        json_dir = get_json_dir(image_dir)

        if not image_dir.exists():
            print(f'警告: 图片文件夹不存在: {image_dir}')
            continue
        if not json_dir.exists():
            print(f'警告: json 文件夹不存在: {json_dir}')
            continue

        for image_file in image_dir.iterdir():
            if not image_file.is_file() or image_file.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            json_file = json_dir / f'{image_file.stem}.json'
            if not json_file.exists():
                missing += 1
                print(f'警告: 未找到对应 json: {json_file}')
                continue

            dst_name = json_file.name
            if dst_name in copied_names:
                dst_name = f'{json_dir.name}_{dst_name}'
            copied_names.add(dst_name)

            shutil.copy2(json_file, output_path / dst_name)
            total += 1

    print(f'{split_name}: 已复制 {total} 个 json 到 {output_path}, 缺失 {missing} 个')
    return total, missing


def copy_train_val_jsons(yaml_path, output_dir):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    total = 0
    missing = 0
    for split_name in ('train', 'val'):
        split_total, split_missing = copy_jsons_from_split(
            load_dirs(data, split_name), output_dir, split_name
        )
        total += split_total
        missing += split_missing

    print(f'完成: 共复制 {total} 个 json 到 {output_dir}, 缺失 {missing} 个')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从 data.yaml 的 train/val 图片列表复制对应 json')
    parser.add_argument(
        '--yaml',
        default='/root/taojianwei/datasets/IRIS/20260625_datasets_yolo/data.yaml',
        help='data.yaml 文件路径',
    )
    parser.add_argument('--output', required=True, help='保存 json 的输出文件夹')
    args = parser.parse_args()

    copy_train_val_jsons(args.yaml, args.output)
