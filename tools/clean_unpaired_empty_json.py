import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def find_image_by_stem(folder, stem):
    for suffix in IMAGE_SUFFIXES:
        image_file = folder / f'{stem}{suffix}'
        if image_file.exists():
            return image_file
    return None


def has_valid_label(json_file):
    """判断 LabelMe json 中是否有有效 label。"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f'警告: 读取 json 失败，按空 label 处理: {json_file} ({e})')
        return False

    shapes = data.get('shapes', []) if isinstance(data, dict) else []
    if not shapes:
        return False

    for shape in shapes:
        if isinstance(shape, dict) and str(shape.get('label', '')).strip():
            return True
    return False


def delete_file(path, dry_run):
    if dry_run:
        print(f'[DRY-RUN] 删除: {path}')
    else:
        path.unlink()
        print(f'删除: {path}')


def clean_dataset(folder, dry_run=True):
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f'文件夹不存在: {folder}')

    json_files = sorted(folder.glob('*.json'))
    removed_unpaired_json = 0
    removed_empty_json = 0
    removed_empty_images = 0

    for json_file in json_files:
        image_file = find_image_by_stem(folder, json_file.stem)

        # 1. 清除没有配对图片的多余 json
        if image_file is None:
            delete_file(json_file, dry_run)
            removed_unpaired_json += 1
            continue

        # 2. 清除 label 为空的 json 及其对应图片
        if not has_valid_label(json_file):
            delete_file(json_file, dry_run)
            delete_file(image_file, dry_run)
            removed_empty_json += 1
            removed_empty_images += 1

    print('\n统计:')
    print(f'  未配对 json: {removed_unpaired_json}')
    print(f'  空 label json: {removed_empty_json}')
    print(f'  空 label 对应图片: {removed_empty_images}')
    print(f'  模式: {"预览，不实际删除" if dry_run else "已实际删除"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='清理没有配对图片的 json，以及 label 为空的 json 和对应图片')
    parser.add_argument(
        '--folder',
        default='/root/taojianwei/datasets/IRIS/20260703_datasets',
        help='需要清理的数据集文件夹',
    )
    parser.add_argument('--delete', action='store_true', help='实际删除文件；不加该参数时只预览')
    args = parser.parse_args()

    clean_dataset(args.folder, dry_run=not args.delete)
