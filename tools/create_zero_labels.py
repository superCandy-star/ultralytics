import yaml
from pathlib import Path


def create_two_labels(yaml_path):
    """将标签类别转变为：8->0 (hand), 其他->1 (obj)，生成two_train和two_val"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    yaml_dir = Path(yaml_path).parent
    train_dirs = data['train'] if isinstance(data['train'], list) else [data['train']]
    val_dirs = data['val'] if isinstance(data['val'], list) else [data['val']]

    two_train_paths = []
    two_val_paths = []

    for train_dir in train_dirs:
        labels_dir = Path(train_dir).parent.parent / 'labels' / 'train'
        two_dir = labels_dir.parent / 'two_train'
        two_dir.mkdir(parents=True, exist_ok=True)

        if labels_dir.exists():
            for txt_file in labels_dir.glob('*.txt'):
                with open(txt_file, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = '0' if parts[0] == '8' else '1'
                        new_lines.append(' '.join(parts) + '\n')

                two_file = two_dir / txt_file.name
                with open(two_file, 'w') as f:
                    f.writelines(new_lines)

        two_train_paths.append(str(two_dir))

    for val_dir in val_dirs:
        labels_dir = Path(val_dir).parent.parent / 'labels' / 'val'
        two_dir = labels_dir.parent / 'two_val'
        two_dir.mkdir(parents=True, exist_ok=True)

        if labels_dir.exists():
            for txt_file in labels_dir.glob('*.txt'):
                with open(txt_file, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = '0' if parts[0] == '8' else '1'
                        new_lines.append(' '.join(parts) + '\n')

                two_file = two_dir / txt_file.name
                with open(two_file, 'w') as f:
                    f.writelines(new_lines)

        two_val_paths.append(str(two_dir))

    # 生成新的yaml文件
    new_data = {
        'names': {0: 'hand', 1: 'obj'},
        'nc': 2,
        'train': two_train_paths,
        'val': two_val_paths
    }

    new_yaml_path = yaml_dir / 'data_two.yaml'
    with open(new_yaml_path, 'w') as f:
        yaml.dump(new_data, f, default_flow_style=False)

    print(f"完成!")
    print(f"two_train数据集: {len(two_train_paths)}")
    print(f"two_val数据集: {len(two_val_paths)}")
    print(f"新yaml文件: {new_yaml_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='创建hand/obj二分类标签数据集')
    parser.add_argument('--yaml', required=True, help='原始data.yaml文件路径')
    args = parser.parse_args()

    create_two_labels(args.yaml)
