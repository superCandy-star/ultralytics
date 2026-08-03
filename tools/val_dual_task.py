#!/usr/bin/env python3
"""
YOLO11 双任务模型验证脚本。

用法:
    # 验证手物检测 (task=0, hand+obj)
    python tools/val_dual_task.py --weight runs/train/dual_task/last.pt --task 0 \
      --data /root/taojianwei/datasets/IRIS/20260727_datasets_yolo/data.yaml

    # 验证虚拟门检测 (task=1, cabinet+junction+side_cabinet+box)
    python tools/val_dual_task.py --weight runs/train/dual_task/last.pt --task 1 \
      --data /root/taojianwei/datasets/IRIS/auxiliary_datasets/20260729_datasets_yolo/data.yaml

    # 同时验证两个任务
    python tools/val_dual_task.py --weight runs/train/dual_task/last.pt --task all \
      --data_det ... --data_door ...
"""

from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_dual_task import DualHeadModel, MODEL_CFG, _init_strides, _validate


def load_model(weight_path: str, device: str, imgsz: int):
    """Load dual-task checkpoint and return model."""
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    nc = ckpt.get("args", {}).get("nc", [2, 4]) if "args" in ckpt else ckpt.get("nc", [2, 4])
    if isinstance(nc, int):
        nc = [2, 4]

    model = DualHeadModel(MODEL_CFG, nc=nc)
    state = (ckpt.get("ema") or ckpt.get("model") or ckpt) if isinstance(ckpt, dict) else ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    from ultralytics.utils import IterableSimpleNamespace
    model.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.to(device)
    _init_strides(model, imgsz, torch.device(device))
    model.eval()
    return model, nc


def main():
    parser = argparse.ArgumentParser(description="YOLO11 双任务模型验证")
    parser.add_argument("--weight", type=str, required=True, help="模型权重路径 (last.pt)")
    parser.add_argument("--task", type=str, default="all",
                        choices=["0", "1", "all"], help="验证任务: 0=手物, 1=虚拟门, all=全部")
    parser.add_argument("--data_det", type=str,
                        default="/root/taojianwei/datasets/IRIS/20260727_datasets_yolo/data.yaml",
                        help="手物检测 data.yaml")
    parser.add_argument("--data_door", type=str,
                        default="/root/taojianwei/datasets/IRIS/auxiliary_datasets/20260729_datasets_yolo/data.yaml",
                        help="虚拟门检测 data.yaml")
    parser.add_argument("--imgsz", type=int, default=320, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=64, help="批量大小")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    args.hand_obj_data = args.data_det
    args.virtual_door_data = args.data_door

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Loading: {args.weight}")
    model, nc = load_model(args.weight, device, args.imgsz)
    print(f"Model: heads={model._h_idx} nc={nc}")

    t0 = time.time()
    task_ids = (0, 1) if args.task == "all" else (int(args.task),)
    metrics = _validate(model, args, torch.device(device), task_ids=task_ids)
    selected = ["det", "door"] if args.task == "all" else ["det" if args.task == "0" else "door"]
    for task_name in selected:
        r = metrics[task_name]
        print(f"\n{task_name}: P={r['precision']:.4f} R={r['recall']:.4f} "
              f"mAP@0.5={r['mAP50']:.4f} mAP@0.5:0.95={r['mAP50-95']:.4f}")
    print(f"Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
