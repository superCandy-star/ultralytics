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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_dual_task import (DualHeadModel, MODEL_CFG, YOLODataset, _load_batch,
                                   _validate_one_head, _compute_ap_from_pr, _compute_map)


def load_model(weight_path: str, device: str):
    """Load dual-task checkpoint and return model."""
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    nc = ckpt.get("args", {}).get("nc", [2, 4]) if "args" in ckpt else ckpt.get("nc", [2, 4])
    if isinstance(nc, int):
        nc = [2, 4]

    model = DualHeadModel(MODEL_CFG, nc=nc)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    from ultralytics.utils import IterableSimpleNamespace
    model.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.to(device)
    # Note: model stays in eval mode here; _validate_one_head internally
    # switches to train mode for dict output from Detect.forward()
    return model, nc


@torch.no_grad()
def val_one_task(model: DualHeadModel, data_yaml: str, task_id: int,
                 imgsz: int, batch: int, device: str, conf: float):
    """Validate a single task head on given dataset using proper mAP."""
    ds = YOLODataset(data_yaml, imgsz)
    head_nc = [2, 4][task_id]

    print(f"  val images: {min(len(ds), 2000)}, nc={head_nc}")

    r = _validate_one_head(model, ds, task_id, imgsz, batch,
                           torch.device(device), max_imgs=2000)
    r["images"] = min(len(ds), 2000)
    return r


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
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批量大小")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Loading: {args.weight}")
    model, nc = load_model(args.weight, device)
    print(f"Model: heads={model._h_idx} nc={nc}")

    tasks = []
    if args.task == "all":
        tasks = [("Hand+Obj (task 0)", args.data_det, 0),
                 ("Door (task 1)", args.data_door, 1)]
    elif args.task == "0":
        tasks = [("Hand+Obj", args.data_det, 0)]
    else:
        tasks = [("Door", args.data_door, 1)]

    for name, data_yaml, tid in tasks:
        print(f"\n{'='*50}")
        print(f"Validating: {name}")
        t0 = time.time()
        metrics = val_one_task(model, data_yaml, tid, args.imgsz, args.batch, device, args.conf)
        print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        print(f"  Images: {metrics['images']}")
        print(f"  Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
