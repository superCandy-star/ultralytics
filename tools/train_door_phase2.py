#!/usr/bin/env python3
"""
独立调试脚本：只跑虚拟门任务训练（复刻项目中 phase2 的冻结/学习率配置）。
不修改任何原始代码，复用 train_dual_task 的模型/数据/验证逻辑。

加载 phase1 checkpoint 的 backbone + NeckA + HeadA；
NeckB + HeadB 保持模型初始化的随机权重（跳过 checkpoint 加载）。
每 epoch 同时验证 door 和 det，用于定位：
  1) door 任务单独训练能否正常收敛（项目里只有 1 个类别有精度且不正常）
  2) 训练 door 时 det 精度快速掉点的原因

用法:
    python tools/train_door_phase2.py --epochs 50 --batch 128 --imgsz 320 --num_workers 4
    python tools/train_door_phase2.py --test_iters 13   # 快速跑 1 个 epoch 验证
"""

import argparse, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dual_task import (  # noqa: E402
    DualHeadModel, YOLODataset, collate_fn, _init_strides, _run_phase,
    _validate, MODEL_CFG, NECK_B_START, HEAD_B_IDX,
)
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",
                   default=str(ROOT / "runs/train/dual_v2/phase1_handobj_epoch50.pt"))
    p.add_argument("--hand_obj_data",
                   default="/root/taojianwei/datasets/IRIS/20260727_datasets_yolo/data.yaml")
    p.add_argument("--virtual_door_data",
                   default="/root/taojianwei/datasets/IRIS/auxiliary_datasets/20260729_datasets_yolo/data.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--test_iters", type=int, default=0,
                   help="run N iters then validate & exit (0=full epochs)")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="door_phase2_only")
    p.add_argument("--backbone_lr", type=float, default=5e-4)
    p.add_argument("--neck_b_lr", type=float, default=0.001)
    p.add_argument("--head_b_lr", type=float, default=0.005)
    args = p.parse_args()

    init_seeds(42)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Device: {device}")

    # ---------- Model: same construction as train_dual_task.train() ----------
    model = DualHeadModel(cfg=MODEL_CFG, nc=[2, 4])
    from ultralytics.utils import IterableSimpleNamespace
    model.args = IterableSimpleNamespace(**{
        "box": 7.5, "cls": 0.5, "dfl": 1.5, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
        "copy_paste_mode": "flip", "cutmix": 0.0, "augmentations": None,
    })
    model.to(device)

    # Load phase1 checkpoint: backbone + NeckA + HeadA only.
    # NeckB (24-35) + HeadB (36) keep the fresh random init from construction
    # (same as the project: they were never trained in phase 1).
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    skip_prefixes = tuple(f"model.{i}." for i in range(NECK_B_START, HEAD_B_IDX + 1))
    loaded_sd = {k: v for k, v in sd.items() if not k.startswith(skip_prefixes)}
    missing, unexpected = model.load_state_dict(loaded_sd, strict=False)
    n_skip = len(sd) - len(loaded_sd)
    print(f"Loaded phase1 from {args.weights}: {len(loaded_sd)}/{len(sd)} tensors "
          f"({n_skip} NeckB/HeadB skipped, keep fresh init)")
    print(f"  missing (expected = NeckB/HeadB fresh): {len(missing)}")

    _init_strides(model, args.imgsz, device)
    model.train()

    # ---------- Data: door train + both val sets ----------
    bs, num_workers = args.batch, args.num_workers
    door_ds = YOLODataset(args.virtual_door_data, args.imgsz, hyp=model.args)
    print(f"door dataset: {len(door_ds)} images")
    door_loader = DataLoader(door_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
                             collate_fn=collate_fn, drop_last=True, pin_memory=True)
    det_val_ds = YOLODataset(args.hand_obj_data, args.imgsz, split="val")
    door_val_ds = YOLODataset(args.virtual_door_data, args.imgsz, split="val")

    # ---------- Phase 2 (exact project config): train backbone+NeckB+HeadB on door data
    #   {'backbone': 5e-4, 'neck_b': 0.001, 'head_b': 0.005}, warmup 3 epochs
    # Start epoch 51 to match the project's phase numbering.
    # NOTE: _run_phase's test_iters only caps steps-per-epoch, not epoch count,
    # so in debug mode run a single epoch then exit.
    epochs_to_run = 1 if args.test_iters else args.epochs
    _run_phase(model, door_loader, (det_val_ds, door_val_ds),
               "phase2_door", epochs_to_run,
               {'backbone': args.backbone_lr, 'neck_b': args.neck_b_lr, 'head_b': args.head_b_lr},
               51, args, device, warmup_epochs=3)

    print(f"Done. Checkpoints: {Path(args.project) / args.name}", flush=True)


if __name__ == "__main__":
    main()
