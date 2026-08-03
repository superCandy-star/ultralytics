#!/usr/bin/env python3
"""
验证实验：双头 phase2 用官方 auto optimizer（AdamW + 分组 weight decay）训练，
对照官方单头 5 epoch mAP50=0.981 的结果。
若 door val 快速收敛 → 根因是 SGD vs AdamW 优化器差异（Neck B 深层梯度消失）。

官方 build_optimizer 逻辑（nc=4, iterations<10000）:
  - AdamW lr = 0.002*5/(4+nc) = 0.00125, betas=(0.9, 0.999)
  - 分组: weight decay=0.0005 (weight), 0 (bias), 0 (BN)

用法:
    python tools/train_door_phase2_adamw.py --epochs 5 --batch 64 --imgsz 320
"""

import argparse, sys, time
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dual_task import (  # noqa: E402
    DualHeadModel, YOLODataset, collate_fn, _init_strides, _step,
    _validate, MODEL_CFG, NECK_B_START, HEAD_B_IDX, _freeze_all,
    _unfreeze_module_list, BACKBONE_END, NECK_B_END,
)
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402


def build_adamw_groups(model, lr, decay=0.0005):
    """Official-style AdamW parameter groups: weight decay only on conv/linear weights."""
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    w, b, n = [], [], []
    for name, mod in model.named_modules():
        for pname, p in mod.named_parameters(recurse=False):
            full = f"{name}.{pname}" if name else pname
            if "bias" in full:
                b.append(p)
            elif isinstance(mod, bn):
                n.append(p)
            else:
                w.append(p)
    return [
        {"params": w, "lr": lr, "weight_decay": decay},
        {"params": b, "lr": lr, "weight_decay": 0.0},
        {"params": n, "lr": lr, "weight_decay": 0.0},
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(ROOT / "runs/train/dual_v2/phase1_handobj_epoch50.pt"))
    p.add_argument("--hand_obj_data",
                   default="/root/taojianwei/datasets/IRIS/20260727_datasets_yolo/data.yaml")
    p.add_argument("--virtual_door_data",
                   default="/root/taojianwei/datasets/IRIS/auxiliary_datasets/20260729_datasets_yolo/data.yaml")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--lr", type=float, default=0.00125)  # official lr_fit for nc=4
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--door_pretrained", default="",
                   help="optional yolo11n pretrained weight mapped to NeckB/HeadB")
    p.add_argument("--schedule_epochs", type=int, default=100,
                   help="scheduler horizon matching official args epochs")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="freeze backbone (protect hand-obj features)")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="door_phase2_adamw")
    args = p.parse_args()

    init_seeds(42)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Device: {device} | AdamW lr={args.lr} momentum={args.momentum}")

    # ---------- Model (same as project phase2 entry) ----------
    model = DualHeadModel(cfg=MODEL_CFG, nc=[2, 4])
    model.args = IterableSimpleNamespace(**{
        "box": 7.5, "cls": 0.5, "dfl": 1.5, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
        "copy_paste_mode": "flip", "cutmix": 0.0, "augmentations": None,
    })
    model.to(device)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    skip = tuple(f"model.{i}." for i in range(NECK_B_START, HEAD_B_IDX + 1))
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith(skip)}, strict=False)
    print("Loaded phase1 backbone+NeckA+HeadA")

    if args.door_pretrained:
        # Reproduce official DetectionModel(nc=4) initialization + pretrained intersection,
        # then map single-head layers 11-22 -> NeckB 24-35 and layer 23 -> HeadB 36.
        ref = DetectionModel(str(ROOT / "ultralytics/cfg/models/11/yolo11n.yaml"), nc=4, verbose=False)
        raw = torch.load(args.door_pretrained, map_location="cpu", weights_only=False)
        source = raw.get("ema") or raw.get("model")
        ref.load(source, verbose=True)
        ref_sd = ref.state_dict()
        dual_sd = model.state_dict()
        mapped = 0
        for key, value in ref_sd.items():
            if not key.startswith("model."):
                continue
            parts = key.split(".")
            idx = int(parts[1])
            if 11 <= idx <= 22:
                dst_idx = idx + 13
            elif idx == 23:
                dst_idx = HEAD_B_IDX
            else:
                continue
            dst = ".".join(["model", str(dst_idx)] + parts[2:])
            if dst in dual_sd and dual_sd[dst].shape == value.shape:
                dual_sd[dst] = value.to(dual_sd[dst].dtype)
                mapped += 1
        model.load_state_dict(dual_sd, strict=False)
        print(f"Mapped {mapped} reference tensors to NeckB+HeadB from {args.door_pretrained}")
    else:
        print("NeckB+HeadB remain fresh initialization")

    _init_strides(model, args.imgsz, device)
    model.train()

    # Freeze per config
    _freeze_all(model)
    if args.freeze_backbone:
        _unfreeze_module_list(model, list(range(NECK_B_START, NECK_B_END + 1)))
        _unfreeze_module_list(model, [HEAD_B_IDX])
        print("Freeze: backbone frozen, NeckB+HeadB trainable")
    else:
        _unfreeze_module_list(model, list(range(BACKBONE_END + 1)))
        _unfreeze_module_list(model, list(range(NECK_B_START, NECK_B_END + 1)))
        _unfreeze_module_list(model, [HEAD_B_IDX])
        print("Freeze: backbone+NeckB+HeadB trainable (like project phase2)")

    optimizer = optim.AdamW(build_adamw_groups(model, args.lr),
                            lr=args.lr, betas=(args.momentum, 0.999), weight_decay=0.0)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {total:,} params")

    # ---------- Data ----------
    door_ds = YOLODataset(args.virtual_door_data, args.imgsz, hyp=model.args)
    door_loader = DataLoader(door_ds, batch_size=args.batch, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate_fn,
                             drop_last=True, pin_memory=True)
    det_val_ds = YOLODataset(args.hand_obj_data, args.imgsz, split="val")
    door_val_ds = YOLODataset(args.virtual_door_data, args.imgsz, split="val")
    print(f"door train: {len(door_ds)} imgs ({len(door_loader)} steps/epoch)")

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    nb = len(door_loader)
    warmup_iters = max(round(3.0 * nb), 100)
    lrf = 0.01
    print(f"Official schedule: warmup_iters={warmup_iters}, horizon={args.schedule_epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        # Official linear scheduler target for this epoch (exp-58 cos_lr=False).
        lf = (1 - (epoch - 1) / args.schedule_epochs) * (1.0 - lrf) + lrf
        pbar = tqdm(door_loader, desc=f"adamw epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            ni = step + nb * (epoch - 1)
            # optimizer=auto sets warmup_bias_lr=0.0 for AdamW: all groups rise 0 -> target.
            if ni <= warmup_iters:
                warmup_lr = args.lr * lf * ni / warmup_iters
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr
            else:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lf

            batch["_task"] = 1
            total_l, box_l, cls_l, dfl_l, n = _step(model, batch, 1.0, optimizer, device)
            losses.append(total_l)
            pbar.set_postfix({"box": f"{box_l:.2f}", "cls": f"{cls_l:.2f}", "dfl": f"{dfl_l:.2f}",
                              "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
        avg = sum(losses) / len(losses)
        print(f"  epoch {epoch} loss={avg:.2f}", flush=True)
        val = _validate(model, args, device)
        print(f"  door mAP50={val['door']['mAP50']:.4f} det mAP50={val['det']['mAP50']:.4f}", flush=True)
        torch.save({"epoch": 50 + epoch, "phase": "phase2_adamw", "model": model.state_dict(),
                    "args": vars(args)}, save_dir / f"phase2_adamw_epoch{50 + epoch}.pt")

    print("Done", flush=True)


if __name__ == "__main__":
    main()
