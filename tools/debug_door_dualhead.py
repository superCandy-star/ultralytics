#!/usr/bin/env python3
"""
双头 Neck B 对照诊断：与 debug_door_official.py 相同的 door 数据、相同 seed、
相同 batch，打印 P3/P4/P5 正样本配对数量 + box/cls/dfl 损失。

用于对比官方单头 vs 双头 Neck B 的 TAL 分配，定位 door 任务训练异常原因。
加载 phase1 checkpoint 的 backbone（与项目 phase2 相同），NeckB/HeadB 保持随机初始化。

用法:
    python tools/debug_door_dualhead.py --batch 64 --imgsz 320
"""

import argparse, sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dual_task import (  # noqa: E402
    DualHeadModel, YOLODataset, collate_fn, _init_strides, _load_pretrained_weights,
    MODEL_CFG, BACKBONE_END, NECK_B_START, NECK_B_END, HEAD_B_IDX,
    _freeze_all, _unfreeze_module_list, _create_param_groups,
)
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402
from debug_door_official import DebugDetectionLoss  # noqa: E402


def make_door_criterion(model):
    """Create a DebugDetectionLoss bound to Head B (model.model[36]).
    v8DetectionLoss.__init__ reads model.model[-1], so temporarily swap it."""
    orig = model.model[-1]
    model.model[-1] = model.model[HEAD_B_IDX]
    crit = DebugDetectionLoss(model)
    model.model[-1] = orig
    return crit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",
                   default=str(ROOT / "runs/train/dual_v2/phase1_handobj_epoch50.pt"))
    p.add_argument("--data_yaml",
                   default="/root/taojianwei/datasets/IRIS/auxiliary_datasets/20260729_datasets_yolo/data.yaml")
    p.add_argument("--steps", type=int, default=26)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    init_seeds(42)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Device: {device}")

    # ---------- Model: same as train_door_phase2.py ----------
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
    skip_prefixes = tuple(f"model.{i}." for i in range(NECK_B_START, HEAD_B_IDX + 1))
    loaded_sd = {k: v for k, v in sd.items() if not k.startswith(skip_prefixes)}
    model.load_state_dict(loaded_sd, strict=False)
    print(f"Loaded phase1 backbone+NeckA+HeadA ({len(loaded_sd)}/{len(sd)} tensors, "
          f"NeckB/HeadB fresh init)")

    _init_strides(model, args.imgsz, device)
    model.train()

    # Debug loss bound to Head B
    model._criteria = {1: make_door_criterion(model)}
    criterion = model._criteria[1]

    # ---------- Data: same door data, same seed/order as official debug ----------
    ds = YOLODataset(args.data_yaml, args.imgsz, hyp=model.args)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate_fn, drop_last=True, pin_memory=True)
    print(f"door dataset: {len(ds)} images, {len(loader)} steps")

    # Phase-2 project optimizer config (freeze NeckA/HeadA, train backbone+NeckB+HeadB)
    _freeze_all(model)
    _unfreeze_module_list(model, list(range(BACKBONE_END + 1)))
    _unfreeze_module_list(model, list(range(NECK_B_START, NECK_B_END + 1)))
    _unfreeze_module_list(model, [HEAD_B_IDX])
    lr_map = {'backbone': 5e-4, 'neck_b': 0.001, 'head_b': 0.005}
    groups = _create_param_groups(model, lr_map)
    optimizer = optim.SGD(groups, lr=0.01, momentum=0.937, weight_decay=0.0005, nesterov=True)
    # warmup 3 epochs like project: step 0 uses lr/3
    for g in optimizer.param_groups:
        g['lr'] = g['initial_lr'] / 3

    total_steps = min(args.steps, len(loader))
    for step in range(total_steps):
        batch = next(iter(loader))
        batch['_task'] = 1
        batch["img"] = batch["img"].to(device)
        batch["batch_idx"] = batch["batch_idx"].to(device)
        batch["cls"] = batch["cls"].to(device)
        batch["bboxes"] = batch["bboxes"].to(device)
        optimizer.zero_grad()
        loss, loss_items = model.dual_loss(batch, 1)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

    print(f"\nPer-stride fg totals over {criterion.step} steps: {criterion.fg_by_stride} "
          f"(per batch avg: { {k: round(v / criterion.n_batches, 1) for k, v in criterion.fg_by_stride.items()} })")


if __name__ == "__main__":
    main()
