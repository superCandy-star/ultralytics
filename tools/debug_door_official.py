#!/usr/bin/env python3
"""
官方单头流程诊断：在虚拟门数据上训练，打印每一步 P3/P4/P5 (stride 8/16/32)
的正样本配对数量（单图 + 整 batch）和 box/cls/dfl 损失分量。

与双头 Neck B 的 TAL 分配对比，定位 door 任务训练异常（val 全 0、只有 1 类有精度）的原因。

用法:
    python tools/debug_door_official.py --cfg ultralytics/cfg/models/11/yolo11n.yaml \
        --data_yaml <door data.yaml> --batch 64 --imgsz 320 --lr 0.0001
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

import yaml  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402
from train_dual_task import YOLODataset, collate_fn  # noqa: E402


class DebugDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss + 每步打印 P3/P4/P5 正样本数与 loss 分量."""

    def __init__(self, model, *a, **kw):
        super().__init__(model, *a, **kw)
        self.step = 0
        self.fg_by_stride = {8: 0, 16: 0, 32: 0}
        self.n_batches = 0

    def get_assigned_targets_and_loss(self, preds, batch):
        result = super().get_assigned_targets_and_loss(preds, batch)
        (fg_mask, _, _, _, stride_tensor), loss, _ = result
        self.step += 1
        self.n_batches += fg_mask.shape[0]

        st = stride_tensor.squeeze(-1)  # [n_anchors]

        def fmt(fg_1d):
            return " ".join(
                f"s{int(s)}:fg={int(fg_1d[st == s].sum())}"
                for s in torch.unique(st))

        b, c, d = loss.detach().tolist()
        print(f"  [step {self.step}] img0: {fmt(fg_mask[0].float())} | "
              f"batch: {fmt(fg_mask.float().sum(0))} | "
              f"loss box={b:.3f} cls={c:.3f} dfl={d:.3f}", flush=True)
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default=str(ROOT / "ultralytics/cfg/models/11/yolo11n.yaml"))
    p.add_argument("--data_yaml",
                   default="/root/taojianwei/datasets/IRIS/auxiliary_datasets/20260729_datasets_yolo/data.yaml")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    init_seeds(42)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Device: {device}")

    with open(args.data_yaml) as f:
        nc = yaml.safe_load(f)["nc"]
    print(f"Door data nc={nc}, cfg={args.cfg}")

    # Official single-head model, random init (same as train.py --cfg without weights)
    model = DetectionModel(args.cfg, nc=nc, verbose=False)
    model.args = IterableSimpleNamespace(**{
        "box": 7.5, "cls": 0.5, "dfl": 1.5, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
        "copy_paste_mode": "flip", "cutmix": 0.0, "augmentations": None,
    })
    model.to(device)
    model.criterion = DebugDetectionLoss(model)  # debug loss, auto-inited on first forward
    criterion = model.criterion
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    ds = YOLODataset(args.data_yaml, args.imgsz, hyp=model.args)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate_fn, drop_last=True, pin_memory=True)
    print(f"door dataset: {len(ds)} images, {len(loader)} steps/epoch")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.937,
                          weight_decay=0.0005, nesterov=True)
    print(f"lr={args.lr}")

    for epoch in range(args.epochs):
        model.train()
        losses = []
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch["img"] = batch["img"].to(device)
            batch["batch_idx"] = batch["batch_idx"].to(device)
            batch["cls"] = batch["cls"].to(device)
            batch["bboxes"] = batch["bboxes"].to(device)
            optimizer.zero_grad()
            loss, loss_items = model.loss(batch)  # official: forward + criterion
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            losses.append(loss.sum().item())
            pbar.set_postfix({"loss": f"{loss.sum().item()/args.batch:.2f}"})

        print(f"epoch {epoch+1}: avg batch-total loss={sum(losses)/len(losses):.2f} "
              f"(per-image={sum(losses)/len(losses)/args.batch:.4f})")

    print(f"\nPer-stride fg totals over {criterion.step} steps: {criterion.fg_by_stride} "
          f"(per batch avg: { {k: round(v / criterion.n_batches, 1) for k, v in criterion.fg_by_stride.items()} })")


if __name__ == "__main__":
    main()
