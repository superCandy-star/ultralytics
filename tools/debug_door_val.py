#!/usr/bin/env python3
"""
door val 验证链路诊断：用训练过的 Neck B checkpoint 跑 door val，
统计 NMS 输出（pred 数量/conf/cls 分布）与 GT（数量/cls 分布）的匹配断点。
定位 door val mAP=0 是验证侧问题还是模型未学到。

用法:
    python tools/debug_door_val.py --weights runs/train/door_phase2_only_test/phase2_door_epoch51.pt
"""

import argparse, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dual_task import (  # noqa: E402
    DualHeadModel, YOLODataset, _init_strides, _load_batch, MODEL_CFG,
)
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(ROOT / "runs/train/door_phase2_only_test/phase2_door_epoch51.pt"))
    p.add_argument("--data_yaml",
                   default="/root/taojianwei/datasets/IRIS/auxiliary_datasets/20260729_datasets_yolo/data.yaml")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", type=str, default="0")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Device: {device}")

    model = DualHeadModel(cfg=MODEL_CFG, nc=[2, 4])
    model.args = IterableSimpleNamespace(**{"box": 7.5, "cls": 0.5, "dfl": 1.5})
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.to(device)
    _init_strides(model, args.imgsz, device)
    print(f"Loaded {args.weights} (epoch {ckpt.get('epoch', '?')})")

    hi = model._h_idx[1]
    h = model.model[hi]
    val_ds = YOLODataset(args.data_yaml, args.imgsz, split="val")
    print(f"door val: {len(val_ds)} images")

    model.eval()
    total_preds, n_imgs_with_pred = 0, 0
    confs, pred_cls = [], []
    gt_counts, gt_cls_counts = {}, {}
    iou_total = torch.zeros(0, 4, device=device)

    from ultralytics.utils.metrics import box_iou
    matched_imgs = 0

    for start in range(0, len(val_ds), args.batch):
        end = min(start + args.batch, len(val_ds))
        indices = list(range(start, end))
        batch_data = _load_batch(val_ds, indices, 1, device)
        with torch.no_grad():
            out = model._forward_to(batch_data["img"], hi)
        pred_all = out[0] if isinstance(out, tuple) else out
        if not isinstance(pred_all, torch.Tensor):
            print(f"  WARN: pred_all type={type(pred_all)}")
            continue
        dets = non_max_suppression(pred_all, conf_thres=0.001, iou_thres=0.65, nc=4, max_det=300)

        for bi in range(len(batch_data["img"])):
            pred = dets[bi]
            gt_labels = batch_data["label"][bi]
            n_gt = gt_labels.numel() // 5 if gt_labels.numel() else 0
            gt_cls = gt_labels[:, 0].long().tolist() if n_gt else []
            for c in gt_cls:
                gt_cls_counts[c] = gt_cls_counts.get(c, 0) + 1
            gt_counts[len(gt_cls)] = gt_counts.get(len(gt_cls), 0) + 1

            if pred is not None and len(pred):
                total_preds += len(pred)
                n_imgs_with_pred += 1
                confs.extend(pred[:, 4].tolist())
                pred_cls.extend(pred[:, 5].long().tolist())
                # IoU with GT (letterboxed space, both in 320 scale)
                pb = pred[:, :4].float()
                if n_gt:
                    gt_cx, gt_cy = gt_labels[:, 1], gt_labels[:, 2]
                    gt_w, gt_h = gt_labels[:, 3], gt_labels[:, 4]
                    gt_x1 = (gt_cx * args.imgsz - gt_w * args.imgsz / 2)
                    gt_y1 = (gt_cy * args.imgsz - gt_h * args.imgsz / 2)
                    gt_x2 = (gt_cx * args.imgsz + gt_w * args.imgsz / 2)
                    gt_y2 = (gt_cy * args.imgsz + gt_h * args.imgsz / 2)
                    gtb = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], -1).to(device)
                    iou = box_iou(gtb, pb)
                    if (iou >= 0.5).any():
                        matched_imgs += 1

    import numpy as np
    confs = np.array(confs)
    pred_cls_arr = np.array(pred_cls)
    print(f"\n=== NMS pred stats ===")
    print(f"images with preds: {n_imgs_with_pred}/{len(val_ds)}")
    print(f"total preds: {total_preds} (avg {total_preds / max(n_imgs_with_pred, 1):.1f}/img with pred)")
    if len(confs):
        print(f"conf: min={confs.min():.4f} p50={np.median(confs):.4f} p90={np.percentile(confs, 90):.4f} max={confs.max():.4f}")
        print(f"pred cls distribution: { {int(c): int((pred_cls_arr == c).sum()) for c in np.unique(pred_cls_arr)} }")
    print(f"\n=== GT stats ===")
    print(f"GT per-image: {dict(sorted(gt_counts.items()))}")
    print(f"GT cls distribution: {dict(sorted(gt_cls_counts.items()))}")
    print(f"images with any IoU>=0.5 match: {matched_imgs}")


if __name__ == "__main__":
    main()
