#!/usr/bin/env python3
"""
YOLO11 三阶段双任务训练：手物检测 + 虚拟门检测
两个独立 neck 和 head 共享 backbone，分阶段训练避免任务冲突。

阶段1 (1-50):  训练 backbone + Neck A + Head A (手物), 冻结 Neck B + Head B
阶段2 (51-100): 训练 backbone + Neck B + Head B (门), 冻结 Neck A + Head A
阶段3a (101-130): 冻结 backbone, 微调 necks + heads
阶段3b (131-150): 解冻全部, 小 LR 联合微调

用法:
    # 单独跑阶段1
    python tools/train_dual_task.py --phase 1 --hand_obj_data ... --epochs 50 ...
    # 单独跑阶段2 (从阶段1 checkpoint 继续)
    python tools/train_dual_task.py --phase 2 --virtual_door_data ... --weights runs/train/.../last.pt --epochs 50 ...
    # 或一次跑完三个阶段
    python tools/train_dual_task.py \
      --hand_obj_data /root/taojianwei/datasets/IRIS/20260727_datasets_yolo/data.yaml \
      --virtual_door_data /root/taojianwei/datasets/IRIS/auxiliary_datasets/20260729_datasets_yolo/data.yaml \
      --weights /root/taojianwei/projects/ultralytics/weights/yolo11n.pt \
      --batch 64 --imgsz 320 --device 0 --num_workers 4
"""

import argparse, math, sys, time, copy
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import init_seeds
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.utils import img2label_paths

MODEL_CFG = str(ROOT / "ultralytics/cfg/models/11/dual_task_handobj_door.yaml")

# Module index ranges for the dual-task model with independent necks
# backbone: 0-10, Neck A: 11-22, Head A: 23, Neck B: 24-35, Head B: 36
BACKBONE_END = 10
NECK_A_START, NECK_A_END = 11, 22
HEAD_A_IDX = 23
NECK_B_START, NECK_B_END = 24, 35
HEAD_B_IDX = 36


# ======================================================================
#  1.  Dataset (YOLO cache verify + DataLoader multi-thread)
# ======================================================================
class YOLODataset(Dataset):
    """YOLO detection dataset with cache-based label loading (like official YOLO).

    First run scans + caches label info (skips corrupt images).
    Subsequent runs load the cache instantly.
    """

    def __init__(self, yaml_path: str, imgsz: int, split: str = "train"):
        self.data = check_det_dataset(yaml_path)
        raw = self.data.get(split, self.data.get("train", []))

        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.im_files = []
        for p in raw:
            p = Path(p)
            if p.is_dir():
                self.im_files.extend(sorted([str(f) for f in p.iterdir() if f.suffix.lower() in exts]))
            else:
                self.im_files.append(str(p))
        self.label_files = img2label_paths(self.im_files)
        self.nc = self.data["nc"]
        self.imgsz = imgsz
        self.labels = self._load_labels()

    def _load_labels(self) -> list:
        """Load labels via cache file, or build cache on first run."""
        from ultralytics.data.utils import load_dataset_cache_file, save_dataset_cache_file, verify_image_label, get_hash
        from multiprocessing.pool import ThreadPool
        from itertools import repeat

        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        # Try loading cached labels
        try:
            cache = load_dataset_cache_file(cache_path)
            if cache.get("hash") == get_hash(self.label_files + self.im_files):
                nf, nm, ne, nc, n = cache["results"]
                LOGGER.info(f"Cache OK ({cache_path.stem}.cache): {nf} imgs, {nm+ne} bg, {nc} corrupt")
                return cache["labels"]
        except Exception:
            pass

        # First run: scan all images in parallel
        labels = []; nf = nm = ne = nc = 0
        prefix = f"{Path(self.im_files[0]).parent.name}: " if self.im_files else ""
        desc = f"Scanning {cache_path.parent.name}..."
        results = ThreadPool(8).imap(verify_image_label,
            zip(self.im_files, self.label_files, repeat(prefix),
                repeat(False), repeat(self.nc), repeat(0), repeat(0), repeat(False)))

        pbar = tqdm(results, desc=desc, total=len(self.im_files))
        for im_file, lb, shape, seg, kpt, nm_f, nf_f, ne_f, nc_f, msg in pbar:
            nm += nm_f; nf += nf_f; ne += ne_f; nc += nc_f
            if im_file:
                labels.append({"im_file": im_file, "shape": shape,
                    "cls": lb[:, 0:1] if len(lb) else np.zeros((0, 1), dtype=np.float32),
                    "bboxes": lb[:, 1:] if len(lb) else np.zeros((0, 4), dtype=np.float32),
                    "normalized": True, "bbox_format": "xywh"})
            pbar.desc = f"{desc} {nf} imgs, {nm+ne} bg, {nc} corrupt"
        pbar.close()

        # Save cache for next time
        try:
            from ultralytics.data.dataset import DATASET_CACHE_VERSION
            save_dataset_cache_file(prefix.replace(": ", ""), cache_path, {
                "labels": labels,
                "hash": get_hash(self.label_files + self.im_files),
                "results": (nf, nm, ne, nc, len(self.im_files)),
            }, DATASET_CACHE_VERSION)
        except Exception:
            pass

        msg = f"Scanned: {nf} imgs, {nm+ne} bg, {nc} corrupt"
        if nc:
            msg += f"  ({nc} corrupt images skipped)"
        LOGGER.info(msg)
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lb = self.labels[idx]
        img = cv2.imread(lb["im_file"])
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {lb['im_file']}")
        img = img[..., ::-1]  # BGR→RGB
        oh, ow = img.shape[:2]

        # Letterbox resize
        S = self.imgsz
        scale = S / max(oh, ow)
        nw, nh = int(ow * scale), int(oh * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out = np.full((S, S, 3), 114, dtype=np.uint8)
        dx, dy = (S - nw) // 2, (S - nh) // 2
        out[dy:dy+nh, dx:dx+nw] = img

        # Labels: [cls, cx, cy, w, h] normalized
        cls = lb["cls"].copy()
        bboxes = lb["bboxes"].copy()
        if len(cls):
            valid = cls[:, 0] < self.nc
            cls = cls[valid]
            bboxes = bboxes[valid]
            if len(bboxes):
                bboxes[:, 0] = (bboxes[:, 0] * ow * scale + dx) / S   # cx
                bboxes[:, 1] = (bboxes[:, 1] * oh * scale + dy) / S   # cy
                bboxes[:, 2] *= ow * scale / S                         # w
                bboxes[:, 3] *= oh * scale / S                         # h

        label = np.concatenate([cls, bboxes], axis=1) if len(cls) else np.zeros((0, 5), dtype=np.float32)
        img_t = torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1).astype(np.float32) / 255.0))
        return {"img": img_t, "label": torch.from_numpy(label)}


def collate_fn(batch: list) -> dict:
    """Collate individual samples into a batch dict compatible with _step."""
    imgs = torch.stack([b["img"] for b in batch], 0)
    labels = [b["label"] for b in batch]

    # Build YOLO-format targets [N, 6] = [batch_idx, cls, cx, cy, w, h]
    targets = []
    for bi, lbl in enumerate(labels):
        if lbl.numel():
            n = lbl.shape[0]
            bi_t = torch.full((n, 1), bi, dtype=torch.float32)
            targets.append(torch.cat([bi_t, lbl], dim=1))
    targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))
    return {
        "img": imgs,
        "label": labels,
        "_task": -1,  # filled by caller
        "batch_idx": targets[:, 0].long(),
        "cls": targets[:, 1].long(),
        "bboxes": targets[:, 2:6],
    }


# ======================================================================
#  2.  Model
# ======================================================================
class DualHeadModel(DetectionModel):
    """Dual-head model: two Detect modules sharing backbone+neck."""

    def __init__(self, cfg=MODEL_CFG, ch=3, nc=None, verbose=True):
        total = sum(nc) if isinstance(nc, (list, tuple)) else nc
        super().__init__(cfg, ch, total, verbose=False)
        self.yaml["nc"] = nc
        self._h_idx = [i for i, m in enumerate(self.model) if hasattr(m, "stride")]

    def _forward_to(self, x, head_idx: int):
        y = []
        for i, m in enumerate(self.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if i == head_idx:
                break
        return x

    def dual_loss(self, batch, task_id: int):
        """Forward + loss for one head."""
        img = batch["img"].to(next(self.parameters()).device)
        hi = self._h_idx[task_id]
        preds = self._forward_to(img, hi)
        # YOLOv11 Detect outputs 'scores' but v8DetectionLoss uses both 'scores' and 'cls'
        if "scores" in preds and "cls" not in preds:
            preds["cls"] = preds["scores"]

        if not hasattr(self, "_criteria"):
            self._criteria = {}
        if task_id not in self._criteria:
            from ultralytics.utils.loss import v8DetectionLoss
            orig = self.model[-1]
            self.model[-1] = self.model[hi]
            self._criteria[task_id] = v8DetectionLoss(self)
            self.model[-1] = orig

        return self._criteria[task_id](preds, batch)


# ======================================================================
#  3.  Model utilities (freeze / param groups)
# ======================================================================
def _freeze_module_list(model, indices: list[int]):
    """Set requires_grad=False for all parameters in given module indices."""
    for idx in indices:
        for p in model.model[idx].parameters():
            p.requires_grad = False


def _unfreeze_module_list(model, indices: list[int]):
    """Set requires_grad=True for all parameters in given module indices."""
    for idx in indices:
        for p in model.model[idx].parameters():
            p.requires_grad = True


def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def _create_param_groups(model, lr_map: dict) -> list[dict]:
    """Create optimizer param_groups based on component LR map.

    lr_map keys: 'backbone', 'neck_a', 'head_a', 'neck_b', 'head_b'
    Values: learning rate (0 = freeze, skip from optimizer)
    """
    groups = []
    for component, lr in lr_map.items():
        if lr == 0:
            continue
        if component == 'backbone':
            indices = list(range(BACKBONE_END + 1))
        elif component == 'neck_a':
            indices = list(range(NECK_A_START, NECK_A_END + 1))
        elif component == 'head_a':
            indices = [HEAD_A_IDX]
        elif component == 'neck_b':
            indices = list(range(NECK_B_START, NECK_B_END + 1))
        elif component == 'head_b':
            indices = [HEAD_B_IDX]
        else:
            continue

        params = []
        for idx in indices:
            for p in model.model[idx].parameters():
                if p.requires_grad:
                    params.append(p)

        if params:
            groups.append({'params': params, 'lr': lr, 'initial_lr': lr})

    return groups


def _init_strides(model, imgsz: int, device: torch.device):
    """Forward warmup to initialize Detect head strides."""
    from ultralytics.utils.tal import make_anchors
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    for hi in model._h_idx:
        h = model.model[hi]
        with torch.no_grad():
            raw = model._forward_to(dummy, hi)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        if isinstance(raw, dict) and h.stride.sum() == 0 and "feats" in raw:
            feats = [f for f in raw["feats"]]
            h.stride = torch.tensor([imgsz / f.shape[-1] for f in feats], device=device)


def _load_pretrained_weights(model, weight_path: str):
    """Load pretrained weights, skipping shape-mismatched layers."""
    raw = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = raw if isinstance(raw, dict) and "model" not in raw else raw.get("model", raw)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    sd = model.state_dict()
    lo, sk = 0, 0
    for k, v in state.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v.to(sd[k].dtype)
            lo += 1
        else:
            sk += 1
    model.load_state_dict(sd, strict=False)
    LOGGER.info(f"Pretrained: loaded {lo}, skipped {sk}")
    return model


# ======================================================================
#  4.  Phase-specific training
# ======================================================================
def _run_phase(model, loader_train, loader_val, phase_name: str,
               epochs: int, lr_map: dict, start_epoch: int,
               args, device: torch.device,
               warmup_epochs: int = 0):
    """Generic phase runner: freezes model per lr_map, trains, validates."""
    # Freeze/unfreeze based on lr_map
    _freeze_all(model)
    for component, lr in lr_map.items():
        if lr > 0:
            if component == 'backbone':
                _unfreeze_module_list(model, list(range(BACKBONE_END + 1)))
            elif component == 'neck_a':
                _unfreeze_module_list(model, list(range(NECK_A_START, NECK_A_END + 1)))
            elif component == 'head_a':
                _unfreeze_module_list(model, [HEAD_A_IDX])
            elif component == 'neck_b':
                _unfreeze_module_list(model, list(range(NECK_B_START, NECK_B_END + 1)))
            elif component == 'head_b':
                _unfreeze_module_list(model, [HEAD_B_IDX])

    # Build optimizer
    param_groups = _create_param_groups(model, lr_map)
    if not param_groups:
        LOGGER.warning("No trainable parameters!")
        return

    optimizer = optim.SGD(param_groups, lr=args.lr0, momentum=0.937, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bs = args.batch

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"[{phase_name}] Trainable: {trainable:,}/{total_params:,} params, "
                f"groups: {[(g['lr'], len(g['params'])) for g in param_groups]}")

    def _infinite_loader(loader):
        while True:
            yield from loader

    # Handle alternating loaders for joint phases
    if isinstance(loader_train, (list, tuple)):
        iter_a = _infinite_loader(loader_train[0])
        iter_b = _infinite_loader(loader_train[1])
        joint_mode = True
    else:
        iter_a = _infinite_loader(loader_train)
        iter_b = None
        joint_mode = False

    det_val_ds, door_val_ds = loader_val if loader_val else (None, None)

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()

        if epoch - start_epoch < warmup_epochs:
            warmup_factor = (epoch - start_epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                base_lr = pg.get('initial_lr', args.lr0)
                pg['lr'] = base_lr * warmup_factor

        if joint_mode:
            total_a = len(loader_train[0].dataset) // bs
            total_b = len(loader_train[1].dataset) // bs
            total_steps = min(total_a, total_b) * 2
        else:
            total_steps = len(loader_train.dataset) // bs
        if args.test_iters:
            total_steps = min(total_steps, args.test_iters)
        max_steps = total_steps
        losses = []
        t_epoch = time.time()

        pbar = tqdm(total=total_steps, desc=f"{phase_name} Epoch {epoch}/{start_epoch+epochs-1}")
        for step in range(max_steps):
            if joint_mode:
                is_det = (step % 2 == 0)
                batch = next(iter_a) if is_det else next(iter_b)
                batch['_task'] = 0 if is_det else 1
            else:
                batch = next(iter_a)
                batch['_task'] = 0 if 'hand' in phase_name else 1
            total_l, box_l, cls_l, dfl_l, n = _step(model, batch, 1.0, optimizer, device)
            losses.append(total_l)
            t = "det" if batch['_task'] == 0 else "door"
            pbar.set_postfix({"task": t, "box": f"{box_l:.2f}", "cls": f"{cls_l:.2f}", "dfl": f"{dfl_l:.2f}", "n": n})
            pbar.update(1)

        pbar.close()
        scheduler.step()

        avg_loss = float(np.mean(losses)) if losses else 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  {phase_name} epoch {epoch}  loss={avg_loss:.2f}  lr={current_lr:.2e}  ({time.time()-t_epoch:.0f}s)", flush=True)

        # Validation
        if det_val_ds is not None:
            t_val = time.time()
            val_results = _validate(model, det_val_ds, door_val_ds, args, device)
            det_m = val_results.get('det', {}).get('mAP50', 0)
            door_m = val_results.get('door', {}).get('mAP50', 0)
            print(f"  val mAP@0.5: det={det_m:.4f} door={door_m:.4f}  ({time.time()-t_val:.0f}s)", flush=True)

        ckpt_path = Path(args.project) / args.name / f"{phase_name}_epoch{epoch}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "phase": phase_name, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "args": vars(args)}, ckpt_path)


def train(args):
    init_seeds(42)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Device: {device}")

    # ---------- Model ----------
    model = DualHeadModel(cfg=MODEL_CFG, nc=[2, 4])
    from ultralytics.utils import IterableSimpleNamespace
    model.args = IterableSimpleNamespace(**{
        "box": 7.5, "cls": 0.5, "dfl": 1.5, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
    })
    model.to(device)

    # Align cv3 initialization to match single-head reference
    from ultralytics.nn.tasks import DetectionModel as _RefModel
    _ref_a = _RefModel('ultralytics/cfg/models/11/yolo11.yaml', nc=2, verbose=False)
    _ref_a.to(device)
    # Copy backbone+neck from ref (Neck A shares same structure as yolo11 neck)
    for i in range(len(_ref_a.model) - 1):
        model.model[i].load_state_dict(_ref_a.model[i].state_dict())
    _ref_det = _ref_a.model[-1]; _our_a = model.model[model._h_idx[0]]
    for j in range(len(_ref_det.cv3)):
        _our_a.cv3[j].load_state_dict(_ref_det.cv3[j].state_dict())

    # Neck B is independent, so we don't align it with ref

    LOGGER.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params, heads at {model._h_idx}")

    # Load pretrained weights (backbone + Neck A only; Neck B + Heads skipped)
    if args.weights and Path(args.weights).exists():
        model = _load_pretrained_weights(model, args.weights)

    _init_strides(model, args.imgsz, device)
    model.train()

    # ---------- Data ----------
    bs = args.batch
    num_workers = args.num_workers
    det_ds = YOLODataset(args.hand_obj_data, args.imgsz)
    door_ds = YOLODataset(args.virtual_door_data, args.imgsz)
    print(f"det dataset: {len(det_ds)} images  |  door dataset: {len(door_ds)} images")

    det_loader = DataLoader(det_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
                            collate_fn=collate_fn, drop_last=True, pin_memory=True)
    door_loader = DataLoader(door_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
                             collate_fn=collate_fn, drop_last=True, pin_memory=True)
    det_val_ds = YOLODataset(args.hand_obj_data, args.imgsz, split="val")
    door_val_ds = YOLODataset(args.virtual_door_data, args.imgsz, split="val")

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────
    #  Phase dispatch
    # ──────────────────────────────────────────────
    phases_to_run = [int(x.strip()) for x in args.phases.split(",")]
    start_epoch = 1

    for phase in phases_to_run:
        if phase == 1:
            _run_phase(model, det_loader, (det_val_ds, door_val_ds),
                       "phase1_handobj", args.epochs_per_phase,
                       {'backbone': 0.01, 'neck_a': 0.01, 'head_a': 0.01},
                       start_epoch, args, device, warmup_epochs=3)
            start_epoch += args.epochs_per_phase
        elif phase == 2:
            _run_phase(model, door_loader, (det_val_ds, door_val_ds),
                       "phase2_door", args.epochs_per_phase,
                       {'backbone': 5e-4, 'neck_b': 0.001, 'head_b': 0.005},
                       start_epoch, args, device, warmup_epochs=3)
            start_epoch += args.epochs_per_phase
        elif phase == 3:
            _run_phase(model, (det_loader, door_loader), (det_val_ds, door_val_ds),
                       "phase3a_joint", 30,
                       {'neck_a': 5e-4, 'neck_b': 5e-4, 'head_a': 1e-3, 'head_b': 1e-3},
                       start_epoch, args, device)
            start_epoch += 30
            _run_phase(model, (det_loader, door_loader), (det_val_ds, door_val_ds),
                       "phase3b_fullft", 20,
                       {'backbone': 1e-5, 'neck_a': 1e-4, 'neck_b': 1e-4,
                        'head_a': 5e-4, 'head_b': 5e-4},
                       start_epoch, args, device)

    print(f"Done. Final weights: {save_dir}", flush=True)


# ======================================================================
#   Validation helpers (shared with val_dual_task.py)
# ======================================================================
def _compute_ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP from recall/precision curve (COCO 101-point interpolation)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _compute_map(image_stats: list, head_nc: int, iou_thresholds: torch.Tensor,
                 n_gt_per_class: dict, device: torch.device) -> dict:
    """Compute mAP@0.5 and mAP@0.5:0.95 from per-image prediction/GT data.

    Uses greedy IoU matching (highest-IoU prediction assigned to each GT,
    one GT per prediction) per image + class, then accumulates over all images
    to build per-class precision-recall curves.
    """
    from ultralytics.utils.metrics import box_iou

    niou = len(iou_thresholds)
    all_tp = {c: [] for c in range(head_nc)}
    all_conf = {c: [] for c in range(head_nc)}

    for img_stat in image_stats:
        pred_boxes = img_stat["pred_boxes"]
        pred_confs = img_stat["pred_confs"]
        pred_cls = img_stat["pred_cls"]
        gt_boxes = img_stat["gt_boxes"]
        gt_cls = img_stat["gt_cls"]

        M = len(pred_boxes)
        N = len(gt_boxes)
        if M == 0:
            continue

        for cls_id in range(head_nc):
            cls_mask = pred_cls == cls_id
            n_pred_cls = cls_mask.sum().item()
            if n_pred_cls == 0:
                continue

            # Sort by confidence descending
            cls_pred_boxes = pred_boxes[cls_mask]
            cls_pred_confs = pred_confs[cls_mask]
            sort_idx = torch.argsort(cls_pred_confs, descending=True)
            cls_pred_boxes = cls_pred_boxes[sort_idx]
            cls_pred_confs = cls_pred_confs[sort_idx]

            # GTs of this class in this image
            gt_cls_mask = gt_cls == cls_id
            n_gt_cls_img = gt_cls_mask.sum().item()

            tp_per_pred = torch.zeros((n_pred_cls, niou), device=device)

            if n_gt_cls_img > 0:
                cls_gt_boxes = gt_boxes[gt_cls_mask]
                iou = box_iou(cls_gt_boxes, cls_pred_boxes)  # [n_gt, n_pred]

                for t in range(niou):
                    matched_gt = [False] * n_gt_cls_img
                    for p_idx in range(n_pred_cls):
                        best_iou = 0.0
                        best_gt_idx = -1
                        for g_idx in range(n_gt_cls_img):
                            if matched_gt[g_idx]:
                                continue
                            iou_val = iou[g_idx, p_idx].item()
                            if iou_val > best_iou:
                                best_iou = iou_val
                                best_gt_idx = g_idx
                        if best_iou >= iou_thresholds[t].item():
                            tp_per_pred[p_idx, t] = 1.0
                            matched_gt[best_gt_idx] = True

            all_tp[cls_id].append(tp_per_pred.cpu().numpy())
            all_conf[cls_id].append(cls_pred_confs.cpu().numpy())

    # Compute AP per class
    aps50 = []
    aps50_95 = []

    for cls_id in range(head_nc):
        if not all_tp[cls_id]:
            aps50.append(0.0)
            aps50_95.append(0.0)
            continue

        tp = np.concatenate(all_tp[cls_id], axis=0)      # [N_preds, 10]
        conf = np.concatenate(all_conf[cls_id], axis=0)   # [N_preds]
        sort_idx = np.argsort(-conf)
        tp = tp[sort_idx]

        n_gt = n_gt_per_class.get(cls_id, 0)
        if n_gt == 0:
            aps50.append(0.0)
            aps50_95.append(0.0)
            continue

        tp_cum = np.cumsum(tp, axis=0)
        fp_cum = np.arange(1, len(tp) + 1).reshape(-1, 1) - tp_cum
        rec = tp_cum / max(n_gt, 1)
        prec = tp_cum / (tp_cum + fp_cum + 1e-16)

        ap50 = _compute_ap_from_pr(rec[:, 0], prec[:, 0])
        aps50.append(ap50)

        ap_per_iou = [_compute_ap_from_pr(rec[:, t], prec[:, t]) for t in range(niou)]
        aps50_95.append(float(np.mean(ap_per_iou)))

    return {
        "mAP50": float(np.mean(aps50)) if aps50 else 0.0,
        "mAP50-95": float(np.mean(aps50_95)) if aps50_95 else 0.0,
        "per_class_ap50": aps50,
        "per_class_ap50_95": aps50_95,
    }


def _validate_one_head(model, val_ds, task_id: int, imgsz: int, batch: int,
                       device: torch.device, max_imgs: int = 500) -> dict:
    """Validate a single task head on its val dataset with proper mAP.

    - Keeps model in train mode so Detect.forward() returns dict
    - Decodes DFL boxes to pixel coords (grid * stride)
    - Converts GT from normalized [0,1] to pixel xyxy
    - Greedy per-image IoU matching → per-class PR curves → mAP
    """
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.loss import v8DetectionLoss

    hi = model._h_idx[task_id]
    h = model.model[hi]
    head_nc = [2, 4][task_id]

    # Warmup forward to init stride
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        raw = model._forward_to(dummy, hi)
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    if isinstance(raw, dict) and h.stride.sum() == 0 and "feats" in raw:
        feats = [f for f in raw["feats"]]
        h.stride = torch.tensor([imgsz / f.shape[-1] for f in feats], device=device)

    iou_thresholds = torch.linspace(0.5, 0.95, 10, device=device)
    n_samples = min(len(val_ds), max_imgs)
    n_batches = (n_samples + batch - 1) // batch

    image_stats = []
    n_gt_per_class = {}

    pbar = tqdm(total=n_batches, desc=f"  val task={task_id}")
    for start in range(0, n_samples, batch):
        end = min(start + batch, n_samples)
        indices = list(range(start, end))
        batch_data = _load_batch(val_ds, indices, task_id, device)

        with torch.no_grad():
            raw = model._forward_to(batch_data["img"], hi)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        if not isinstance(raw, dict):
            continue
        preds = raw
        if "scores" in preds and "cls" not in preds:
            preds["cls"] = preds["scores"]

        # Decode boxes: DFL + dist2bbox(xywh=True) to match official Detect output.
        # non_max_suppression expects xywh input (it converts to xyxy internally).
        from ultralytics.utils.tal import dist2bbox
        anc, stride_t = make_anchors(preds["feats"], h.stride, 0.5)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()  # [B, anchors, 4*reg_max]
        # Temporarily replace model[-1] so v8DetectionLoss reads the correct head
        orig_last = model.model[-1]
        model.model[-1] = h
        tmp_criterion = v8DetectionLoss(model)
        model.model[-1] = orig_last
        # DFL decode, then xywh output (grid coords)
        if tmp_criterion.use_dfl:
            b, a, c = pred_distri.shape
            dist = pred_distri.view(b, a, 4, c // 4).softmax(3).matmul(
                tmp_criterion.proj.type(pred_distri.dtype))
        else:
            dist = pred_distri
        boxes_xywh_grid = dist2bbox(dist, anc, xywh=True)              # [B, anchors, 4] xywh grid
        boxes_xywh = boxes_xywh_grid * stride_t.view(1, -1, 1)          # → pixel xywh

        for bi in range(len(batch_data["img"])):
            # --- predictions ---
            # BCN format [1, 4+nc, anchors] in xywh — CRITICAL:
            # 1) last dim must NOT be 6 (else NMS treats as end2end and misparses classes)
            # 2) boxes must be xywh (NMS converts xywh→xyxy internally)
            bi_cls = torch.sigmoid(preds["cls"][bi])               # [nc, anchors]
            bi_box = boxes_xywh[bi].permute(1, 0)                   # [4, anchors] xywh
            pred = torch.cat([bi_box, bi_cls], dim=0).unsqueeze(0)  # [1, 4+nc, anchors]
            pred = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.65,
                                       nc=head_nc, max_det=300)[0]

            # DEBUG: print first valid pred vs GT for door task
            if task_id == 1 and pred is not None and len(pred) > 0 and bi == 0 and start == 0:
                gt_lbl = batch_data["label"][0]
                print(f"\n  [DEBUG door] pred[0]={pred[0][:4].tolist()} (pixel xyxy)  "
                      f"gt[0]={gt_lbl[0].tolist() if gt_lbl.numel() else 'N/A'} (norm cxcywh)",
                      flush=True)

            if pred is None or len(pred) == 0:
                pred_boxes = torch.zeros((0, 4), device=device)
                pred_confs = torch.zeros(0, device=device)
                pred_cls = torch.zeros(0, dtype=torch.long, device=device)
            else:
                pred_boxes = pred[:, :4]
                pred_confs = pred[:, 4]
                pred_cls = pred[:, 5].long()

            # --- ground truth ---
            gt_labels = batch_data["label"][bi]  # [N, 5] normalized [cls, cx, cy, w, h]
            if gt_labels.numel():
                gt_cls_img = gt_labels[:, 0].long()
                gt_cx, gt_cy = gt_labels[:, 1], gt_labels[:, 2]
                gt_w, gt_h = gt_labels[:, 3], gt_labels[:, 4]
                gt_x1 = (gt_cx - gt_w / 2) * imgsz
                gt_y1 = (gt_cy - gt_h / 2) * imgsz
                gt_x2 = (gt_cx + gt_w / 2) * imgsz
                gt_y2 = (gt_cy + gt_h / 2) * imgsz
                gt_boxes = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1).to(device)
                gt_boxes = gt_boxes.clamp(0, imgsz)
            else:
                gt_boxes = torch.zeros((0, 4), device=device)
                gt_cls_img = torch.zeros(0, dtype=torch.long, device=device)

            # Count GTs per class
            for c in gt_cls_img.tolist():
                n_gt_per_class[c] = n_gt_per_class.get(c, 0) + 1

            image_stats.append({
                "pred_boxes": pred_boxes, "pred_confs": pred_confs,
                "pred_cls": pred_cls, "gt_boxes": gt_boxes, "gt_cls": gt_cls_img,
            })

        pbar.update(1)

    pbar.close()

    if not image_stats:
        return {"mAP50": 0.0, "mAP50-95": 0.0}

    return _compute_map(image_stats, head_nc, iou_thresholds, n_gt_per_class, device)


def _validate(model, det_val_ds, door_val_ds, args, device):
    """Validate both heads on their respective val datasets."""
    model.train()  # keep train mode — Detect.forward() returns dict in train mode

    results = {}
    for task_name, val_ds, task_id in [("det", det_val_ds, 0), ("door", door_val_ds, 1)]:
        if val_ds is None or len(val_ds) == 0:
            results[task_name] = {"mAP50": 0.0, "mAP50-95": 0.0}
            continue
        t0 = time.time()
        r = _validate_one_head(model, val_ds, task_id, args.imgsz, args.batch, device)
        results[task_name] = r
        print(f"  {task_name} val: mAP@0.5={r['mAP50']:.4f}  mAP@0.5:0.95={r['mAP50-95']:.4f}"
              f"  per_class={[f'{v:.3f}' for v in r['per_class_ap50']]}  ({time.time()-t0:.0f}s)",
              flush=True)

    return results


def _load_batch(ds, indices, task_id, device):
    imgs, labels = [], []
    for i in indices:
        item = ds[i]
        imgs.append(item["img"])
        labels.append(item["label"])
    imgs = torch.stack(imgs, 0).to(device)
    batch = {"img": imgs, "label": labels, "_task": task_id}

    # Build YOLO-format targets [N, 6] = [batch_idx, cls, x, y, w, h]
    targets = []
    for bi, lbl in enumerate(labels):
        if lbl.numel():
            n = lbl.shape[0]
            bi_t = torch.full((n, 1), bi, dtype=torch.float32)
            targets.append(torch.cat([bi_t, lbl[:, :5].float()], 1))
    targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))
    batch["batch_idx"] = targets[:, 0].long()
    batch["cls"] = targets[:, 1].long()
    batch["bboxes"] = targets[:, 2:6]
    return batch


def _step(model, batch, weight, optimizer, device):
    batch["img"] = batch["img"].to(device)
    batch["batch_idx"] = batch["batch_idx"].to(device)
    batch["cls"] = batch["cls"].to(device)
    batch["bboxes"] = batch["bboxes"].to(device)

    optimizer.zero_grad()
    result = model.dual_loss(batch, batch["_task"])
    if isinstance(result, (list, tuple)):
        loss = result[0]
        loss_items = result[1] if len(result) > 1 else None
    else:
        loss = result
        loss_items = None

    if hasattr(loss, 'shape') and loss.numel() > 1:
        loss_sum = loss.sum()
    else:
        loss_sum = loss

    (weight * loss_sum).backward()
    # Gradient clipping (official YOLO default max_norm=10.0) — prevents
    # explosion from randomly-initialized head layers in early training
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    # Parse loss items: [box, cls, dfl] * batch_size
    if loss_items is not None:
        items = loss_items.tolist() if torch.is_tensor(loss_items) else loss_items
        box_l, cls_l, dfl_l = items[0], items[1], items[2]
    elif hasattr(loss, 'shape') and loss.numel() >= 3:
        raw = loss.tolist() if isinstance(loss, torch.Tensor) else loss
        box_l, cls_l, dfl_l = raw[0], raw[1], raw[2]
    else:
        box_l, cls_l, dfl_l = 0, 0, 0

    batch_size = batch["img"].shape[0]
    n_instances = len(batch["cls"])  # actual GT objects, not images
    per_image_total = loss_sum.item() / batch_size
    weighted_total = per_image_total * weight
    return weighted_total, box_l, cls_l, dfl_l, n_instances


# ======================================================================
#  4.  Entry
# ======================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand_obj_data", required=True)
    p.add_argument("--virtual_door_data", required=True)
    p.add_argument("--weights", default=str(ROOT / "weights/yolo11n.pt"))
    p.add_argument("--det_loss_weight", type=float, default=1.0)
    p.add_argument("--door_loss_weight", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="dual_task")
    p.add_argument("--save_period", type=int, default=10)
    p.add_argument("--test_iters", type=int, default=0,
                   help="run N iters then validate & exit (0=full epochs)")
    p.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader num_workers for multi-thread loading")
    p.add_argument("--door_ratio", type=int, default=0,
                   help="override auto door_ratio (det:door). 0=auto from dataset sizes")
    p.add_argument("--phases", type=str, default="1,2,3",
                   help="comma-separated phases to run (1=handobj, 2=door, 3=joint)")
    p.add_argument("--epochs_per_phase", type=int, default=50,
                   help="epochs for phase 1 and 2 (phase 3a=30, 3b=20)")
    train(p.parse_args())


if __name__ == "__main__":
    main()
