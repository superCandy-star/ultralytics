#!/usr/bin/env python3
"""
YOLO11 三阶段双任务训练：手物检测 + 虚拟门检测
两个独立 neck 和 head 共享 backbone，分阶段训练避免任务冲突。

阶段1 (1-50):  使用官方 YOLO 单头训练手物，映射 backbone + Neck A + Head A
阶段2 (51-100): 加载阶段1 backbone/A分支；Neck B + Head B 从原始 YOLO11n 初始化并训练
阶段3a (101-130): 冻结 backbone, 微调 necks + heads
阶段3b (131-150): 解冻全部, 小 LR 联合微调

用法:
    # 单独跑阶段1（内部使用官方 trainer）
    python tools/train_dual_task.py --phases 1 --hand_obj_data ... --epochs_per_phase 50 ...
    # 单独跑阶段2（只从阶段1 checkpoint 加载 backbone/NeckA/HeadA）
    python tools/train_dual_task.py --phases 2 --phase1_checkpoint runs/train/.../phase1_handobj_epoch50.pt ...
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

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import ModelEMA, init_seeds
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.utils import img2label_paths

MODEL_CFG = str(ROOT / "ultralytics/cfg/models/11/dual_task_handobj_door.yaml")
OFFICIAL_CFG = str(ROOT / "ultralytics/cfg/models/11/yolo11n.yaml")

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
    """YOLO detection dataset with cache-based label loading and YOLO-style
    data augmentation (Mosaic, MixUp, HSV jitter, flip, random perspective).

    First run scans + caches label info (skips corrupt images).
    Subsequent runs load the cache instantly.
    """

    def __init__(self, yaml_path: str, imgsz: int, split: str = "train",
                 hyp=None):
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
        # Image caching: npy files store imgsz-scaled images (like official
        # cache='disk'); np.load is ~20x faster than imread+resize of big images.
        # RAM cache accelerates within-process repeats (validation loop).
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.ims = [None] * len(self.labels)
        # Pre-generate npy cache (one-time cost, parallel) — otherwise Mosaic's
        # random mix_labels hit cv2.imread of big images every step.
        self._preload_npy()
        # Data augmentation (train split only)
        self.augment = (split == "train") and hyp is not None
        self.hyp = hyp
        self.stride = 32
        self.cache = "ram"  # tells Mosaic to pick random indexes directly
        self.transforms = self.build_transforms()

    def _preload_npy(self):
        """Pre-generate imgsz-scaled .npy cache for all images in parallel."""
        from multiprocessing.pool import ThreadPool

        missing = [i for i, f in enumerate(self.npy_files) if not f.exists()]
        if not missing:
            return

        def work(i):
            try:
                img = cv2.imread(self.im_files[i])  # BGR
                if img is None:
                    return
                h, w = img.shape[:2]
                r = self.imgsz / max(h, w)
                if r != 1:
                    img = cv2.resize(img, (min(round(w * r), self.imgsz), min(round(h * r), self.imgsz)),
                                     interpolation=cv2.INTER_LINEAR)
                np.save(self.npy_files[i], img, allow_pickle=False)
                self.ims[i] = img
            except Exception:
                pass  # corrupt image — skip, get_image_and_label will raise

        desc = f"Preloading npy cache {Path(self.im_files[0]).parent.parent.name if self.im_files else ''}"
        pbar = tqdm(total=len(missing), desc=desc)
        with ThreadPool(8) as pool:
            for _ in pool.imap_unordered(work, missing):
                pbar.update(1)
        pbar.close()

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

    def get_image_and_label(self, index: int) -> dict:
        """Return official-format label dict with loaded image (BGR uint8).

        Used by Mosaic/MixUp/CopyPaste augmentations. Label format matches
        ultralytics BaseDataset: {im_file, shape, cls, instances, img, ...}
        """
        from ultralytics.utils.instance import Instances

        label = copy.deepcopy(self.labels[index])
        img = self.ims[index]
        if img is None:
            if self.augment:
                # Training: npy cache, INTER_LINEAR like official augment load_image
                fn = self.npy_files[index]
                if fn.exists():
                    img = np.load(fn)
                else:
                    img = cv2.imread(label["im_file"])  # BGR
                    if img is None:
                        raise FileNotFoundError(f"Cannot read image: {label['im_file']}")
                    h, w = img.shape[:2]
                    # Resize to imgsz scale like official load_image — Mosaic layout
                    # math assumes patches are ~imgsz (padw/padh = x1a - x1b can go
                    # negative otherwise, wiping out all instances after clipping)
                    r = self.imgsz / max(h, w)
                    if r != 1:
                        img = cv2.resize(img, (min(round(w * r), self.imgsz), min(round(h * r), self.imgsz)),
                                         interpolation=cv2.INTER_LINEAR)
                    try:
                        np.save(fn, img, allow_pickle=False)  # npy disk cache
                    except Exception:
                        pass
            else:
                # Validation: match official (INTER_AREA downscale) — LINEAR
                # shifts pixels 1-2px, pushing IoU below 0.5 on hard boxes.
                img = cv2.imread(label["im_file"])  # BGR
                if img is None:
                    raise FileNotFoundError(f"Cannot read image: {label['im_file']}")
                h, w = img.shape[:2]
                r = self.imgsz / max(h, w)
                if r != 1:
                    img = cv2.resize(img, (min(round(w * r), self.imgsz), min(round(h * r), self.imgsz)),
                                     interpolation=cv2.INTER_AREA)
            self.ims[index] = img
        h, w = img.shape[:2]
        label["ori_shape"] = (h, w)
        label["img"] = img
        label["resized_shape"] = img.shape[:2]
        label["ratio_pad"] = (1.0, 1.0)
        # Instances object required by Mosaic/MixUp/Format pipeline.
        # segments must be a non-None array (denormalize indexes it directly).
        label["instances"] = Instances(
            label["bboxes"],
            segments=np.empty((0, 2), dtype=np.float32),
            keypoints=None,
            bbox_format="xywh",
            normalized=True)
        return label

    def build_transforms(self):
        """Build YOLO-style transforms: train = mosaic/HSV/flip/perspective,
        val = letterbox only. Outputs official Format dict (BGR img, xywh norm)."""
        from ultralytics.data.augment import (Compose, LetterBox, Format, v8_transforms)

        if self.augment:
            transforms = v8_transforms(self, imgsz=self.imgsz, hyp=self.hyp)
        else:
            transforms = Compose([
                LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, stride=self.stride)
            ])
        transforms.append(
            Format(bbox_format="xywh", normalize=True,
                   return_mask=False, return_keypoint=False, return_obb=False,
                   batch_idx=True, bgr=0.0)
        )
        return transforms

    def __getitem__(self, idx):
        out = self.transforms(self.get_image_and_label(idx))

        # Format outputs CHW uint8 RGB torch.Tensor (handles BGR→RGB internally)
        img_t = out["img"].to(torch.float32) / 255.0

        cls = out["cls"]
        bboxes = out["bboxes"]  # xywh normalized
        if cls.ndim == 1:
            cls = cls[:, None]
        label = np.concatenate([cls, bboxes], axis=1) if len(cls) else np.zeros((0, 5), dtype=np.float32)
        return {"img": img_t,
                "label": torch.from_numpy(label.astype(np.float32)),
                "ratio_pad": out.get("ratio_pad", (1.0, (0.0, 0.0)))}


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

    def task_forward(self, x, task_id: int):
        """Forward one task path without executing the other task's neck/head.

        In phase 2, skipping layers 11-23 is essential: requires_grad=False does
        not freeze BatchNorm running statistics, so executing frozen Neck A on
        door images would still damage the phase-1 hand/object branch.
        """
        head_idx = self._h_idx[task_id]
        skip = set(range(NECK_A_START, HEAD_A_IDX + 1)) if task_id == 1 else set()
        y = []
        for i, m in enumerate(self.model):
            if i in skip:
                y.append(None)
                continue
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if i == head_idx:
                break
        return x

    def _forward_to(self, x, head_idx: int):
        """Backward-compatible task forward selected by Detect module index."""
        return self.task_forward(x, self._h_idx.index(head_idx))

    def dual_loss(self, batch, task_id: int):
        """Forward + loss for one head."""
        img = batch["img"].to(next(self.parameters()).device)
        hi = self._h_idx[task_id]
        preds = self.task_forward(img, task_id)
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


class TaskHeadAdapter(nn.Module):
    """Expose one dual-model task as a standard Ultralytics detection model."""

    def __init__(self, model: DualHeadModel, task_id: int, names: dict):
        super().__init__()
        self.dual = model
        self.task_id = task_id
        self.names = names
        self.nc = len(names)
        self.stride = model.model[model._h_idx[task_id]].stride
        self.yaml = {"channels": 3}
        self.task = "detect"
        self.end2end = False

    def forward(self, x, augment=False, visualize=False, embed=None, **kwargs):
        return self.dual.task_forward(x, self.task_id)


def _checkpoint_state(path: str | Path) -> dict:
    """Load a state dict from either an official or dual-task checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        state = ckpt.get("ema") or ckpt.get("model") or ckpt
    return state.state_dict() if hasattr(state, "state_dict") else state


def _official_reference(nc: int, weights: str) -> DetectionModel:
    """Build an official nc-specific YOLO11n and apply shape-compatible pretrained weights."""
    ref = DetectionModel(OFFICIAL_CFG, nc=nc, verbose=False)
    raw = torch.load(weights, map_location="cpu", weights_only=False)
    source = raw.get("ema") if isinstance(raw, dict) else None
    if source is None and isinstance(raw, dict):
        source = raw.get("model")
    ref.load(source if source is not None else raw, verbose=True)
    return ref


def _map_reference_branch(ref: DetectionModel, model: DualHeadModel, task_id: int,
                          include_backbone: bool = False) -> int:
    """Map official single-head tensors into one dual neck/head branch."""
    src, dst = ref.state_dict(), model.state_dict()
    mapped = 0
    for key, value in src.items():
        if not key.startswith("model."):
            continue
        parts = key.split(".")
        idx = int(parts[1])
        if idx <= BACKBONE_END and include_backbone:
            dst_idx = idx
        elif NECK_A_START <= idx <= NECK_A_END:
            dst_idx = idx if task_id == 0 else idx + (NECK_B_START - NECK_A_START)
        elif idx == HEAD_A_IDX:
            dst_idx = HEAD_A_IDX if task_id == 0 else HEAD_B_IDX
        else:
            continue
        dst_key = ".".join(["model", str(dst_idx)] + parts[2:])
        if dst_key in dst and dst[dst_key].shape == value.shape:
            dst[dst_key] = value.to(dst[dst_key].dtype)
            mapped += 1
    model.load_state_dict(dst, strict=False)
    return mapped


def _load_phase1_branch(model: DualHeadModel, checkpoint: str) -> int:
    """Load only backbone + Neck A + Head A from a phase-1 dual checkpoint."""
    src, dst = _checkpoint_state(checkpoint), model.state_dict()
    loaded = 0
    for key, value in src.items():
        if not key.startswith("model."):
            continue
        idx = int(key.split(".")[1])
        if idx <= HEAD_A_IDX and key in dst and dst[key].shape == value.shape:
            dst[key] = value.to(dst[key].dtype)
            loaded += 1
    model.load_state_dict(dst, strict=False)
    return loaded


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


def _component_indices(component: str) -> list[int]:
    return {
        "backbone": list(range(BACKBONE_END + 1)),
        "neck_a": list(range(NECK_A_START, NECK_A_END + 1)),
        "head_a": [HEAD_A_IDX],
        "neck_b": list(range(NECK_B_START, NECK_B_END + 1)),
        "head_b": [HEAD_B_IDX],
    }.get(component, [])


def _set_phase_mode(model: DualHeadModel) -> None:
    """Train trainable modules while keeping frozen BatchNorm modules in eval mode."""
    model.train()
    for module in model.model:
        if not any(p.requires_grad for p in module.parameters()):
            module.eval()


def _create_param_groups(model, lr_map: dict, weight_decay: float = 0.0005,
                         optimizer_name: str = "AdamW") -> list[dict]:
    """Create official weight/BN/bias groups while preserving per-component learning rates."""
    norm_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    groups = []
    for component, lr in lr_map.items():
        indices = _component_indices(component)
        if not indices or lr <= 0:
            continue
        buckets = {"weight": [], "bn": [], "bias": [], "muon": []}
        for idx in indices:
            prefix = f"model.{idx}"
            for module_name, module in model.model[idx].named_modules():
                full_module = f"{prefix}.{module_name}" if module_name else prefix
                for param_name, param in module.named_parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    fullname = f"{full_module}.{param_name}"
                    if optimizer_name == "MuSGD" and param.ndim >= 2:
                        buckets["muon"].append(param)
                    elif "bias" in fullname:
                        buckets["bias"].append(param)
                    elif isinstance(module, norm_types):
                        buckets["bn"].append(param)
                    else:
                        buckets["weight"].append(param)
        for kind, params in buckets.items():
            if not params:
                continue
            decay = weight_decay if kind in {"weight", "muon"} else 0.0
            groups.append({"params": params, "lr": lr, "initial_lr": lr,
                           "weight_decay": decay, "param_group": kind,
                           "component": component, "use_muon": kind == "muon"})
    return groups


def _init_strides(model, imgsz: int, device: torch.device):
    """Initialize both Detect strides without changing any BatchNorm statistics."""
    was_training = model.training
    model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    for task_id, hi in enumerate(model._h_idx):
        head = model.model[hi]
        head.training = True  # Detect returns raw dict; all BN modules remain eval.
        with torch.no_grad():
            raw = model.task_forward(dummy, task_id)
        head.training = False
        if isinstance(raw, dict) and "feats" in raw:
            head.stride = torch.tensor([imgsz / feat.shape[-1] for feat in raw["feats"]], device=device)
    model.train(was_training)


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
def _run_phase(model, loader_train, validate: bool, phase_name: str,
               epochs: int, lr_map: dict, start_epoch: int,
               args, device: torch.device, warmup_epochs: int = 3,
               phase_start_epoch: int | None = None, total_phase_epochs: int | None = None,
               resume_checkpoint: str = "", restore_optimizer: bool = True):
    """Train or resume one dual-task phase with finite-state safeguards."""
    phase_start_epoch = phase_start_epoch or start_epoch
    total_phase_epochs = total_phase_epochs or epochs
    # A new optimizer cannot clear gradients owned by the previous phase.
    # Clear the whole model before changing requires_grad/parameter groups.
    model.zero_grad(set_to_none=True)
    _freeze_all(model)
    for component, lr in lr_map.items():
        if lr > 0:
            _unfreeze_module_list(model, _component_indices(component))

    if isinstance(loader_train, (list, tuple)):
        steps_per_epoch = min(len(loader_train[0]), len(loader_train[1])) * 2
    else:
        steps_per_epoch = len(loader_train)
    iterations = steps_per_epoch * total_phase_epochs
    optimizer_name = "MuSGD" if iterations > 10000 else "AdamW"
    accumulate = max(round(64 / args.batch), 1)
    scaled_decay = args.weight_decay * args.batch * accumulate / 64
    param_groups = _create_param_groups(model, lr_map, scaled_decay, optimizer_name)
    if not param_groups:
        LOGGER.warning("No trainable parameters!")
        return

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(param_groups, lr=1.0, betas=(0.9, 0.999), weight_decay=0.0)
    else:
        from ultralytics.optim import MuSGD
        optimizer = MuSGD(param_groups, lr=1.0, momentum=0.9, nesterov=True, muon=0.2, sgd=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = ModelEMA(model)
    resume_raw = None
    if resume_checkpoint:
        resume_raw = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if restore_optimizer and resume_raw.get("optimizer"):
            optimizer.load_state_dict(resume_raw["optimizer"])
        if resume_raw.get("scaler"):
            scaler.load_state_dict(resume_raw["scaler"])
        if resume_raw.get("ema"):
            ema.ema.load_state_dict(resume_raw["ema"], strict=False)
        restored = "optimizer/scaler/EMA" if restore_optimizer else "scaler/EMA"
        LOGGER.info(f"[{phase_name}] Restored {restored} from {resume_checkpoint}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"[{phase_name}] {optimizer_name}, trainable: {trainable:,}/{total_params:,}, "
                f"weight_decay={scaled_decay:.6g}, groups: "
                f"{[(g['component'], g['param_group'], g['initial_lr'], len(g['params'])) for g in param_groups]}")

    def _infinite_loader(loader):
        while True:
            yield from loader

    if isinstance(loader_train, (list, tuple)):
        iter_a = _infinite_loader(loader_train[0])
        iter_b = _infinite_loader(loader_train[1])
        joint_mode = True
    else:
        iter_a = _infinite_loader(loader_train)
        iter_b = None
        joint_mode = False

    warmup_iters = max(round(warmup_epochs * steps_per_epoch), 100) if warmup_epochs else 0
    lrf = getattr(args, "lrf", 0.01)
    best_fitness = -float("inf")
    phase_dir = Path(args.project) / args.name
    phase_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, start_epoch + epochs):
        phase_epoch = epoch - phase_start_epoch
        if total_phase_epochs > 10 and phase_epoch == total_phase_epochs - 10:
            loaders = loader_train if isinstance(loader_train, (list, tuple)) else (loader_train,)
            for loader in loaders:
                if hasattr(loader.dataset, "close_mosaic"):
                    LOGGER.info(f"[{phase_name}] Closing dataloader mosaic")
                    loader.dataset.close_mosaic(hyp=loader.dataset._dual_hyp)
        _set_phase_mode(model)
        lr_factor = max(1 - phase_epoch / total_phase_epochs, 0) * (1.0 - lrf) + lrf
        total_steps = steps_per_epoch
        if args.test_iters:
            total_steps = min(total_steps, args.test_iters)
        losses = []
        t_epoch = time.time()

        pbar = tqdm(total=total_steps, desc=f"{phase_name} Epoch {epoch}/{start_epoch+epochs-1}")
        for step in range(total_steps):
            ni = phase_epoch * steps_per_epoch + step
            for pg in optimizer.param_groups:
                target_lr = pg["initial_lr"] * lr_factor
                pg["lr"] = target_lr if not warmup_iters or ni > warmup_iters else target_lr * ni / warmup_iters

            if joint_mode:
                is_det = step % 2 == 0
                batch = next(iter_a) if is_det else next(iter_b)
                batch["_task"] = 0 if is_det else 1
            else:
                batch = next(iter_a)
                batch["_task"] = 0 if "hand" in phase_name else 1
            total_l, box_l, cls_l, dfl_l, n = _step(
                model, batch, 1.0, optimizer, device, scaler=scaler, ema=ema)
            losses.append(total_l)
            task_name = "det" if batch["_task"] == 0 else "door"
            pbar.set_postfix({"task": task_name, "box": f"{box_l:.2f}", "cls": f"{cls_l:.2f}",
                              "dfl": f"{dfl_l:.2f}", "n": n})
            pbar.update(1)
        pbar.close()

        avg_loss = float(np.mean(losses)) if losses else 0
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  {phase_name} epoch {epoch}  loss={avg_loss:.2f}  lr={current_lr:.2e}  "
              f"({time.time()-t_epoch:.0f}s)", flush=True)

        fitness = -float("inf")
        if validate:
            t_val = time.time()
            val_results = _validate(ema.ema, args, device)
            det_m = val_results.get("det", {}).get("mAP50", 0)
            door_m = val_results.get("door", {}).get("mAP50", 0)
            fitness = (det_m + door_m) / 2 if "joint" in phase_name or "fullft" in phase_name else door_m
            print(f"  val mAP@0.5: det={det_m:.4f} door={door_m:.4f}  "
                  f"({time.time()-t_val:.0f}s)", flush=True)

        checkpoint = {"epoch": epoch, "phase": phase_name, "model": model.state_dict(),
                      "ema": ema.ema.state_dict(), "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(), "args": vars(args)}
        ckpt_path = phase_dir / f"{phase_name}_epoch{epoch}.pt"
        torch.save(checkpoint, ckpt_path)
        torch.save(checkpoint, phase_dir / "last.pt")
        if fitness > best_fitness:
            best_fitness = fitness
            torch.save(checkpoint, phase_dir / "best.pt")

    return str(phase_dir / "last.pt")


def _official_phase1(args) -> Path:
    """Train hand/object detection through the unmodified official YOLO trainer."""
    LOGGER.info("Phase 1 uses the official YOLO trainer; its result is the mandatory single-head baseline.")
    official = YOLO(OFFICIAL_CFG)
    official.train(
        data=args.hand_obj_data,
        epochs=args.epochs_per_phase,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.num_workers,
        pretrained=args.weights,
        optimizer="auto",
        project=args.project,
        name=f"{args.name}_phase1_official",
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        val=True,
        plots=False,
    )
    best = Path(official.trainer.best)
    if not best.exists():
        raise FileNotFoundError(f"Official phase-1 best checkpoint not found: {best}")
    return best


def _build_official_loader(data_yaml: str, args, mode: str):
    """Build the same dataset/dataloader used by DetectionTrainer."""
    data = check_det_dataset(data_yaml)
    cfg = get_cfg(overrides={
        "task": "detect", "mode": "train", "data": data_yaml,
        "imgsz": args.imgsz, "batch": args.batch, "workers": args.num_workers,
        "cache": False, "rect": mode == "val", "fraction": 1.0,
        "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5,
        "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "bgr": 0.0, "close_mosaic": 10, "seed": args.seed,
    })
    dataset = build_yolo_dataset(cfg, data[mode], args.batch, data, mode=mode,
                                 rect=mode == "val", stride=32)
    dataset._dual_hyp = cfg  # official trainer passes its args back to close_mosaic()
    return build_dataloader(dataset, batch=args.batch,
                            workers=args.num_workers if mode == "train" else args.num_workers * 2,
                            shuffle=mode == "train", rank=-1, drop_last=False)


def _initialize_dual_model(args, device: torch.device) -> DualHeadModel:
    """Initialize both branches from official nc-specific YOLO11n references."""
    model = DualHeadModel(cfg=MODEL_CFG, nc=[2, 4])
    model.args = get_cfg(overrides={"box": 7.5, "cls": 0.5, "dfl": 1.5})
    ref_a = _official_reference(2, args.weights)
    ref_b = _official_reference(4, args.weights)
    mapped_a = _map_reference_branch(ref_a, model, task_id=0, include_backbone=True)
    mapped_b = _map_reference_branch(ref_b, model, task_id=1, include_backbone=False)
    LOGGER.info(f"Dual init from {args.weights}: task A {mapped_a} tensors, task B {mapped_b} tensors")
    model.to(device)
    _init_strides(model, args.imgsz, device)
    return model


def train(args):
    init_seeds(args.seed, deterministic=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Device: {device}")
    phases_to_run = [int(x.strip()) for x in args.phases.split(",")]
    model = _initialize_dual_model(args, device)

    # Phase 1 is intentionally not reimplemented here. It is trained by the
    # official single-head pipeline and then mapped losslessly to task A.
    if 1 in phases_to_run:
        phase1_best = _official_phase1(args)
        ref_phase1 = DetectionModel(OFFICIAL_CFG, nc=2, verbose=False)
        phase1_raw = torch.load(phase1_best, map_location="cpu", weights_only=False)
        phase1_source = phase1_raw.get("ema") or phase1_raw.get("model")
        ref_phase1.load(phase1_source, verbose=True)
        mapped = _map_reference_branch(ref_phase1, model, task_id=0, include_backbone=True)
        LOGGER.info(f"Mapped official phase-1 best into task A: {mapped} tensors")
    elif args.phase1_checkpoint:
        loaded = _load_phase1_branch(model, args.phase1_checkpoint)
        LOGGER.info(f"Loaded phase-1 backbone/NeckA/HeadA: {loaded} tensors from {args.phase1_checkpoint}")
        # NeckB/HeadB deliberately remain initialized from the original
        # yolo11n.pt reference, never from the stale random branch in phase1 ckpt.
    elif 2 in phases_to_run and not args.dual_checkpoint:
        raise ValueError("Phase 2 requires phase 1, --phase1_checkpoint, or --dual_checkpoint")

    resume_epoch = None
    resume_phase = None
    if args.dual_checkpoint:
        dual_raw = torch.load(args.dual_checkpoint, map_location="cpu", weights_only=False)
        state = dual_raw.get("model", dual_raw) if isinstance(dual_raw, dict) else dual_raw
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        model.load_state_dict(state, strict=False)
        resume_epoch = dual_raw.get("epoch") if isinstance(dual_raw, dict) else None
        resume_phase = dual_raw.get("phase") if isinstance(dual_raw, dict) else None
        LOGGER.info(f"Loaded full dual checkpoint: {args.dual_checkpoint} "
                    f"(phase={resume_phase}, epoch={resume_epoch})")

    _init_strides(model, args.imgsz, device)
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    phase1_ckpt = save_dir / f"phase1_handobj_epoch{args.epochs_per_phase}.pt"
    if 1 in phases_to_run:
        torch.save({"epoch": args.epochs_per_phase, "phase": "phase1_handobj",
                    "model": model.state_dict(), "args": vars(args),
                    "official_phase1": str(phase1_best)}, phase1_ckpt)
        LOGGER.info(f"Saved mapped dual phase-1 checkpoint: {phase1_ckpt}")
        phase1_metrics = _validate(model, args, device)
        LOGGER.info(f"Mapped phase-1 validation: {phase1_metrics['det']}")

    det_loader = door_loader = None
    if 3 in phases_to_run:
        det_loader = _build_official_loader(args.hand_obj_data, args, "train")
    if 2 in phases_to_run or 3 in phases_to_run:
        door_loader = _build_official_loader(args.virtual_door_data, args, "train")

    start_epoch = args.epochs_per_phase + 1
    phase2_last = ""
    if 2 in phases_to_run:
        phase2_end = args.epochs_per_phase * 2
        phase2_epochs = args.epochs_per_phase
        phase2_resume = ""
        if resume_phase == "phase2_door" and resume_epoch is not None:
            start_epoch = resume_epoch + 1
            phase2_epochs = max(phase2_end - resume_epoch, 0)
            phase2_resume = args.dual_checkpoint
            LOGGER.info(f"Resuming phase2_door at epoch {start_epoch}; {phase2_epochs} epochs remain")
        if phase2_epochs:
            phase2_last = _run_phase(
                model, door_loader, True, "phase2_door", phase2_epochs,
                {"backbone": 5e-4, "neck_b": 1e-3, "head_b": 5e-3},
                start_epoch, args, device, warmup_epochs=3,
                phase_start_epoch=args.epochs_per_phase + 1,
                total_phase_epochs=args.epochs_per_phase,
                resume_checkpoint=phase2_resume)
        else:
            phase2_last = phase2_resume
        start_epoch = phase2_end + 1
    if 3 in phases_to_run:
        phase3a_start = args.epochs_per_phase * 2 + 1
        phase3a_epochs = 30
        phase3a_state = ""
        phase3a_restore_optimizer = False
        if resume_phase == "phase2_door" and resume_epoch is not None:
            start_epoch = phase3a_start
            phase3a_state = args.dual_checkpoint
            LOGGER.info(f"Starting phase3a_joint from phase2 checkpoint at epoch {start_epoch}")
        elif resume_phase == "phase3a_joint" and resume_epoch is not None:
            start_epoch = resume_epoch + 1
            phase3a_epochs = max(phase3a_start + 30 - start_epoch, 0)
            phase3a_state = args.dual_checkpoint
            phase3a_restore_optimizer = True
            LOGGER.info(f"Resuming phase3a_joint at epoch {start_epoch}; {phase3a_epochs} epochs remain")
        if not phase3a_state and phase2_last:
            phase3a_state = phase2_last
        phase3a_last = phase3a_state
        if phase3a_epochs:
            phase3a_last = _run_phase(
                model, (det_loader, door_loader), True, "phase3a_joint", phase3a_epochs,
                {"neck_a": 5e-4, "neck_b": 5e-4, "head_a": 1e-3, "head_b": 1e-3},
                start_epoch, args, device, warmup_epochs=3,
                phase_start_epoch=phase3a_start, total_phase_epochs=30,
                resume_checkpoint=phase3a_state, restore_optimizer=phase3a_restore_optimizer)
        start_epoch = phase3a_start + 30
        _run_phase(
            model, (det_loader, door_loader), True, "phase3b_fullft", 20,
            {"backbone": 1e-5, "neck_a": 1e-4, "neck_b": 1e-4,
             "head_a": 5e-4, "head_b": 5e-4},
            start_epoch, args, device, warmup_epochs=3,
            phase_start_epoch=start_epoch, total_phase_epochs=20,
            resume_checkpoint=phase3a_last, restore_optimizer=False)

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

                # Vectorized greedy matching (official match_predictions order):
                # sort by IoU desc → each pred once → re-sort → each GT once
                for t in range(niou):
                    thr = iou_thresholds[t].item()
                    x = torch.nonzero(iou >= thr)  # [n_matches, 2] (gt_idx, pred_idx)
                    if x.shape[0] == 0:
                        continue
                    x = torch.cat((x, iou[x[:, 0], x[:, 1]][:, None]), dim=1)
                    x = x[x[:, 2].argsort(descending=True)]  # sort by IoU desc
                    x = x.cpu().numpy()
                    x = x[np.unique(x[:, 1], return_index=True)[1]]  # each pred once
                    x = x[x[:, 2].argsort()[::-1]]                    # re-sort by IoU desc
                    x = x[np.unique(x[:, 0], return_index=True)[1]]  # each GT once
                    tp_per_pred[torch.from_numpy(x[:, 1]).long().to(device), t] = 1.0

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

    Uses official eval mode: Detect._inference output (xywh + sigmoid scores)
    fed to non_max_suppression — identical to the official validator so BN
    running stats match training (train-mode batch stats would shift
    predictions and crash mAP on out-of-distribution images).
    """
    from ultralytics.utils.nms import non_max_suppression

    hi = model._h_idx[task_id]
    h = model.model[hi]
    head_nc = [2, 4][task_id]

    # Warmup forward in train mode to init stride (dict output)
    model.train()
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        raw = model._forward_to(dummy, hi)
    if isinstance(raw, dict) and h.stride.sum() == 0 and "feats" in raw:
        feats = [f for f in raw["feats"]]
        h.stride = torch.tensor([imgsz / f.shape[-1] for f in feats], device=device)

    iou_thresholds = torch.linspace(0.5, 0.95, 10, device=device)
    n_samples = min(len(val_ds), max_imgs)
    n_batches = (n_samples + batch - 1) // batch

    image_stats = []
    n_gt_per_class = {}

    model.eval()
    pbar = tqdm(total=n_batches, desc=f"  val task={task_id}")
    for start in range(0, n_samples, batch):
        end = min(start + batch, n_samples)
        indices = list(range(start, end))
        batch_data = _load_batch(val_ds, indices, task_id, device)

        with torch.no_grad():
            out = model._forward_to(batch_data["img"], hi)
        if isinstance(out, tuple):
            pred_all = out[0]  # official _inference: [B, 4+nc, anchors] xywh + sigmoid
        else:
            pred_all = out
        if not isinstance(pred_all, torch.Tensor):
            continue

        # Whole-batch NMS (official format: xywh boxes, scores sigmoided)
        dets_list = non_max_suppression(pred_all, conf_thres=0.001, iou_thres=0.65,
                                        nc=head_nc, max_det=300)

        for bi in range(len(batch_data["img"])):
            pred = dets_list[bi]
            # Scale detections from letterboxed space back to original image
            # coords (official validator does this too). ratio_pad from
            # LetterBox: (ratio, (padw, padh)); orig = (lb - pad) / ratio.
            ratio_pad = batch_data["ratio_pad"][bi]
            gain = ratio_pad[0] if isinstance(ratio_pad, tuple) else (1.0, 1.0)
            padw, padh = ratio_pad[1] if isinstance(ratio_pad, tuple) else (0.0, 0.0)
            if isinstance(gain, (tuple, list)):
                gain_x, gain_y = gain
            else:
                gain_x = gain_y = gain
            if pred is None or len(pred) == 0:
                pred_boxes = torch.zeros((0, 4), device=device)
                pred_confs = torch.zeros(0, device=device)
                pred_cls = torch.zeros(0, dtype=torch.long, device=device)
            else:
                pb = pred[:, :4].to(device).float()
                pb[:, [0, 2]] = (pb[:, [0, 2]] - padw) / gain_x
                pb[:, [1, 3]] = (pb[:, [1, 3]] - padh) / gain_y
                pred_boxes = pb
                pred_confs = pred[:, 4]
                pred_cls = pred[:, 5].long()

            # --- ground truth (letterboxed normalized → original pixels) ---
            gt_labels = batch_data["label"][bi]  # [N, 5] normalized [cls, cx, cy, w, h]
            if gt_labels.numel():
                gt_cls_img = gt_labels[:, 0].long()
                gt_cx, gt_cy = gt_labels[:, 1], gt_labels[:, 2]
                gt_w, gt_h = gt_labels[:, 3], gt_labels[:, 4]
                gt_x1 = (gt_cx * imgsz - padw) / gain_x - (gt_w * imgsz) / gain_x / 2
                gt_y1 = (gt_cy * imgsz - padh) / gain_y - (gt_h * imgsz) / gain_y / 2
                gt_x2 = (gt_cx * imgsz - padw) / gain_x + (gt_w * imgsz) / gain_x / 2
                gt_y2 = (gt_cy * imgsz - padh) / gain_y + (gt_h * imgsz) / gain_y / 2
                gt_boxes = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1).to(device)
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


def _validate(model: DualHeadModel, args, device: torch.device, task_ids=(0, 1)):
    """Validate selected task heads with the unmodified official DetectionValidator."""
    results = {}
    requires_grad = {name: param.requires_grad for name, param in model.named_parameters()}
    tasks = [
        ("det", args.hand_obj_data, 0),
        ("door", args.virtual_door_data, 1),
    ]
    was_training = model.training
    for task_name, data_yaml, task_id in tasks:
        if task_id not in task_ids:
            continue
        data = check_det_dataset(data_yaml)
        names = data["names"]
        if isinstance(names, list):
            names = dict(enumerate(names))
        adapter = TaskHeadAdapter(model, task_id, names)
        validator = DetectionValidator(args={
            "data": data_yaml,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": str(device),
            "workers": args.num_workers,
            "plots": False,
            "verbose": True,
            "conf": None,
            "iou": 0.7,
            "max_det": 300,
            "split": "val",
            "save_json": False,
            "save_txt": False,
            "half": False,
            "augment": False,
            "rect": True,
        })
        t0 = time.time()
        stats = validator(model=adapter)
        results[task_name] = {
            "precision": float(stats["metrics/precision(B)"]),
            "recall": float(stats["metrics/recall(B)"]),
            "mAP50": float(stats["metrics/mAP50(B)"]),
            "mAP50-95": float(stats["metrics/mAP50-95(B)"]),
        }
        print(f"  {task_name} official val: P={results[task_name]['precision']:.4f} "
              f"R={results[task_name]['recall']:.4f} "
              f"mAP@0.5={results[task_name]['mAP50']:.4f} "
              f"mAP@0.5:0.95={results[task_name]['mAP50-95']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
    model.train(was_training)
    return results


def _load_batch(ds, indices, task_id, device):
    imgs, labels, ratio_pads = [], [], []
    for i in indices:
        item = ds[i]
        imgs.append(item["img"])
        labels.append(item["label"])
        ratio_pads.append(item.get("ratio_pad", (1.0, (0.0, 0.0))))
    imgs = torch.stack(imgs, 0).to(device)
    batch = {"img": imgs, "label": labels, "ratio_pad": ratio_pads, "_task": task_id}

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


def _step(model, batch, weight, optimizer, device, scaler=None, ema=None):
    batch["img"] = batch["img"].to(device, non_blocking=True)
    if batch["img"].dtype == torch.uint8:
        batch["img"] = batch["img"].float() / 255.0
    else:
        batch["img"] = batch["img"].float()
    batch["batch_idx"] = batch["batch_idx"].to(device, non_blocking=True)
    batch["cls"] = batch["cls"].to(device, non_blocking=True)
    batch["bboxes"] = batch["bboxes"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    optimizer_params = [p for group in optimizer.param_groups for p in group["params"]]
    amp_enabled = scaler is not None and scaler.is_enabled()
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        result = model.dual_loss(batch, batch["_task"])
        if isinstance(result, (list, tuple)):
            loss = result[0]
            loss_items = result[1] if len(result) > 1 else None
        else:
            loss = result
            loss_items = None
        loss_sum = loss.sum() if hasattr(loss, "shape") and loss.numel() > 1 else loss
        loss_sum = weight * loss_sum

    if not torch.isfinite(loss_sum):
        raise FloatingPointError(f"Non-finite loss before backward: {loss_sum.detach().item()}")

    stepped = True
    if scaler is not None:
        scale_before = scaler.get_scale()
        scaler.scale(loss_sum).backward()
        scaler.unscale_(optimizer)
        grads_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in optimizer_params)
        if grads_finite:
            grad_norm = torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=10.0, error_if_nonfinite=True)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm: {grad_norm.item()}")
        scaler.step(optimizer)  # skips internally when unscale_ found inf/nan
        scaler.update()
        stepped = grads_finite and scaler.get_scale() >= scale_before
    else:
        loss_sum.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=10.0, error_if_nonfinite=True)
        optimizer.step()

    if stepped:
        bad_params = [name for name, p in model.named_parameters() if not torch.isfinite(p).all()]
        if bad_params:
            raise FloatingPointError(f"Non-finite parameters after optimizer step: {bad_params[:5]}")
        if ema is not None:
            ema.update(model)

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
    weighted_total = loss_sum.item() / batch_size
    return weighted_total, box_l, cls_l, dfl_l, n_instances


# ======================================================================
#  4.  Entry
# ======================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand_obj_data", required=True)
    p.add_argument("--virtual_door_data", required=True)
    p.add_argument("--weights", default=str(ROOT / "weights/yolo11n.pt"),
                   help="original YOLO11n pretrained weights for both task branches")
    p.add_argument("--phase1_checkpoint", default="",
                   help="mapped dual phase-1 checkpoint when starting from phase 2")
    p.add_argument("--dual_checkpoint", default="",
                   help="full dual checkpoint when resuming phase 3")
    p.add_argument("--det_loss_weight", type=float, default=1.0)
    p.add_argument("--door_loss_weight", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--seed", type=int, default=0)
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
