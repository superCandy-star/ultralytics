# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from ultralytics.utils import LOGGER


PHASE_BEFORE_APPROACH = "before_approach"
PHASE_HOLDING = "holding"
PHASE_MOVING = "moving"
PHASE_PUT_DOWN = "put_down"
PHASE_GLOBAL = "global"


@dataclass
class ObjClassificationSample:
    """One object-track sample used for track-level classification."""

    frame_id: int
    xyxy: np.ndarray
    xywh: np.ndarray
    score: float
    area: float
    center: np.ndarray
    anchor_enabled: bool
    is_synthetic: bool
    crop_image: Image.Image
    image_area: float
    speed: float = 0.0
    smooth_speed: float = 0.0
    phase: str = PHASE_GLOBAL
    quality: float = 0.0


class ObjTrackClassifier:
    """Track-level object classifier using obj-centered key frames and MobileNetV3."""

    PHASE_WEIGHTS = {
        PHASE_BEFORE_APPROACH: 1.0,
        PHASE_HOLDING: 0.8,
        PHASE_MOVING: 0.7,
        PHASE_PUT_DOWN: 1.0,
        PHASE_GLOBAL: 1.0,
    }

    def __init__(self, args: Any):
        self.enabled = self._as_bool(getattr(args, "obj_classification_enabled", False))
        self.model_path = getattr(
            args,
            "obj_classification_model",
            "/root/taojianwei/projects/mobilenetv3.pytorch-master/runs/obj_cls/mobilenetv3_small_224/best.pt",
        )
        self.repo_path = Path(
            getattr(args, "obj_classification_repo", "/root/taojianwei/projects/mobilenetv3.pytorch-master")
        )
        self.device_arg = getattr(args, "obj_classification_device", None)
        self.target_min_dim = float(getattr(args, "obj_classification_target_min_dim", 160.0))
        self.context_scale = float(getattr(args, "obj_classification_context_scale", 2.0))
        self.max_context_ratio = float(getattr(args, "obj_classification_max_context_ratio", 4.0))
        self.min_track_len = max(1, int(getattr(args, "obj_classification_min_track_len", 8)))
        self.max_samples = max(1, int(getattr(args, "obj_classification_max_samples", 120)))
        self.max_frames_per_phase = max(1, int(getattr(args, "obj_classification_max_frames_per_phase", 2)))
        self.min_frame_gap = max(0, int(getattr(args, "obj_classification_min_frame_gap", 5)))
        self.move_speed_thresh = float(getattr(args, "obj_classification_move_speed_thresh", 10.0))
        self.stable_speed_thresh = float(getattr(args, "obj_classification_stable_speed_thresh", 4.0))
        self.move_confirm_frames = max(1, int(getattr(args, "obj_classification_move_confirm_frames", 2)))
        self.stable_confirm_frames = max(1, int(getattr(args, "obj_classification_stable_confirm_frames", 3)))
        self.area_full_ratio = max(1e-9, float(getattr(args, "obj_classification_area_full_ratio", 0.01)))
        self.speed_norm = max(1e-9, float(getattr(args, "obj_classification_speed_norm", 80.0)))
        self.bbox_iou_diversity_thresh = float(getattr(args, "obj_classification_bbox_iou_diversity_thresh", 0.85))

        self.model = None
        self.input_size = int(getattr(args, "obj_classification_input_size", 224))
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}
        self.device = None
        self._torch = None
        self._load_failed = False

        if self.enabled:
            self._load_model()

    def add_sample(self, track: Any, frame_id: int, img: np.ndarray | None) -> None:
        """Append a real-detection object sample to a track's classification sample buffer."""
        if not self.enabled or self.model is None or img is None:
            return
        if int(getattr(track, "idx", -1)) < 0:
            return

        xyxy = np.asarray(track.xyxy, dtype=np.float32).copy()
        xywh = np.asarray(track.xywh, dtype=np.float32).copy()
        x1, y1, x2, y2 = xyxy.tolist()
        if x2 <= x1 or y2 <= y1:
            return
        crop = self._crop_track_image(img, xyxy)
        area = float((x2 - x1) * (y2 - y1))
        sample = ObjClassificationSample(
            frame_id=int(frame_id),
            xyxy=xyxy,
            xywh=xywh,
            score=float(getattr(track, "score", 0.0)),
            area=area,
            center=xywh[:2].copy(),
            anchor_enabled=bool(getattr(track, "anchor_enabled", False)),
            is_synthetic=False,
            crop_image=crop,
            image_area=float(img.shape[0] * img.shape[1]),
        )
        samples = getattr(track, "classification_samples", None)
        if samples is None:
            track.classification_samples = []
            samples = track.classification_samples
        if samples and samples[-1].frame_id == frame_id:
            samples[-1] = sample
        else:
            samples.append(sample)
        if len(samples) > self.max_samples:
            del samples[: len(samples) - self.max_samples]

    def classify_track(self, track: Any) -> dict[str, Any] | None:
        """Classify a track from selected key-frame crops and update track attributes."""
        if not self.enabled or self.model is None:
            return None
        samples = [s for s in getattr(track, "classification_samples", []) if not s.is_synthetic]
        if not samples:
            return None

        selected = self.select_keyframes(samples)
        if not selected:
            return None

        votes: dict[str, float] = {}
        frame_results: list[dict[str, Any]] = []
        for sample in selected:
            pred = self.predict_crop(sample.crop_image)
            if pred is None:
                continue
            label, conf, scores = pred
            weight = float(sample.quality) * self.PHASE_WEIGHTS.get(sample.phase, 1.0)
            votes[label] = votes.get(label, 0.0) + conf * max(weight, 1e-6)
            frame_results.append(
                {
                    "frame_id": sample.frame_id,
                    "phase": sample.phase,
                    "quality": float(sample.quality),
                    "label": label,
                    "conf": float(conf),
                }
            )

        if not votes:
            return None
        total = sum(votes.values()) or 1.0
        normalized_votes = {k: float(v / total) for k, v in sorted(votes.items())}
        label, score = max(normalized_votes.items(), key=lambda item: item[1])
        result = {
            "label": label,
            "conf": float(score),
            "scores": normalized_votes,
            "frame_id": int(frame_results[0]["frame_id"]),
            "detail": {"frames": frame_results},
        }
        track.classification_label = result["label"]
        track.classification_conf = result["conf"]
        track.classification_scores = result["scores"]
        track.classification_frame_id = result["frame_id"]
        track.classification_detail = result["detail"]
        return result

    def select_keyframes(self, samples: list[ObjClassificationSample]) -> list[ObjClassificationSample]:
        """Select diverse high-quality key frames from trajectory phases."""
        if not samples:
            return []
        samples = sorted(samples, key=lambda sample: sample.frame_id)
        self._annotate_motion(samples)
        self._assign_phases(samples)
        self._score_samples(samples)

        if len(samples) < self.min_track_len:
            return self._select_diverse(samples, self.max_frames_per_phase * 2)

        selected: list[ObjClassificationSample] = []
        for phase in (PHASE_BEFORE_APPROACH, PHASE_HOLDING, PHASE_MOVING, PHASE_PUT_DOWN):
            phase_samples = [sample for sample in samples if sample.phase == phase]
            selected.extend(self._select_diverse(phase_samples, self.max_frames_per_phase))
        if not selected:
            selected = self._select_diverse(samples, self.max_frames_per_phase * 2)
        return sorted(selected, key=lambda sample: sample.frame_id)

    def predict_crop(self, crop: Image.Image) -> tuple[str, float, dict[str, float]] | None:
        """Run classifier inference on one crop image."""
        if self.model is None or self._torch is None:
            return None
        torch = self._torch
        arr = np.asarray(crop.resize((self.input_size, self.input_size), Image.BILINEAR), dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        label = self.idx_to_class.get(pred_idx, str(pred_idx))
        scores = {self.idx_to_class.get(i, str(i)): float(p) for i, p in enumerate(probs)}
        return label, float(probs[pred_idx]), scores

    def classification_payload(self, track: Any) -> dict[str, Any]:
        """Return a JSON-friendly classification payload for an event."""
        return {
            "label": getattr(track, "classification_label", None),
            "conf": getattr(track, "classification_conf", None),
            "scores": getattr(track, "classification_scores", None),
            "frame_id": getattr(track, "classification_frame_id", None),
            "detail": getattr(track, "classification_detail", None),
        }

    def _load_model(self) -> None:
        """Load MobileNetV3 checkpoint with fail-open behavior."""
        try:
            import torch
        except Exception as exc:
            LOGGER.warning(f"Object classification disabled: failed to import torch: {exc}")
            self.enabled = False
            self._load_failed = True
            return

        ckpt_path = Path(self.model_path)
        if not ckpt_path.exists():
            LOGGER.warning(f"Object classification disabled: model not found: {ckpt_path}")
            self.enabled = False
            self._load_failed = True
            return
        mobilenet_path = self.repo_path / "mobilenetv3.py"
        if not mobilenet_path.exists():
            LOGGER.warning(f"Object classification disabled: mobilenetv3.py not found: {mobilenet_path}")
            self.enabled = False
            self._load_failed = True
            return

        try:
            module_name = "mobilenetv3_obj_classifier_dynamic"
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, mobilenet_path)
                module = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            self.input_size = int(ckpt.get("input_size", self.input_size))
            self.class_to_idx = {str(k): int(v) for k, v in ckpt.get("class_to_idx", {}).items()}
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            num_classes = int(ckpt.get("num_classes", len(self.class_to_idx)))
            model = module.mobilenetv3_small(num_classes=num_classes)
            model.load_state_dict(ckpt["model"])
            if self.device_arg is None:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(str(self.device_arg))
            model.to(self.device).eval()
            self.model = model
            self._torch = torch
        except Exception as exc:
            LOGGER.warning(f"Object classification disabled: failed to load checkpoint {ckpt_path}: {exc}")
            self.enabled = False
            self.model = None
            self._load_failed = True

    def _crop_track_image(self, img: np.ndarray, xyxy: np.ndarray) -> Image.Image:
        """Build an obj-centered adaptive crop from the current BGR frame."""
        crop_box = self._build_obj_centered_crop_box(xyxy)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        crop = self._crop_with_padding(image, crop_box)
        return crop.resize((self.input_size, self.input_size), Image.BILINEAR)

    def _build_obj_centered_crop_box(self, xyxy: np.ndarray) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        obj_w, obj_h = x2 - x1, y2 - y1
        obj_long_side = max(obj_w, obj_h)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        crop_size = max(obj_long_side * self.context_scale, self.target_min_dim)
        max_crop_size = max(obj_long_side * self.max_context_ratio, self.target_min_dim)
        crop_size = min(crop_size, max_crop_size)
        half = crop_size / 2.0
        return cx - half, cy - half, cx + half, cy + half

    @staticmethod
    def _crop_with_padding(image: Image.Image, crop_box: tuple[float, float, float, float], fill=(114, 114, 114)):
        x1, y1, x2, y2 = crop_box
        left, top = int(round(x1)), int(round(y1))
        right, bottom = int(round(x2)), int(round(y2))
        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - image.width)
        pad_bottom = max(0, bottom - image.height)
        clipped = image.crop((max(0, left), max(0, top), min(image.width, right), min(image.height, bottom)))
        if any((pad_left, pad_top, pad_right, pad_bottom)):
            clipped = ImageOps.expand(clipped, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)
        return clipped

    def _annotate_motion(self, samples: list[ObjClassificationSample]) -> None:
        speeds = [0.0]
        for prev, cur in zip(samples, samples[1:]):
            speeds.append(float(np.linalg.norm(cur.center - prev.center)))
        for sample, speed in zip(samples, speeds):
            sample.speed = speed
        radius = 1
        for i, sample in enumerate(samples):
            lo, hi = max(0, i - radius), min(len(samples), i + radius + 1)
            sample.smooth_speed = float(np.mean(speeds[lo:hi]))

    def _assign_phases(self, samples: list[ObjClassificationSample]) -> None:
        n = len(samples)
        holding_start = next((i for i, sample in enumerate(samples) if sample.anchor_enabled), None)
        search_start = holding_start if holding_start is not None else 0
        moving_start = self._first_confirmed(
            [sample.smooth_speed > self.move_speed_thresh for sample in samples], search_start, self.move_confirm_frames
        )
        moving_end = None
        if moving_start is not None:
            moving_end = self._first_confirmed(
                [sample.smooth_speed < self.stable_speed_thresh for sample in samples],
                moving_start + 1,
                self.stable_confirm_frames,
            )

        if holding_start is None and moving_start is None:
            # Fallback time split when behavior boundaries are absent.
            for i, sample in enumerate(samples):
                ratio = i / max(n - 1, 1)
                if ratio < 0.3:
                    sample.phase = PHASE_BEFORE_APPROACH
                elif ratio < 0.7:
                    sample.phase = PHASE_MOVING
                else:
                    sample.phase = PHASE_PUT_DOWN
            return

        for i, sample in enumerate(samples):
            if holding_start is not None and i < holding_start:
                sample.phase = PHASE_BEFORE_APPROACH
            elif moving_start is not None and i < moving_start:
                sample.phase = PHASE_HOLDING if holding_start is not None else PHASE_BEFORE_APPROACH
            elif moving_start is not None and (moving_end is None or i < moving_end):
                sample.phase = PHASE_MOVING
            elif moving_end is not None and i >= moving_end:
                sample.phase = PHASE_PUT_DOWN
            else:
                sample.phase = PHASE_HOLDING

    def _score_samples(self, samples: list[ObjClassificationSample]) -> None:
        for sample in samples:
            area_ratio = sample.area / max(sample.image_area, 1.0)
            area_score = min(area_ratio / self.area_full_ratio, 1.0)
            conf_score = max(0.0, min(float(sample.score), 1.0))
            stability_score = 1.0 - min(float(sample.smooth_speed) / self.speed_norm, 1.0)
            sample.quality = float(0.4 * area_score + 0.4 * conf_score + 0.2 * stability_score)

    def _select_diverse(self, samples: list[ObjClassificationSample], limit: int) -> list[ObjClassificationSample]:
        selected: list[ObjClassificationSample] = []
        for sample in sorted(samples, key=lambda item: item.quality, reverse=True):
            if any(abs(sample.frame_id - kept.frame_id) < self.min_frame_gap for kept in selected):
                continue
            if any(self._bbox_iou(sample.xyxy, kept.xyxy) > self.bbox_iou_diversity_thresh for kept in selected):
                continue
            selected.append(sample)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _first_confirmed(flags: list[bool], start: int, confirm: int) -> int | None:
        count = 0
        for i in range(max(0, start), len(flags)):
            count = count + 1 if flags[i] else 0
            if count >= confirm:
                return i - confirm + 1
        return None

    @staticmethod
    def _bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
        area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
