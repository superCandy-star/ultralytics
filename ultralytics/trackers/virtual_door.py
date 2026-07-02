# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ultralytics.utils import LOGGER


UNKNOWN = "UNKNOWN"
INSIDE = "INSIDE"
OUTSIDE = "OUTSIDE"
PUT_IN = "put_in"
TAKE_OUT = "take_out"


@dataclass
class DoorObservation:
    """Virtual door detections observed in one frame."""

    frame_id: int
    front_y: float | None = None
    junction_xyxy: np.ndarray | None = None
    cabinet_xyxy: np.ndarray | None = None
    side_boxes: list[np.ndarray] = field(default_factory=list)


@dataclass
class DoorStateRecord:
    """Debounced inside/outside state for one object at one door."""

    state: str = UNKNOWN
    candidate: str | None = None
    candidate_count: int = 0


@dataclass
class ObjectDoorState:
    """All virtual-door states owned by one object track."""

    front: DoorStateRecord = field(default_factory=DoorStateRecord)
    sides: dict[str, DoorStateRecord] = field(default_factory=dict)
    last_seen_frame: int = 0


class VirtualDoorManager:
    """Initialize fixed virtual doors and emit object put-in/take-out events.

    The manager is deliberately independent from tracker association. It observes finalized object tracks, keeps a small
    state machine per track ID, and exposes events as dictionaries without changing the tracker output array format.
    """

    FRONT_DIRECTIONS = {"y_greater_inside", "y_less_inside", "x_less_inside", "x_greater_inside"}

    def __init__(self, args: Any):
        self.enabled = self._as_bool(getattr(args, "virtual_door_enabled", False))
        self.model_path = getattr(args, "virtual_door_model", None)
        self.conf = float(getattr(args, "virtual_door_conf", 0.25))
        self.imgsz = getattr(args, "virtual_door_imgsz", 640)
        self.device = getattr(args, "virtual_door_device", None)
        self.verbose = self._as_bool(getattr(args, "virtual_door_verbose", False))
        self.side_cls = getattr(args, "virtual_door_side_cabinet_cls", "side_cabinet")
        self.junction_cls = getattr(args, "virtual_door_junction_cls", "junction")
        self.cabinet_cls = getattr(args, "virtual_door_cabinet_cls", "cabinet")
        self.init_frames = max(1, int(getattr(args, "virtual_door_init_frames", 30)))
        self.stability_frames = max(1, int(getattr(args, "virtual_door_stability_frames", 5)))
        self.position_tolerance = float(getattr(args, "virtual_door_position_tolerance", 10.0))
        self.side_iou_tolerance = float(getattr(args, "virtual_door_side_iou_tolerance", 0.8))
        self.confirm_frames = max(1, int(getattr(args, "virtual_door_confirm_frames", 2)))
        self.unknown_confirm_frames = max(1, int(getattr(args, "virtual_door_unknown_confirm_frames", 1)))
        self.front_direction = getattr(args, "virtual_door_front_direction", "y_greater_inside")
        self.front_margin = max(0.0, float(getattr(args, "virtual_door_front_margin", 3.0)))
        self.track_prune_frames = max(1, int(getattr(args, "virtual_door_track_prune_frames", 120)))
        self.fail_open = self._as_bool(getattr(args, "virtual_door_fail_open", True))

        self._model = None
        self._class_ids: dict[str, int] | None = None
        self._observations: list[DoorObservation] = []
        self._init_seen = 0
        self.locked = False
        self.failed = False

        if self.front_direction not in self.FRONT_DIRECTIONS:
            self._handle_failure(
                f"Unsupported virtual_door_front_direction={self.front_direction!r}; "
                f"expected one of {sorted(self.FRONT_DIRECTIONS)}."
            )
        self.front_y: float | None = None
        self.front_source: dict[str, list[float]] = {}
        self.side_doors: dict[str, np.ndarray] = {}
        self.locked_frame_id: int | None = None
        self.object_states: dict[int, ObjectDoorState] = {}

    def reset(self) -> None:
        """Reset all door geometry and per-track state for a new source/video."""
        self._model = None
        self._class_ids = None
        self._observations = []
        self._init_seen = 0
        self.locked = False
        self.failed = False
        self.front_y = None
        self.front_source = {}
        self.side_doors = {}
        self.locked_frame_id = None
        self.object_states = {}

    def update(self, frame_id: int, img: np.ndarray | None, object_tracks: list[Any]) -> list[dict[str, Any]]:
        """Update virtual-door initialization and return events generated in this frame."""
        if not self.enabled or self.failed:
            return []

        if not self.locked:
            self._update_initialization(frame_id, img)
            if not self.locked:
                return []

        tracks = sorted(object_tracks, key=lambda track: int(track.track_id))
        events: list[dict[str, Any]] = []
        active_track_ids: set[int] = set()
        for track in tracks:
            track_id = int(track.track_id)
            active_track_ids.add(track_id)
            state = self.object_states.setdefault(track_id, ObjectDoorState())
            state.last_seen_frame = frame_id
            events.extend(self._update_track_state(frame_id, track, state))

        self._prune_states(frame_id, active_track_ids)
        return events

    def _update_initialization(self, frame_id: int, img: np.ndarray | None) -> None:
        """Run door inference until stable geometry is found or initialization times out."""
        if img is None:
            self._init_seen += 1
            if self._init_seen >= self.init_frames:
                self._handle_failure("Virtual door initialization timed out because frames were unavailable.")
            return

        self._init_seen += 1
        observation = self._observe_doors(frame_id, img)
        if observation is not None:
            self._observations.append(observation)
            self._observations = self._observations[-self.stability_frames :]
            self._try_lock(frame_id)
        else:
            self._observations = []

        if not self.locked and self._init_seen >= self.init_frames:
            self._handle_failure("Virtual door initialization timed out before stable geometry was detected.")

    def _observe_doors(self, frame_id: int, img: np.ndarray) -> DoorObservation | None:
        """Run the virtual-door detector once and extract the best door observations."""
        model = self._load_model()
        if model is None:
            return None

        try:
            kwargs = {"conf": self.conf, "verbose": self.verbose}
            if self.imgsz is not None:
                kwargs["imgsz"] = int(self.imgsz)
            if self.device is not None:
                kwargs["device"] = self.device
            results = model.predict(img, **kwargs)
        except Exception as exc:
            self._handle_failure(f"Virtual door model prediction failed: {exc}")
            return None

        if not results:
            return None
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        try:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = boxes.conf.cpu().numpy().astype(np.float32)
            cls = boxes.cls.cpu().numpy().astype(np.int32)
        except Exception as exc:
            self._handle_failure(f"Could not read virtual door prediction boxes: {exc}")
            return None

        class_ids = self._resolve_class_ids(model)
        if class_ids is None:
            return None

        junction = self._best_box_for_cls(xyxy, conf, cls, class_ids["junction"])
        cabinet = self._best_box_for_cls(xyxy, conf, cls, class_ids["cabinet"])
        side_boxes = [box.copy() for box in xyxy[cls == class_ids["side"]]]
        side_boxes.sort(key=lambda box: (float(box[0]), float(box[1]), float(box[2]), float(box[3])))

        front_y = None
        junction_xyxy = None
        cabinet_xyxy = None
        if junction is not None and cabinet is not None:
            junction_xyxy = junction.copy()
            cabinet_xyxy = cabinet.copy()
            front_y = float(max(junction_xyxy[1], cabinet_xyxy[1]))

        if front_y is None and not side_boxes:
            return None
        return DoorObservation(frame_id, front_y, junction_xyxy, cabinet_xyxy, side_boxes)

    def _load_model(self):
        """Lazily load the door model only while initialization is active."""
        if self._model is not None:
            return self._model
        if not self.model_path:
            self._handle_failure("virtual_door_model is empty.")
            return None
        if isinstance(self.model_path, str) and self.model_path not in {"auto", "none", "None"}:
            model_file = Path(self.model_path)
            if not model_file.exists():
                self._handle_failure(f"Virtual door model does not exist: {self.model_path}")
                return None
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        except Exception as exc:
            self._handle_failure(f"Could not load virtual door model {self.model_path!r}: {exc}")
            return None
        return self._model

    def _resolve_class_ids(self, model) -> dict[str, int] | None:
        """Resolve configured class names/ids against the door model names."""
        if self._class_ids is not None:
            return self._class_ids
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            name_to_id = {str(name): int(cls_id) for cls_id, name in names.items()}
        else:
            name_to_id = {str(name): cls_id for cls_id, name in enumerate(names)}

        side = self._resolve_one_class(self.side_cls, name_to_id, "side_cabinet")
        junction = self._resolve_one_class(self.junction_cls, name_to_id, "junction")
        cabinet = self._resolve_one_class(self.cabinet_cls, name_to_id, "cabinet")
        if side is None or junction is None or cabinet is None:
            return None
        self._class_ids = {"side": side, "junction": junction, "cabinet": cabinet}
        return self._class_ids

    def _resolve_one_class(self, value: Any, name_to_id: dict[str, int], label: str) -> int | None:
        """Resolve one configured class value as either numeric id or model class name."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lstrip("+-").isdigit():
                return int(stripped)
            if stripped in name_to_id:
                return name_to_id[stripped]
            self._handle_failure(
                f"Virtual door class {label!r}={value!r} was not found in model names {sorted(name_to_id)}."
            )
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            self._handle_failure(f"Virtual door class {label!r}={value!r} is not a valid name or id.")
            return None

    @staticmethod
    def _best_box_for_cls(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, cls_id: int) -> np.ndarray | None:
        """Return the highest-confidence box for one class id."""
        indices = np.where(cls == cls_id)[0]
        if len(indices) == 0:
            return None
        best = indices[int(np.argmax(conf[indices]))]
        return xyxy[best]

    def _try_lock(self, frame_id: int) -> None:
        """Lock stable virtual-door geometry from recent observations."""
        recent = self._observations[-self.stability_frames :]
        if len(recent) < self.stability_frames:
            return

        front_values = [obs.front_y for obs in recent if obs.front_y is not None]
        front_locked = len(front_values) == self.stability_frames and (
            max(front_values) - min(front_values) <= self.position_tolerance
        )
        side_doors = self._lock_side_doors(recent)
        if not front_locked and not side_doors:
            return

        self.front_y = None
        self.front_source = {}
        if front_locked:
            self.front_y = float(np.median(np.asarray(front_values, dtype=np.float32)))
            front_obs = [obs for obs in recent if obs.front_y is not None][-1]
            if front_obs.junction_xyxy is not None:
                self.front_source["junction_xyxy"] = self._box_to_list(front_obs.junction_xyxy)
            if front_obs.cabinet_xyxy is not None:
                self.front_source["cabinet_xyxy"] = self._box_to_list(front_obs.cabinet_xyxy)
        self.side_doors = side_doors
        self.locked = True
        self.locked_frame_id = frame_id
        self._model = None

    def _lock_side_doors(self, observations: list[DoorObservation]) -> dict[str, np.ndarray]:
        """Cluster stable side-cabinet boxes and assign deterministic side door IDs."""
        entries: list[tuple[int, np.ndarray]] = []
        for obs in observations:
            entries.extend((obs.frame_id, box) for box in obs.side_boxes)
        if not entries:
            return {}

        clusters: list[list[tuple[int, np.ndarray]]] = []
        for frame_id, box in entries:
            matched = None
            for cluster in clusters:
                rep = np.median(np.asarray([item[1] for item in cluster], dtype=np.float32), axis=0)
                if self._bbox_iou(rep, box) >= self.side_iou_tolerance:
                    matched = cluster
                    break
            if matched is None:
                clusters.append([(frame_id, box)])
            else:
                matched.append((frame_id, box))

        locked_boxes: list[np.ndarray] = []
        for cluster in clusters:
            seen_frames = {item[0] for item in cluster}
            if len(seen_frames) >= self.stability_frames:
                locked_boxes.append(np.median(np.asarray([item[1] for item in cluster], dtype=np.float32), axis=0))

        locked_boxes.sort(key=lambda box: (float(box[0]), float(box[1]), float(box[2]), float(box[3])))
        return {f"side_{i}": box.astype(np.float32) for i, box in enumerate(locked_boxes)}

    def _update_track_state(self, frame_id: int, track: Any, state: ObjectDoorState) -> list[dict[str, Any]]:
        """Update one object track against all locked doors."""
        events: list[dict[str, Any]] = []
        front_observed = self._classify_front(track)
        if front_observed is not None:
            transition = self._advance_state(state.front, front_observed)
            if transition is not None:
                event, from_state, to_state = transition
                events.append(self._make_event(frame_id, track, "front", "front", event, from_state, to_state))

        for door_id, door_box in sorted(self.side_doors.items()):
            observed = self._classify_side(track, door_box)
            side_state = state.sides.setdefault(door_id, DoorStateRecord())
            transition = self._advance_state(side_state, observed)
            if transition is not None:
                event, from_state, to_state = transition
                events.append(self._make_event(frame_id, track, door_id, "side", event, from_state, to_state))
        return events

    def _classify_front(self, track: Any) -> str | None:
        """Classify an object center as inside/outside the front horizontal door line."""
        if self.front_y is None:
            return None
        cy = float(track.xywh[1])
        # if self.front_direction in {"x_less_inside", "x_greater_inside"}:
        #     if abs(cx - self.front_y) <= self.front_margin:
        #         return None
        #     if self.front_direction == "x_less_inside":
        #         return INSIDE if cx < self.front_y else OUTSIDE
        #     return INSIDE if cx > self.front_y else OUTSIDE

        if abs(cy - self.front_y) <= self.front_margin:
            return None
        if self.front_direction == "y_less_inside":
            return INSIDE if cy < self.front_y else OUTSIDE
        return INSIDE if cy > self.front_y else OUTSIDE

    @staticmethod
    def _classify_side(track: Any, door_box: np.ndarray) -> str:
        """Classify an object center relative to a side-cabinet door box."""
        cx, cy = float(track.xywh[0]), float(track.xywh[1])
        inside = bool(door_box[0] <= cx <= door_box[2] and door_box[1] <= cy <= door_box[3])
        return INSIDE if inside else OUTSIDE

    def _advance_state(self, record: DoorStateRecord, observed: str) -> tuple[str, str, str] | None:
        """Advance one debounced state record and return an event transition when one occurs."""
        previous = record.state
        if previous == UNKNOWN:
            self._update_candidate(record, observed)
            if record.candidate_count >= self.unknown_confirm_frames:
                record.state = observed
                record.candidate = None
                record.candidate_count = 0
            return None

        if observed == previous:
            record.candidate = None
            record.candidate_count = 0
            return None

        self._update_candidate(record, observed)
        if record.candidate_count < self.confirm_frames:
            return None

        record.state = observed
        record.candidate = None
        record.candidate_count = 0
        if previous == OUTSIDE and observed == INSIDE:
            return PUT_IN, previous, observed
        if previous == INSIDE and observed == OUTSIDE:
            return TAKE_OUT, previous, observed
        return None

    @staticmethod
    def _update_candidate(record: DoorStateRecord, observed: str) -> None:
        """Update candidate state counters for debounce logic."""
        if record.candidate == observed:
            record.candidate_count += 1
        else:
            record.candidate = observed
            record.candidate_count = 1

    @staticmethod
    def _make_event(
        frame_id: int, track: Any, door_id: str, door_type: str, event: str, from_state: str, to_state: str
    ) -> dict[str, Any]:
        """Build a stable event dictionary for downstream consumers."""
        return {
            "frame_id": int(frame_id),
            "track_id": int(track.track_id),
            "event": event,
            "door_id": door_id,
            "door_type": door_type,
            "from_state": from_state,
            "to_state": to_state,
            "center": [float(track.xywh[0]), float(track.xywh[1])],
            "xyxy": VirtualDoorManager._box_to_list(track.xyxy),
            "classification": {
                "label": getattr(track, "classification_label", None),
                "conf": getattr(track, "classification_conf", None),
                "scores": getattr(track, "classification_scores", None),
                "frame_id": getattr(track, "classification_frame_id", None),
                "detail": getattr(track, "classification_detail", None),
            },
        }

    def _prune_states(self, frame_id: int, active_track_ids: set[int]) -> None:
        """Drop stale per-track door state to keep memory bounded."""
        stale = [
            track_id
            for track_id, state in self.object_states.items()
            if track_id not in active_track_ids and frame_id - state.last_seen_frame > self.track_prune_frames
        ]
        for track_id in stale:
            del self.object_states[track_id]

    def _handle_failure(self, message: str) -> None:
        """Handle virtual-door failures according to fail-open/strict configuration."""
        if self.fail_open:
            if self.enabled and not self.failed:
                LOGGER.warning(f"Virtual door disabled: {message}")
            self.failed = True
            self._model = None
            return
        raise RuntimeError(message)

    @staticmethod
    def _bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two xyxy boxes."""
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
    def _box_to_list(box: np.ndarray) -> list[float]:
        """Convert a numpy-like box to a JSON-friendly float list."""
        return [float(v) for v in np.asarray(box, dtype=np.float32).reshape(-1).tolist()]

    @staticmethod
    def _as_bool(value: Any) -> bool:
        """Parse config booleans that may arrive as bools or strings."""
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
