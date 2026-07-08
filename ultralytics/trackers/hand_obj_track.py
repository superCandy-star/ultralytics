# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .obj_classifier import ObjTrackClassifier
from .utils import matching
from .utils.stracks import joint_stracks, parse_bboxes, sub_stracks
from .virtual_door import VirtualDoorManager


class HandObjTrack(STrack):
    """ByteTrack track with optional hand-anchor metadata for object tracks."""

    def __init__(self, xywh: np.ndarray, score: float, cls: Any):
        super().__init__(xywh, score, cls)
        self.anchor_hand_track_id: int | None = None
        self.anchor_hand_xywh: np.ndarray | None = None
        self.anchor_mode = False
        self.anchor_lost_frames = 0
        self.anchor_enabled = False
        self.hold_confirm_count = 0
        self.last_hand_obj_distance: float | None = None
        self.last_hold_update_frame = -1
        self.release_confirm_count = 0
        self.last_hold_obj_xywh: np.ndarray | None = None
        self.last_real_center: np.ndarray | None = None
        self.classification_samples = []
        self.classification_label: str | None = None
        self.classification_conf: float | None = None
        self.classification_scores: dict[str, float] | None = None
        self.classification_frame_id = -1
        self.classification_detail: dict[str, Any] | None = None

    def bind_hand(self, hand_track: HandObjTrack | None, enable_anchor: bool = False) -> None:
        """Bind this object track to a hand track and remember the hand's current box."""
        if hand_track is None:
            return
        self.anchor_hand_track_id = hand_track.track_id
        self.anchor_hand_xywh = hand_track.xywh.copy()
        if enable_anchor:
            self.anchor_enabled = True

    def clear_hand_binding(self) -> None:
        """Clear all hand-anchor state from this track."""
        self.anchor_hand_track_id = None
        self.anchor_hand_xywh = None
        self.anchor_enabled = False
        self.hold_confirm_count = 0
        self.last_hand_obj_distance = None
        self.last_hold_obj_xywh = None
        self.exit_anchor_mode()

    def enter_anchor_mode(self) -> None:
        """Mark this track as hand-anchored lost."""
        if not self.anchor_mode:
            self.anchor_lost_frames = 0
        self.anchor_mode = True

    def exit_anchor_mode(self) -> None:
        """Leave hand-anchored lost mode while preserving the hand binding."""
        self.anchor_mode = False
        self.anchor_lost_frames = 0

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Reactivate this track and clear anchored-lost bookkeeping."""
        super().re_activate(new_track, frame_id, new_id=new_id)
        self.last_real_center = new_track.xywh[:2].copy()
        self.exit_anchor_mode()

    def update(self, new_track: STrack, frame_id: int):
        """Update this track from a detection and clear anchored-lost bookkeeping."""
        super().update(new_track, frame_id)
        self.last_real_center = new_track.xywh[:2].copy()
        self.exit_anchor_mode()

    def propagate_with_hand(self, hand_track: HandObjTrack, frame_id: int) -> None:
        """Move the object center by the bound hand's center delta."""
        current_hand_xywh = hand_track.xywh.copy()
        if self.anchor_hand_xywh is None:
            self.anchor_hand_xywh = current_hand_xywh
            dx = dy = 0.0
        else:
            dx, dy = current_hand_xywh[:2] - self.anchor_hand_xywh[:2]
            self.anchor_hand_xywh = current_hand_xywh

        if self.mean is not None:
            self.mean[0] += dx
            self.mean[1] += dy
        else:
            self._tlwh[:2] += np.asarray([dx, dy], dtype=self._tlwh.dtype)

        self.frame_id = frame_id
        self.state = TrackState.Lost
        self.anchor_mode = True
        self.anchor_lost_frames += 1

    @property
    def synthetic_result(self) -> list[float]:
        """Return a track row whose final idx marks that no current detection exists."""
        result = self.result
        result[-1] = -1
        return result


class HandObjBYTETracker(BYTETracker):
    """ByteTrack variant that anchors missed object tracks to stable hand tracks.

    The intended two-class use case is smart-fridge tracking with class 0 as ``hand`` and class 1 as ``obj``. Hand tracks
    use standard ByteTrack-style association. Object tracks bind to the nearest tracked hand when created; if an object is
    missed while its bound hand is still tracked, the object enters anchored-lost mode and its box follows the hand motion
    for a bounded number of frames. Reappearing object detections are matched back to anchored-lost tracks before new IDs
    are created.
    """

    track_class = HandObjTrack

    def __init__(self, args):
        super().__init__(args)
        self.hand_cls = int(getattr(args, "hand_cls", 0))
        self.obj_cls = int(getattr(args, "obj_cls", 1))
        self.anchor_track_buffer = int(getattr(args, "anchor_track_buffer", 15))
        self.anchor_max_bind_distance = float(getattr(args, "anchor_max_bind_distance", 150.0))
        self.anchor_match_thresh = float(getattr(args, "anchor_match_thresh", getattr(args, "match_thresh", 0.8)))
        self.hold_confirm_frames = int(getattr(args, "hold_confirm_frames", 2))
        self.hold_center_stable_thresh = float(getattr(args, "hold_center_stable_thresh", 8.0))
        self.hold_center_in_hand = bool(getattr(args, "hold_center_in_hand", True))
        self.release_confirm_frames = int(getattr(args, "release_confirm_frames", 3))
        self.hold_motion_min_displacement = max(0.0, float(getattr(args, "hold_motion_min_displacement", 1.0)))
        self.anchored_lost_max_door_frames = int(getattr(args, "virtual_door_anchored_lost_max_frames", 2))
        self.obj_center_match_dist = float(getattr(args, "obj_center_match_dist", 200.0))
        self.hand_center_match_dist = float(getattr(args, "hand_center_match_dist", 100.0))
        self.virtual_door_manager = VirtualDoorManager(args)
        self.obj_classifier = ObjTrackClassifier(args)
        self.obj_classification_results: dict[int, dict[str, Any]] = {}
        self.last_virtual_door_events: list[dict[str, Any]] = []
        self.virtual_door_events: list[dict[str, Any]] = []

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """Update hand/object tracks and return rows in Ultralytics tracker format."""
        self.frame_id += 1
        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        results_high, results_low, mask_high, mask_low = self._split_detections(results)
        detections_high = self.init_track(results_high, self._input_for(img, feats, mask_high))
        detections_low = self.init_track(results_low, self._input_for(img, feats, mask_low))

        hand_high = self._filter_cls(detections_high, self.hand_cls)
        hand_low = self._filter_cls(detections_low, self.hand_cls)
        obj_high = self._filter_cls(detections_high, self.obj_cls)
        obj_low = self._filter_cls(detections_low, self.obj_cls)

        self._update_standard_class(
            self.hand_cls, hand_high, hand_low, activated_stracks, refind_stracks, lost_stracks, removed_stracks
        )
        current_hands = self._current_tracks_for_cls(self.hand_cls, activated_stracks, refind_stracks)
        hand_by_id = {track.track_id: track for track in current_hands}

        self._update_object_tracks(
            obj_high, obj_low, hand_by_id, activated_stracks, refind_stracks, lost_stracks, removed_stracks
        )

        self._remove_stale_lost(removed_stracks)
        self._merge_track_pools(activated_stracks, refind_stracks, lost_stracks, removed_stracks)
        self._update_obj_classification_samples(img)
        self._update_virtual_door_events(img)
        return self._format_output()

    def init_track(self, results, img: np.ndarray | None = None) -> list[HandObjTrack]:
        """Initialize hand/object track instances from detections."""
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        return [self.track_class(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def _update_standard_class(
        self,
        cls: int,
        detections: list[HandObjTrack],
        detections_second: list[HandObjTrack],
        activated: list[HandObjTrack],
        refind: list[HandObjTrack],
        lost: list[HandObjTrack],
        removed: list[HandObjTrack],
    ) -> None:
        """Run ByteTrack-style association for one class without hand-anchor behavior."""
        unconfirmed, tracked = self._split_tracked_for_cls(cls)
        lost_pool = [track for track in self.lost_stracks if self._is_cls(track, cls)]
        strack_pool = joint_stracks(tracked, lost_pool)
        self.multi_predict(strack_pool)

        u_track, u_detection = self._associate_first(strack_pool, detections, activated, refind)
        # Center-distance fallback for hand / other classes missed by IoU matching
        center_dist = self.hand_center_match_dist if cls == self.hand_cls else 100.0
        u_track, u_detection = self._center_distance_fallback(
            strack_pool, detections, u_track, u_detection, activated, refind, center_dist
        )
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        self._associate_second(r_tracked, detections_second, activated, refind, lost)
        u_detection, detections = self._unconfirmed_association(unconfirmed, u_detection, detections, activated, removed)
        self._init_new_tracks(u_detection, detections, activated, refind)

    def _update_object_tracks(
        self,
        detections: list[HandObjTrack],
        detections_second: list[HandObjTrack],
        hand_by_id: dict[int, HandObjTrack],
        activated: list[HandObjTrack],
        refind: list[HandObjTrack],
        lost: list[HandObjTrack],
        removed: list[HandObjTrack],
    ) -> None:
        """Update object tracks with hand-anchored lost/reassociation behavior."""
        anchored_tracks, normal_lost = self._prepare_lost_object_tracks(hand_by_id, removed)
        unconfirmed, tracked = self._split_tracked_for_cls(self.obj_cls)
        strack_pool = joint_stracks(tracked, normal_lost)
        self.multi_predict(strack_pool)

        u_track, u_detection = self._associate_first(strack_pool, detections, activated, refind)
        # Center-distance fallback for obj tracks missed by IoU matching
        u_track, u_detection = self._center_distance_fallback(
            strack_pool, detections, u_track, u_detection, activated, refind, self.obj_center_match_dist
        )
        self._update_held_states(activated, refind, hand_by_id)

        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        u_second = self._associate_second(r_tracked, detections_second, activated, refind, lost_on_unmatched=False)
        self._update_held_states(activated, refind, hand_by_id)
        self._anchor_or_mark_lost([r_tracked[i] for i in u_second], hand_by_id, lost)

        u_detection = self._associate_anchored(anchored_tracks, detections, u_detection, activated, refind, hand_by_id)
        u_detection, detections = self._unconfirmed_association(unconfirmed, u_detection, detections, activated, removed)
        self._update_held_states(activated, refind, hand_by_id)
        self._init_new_object_tracks(u_detection, detections, activated, hand_by_id)

    def _associate_first(
        self,
        strack_pool: list[HandObjTrack],
        detections: list[HandObjTrack],
        activated: list[HandObjTrack],
        refind: list[HandObjTrack],
    ) -> tuple[list[int], list[int]]:
        """Associate high-confidence detections to a track pool."""
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        self._apply_matches(matches, strack_pool, detections, activated, refind)
        return list(u_track), list(u_detection)

    def _center_distance_fallback(
        self,
        strack_pool: list[HandObjTrack],
        detections: list[HandObjTrack],
        u_track: list[int],
        u_detection: list[int],
        activated: list[HandObjTrack],
        refind: list[HandObjTrack],
        max_dist: float,
    ) -> tuple[list[int], list[int]]:
        """Match unmatched tracks to unmatched detections using center-point distance.

        Used as a fallback when IoU matching fails because the object moved
        too far between consecutive frames (e.g. hand-held obj moving fast).
        """
        if not u_track or not u_detection or max_dist <= 0:
            return u_track, u_detection

        unmatched_tracks = [strack_pool[i] for i in u_track]
        unmatched_dets = [detections[i] for i in u_detection]

        track_centers = np.asarray([t.xywh[:2] for t in unmatched_tracks], dtype=np.float32)
        det_centers = np.asarray([d.xywh[:2] for d in unmatched_dets], dtype=np.float32)
        center_dists = np.linalg.norm(track_centers[:, None, :] - det_centers[None, :, :], axis=2)

        # Gate: pairs with distance > max_dist get high cost
        max_val = center_dists.max() * 2.0 + 1.0
        gated = np.where(center_dists <= max_dist, center_dists, max_val)

        matches, new_u_track, new_u_detection = matching.linear_assignment(gated, thresh=max_dist)
        for itracked, idet in matches:
            track = unmatched_tracks[itracked]
            det = unmatched_dets[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        return (
            [u_track[i] for i in new_u_track],
            [u_detection[i] for i in new_u_detection],
        )

    def _associate_second(
        self,
        tracks: list[HandObjTrack],
        detections_second: list[HandObjTrack],
        activated: list[HandObjTrack],
        refind: list[HandObjTrack],
        lost: list[HandObjTrack] | None = None,
        lost_on_unmatched: bool = True,
    ) -> list[int]:
        """Associate low-confidence detections to currently tracked tracks."""
        if tracks and detections_second:
            dists = matching.iou_distance(tracks, detections_second)
            if self.args.fuse_score:
                dists = matching.fuse_score(dists, detections_second)
            matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
            self._apply_matches(matches, tracks, detections_second, activated, refind)
            unmatched = list(u_track)
        else:
            unmatched = list(range(len(tracks)))

        if lost_on_unmatched and lost is not None:
            for it in unmatched:
                track = tracks[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost.append(track)
        return unmatched

    def _associate_anchored(
        self,
        anchored_tracks: list[HandObjTrack],
        detections: list[HandObjTrack],
        u_detection: list[int],
        activated: list[HandObjTrack],
        refind: list[HandObjTrack],
        hand_by_id: dict[int, HandObjTrack],
    ) -> list[int]:
        """Match remaining object detections to hand-propagated anchored-lost object tracks."""
        candidates = [track for track in anchored_tracks if track.anchor_mode and track.state == TrackState.Lost]
        remaining = [detections[i] for i in u_detection]
        if not candidates or not remaining:
            return u_detection

        dists = self._anchored_dists(candidates, remaining, hand_by_id)
        matches, u_track_a, unmatched_detection = matching.linear_assignment(dists, thresh=self.anchor_match_thresh)
        for itracked, idet in matches:
            track = candidates[itracked]
            det = remaining[idet]
            track.re_activate(det, self.frame_id, new_id=False)
            self._update_held_state(track, hand_by_id)
            refind.append(track)

        # Center-distance fallback for anchored tracks that drifted
        if len(u_track_a) > 0 and len(unmatched_detection) > 0:
            still_anchored = [candidates[i] for i in u_track_a]
            leftover_dets = [remaining[i] for i in unmatched_detection]
            track_c = np.asarray([t.xywh[:2] for t in still_anchored], dtype=np.float32)
            det_c = np.asarray([d.xywh[:2] for d in leftover_dets], dtype=np.float32)
            cdist = np.linalg.norm(track_c[:, None, :] - det_c[None, :, :], axis=2)
            c_matches, u_track_c, new_unmatched = matching.linear_assignment(cdist, thresh=self.obj_center_match_dist)
            for itracked, idet in c_matches:
                track = still_anchored[itracked]
                det = leftover_dets[idet]
                track.re_activate(det, self.frame_id, new_id=False)
                self._update_held_state(track, hand_by_id)
                refind.append(track)
            # Anchored tracks still unmatched: check if they are ghosts near active tracks
            for i in u_track_c:
                ghost = still_anchored[i]
                # If anchored track overlaps significantly with any active obj track, remove it
                active_objs = [t for t in self.tracked_stracks
                               if self._is_cls(t, self.obj_cls) and t.state == TrackState.Tracked]
                is_ghost = bool(active_objs) and any(
                    matching.iou_distance([ghost], [active])[0, 0] <= 1.0 - 0.3
                    for active in active_objs
                )
                if is_ghost:
                    ghost.mark_removed()
                    try:
                        self.lost_stracks.remove(ghost)
                    except ValueError:
                        pass
                else:
                    ghost.exit_anchor_mode()
            unmatched_detection = [unmatched_detection[i] for i in new_unmatched]
        return [u_detection[i] for i in unmatched_detection]

    def _anchored_dists(
        self,
        tracks: list[HandObjTrack],
        detections: list[HandObjTrack],
        hand_by_id: dict[int, HandObjTrack],
    ) -> np.ndarray:
        """Return a gated cost matrix for anchored object re-association."""
        dists = matching.iou_distance(tracks, detections)
        if dists.size == 0:
            return dists

        track_centers = np.asarray([track.xywh[:2] for track in tracks], dtype=np.float32)
        det_centers = np.asarray([det.xywh[:2] for det in detections], dtype=np.float32)
        center_dist = np.linalg.norm(track_centers[:, None, :] - det_centers[None, :, :], axis=2)

        gates = np.full_like(dists, fill_value=1e6, dtype=np.float32)
        for i, track in enumerate(tracks):
            hand = hand_by_id.get(track.anchor_hand_track_id)
            if hand is None:
                max_dist = max(track.xywh[2], track.xywh[3]) * 2.0
            else:
                max_dist = max(hand.xywh[2], hand.xywh[3], track.xywh[2], track.xywh[3]) * 2.0
            gates[i] = np.where(center_dist[i] <= max_dist, dists[i], 1e6)
        if self.args.fuse_score:
            finite = np.isfinite(gates)
            fused = matching.fuse_score(np.where(finite, gates, 1.0), detections)
            gates = np.where(finite, fused, 1e6)
        return gates

    def _prepare_lost_object_tracks(
        self, hand_by_id: dict[int, HandObjTrack], removed: list[HandObjTrack]
    ) -> tuple[list[HandObjTrack], list[HandObjTrack]]:
        """Propagate reusable anchored-lost objects and split lost objects into anchored/normal pools."""
        anchored, normal = [], []
        for track in self.lost_stracks:
            if not self._is_cls(track, self.obj_cls):
                continue
            if not track.anchor_mode or not track.anchor_enabled:
                normal.append(track)
                continue

            # Ghost track: propagated >= 3 frames without ever matching a real detection
            if track.anchor_lost_frames >= 3:
                track.exit_anchor_mode()
                normal.append(track)
                continue

            if track.anchor_lost_frames >= self.anchor_track_buffer:
                track.exit_anchor_mode()
                normal.append(track)
                continue

            # Keep propagating only while hand is available

            # Use fallback mechanism to find hand
            hand = self._held_candidate_hand(track, hand_by_id)
            if hand is None:
                track.exit_anchor_mode()
                normal.append(track)
                continue

            track.propagate_with_hand(hand, self.frame_id)
            anchored.append(track)
        return anchored, normal

    def _anchor_or_mark_lost(
        self, tracks: list[HandObjTrack], hand_by_id: dict[int, HandObjTrack], lost: list[HandObjTrack]
    ) -> None:
        """Move unmatched object tracks into anchored-lost mode when their bound hand is tracked."""
        for track in tracks:
            # Use fallback mechanism to find hand
            hand = self._held_candidate_hand(track, hand_by_id)
            if track.anchor_enabled and hand is not None and self.anchor_track_buffer > 0:
                track.mark_lost()
                track.enter_anchor_mode()
                if track.anchor_hand_xywh is None:
                    track.bind_hand(hand)
                track.propagate_with_hand(hand, self.frame_id)
            else:
                track.mark_lost()
            lost.append(track)

    def _init_new_object_tracks(
        self,
        u_detection: list[int],
        detections: list[HandObjTrack],
        activated: list[HandObjTrack],
        hand_by_id: dict[int, HandObjTrack],
    ) -> None:
        """Activate new object tracks and bind them to the nearest current hand."""
        hands = list(hand_by_id.values())
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            track.is_activated = True  # activate immediately, bypass ByteTrack unconfirmed phase
            nearest_hand = self._nearest_hand(track, hands)
            track.bind_hand(nearest_hand, enable_anchor=False)
            activated.append(track)
            self._update_held_state(track, hand_by_id)

    def _update_held_states(
        self,
        activated: list[HandObjTrack],
        refind: list[HandObjTrack],
        hand_by_id: dict[int, HandObjTrack],
    ) -> None:
        """Update held-state evidence for object tracks that matched real detections this frame."""
        for track in [*activated, *refind]:
            if self._is_cls(track, self.obj_cls):
                self._update_held_state(track, hand_by_id)

    def _update_held_state(self, track: HandObjTrack, hand_by_id: dict[int, HandObjTrack]) -> None:
        """Confirm whether an object is currently hand-held before enabling hand anchoring.

        Requires contact evidence (obj inside hand box or hand inside obj box)
        AND the object itself is moving (was stationary, now moving) for consecutive frames.
        """
        if track.last_hold_update_frame == self.frame_id:
            return
        track.last_hold_update_frame = self.frame_id
        hand = self._held_candidate_hand(track, hand_by_id)
        if hand is None:
            track.hold_confirm_count = 0
            track.last_hand_obj_distance = None
            track.last_hold_obj_xywh = None
            track.release_confirm_count += 1
            if track.release_confirm_count >= self.release_confirm_frames and track.anchor_enabled:
                track.anchor_enabled = False
                track.release_confirm_count = 0
            return

        distance = float(np.linalg.norm(track.xywh[:2] - hand.xywh[:2]))
        obj_center_in_hand = self._center_in_box(track.xywh[:2], hand.xyxy) if self.hold_center_in_hand else False
        hand_center_in_obj = self._center_in_box(hand.xywh[:2], track.xyxy)
        contact_evidence = obj_center_in_hand or hand_center_in_obj

        obj_moving = False
        if track.last_hold_obj_xywh is not None:
            obj_speed = float(np.linalg.norm(track.xywh[:2] - track.last_hold_obj_xywh[:2]))
            obj_moving = obj_speed >= self.hold_motion_min_displacement

        is_held = contact_evidence and obj_moving

        track.last_hold_obj_xywh = track.xywh.copy()

        if is_held:
            track.hold_confirm_count += 1
            track.release_confirm_count = 0
            if track.hold_confirm_count >= self.hold_confirm_frames:
                track.bind_hand(hand, enable_anchor=True)
        else:
            track.hold_confirm_count = 0
            track.release_confirm_count += 1
            if track.release_confirm_count >= self.release_confirm_frames and track.anchor_enabled:
                track.anchor_enabled = False
                track.release_confirm_count = 0
            if track.anchor_hand_track_id is None:
                track.anchor_hand_xywh = None
        track.last_hand_obj_distance = distance

    def _held_candidate_hand(
        self, track: HandObjTrack, hand_by_id: dict[int, HandObjTrack]
    ) -> HandObjTrack | None:
        """Return the hand used to evaluate held state without doing handover rebinding.

        If the bound hand ID is lost, attempt to recover it by finding the nearest hand
        to the last known hand position. If found within fallback distance, update the ID.
        """
        if track.anchor_hand_track_id is not None:
            hand = hand_by_id.get(track.anchor_hand_track_id)
            if hand is not None:
                return hand

            # Hand ID fallback: try to recover binding using last known hand position
            if track.anchor_hand_xywh is not None and hand_by_id:
                last_hand_center = track.anchor_hand_xywh[:2]
                nearest = None
                min_dist = float('inf')

                for candidate_hand in hand_by_id.values():
                    dist = float(np.linalg.norm(candidate_hand.xywh[:2] - last_hand_center))
                    if dist < min_dist:
                        min_dist = dist
                        nearest = candidate_hand

                # If nearest hand is within fallback distance, treat as same hand with ID switch
                fallback_distance = max(track.xywh[2], track.xywh[3]) * 1.5
                if nearest is not None and min_dist <= fallback_distance:
                    track.anchor_hand_track_id = nearest.track_id
                    return nearest

        return self._nearest_hand(track, list(hand_by_id.values()))

    @staticmethod
    def _center_in_box(center: np.ndarray, xyxy: np.ndarray) -> bool:
        """Return whether a center point lies inside an xyxy box."""
        return bool(xyxy[0] <= center[0] <= xyxy[2] and xyxy[1] <= center[1] <= xyxy[3])

    def _nearest_hand(self, track: HandObjTrack, hands: list[HandObjTrack]) -> HandObjTrack | None:
        """Return the nearest hand track to an object track, respecting the binding distance threshold."""
        if not hands:
            return None
        obj_center = track.xywh[:2]
        distances = np.asarray([np.linalg.norm(hand.xywh[:2] - obj_center) for hand in hands], dtype=np.float32)
        nearest = int(np.argmin(distances))
        if self.anchor_max_bind_distance > 0 and distances[nearest] > self.anchor_max_bind_distance:
            return None
        return hands[nearest]

    def _current_tracks_for_cls(
        self,
        cls: int,
        activated: list[HandObjTrack],
        refind: list[HandObjTrack],
    ) -> list[HandObjTrack]:
        """Return currently tracked objects of a class, including tracks updated earlier this frame."""
        tracks = [*self.tracked_stracks, *activated, *refind]
        seen, current = set(), []
        for track in tracks:
            if track.track_id in seen or not self._is_cls(track, cls) or track.state != TrackState.Tracked:
                continue
            seen.add(track.track_id)
            current.append(track)
        return current

    def _split_tracked_for_cls(self, cls: int) -> tuple[list[HandObjTrack], list[HandObjTrack]]:
        """Split current tracked tracks of one class into unconfirmed and confirmed pools."""
        unconfirmed, tracked = [], []
        for track in self.tracked_stracks:
            if not self._is_cls(track, cls):
                continue
            (unconfirmed if not track.is_activated else tracked).append(track)
        return unconfirmed, tracked

    def _filter_cls(self, tracks: list[HandObjTrack], cls: int) -> list[HandObjTrack]:
        """Filter tracks by integer class id."""
        return [track for track in tracks if self._is_cls(track, cls)]

    @staticmethod
    def _is_cls(track: HandObjTrack, cls: int) -> bool:
        """Return True when a track belongs to a class id."""
        return int(track.cls) == cls

    def _remove_stale_lost(self, removed: list[HandObjTrack]) -> None:
        """Remove lost tracks, ending anchor mode before applying the normal lost timeout."""
        for track in self.lost_stracks:
            if self._is_cls(track, self.obj_cls) and track.anchor_mode:
                if track.anchor_lost_frames >= self.anchor_track_buffer:
                    track.exit_anchor_mode()
                else:
                    continue
            if self.frame_id - track.end_frame > self.max_frames_lost:
                track.mark_removed()
                removed.append(track)

    def _merge_track_pools(
        self,
        activated: list[HandObjTrack],
        refind: list[HandObjTrack],
        lost: list[HandObjTrack],
        removed: list[HandObjTrack],
        removed_buffer: int = 1000,
    ) -> None:
        """Merge persistent pools with class-aware duplicate removal."""
        self.tracked_stracks = [track for track in self.tracked_stracks if track.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self._remove_duplicate_stracks_by_class(
            self.tracked_stracks, self.lost_stracks
        )
        self.removed_stracks.extend(removed)
        if len(self.removed_stracks) > removed_buffer:
            self.removed_stracks = self.removed_stracks[-removed_buffer:]

    def _remove_duplicate_stracks_by_class(
        self, atracks: list[HandObjTrack], btracks: list[HandObjTrack], dup_thresh: float = 0.15
    ) -> tuple[list[HandObjTrack], list[HandObjTrack]]:
        """Remove duplicate tracks only when they share the same class id."""
        pdist = matching.iou_distance(atracks, btracks)
        pairs = np.where(pdist < dup_thresh)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            if int(atracks[p].cls) != int(btracks[q].cls):
                continue
            timep = atracks[p].frame_id - atracks[p].start_frame
            timeq = btracks[q].frame_id - btracks[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        dupa_set, dupb_set = set(dupa), set(dupb)
        resa = [track for i, track in enumerate(atracks) if i not in dupa_set]
        resb = [track for i, track in enumerate(btracks) if i not in dupb_set]
        return resa, resb

    def _current_object_tracks_for_events(self) -> list[HandObjTrack]:
        """Return current object tracks that can participate in virtual-door events."""
        max_drift = self.obj_center_match_dist
        object_tracks = [
            track
            for track in self.tracked_stracks
            if self._is_cls(track, self.obj_cls) and track.is_activated and track.state == TrackState.Tracked
            and not track.anchor_mode  # skip tracks that are being hand-propagated
            and (track.last_real_center is None
                 or float(np.linalg.norm(track.xywh[:2] - track.last_real_center)) <= max_drift)
        ]
        if bool(getattr(self.args, "virtual_door_include_anchored_lost", False)):
            for track in self.lost_stracks:
                if not (self._is_cls(track, self.obj_cls) and track.is_activated
                        and track.anchor_enabled and track.anchor_mode
                        and track.anchor_lost_frames <= self.anchored_lost_max_door_frames):
                    continue
                # Skip if anchor propagation has drifted too far from last real detection
                if track.last_real_center is not None:
                    drift = float(np.linalg.norm(track.xywh[:2] - track.last_real_center))
                    if drift > max_drift:
                        continue
                object_tracks.append(track)
        return object_tracks

    def _update_obj_classification_samples(self, img: np.ndarray | None) -> None:
        """Collect object-centered crop samples for track-level classification."""
        if not self.obj_classifier.enabled:
            return
        for track in self._current_object_tracks_for_events():
            if track.state == TrackState.Tracked:
                self.obj_classifier.add_sample(track, self.frame_id, img)

    def _classification_payload_for_track(self, track: HandObjTrack | None) -> dict[str, Any]:
        """Return or compute classification payload for one object track."""
        empty = {"label": None, "conf": None, "scores": None, "frame_id": None, "detail": None}
        if track is None or not self.obj_classifier.enabled:
            return empty
        result = None
        if getattr(track, "classification_label", None) is None:
            result = self.obj_classifier.classify_track(track)
        if result is None and getattr(track, "classification_label", None) is None:
            return empty
        payload = self.obj_classifier.classification_payload(track)
        self.obj_classification_results[int(track.track_id)] = payload
        return payload

    def _update_virtual_door_events(self, img: np.ndarray | None) -> None:
        """Update virtual-door event state from finalized current-frame object tracks."""
        self.last_virtual_door_events = []
        if not self.virtual_door_manager.enabled:
            return

        object_tracks = self._current_object_tracks_for_events()
        all_tracks = {int(t.track_id): t for t in (list(self.tracked_stracks) + list(self.lost_stracks))}
        events = self.virtual_door_manager.update(self.frame_id, img, object_tracks)

        # Deduplicate events: same door + overlapping track boxes → keep one
        deduped = []
        for event in events:
            e_tid = int(event.get("track_id", -1))
            e_door = event.get("door_id", "")
            dup = False
            for kept in deduped:
                if kept.get("door_id") != e_door:
                    continue
                k_tid = int(kept.get("track_id", -1))
                t1 = all_tracks.get(e_tid)
                t2 = all_tracks.get(k_tid)
                if t1 is None or t2 is None:
                    continue
                if matching.iou_distance([t1], [t2])[0, 0] <= 1.0 - 0.3:
                    dup = True
                    break
            if not dup:
                deduped.append(event)

        for event in deduped:
            track = all_tracks.get(int(event.get("track_id", -1)))
            classification = self._classification_payload_for_track(track)
            event["classification"] = classification
        # Collapse rapid alternating transitions (put_in→take_out→put_in)
        # that occur when an object crosses nested/adjacent side doors.
        rapid_window = int(getattr(self.args, "virtual_door_rapid_window", 15))
        self.last_virtual_door_events = []
        for event in deduped:
            e_tid = int(event.get("track_id", -1))
            e_frame = int(event.get("frame_id", self.frame_id))
            e_type = event.get("event", "")
            tail = [ev for ev in self.virtual_door_events
                    if int(ev.get("track_id", -1)) == e_tid
                    and e_frame - int(ev.get("frame_id", 0)) <= rapid_window]
            if len(tail) >= 2:
                types_in_tail = [ev.get("event") for ev in tail]
                alternates = any(types_in_tail[i] != types_in_tail[i + 1]
                                 for i in range(len(types_in_tail) - 1))
                if alternates and e_type == tail[0].get("event"):
                    for ev in tail:
                        try:
                            self.virtual_door_events.remove(ev)
                            self.last_virtual_door_events = [
                                x for x in self.last_virtual_door_events if x is not ev
                            ]
                        except ValueError:
                            pass
            self.virtual_door_events.append(event)
            self.last_virtual_door_events.append(event)

    def reset(self):
        """Reset tracker state and virtual-door state for a new source/video."""
        super().reset()
        self.last_virtual_door_events = []
        self.virtual_door_events = []
        self.obj_classification_results = {}
        self.virtual_door_manager.reset()

    def _format_output(self) -> np.ndarray:
        """Format tracked and visible anchored-lost tracks for Ultralytics results."""
        outputs = [track.result for track in self.tracked_stracks if track.is_activated]
        outputs.extend(
            track.synthetic_result
            for track in self.lost_stracks
            if self._is_cls(track, self.obj_cls)
            and track.is_activated
            and track.anchor_enabled
            and track.anchor_mode
            and track.anchor_lost_frames <= self.anchor_track_buffer
        )
        return np.asarray(outputs, dtype=np.float32)
