#!/usr/bin/env python3
"""
YOLOv11 hand-object anchored 视频目标跟踪测试脚本

用法:
    python tools/track_hand_obj_video.py --weight best.pt --video video.mp4 --output result.mp4
    python tools/track_hand_obj_video.py --weight best.pt --video video.mp4 --output result.mp4 --conf 0.5
    python tools/track_hand_obj_video.py --weight best.pt --video video.mp4 --output result.mp4 --anchor_track_buffer 20
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKER = ROOT / "ultralytics/cfg/trackers/handobjtrack.yaml"


EVENT_LABELS = {"take_out": "拿出", "put_in": "放入"}
CHINESE_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)


def _load_overlay_font(size: int = 52):
    """Load a Chinese-capable font for count overlays, returning whether Chinese text is supported."""
    try:
        from PIL import ImageFont
    except Exception:
        return None, False

    for font_path in CHINESE_FONT_CANDIDATES:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size), True
            except Exception:
                continue
    return ImageFont.load_default(), False


def _draw_text_with_background(frame, lines: list[str]) -> None:
    """Draw count lines in the top-left corner of a BGR frame."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        y = 58
        for line in lines:
            line = line.replace("拿出", "Take out").replace("放入", "Put in")
            cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            y += 68
        return

    font, supports_chinese = _load_overlay_font()
    if not supports_chinese:
        lines = [line.replace("拿出", "Take out").replace("放入", "Put in") for line in lines]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    x, y = 24, 24
    padding = 14
    line_height = 68
    width = max(int(draw.textlength(line, font=font)) for line in lines) + padding * 2
    height = line_height * len(lines) + padding
    draw.rounded_rectangle((x - padding, y - padding, x - padding + width, y - padding + height), radius=12, fill=(0, 0, 0))
    for line in lines:
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += line_height
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _draw_virtual_door_mask(frame, door_manager) -> None:
    """Draw locked virtual-door geometry: filled inside-region mask + single horizontal boundary line."""
    if door_manager is None or not getattr(door_manager, "locked", False):
        return

    overlay = frame.copy()
    height, width = frame.shape[:2]
    front_y = getattr(door_manager, "front_y", getattr(door_manager, "front_x", None))
    front_dir = getattr(door_manager, "front_direction", "y_greater_inside")

    # --- front door: fill the entire inside side + one horizontal boundary line ---
    if front_y is not None:
        boundary = int(round(float(front_y)))
        inside_mask_color = (220, 245, 190)   # very light greenish-blue BGR
        if front_dir == "y_less_inside":
            x1, y1, x2, y2 = (0, 0, width - 1, max(0, boundary))
        else:
            x1, y1, x2, y2 = (0, boundary, width - 1, height - 1)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), inside_mask_color, -1)

    # --- side doors: detection boxes themselves ARE the virtual doors, filled only, no border ---
    for door_id, box in sorted(getattr(door_manager, "side_doors", {}).items()):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 160), -1)

    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # --- single horizontal boundary line ---
    if front_y is not None:
        boundary = int(round(float(front_y)))
        cv2.line(frame, (0, boundary), (width - 1, boundary), (0, 0, 255), 4, cv2.LINE_AA)


def overlay_virtual_door_counts(video_path: Path, events: list[dict], door_manager=None) -> None:
    """Overlay cumulative take-out/put-in counts and virtual-door masks on every frame of a saved video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"警告: 无法打开视频进行虚拟门数量叠加: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    suffix = video_path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if suffix == ".mp4" else "XVID"))
    tmp_path = video_path.with_name(f"{video_path.stem}_counts_tmp{video_path.suffix}")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"警告: 无法写入虚拟门数量叠加视频: {tmp_path}")
        return

    events_by_frame: dict[int, list[dict]] = {}
    for event in events or []:
        frame_id = int(event.get("frame_id", 0))
        if frame_id > 0:
            events_by_frame.setdefault(frame_id, []).append(event)

    counts = {"take_out": 0, "put_in": 0}
    persistent_event_lines: list[str] = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        for event in events_by_frame.get(frame_id, []):
            event_name = event.get("event")
            if event_name in counts:
                counts[event_name] += 1
            classification = event.get("classification") or {}
            label = classification.get("label") or "unknown"
            action = "放入" if event_name == "put_in" else "拿出" if event_name == "take_out" else str(event_name)
            persistent_event_lines.append(f"{label} {action}")
            persistent_event_lines = persistent_event_lines[-6:]
        _draw_virtual_door_mask(frame, door_manager)
        lines = [f"拿出: {counts['take_out']}", f"放入: {counts['put_in']}", *persistent_event_lines]
        _draw_text_with_background(frame, lines)
        writer.write(frame)

    cap.release()
    writer.release()
    tmp_path.replace(video_path)


def find_saved_video(output_path: Path, output_dir: Path, predictor_save_dir: Path | None) -> Path | None:
    """Find the fresh video saved by Ultralytics, preferring predictor save_dir over an old output file."""
    search_dirs = []
    for directory in (predictor_save_dir, output_dir):
        if directory is None:
            continue
        directory = Path(directory)
        if directory.exists() and directory not in search_dirs:
            search_dirs.append(directory)

    candidates = []
    for directory in search_dirs:
        candidates.extend(directory.glob("*.avi"))
        candidates.extend(directory.glob("*.mp4"))
    candidates = [p for p in candidates if p.resolve() != output_path.resolve()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return output_path if output_path.exists() else None


def build_tracker_yaml(args, output_dir: Path) -> str:
    """根据命令行参数生成临时tracker配置，未覆盖参数时直接使用默认配置。"""
    overrides = {
        "hand_cls": args.hand_cls,
        "obj_cls": args.obj_cls,
        "anchor_track_buffer": args.anchor_track_buffer,
        "anchor_max_bind_distance": args.anchor_max_bind_distance,
        "anchor_match_thresh": args.anchor_match_thresh,
        "hold_confirm_frames": args.hold_confirm_frames,
        "hold_center_stable_thresh": args.hold_center_stable_thresh,
        "hold_center_in_hand": args.hold_center_in_hand,
        "virtual_door_enabled": args.virtual_door_enabled,
        "virtual_door_model": args.virtual_door_model,
        "virtual_door_conf": args.virtual_door_conf,
        "virtual_door_imgsz": args.virtual_door_imgsz,
        "virtual_door_init_frames": args.virtual_door_init_frames,
        "virtual_door_stability_frames": args.virtual_door_stability_frames,
        "virtual_door_position_tolerance": args.virtual_door_position_tolerance,
        "virtual_door_confirm_frames": args.virtual_door_confirm_frames,
        "virtual_door_front_direction": args.virtual_door_front_direction,
        "obj_classification_enabled": args.obj_classification_enabled,
        "obj_classification_model": args.obj_classification_model,
        "obj_classification_repo": args.obj_classification_repo,
        "obj_classification_device": args.obj_classification_device,
        "obj_classification_target_min_dim": args.obj_classification_target_min_dim,
        "obj_classification_context_scale": args.obj_classification_context_scale,
        "obj_classification_max_context_ratio": args.obj_classification_max_context_ratio,
        "obj_classification_max_frames_per_phase": args.obj_classification_max_frames_per_phase,
        "obj_classification_min_frame_gap": args.obj_classification_min_frame_gap,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if not overrides:
        return str(DEFAULT_TRACKER)

    with open(DEFAULT_TRACKER, "r", encoding="utf-8") as f:
        tracker_cfg = yaml.safe_load(f)
    tracker_cfg.update(overrides)

    runtime_tracker = output_dir / "handobjtrack_runtime.yaml"
    with open(runtime_tracker, "w", encoding="utf-8") as f:
        yaml.safe_dump(tracker_cfg, f, sort_keys=False, allow_unicode=True)
    return str(runtime_tracker)


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 hand-object anchored 视频跟踪")
    parser.add_argument("--weight", type=str, required=True, help="模型权重文件")
    parser.add_argument("--video", type=str, required=True, help="输入视频文件")
    parser.add_argument("--output", type=str, default="output.mp4", help="输出视频文件路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--hand_cls", type=int, default=None, help="hand类别ID，默认读取handobjtrack.yaml")
    parser.add_argument("--obj_cls", type=int, default=None, help="obj类别ID，默认读取handobjtrack.yaml")
    parser.add_argument("--anchor_track_buffer", type=int, default=None, help="obj丢检后由hand锚定传播的最大帧数")
    parser.add_argument("--anchor_max_bind_distance", type=float, default=None, help="新obj绑定hand的最大中心距离像素")
    parser.add_argument("--anchor_match_thresh", type=float, default=None, help="anchored-lost obj重关联匹配阈值")
    parser.add_argument("--hold_confirm_frames", type=int, default=None, help="确认手持所需的连续真实检测帧数")
    parser.add_argument("--hold_center_stable_thresh", type=float, default=None, help="手-物中心距离变化小于该值时认为相对稳定")
    parser.add_argument(
        "--hold_center_in_hand",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否将obj中心在hand框内作为手持证据",
    )
    parser.add_argument(
        "--virtual_door_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用虚拟门put_in/take_out事件检测",
    )
    parser.add_argument("--virtual_door_model", type=str, default=None, help="虚拟门检测模型权重路径")
    parser.add_argument("--virtual_door_conf", type=float, default=None, help="虚拟门初始化检测置信度阈值")
    parser.add_argument("--virtual_door_imgsz", type=int, default=None, help="虚拟门模型推理尺寸")
    parser.add_argument("--virtual_door_init_frames", type=int, default=None, help="虚拟门初始化最大帧数")
    parser.add_argument("--virtual_door_stability_frames", type=int, default=None, help="虚拟门连续稳定确认帧数")
    parser.add_argument("--virtual_door_position_tolerance", type=float, default=None, help="虚拟门位置稳定像素容差")
    parser.add_argument("--virtual_door_confirm_frames", type=int, default=None, help="物体进出状态转换确认帧数")
    parser.add_argument(
        "--virtual_door_front_direction",
        choices=["y_greater_inside", "y_less_inside", "x_less_inside", "x_greater_inside"],
        default=None,
        help="正前方虚拟门内侧方向，默认y_greater_inside；旧x方向选项保留兼容",
    )
    parser.add_argument(
        "--obj_classification_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用轨迹级物体分类并将类别挂到虚拟门事件",
    )
    parser.add_argument("--obj_classification_model", type=str, default=None, help="MobileNetV3分类模型best.pt路径")
    parser.add_argument("--obj_classification_repo", type=str, default=None, help="mobilenetv3.py所在仓库路径")
    parser.add_argument("--obj_classification_device", type=str, default=None, help="分类模型推理设备，如cuda:0或cpu")
    parser.add_argument("--obj_classification_target_min_dim", type=float, default=None, help="obj-centered crop最小边长")
    parser.add_argument("--obj_classification_context_scale", type=float, default=None, help="obj长边上下文扩展倍率")
    parser.add_argument("--obj_classification_max_context_ratio", type=float, default=None, help="crop最大上下文倍率")
    parser.add_argument("--obj_classification_max_frames_per_phase", type=int, default=None, help="每个阶段最多选择的关键帧数")
    parser.add_argument("--obj_classification_min_frame_gap", type=int, default=None, help="分类关键帧最小间隔")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"错误: 视频文件不存在: {args.video}")
        return
    if not Path(args.weight).exists():
        print(f"错误: 权重文件不存在: {args.weight}")
        return
    if not DEFAULT_TRACKER.exists():
        print(f"错误: tracker配置不存在: {DEFAULT_TRACKER}")
        return

    output_path = Path(args.output)
    output_dir = output_path.parent if output_path.parent != Path(".") else Path(".")
    output_name = output_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker_yaml = build_tracker_yaml(args, output_dir)

    print(f"加载模型: {args.weight}")
    model = YOLO(args.weight)

    print(f"开始hand-object anchored跟踪视频: {args.video}")
    print(f"tracker配置: {tracker_yaml}")
    results = model.track(
        source=args.video,
        conf=args.conf,
        tracker=tracker_yaml,
        persist=True,
        save=True,
        project=str(output_dir),
        name="",
        verbose=False,
    )

    predictor_save_dir = Path(model.predictor.save_dir) if getattr(model, "predictor", None) is not None else None
    src_video = find_saved_video(output_path, output_dir, predictor_save_dir)
    dst_video = None
    if src_video is not None:
        dst_video = output_dir / output_name
        if src_video.resolve() != dst_video.resolve():
            shutil.move(str(src_video), str(dst_video))
        print(f"✓ hand-object anchored跟踪完成! 结果已保存到: {dst_video}")
    elif results:
        print(f"警告: 未找到Ultralytics保存的视频文件，请检查: {predictor_save_dir}")

    tracker = getattr(getattr(model, "predictor", None), "trackers", [None])[0]
    events = getattr(tracker, "virtual_door_events", []) or []
    door_manager = getattr(tracker, "virtual_door_manager", None)
    if dst_video is not None and dst_video.exists():
        overlay_virtual_door_counts(dst_video, events, door_manager)
        print(f"✓ 已在视频中叠加虚拟门和数量: 拿出 {sum(e.get('event') == 'take_out' for e in events)}, 放入 {sum(e.get('event') == 'put_in' for e in events)}")
    if events:
        print(f"虚拟门事件数量: {len(events)}")
        for event in events:
            print(event)


if __name__ == "__main__":
    main()
