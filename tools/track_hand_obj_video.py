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

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKER = ROOT / "ultralytics/cfg/trackers/handobjtrack.yaml"


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
        choices=["x_less_inside", "x_greater_inside"],
        default=None,
        help="正前方虚拟门内侧方向，按相机视角选择",
    )
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

    if results:
        video_files = list(output_dir.glob("*.avi")) + list(output_dir.glob("*.mp4"))
        video_files = [p for p in video_files if p.resolve() != output_path.resolve()]
        if video_files:
            src_video = video_files[0]
            dst_video = output_dir / output_name
            if src_video.resolve() != dst_video.resolve():
                shutil.move(str(src_video), str(dst_video))
            print(f"✓ hand-object anchored跟踪完成! 结果已保存到: {dst_video}")

    tracker = getattr(getattr(model, "predictor", None), "trackers", [None])[0]
    events = getattr(tracker, "virtual_door_events", None)
    if events:
        print(f"虚拟门事件数量: {len(events)}")
        for event in events:
            print(event)


if __name__ == "__main__":
    main()
