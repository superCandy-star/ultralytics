# 完整训练
python tools/train_dual_task.py \
  --hand_obj_data /root/taojianwei/datasets/IRIS/20260727_datasets_yolo/data.yaml \
  --virtual_door_data /root/taojianwei/datasets/IRIS/auxiliary_datasets/20260729_datasets_yolo/data.yaml \
  --weights /root/taojianwei/projects/ultralytics/weights/yolo11n.pt \
  --phases 1,2,3 --epochs_per_phase 50 --batch 128 --imgsz 320 \
  --num_workers 4 --device 0 --name dual_v2

