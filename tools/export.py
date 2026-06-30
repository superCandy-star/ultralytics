from ultralytics import YOLO
model = YOLO("/root/taojianwei/projects/ultralytics/runs/detect/runs/train/exp-12/weights/best.pt")
model.export(format="onnx")